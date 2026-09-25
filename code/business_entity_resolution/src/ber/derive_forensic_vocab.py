"""Derive the street/admin vocabulary for ``ber.forensic_features`` from the supplied data.

Uses only the released test TSVs (Source 1, 2 and 3 of each country). No external
list of places, streets or businesses. Rules, per country (all counts are over
distinct folded addresses):

1. **Admin phrases.** A comma component without digits is an admin phrase (city,
   region, departement, state) when it occurs in at least the configured minimum of distinct
   addresses and is not street-like. Street-like: the same phrase also occurs as
   the non-digit residue of a digit-bearing component ("12 rue monge" -> "rue
   monge") in >= 10% as many addresses. Level: a phrase is "region" when at least
   three other admin phrases each occur in >= 5% of its addresses; otherwise "city".
2. **Abbreviations and noise tokens.** Pairs that share an exact key (normalized name + address
   digits; Source 1 key unique in the country) are aligned. When the street tokens
   differ by exactly one token on each side, the substitution record-token ->
   S1-token is counted. A substitution becomes an abbreviation when it occurs
   >= ``--min-abbrev`` times, takes >= 75% of that record token's substitutions,
   and the record token is a same-initial subsequence of the S1 token ("bd" ->
   "boulevard", "r" -> "rue") or, for record tokens of >= 4 letters, within
   Damerau-Levenshtein distance 2 ("aveue" -> "avenue"). The task brief's French
   forms (r, av, bd, pl, chem, all, imp, fbg) are added as seeds when not mined;
   "chem" and "fbg" never occur in French addresses. A token that is inserted (or
   dropped) by the record in >= 50% of its key-pair occurrences and >= 25 times is
   a noise token ("n" of "N° 225") and joins the generic words.
3. **Generic street words.** After abbreviation expansion: street tokens with
   document frequency >= 1% of the country's Source 1 addresses; tokens of <= 2
   letters with frequency >= 0.1%; and positional street types: if one token
   follows the house number in >= 30% of Source 1 addresses (France "5 rue"), every
   token seen in that position in >= 0.1% of addresses; likewise for the last token
   of a component that starts with a number when the top one covers >= 15% (US
   "53 brook avenue").

Run: ``PYTHONPATH=src python -m ber.derive_forensic_vocab --data <dataset/test>
--output resources/forensic_vocab.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
from rapidfuzz.distance import DamerauLevenshtein

from .forensic_features import CountryVocabulary, exact_keys, fold_ascii, normalize_names

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
PARAMETERS = {
    "min_admin": {"France": 500, "US": 100, "India": 100},
    "street_like_ratio": 0.10,
    "region_partner_share": 0.05,
    "region_min_partners": 3,
    "noise_share": 0.50,
    # Listed in the task brief; kept even when a form never occurs in the data
    # (France: "chem" and "fbg" have no address occurrence). Mined forms win.
    "seed_abbreviations": {"France": {"r": "rue", "av": "avenue", "bd": "boulevard", "pl": "place",
                                      "chem": "chemin", "all": "allee", "imp": "impasse",
                                      "fbg": "faubourg"}},
    "min_abbrev": 25,
    "abbrev_dominance": 0.75,
    "generic_df": 0.01,
    "short_generic_df": 0.001,
    "short_generic_max_len": 2,
    "positional_top_share": {"after_number": 0.30, "last_token": 0.15},
    "positional_df": 0.001,
}


def read_country(path: Path, country: str, columns=("business_name", "business_address")) -> pa.Table:
    reader = pacsv.open_csv(
        path, read_options=pacsv.ReadOptions(block_size=64 << 20),
        parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
        convert_options=pacsv.ConvertOptions(column_types={c: pa.string() for c in COLUMNS},
                                             include_columns=list(columns) + ["country"],
                                             strings_can_be_null=False))
    parts = []
    for batch in reader:
        table = pa.Table.from_batches([batch])
        parts.append(table.filter(pc.equal(table["country"], country)).select(list(columns)))
    return pa.concat_tables(parts)


def file_identity(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def _has_digit(text: str) -> bool:
    return any(ch.isdigit() for ch in text)


def component_counts(addresses: set[str]) -> tuple[Counter, Counter]:
    free, residual = Counter(), Counter()
    for address in addresses:
        seen_free, seen_residual = set(), set()
        for component in address.split(","):
            if not component:
                continue
            if _has_digit(component):
                rest = " ".join(w for w in component.split() if not _has_digit(w))
                if rest:
                    seen_residual.add(rest)
            else:
                seen_free.add(component)
        free.update(seen_free)
        residual.update(seen_residual)
    return free, residual


def admin_levels(addresses: set[str], admin: set[str], share: float, min_partners: int) -> dict[str, str]:
    """"region" when >= ``min_partners`` other admin phrases each occur in >= ``share``
    of its addresses: a city co-occurs with its region and departement (2), a region
    or departement with several cities. Heuristic; Pas-de-Calais holds only Calais
    in this data and stays "city".
    """
    count, pair = Counter(), Counter()
    for address in addresses:
        present = sorted({c for c in address.split(",") if c in admin})
        count.update(present)
        for i, a in enumerate(present):
            for b in present[i + 1:]:
                pair[(a, b)] += 1
    partners = defaultdict(set)
    for (a, b), n in pair.items():
        if n >= share * count[a]:
            partners[a].add(b)
        if n >= share * count[b]:
            partners[b].add(a)
    return {a: ("region" if len(partners[a]) >= min_partners else "city") for a in sorted(admin)}


def _is_subsequence_abbreviation(short: str, full: str) -> bool:
    if not short or len(short) >= len(full) or short[0] != full[0]:
        return False
    it = iter(full)
    return all(ch in it for ch in short)


def mine_abbreviations(s1: pa.Table, records: list[pa.Table], vocab: CountryVocabulary,
                       min_count: int, dominance: float, noise_share: float,
                       seeds: dict) -> tuple[dict, dict, dict, int]:
    s1_keys = exact_keys(s1["business_name"], s1["business_address"])
    s1_frame = pd.DataFrame({"key": s1_keys, "s1_address": fold_ascii(
        s1["business_address"], keep_commas=True).to_numpy(zero_copy_only=False)})
    s1_frame = s1_frame[s1_frame.key != ""]
    s1_frame = s1_frame[~s1_frame.key.duplicated(keep=False)]
    substitutions = Counter()
    per_record_token = Counter()
    inserted, deleted, in_record, in_s1 = Counter(), Counter(), Counter(), Counter()
    key_pairs = 0
    for table in records:
        frame = pd.DataFrame({"key": exact_keys(table["business_name"], table["business_address"]),
                              "address": fold_ascii(table["business_address"], keep_commas=True)
                              .to_numpy(zero_copy_only=False)})
        joined = frame.merge(s1_frame, on="key", how="inner")
        key_pairs += len(joined)
        for left, right in zip(joined.s1_address.tolist(), joined.address.tolist()):
            a, _ = vocab.parse(left, False)
            b, _ = vocab.parse(right, False)
            in_s1.update(a)
            in_record.update(b)
            only_a, only_b = a - b, b - a
            if len(only_a) == 1 and len(only_b) == 1:
                (ta,), (tb,) = only_a, only_b
                substitutions[(tb, ta)] += 1
                per_record_token[tb] += 1
            elif only_b and not only_a:
                inserted.update(only_b)
            elif only_a and not only_b:
                deleted.update(only_a)
    # Tokens a vendor inserts (or drops) in at least half of their key-pair
    # occurrences carry no street identity ("n" of "N° 225", "bp").
    noise = {}
    for counter, base, kind in ((inserted, in_record, "inserted"), (deleted, in_s1, "deleted")):
        for token, n in counter.items():
            if n >= min_count and n >= noise_share * base[token] and not token.isdigit():
                noise[token] = {"kind": kind, "count": n, "share": round(n / base[token], 4)}
    mapping, evidence = {}, {}
    for (short, full), n in substitutions.most_common():
        if n < min_count:
            break
        share = n / per_record_token[short]
        if short in mapping or share < dominance or short.isdigit():
            continue
        abbreviation = _is_subsequence_abbreviation(short, full)
        typo = len(short) >= 4 and DamerauLevenshtein.distance(short, full) <= 2
        if abbreviation or typo:
            mapping[short] = full
            evidence[short] = {"to": full, "count": n, "share": round(share, 4),
                               "kind": "abbreviation" if abbreviation else "typo"}
    for short, full in seeds.items():
        if short not in mapping:
            mapping[short] = full
            evidence[short] = {"to": full, "count": int(sum(n for (s, _), n in substitutions.items() if s == short)),
                               "share": None, "kind": "task_seed"}
    # No chains: a target is never itself rewritten.
    mapping = {s: f for s, f in mapping.items() if f not in mapping}
    noise = {t: v for t, v in noise.items() if t not in mapping}
    return mapping, {s: evidence[s] for s in mapping}, noise, key_pairs


def generic_tokens(s1_addresses: set[str], vocab: CountryVocabulary, params: dict) -> tuple[set, dict]:
    df, after, last = Counter(), Counter(), Counter()
    n = 0
    abbreviations = vocab.abbreviations
    for address in s1_addresses:
        n += 1
        street, _ = vocab.parse(address, False)
        df.update(street)
        seen_after, seen_last = set(), set()
        for component in address.split(","):
            if not component or component in vocab.admin:
                continue
            words = [abbreviations.get(w, w) for w in component.split()]
            for i in range(len(words) - 1):
                if _has_digit(words[i]) and not _has_digit(words[i + 1]):
                    seen_after.add(words[i + 1])
                    break
            if len(words) >= 2 and _has_digit(words[0]) and not _has_digit(words[-1]):
                seen_last.add(words[-1])
        after.update(seen_after)
        last.update(seen_last)
    generic = {t for t, c in df.items() if c >= params["generic_df"] * n}
    generic |= {t for t, c in df.items() if len(t) <= params["short_generic_max_len"]
                and c >= params["short_generic_df"] * n}
    positional = {}
    for name, counter in (("after_number", after), ("last_token", last)):
        top_token, top = counter.most_common(1)[0] if counter else ("", 0)
        used = top >= params["positional_top_share"][name] * n
        tokens = sorted(t for t, c in counter.items() if c >= params["positional_df"] * n) if used else []
        generic |= set(tokens)
        positional[name] = {"top_token": top_token, "top_share": round(top / max(n, 1), 4),
                            "applied": used, "tokens": tokens}
    evidence = {"s1_addresses": n, "document_frequency": {t: round(df[t] / n, 5) for t in sorted(generic) if t in df},
                "positional": positional}
    return generic, evidence


def derive_country(data: Path, country: str, params: dict, log) -> dict:
    started = time.monotonic()
    s1 = read_country(data / "test_source1.tsv", country)
    records = [read_country(data / f"test_source{i}.tsv", country) for i in (2, 3)]
    s1_folded = set(fold_ascii(s1["business_address"], keep_commas=True).to_pylist())
    all_folded = set(s1_folded)
    for table in records:
        all_folded.update(fold_ascii(table["business_address"], keep_commas=True).to_pylist())
    free, residual = component_counts(all_folded)
    threshold = params["min_admin"].get(country, min(params["min_admin"].values()))
    admin = {p for p, c in free.items()
             if c >= threshold and residual.get(p, 0) < params["street_like_ratio"] * c}
    rejected_street_like = sorted(p for p, c in free.items()
                                  if c >= threshold and p not in admin)
    levels = admin_levels(all_folded, admin, params["region_partner_share"],
                          params["region_min_partners"])
    log(f"{country}: {len(s1)} S1, {sum(len(t) for t in records)} records, "
        f"{len(all_folded)} distinct addresses, {len(admin)} admin phrases")
    base = CountryVocabulary(admin=frozenset(admin))
    abbreviations, abbrev_evidence, noise, key_pairs = mine_abbreviations(
        s1, records, base, params["min_abbrev"], params["abbrev_dominance"], params["noise_share"],
        params["seed_abbreviations"].get(country, {}))
    log(f"{country}: {key_pairs} key pairs, {len(abbreviations)} abbreviations, {len(noise)} noise tokens")
    # Admin phrases also in their expanded spelling ("st nazaire" -> "saint nazaire").
    expanded = {" ".join(abbreviations.get(w, w) for w in p.split()) for p in admin}
    for p in expanded - admin:
        levels[p] = levels.get(next(a for a in admin if " ".join(abbreviations.get(w, w) for w in a.split()) == p), "city")
    admin |= expanded
    with_abbrev = CountryVocabulary(admin=frozenset(admin), abbreviations=abbreviations)
    generic, generic_evidence = generic_tokens(s1_folded, with_abbrev, params)
    generic |= set(noise)
    log(f"{country}: {len(generic)} generic tokens ({time.monotonic() - started:.0f}s)")
    return {
        "admin_phrases": sorted(admin),
        "admin_level": {p: levels[p] for p in sorted(admin)},
        "abbreviations": dict(sorted(abbreviations.items())),
        "generic_tokens": sorted(generic),
        "evidence": {
            "admin_min_distinct_addresses": threshold,
            "admin_counts": {p: free.get(p, 0) for p in sorted(admin)},
            "rejected_street_like": {p: [free[p], residual.get(p, 0)] for p in rejected_street_like[:200]},
            "exact_key_pairs_mined": key_pairs,
            "abbreviations": abbrev_evidence,
            "noise_tokens": noise,
            "generic": generic_evidence,
        },
    }


def parse_countries(raw: str) -> list[str]:
    countries = [country.strip() for country in raw.split(",")]
    known = set(PARAMETERS["min_admin"])
    if (any(not country for country in countries) or len(set(countries)) != len(countries)
            or any(country not in known for country in countries)):
        raise ValueError(f"--countries must be unique values from {sorted(known)}")
    return countries


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dataset/test directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--countries", default="France,US,India")
    args = parser.parse_args()

    def log(message):
        print(message, flush=True)

    countries = {c: derive_country(args.data, c, PARAMETERS, log) for c in parse_countries(args.countries)}
    payload = {
        "schema": 1,
        "generated_by": "ber.derive_forensic_vocab",
        "inputs": [file_identity(args.data / f"test_source{i}.tsv") for i in (1, 2, 3)],
        "parameters": PARAMETERS,
        "countries": countries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=True) + "\n")
    log(f"wrote {args.output}")


if __name__ == "__main__":
    main()
