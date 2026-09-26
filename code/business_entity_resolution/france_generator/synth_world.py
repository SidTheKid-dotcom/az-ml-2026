"""Synthetic labelled world built from real Source 1 records (reference implementation).

Purpose
    France has no labels. This module copies the dataset generator's behaviour, as measured
    on train (generator_params.json), onto real Source 1 records and writes a complete,
    labelled world in the official file format:

        <out>/s1.tsv  <out>/s2.tsv  <out>/s3.tsv  <out>/gt.tsv

    Every Source 1 record gets 0..11 noisy true copies split across S2/S3, and the pool also
    receives look-alike decoys (same core name + different number, same address + different
    name, same template + different street). All Source 1 records compete for all records, so
    the world is competitor-complete like the arena.

Uses
    1. Fidelity gate: build a world from US/India train S1 records, run the unchanged pipeline,
       compare its score and score-band densities with the real arena (B_REST).
    2. France check set: build a world from French test S1 records; the current French rule
       should score close to the implied public France score (~0.941).
    3. France training data: labelled French pairs (copies = positive, decoys and other
       S1s' copies = negative) for continued reranker training with US/India replay.

Rules
    Uses only provided data and rates measured on provided data. No external lists: French
    vocabularies (legal forms, street abbreviations, added words) come from the release text.
    Deterministic given --seed. Keep this file in the final code ZIP if its outputs are used.

Usage
    python synth_world.py --s1 test_source1.tsv --country France --params generator_params.json \
        --out synth_fr --seed 1 [--limit 50000]
"""
from __future__ import annotations

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

import pandas as pd

LEGAL_FR = ["sarl", "sas", "eurl", "sa", "sasu", "sci", "snc", "association"]
LEGAL_EN = ["inc", "llc", "corp", "corporation", "co", "company", "ltd", "limited", "lp",
            "pvt ltd", "private limited", "llp", "incorporated"]
LEGAL_ALL = set(w for x in LEGAL_FR + LEGAL_EN for w in x.split())
ADDED_FR = ["groupe", "holding", "developpement", "participations", "distribution",
            "international", "services", "associes", "et fils"]
ADDED_EN = ["center", "services", "partners", "service", "enterprises", "trading", "labs"]
ABBREV_FR = {"rue": ["R.", "R"], "avenue": ["Av.", "AV"], "boulevard": ["Bd", "BD"],
             "impasse": ["Imp.", "IMP."], "place": ["Pl.", "PL"], "chemin": ["Ch.", "CH"],
             "allee": ["All.", "ALL"], "route": ["Rte", "RTE"]}
ABBREV_EN = {"road": ["Rd"], "street": ["St"], "avenue": ["Ave"], "drive": ["Dr"],
             "lane": ["Ln"], "court": ["Ct"], "boulevard": ["Blvd"], "place": ["Pl"]}
KEYBOARD = {c: n for c, n in zip("qwertyuiopasdfghjklzxcvbnm",
            ["wa", "qes", "wrd", "etf", "ryg", "tuh", "yij", "uok", "ipl", "o", "qsz", "awdx",
             "sefc", "drgv", "fthb", "gyjn", "hukm", "jil", "ko", "asx", "zdc", "xfv", "cgb",
             "vhn", "bjm", "nk"])}


def fold(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()


class Rates:
    """Operation probabilities for one country x source, with France fallbacks.

    Word-level name and digit operations are taken from US/India train (the generator is
    country-independent on every statistic measured); case, abbreviation, component and
    legal-form operations come from French key-linked pairs and French text frequencies.
    """

    def __init__(self, params: dict, country: str, source: str):
        base_country = "US" if country == "France" else country
        self.p = dict(params["copy_ops"][f"{base_country}|{source}"])
        if country == "France":
            fr = params["france_copy_ops_keypairs"][f"France|{source}"]
            for k in ("addr_upper", "abbrev_swap", "components_reordered", "component_dropped",
                      "addr_word_typo", "name_upper", "name_lower", "suffix_added",
                      "suffix_dropped", "suffix_swapped", "name_double_space", "name_bracket"):
                self.p[k] = fr.get(k, 0.0)
            for k in ("addr_junk_token", "name_junk_token", "name_domain_style", "name_leet_0",
                      "name_native_script", "addr_native_script", "landmark_added"):
                self.p[k] = 0.0
        self.country, self.source = country, source

    def __call__(self, op: str) -> float:
        return float(self.p.get(op, 0.0))


def typo(word: str, rng: random.Random, heavy: bool = False) -> str:
    edits = rng.randint(2, 3) if heavy else 1
    w = list(word)
    for _ in range(edits):
        if len(w) < 3:
            break
        i = rng.randrange(1, len(w))
        kind = rng.choice(["sub", "ins", "del", "swap"])
        ch = w[i].lower()
        if kind == "sub":
            w[i] = rng.choice(KEYBOARD.get(ch, "e"))
        elif kind == "ins":
            w.insert(i, w[i])
        elif kind == "del":
            del w[i]
        elif i < len(w) - 1:
            w[i], w[i + 1] = w[i + 1], w[i]
    return "".join(w)


def split_legal(name: str):
    toks = name.split()
    core = [t for t in toks if fold(t.lower().strip(".,")) not in LEGAL_ALL]
    legal = [t for t in toks if fold(t.lower().strip(".,")) in LEGAL_ALL]
    return core, legal


def noisy_name(name: str, r: Rates, rng: random.Random) -> str:
    core, legal = split_legal(name)
    forms = LEGAL_FR if r.country == "France" else LEGAL_EN
    if legal and rng.random() < r("suffix_dropped"):
        legal = []
    elif legal and rng.random() < r("suffix_swapped"):
        legal = [rng.choice(forms).upper()]
    elif not legal and rng.random() < r("suffix_added"):
        legal = [rng.choice(forms).upper()]
    if legal and r.country == "France" and legal[0].lower() == "sas" and rng.random() < 0.15:
        legal = ["S.A.S"]
    words = core[:]
    if len(words) >= 2 and rng.random() < r("word_reorder"):
        i = rng.randrange(len(words) - 1)
        words[i], words[i + 1] = words[i + 1], words[i]
    if words and rng.random() < r("word_typo"):
        j = max(range(len(words)), key=lambda k: len(words[k]) + rng.random())
        words[j] = typo(words[j], rng)
    if words and rng.random() < r("word_substituted"):
        j = rng.randrange(len(words))
        words[j] = typo(words[j], rng, heavy=True)
    if len(words) >= 3 and rng.random() < r("word_dropped"):
        del words[rng.randrange(1, len(words))]
    if rng.random() < r("word_added"):
        vocab = ADDED_FR if r.country == "France" else ADDED_EN
        w = rng.choice(vocab).title()
        pos = rng.choices(["end", "start", "mid"], weights=[0.56, 0.21, 0.23])[0]
        if pos == "end":
            words.append(w)
        elif pos == "start":
            words.insert(0, w)
        else:
            words.insert(rng.randrange(0, len(words) + 1), w)
    if words and rng.random() < r("word_doubled"):
        j = rng.randrange(len(words))
        words.insert(j, words[j])
    out = " ".join(words + legal)
    if r.country == "France" and rng.random() < 0.10:  # accent injection seen in French records
        out = re.sub(r"\bEcole\b", "École", out)
        out = re.sub(r"^E", "É", out)
    if rng.random() < r("name_bracket") and legal:
        out = " ".join(words) + " [" + " ".join(legal) + "]"
    if rng.random() < r("name_double_space"):
        out = out.replace(" ", "  ", 1)
    if rng.random() < r("name_upper"):
        out = out.upper()
    elif rng.random() < r("name_lower"):
        out = out.lower()
    return out


def noisy_address(addr: str, r: Rates, rng: random.Random) -> str:
    if rng.random() < r("addr_empty"):
        return ""
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    abbrev = ABBREV_FR if r.country == "France" else ABBREV_EN
    if rng.random() < r("abbrev_swap"):
        for full, short in abbrev.items():
            pat = re.compile(rf"\b{full}\b", re.I)
            if any(pat.search(fold(p)) for p in parts):
                parts = [pat.sub(rng.choice(short), fold(p)) if r.country != "France" else
                         re.sub(rf"\b{full}\b", rng.choice(short), p, flags=re.I) for p in parts]
                break
    if parts and rng.random() < r("addr_word_typo"):
        j = rng.randrange(len(parts))
        ws = parts[j].split()
        cand = [k for k, w in enumerate(ws) if len(w) >= 4 and not any(c.isdigit() for c in w)]
        if cand:
            k = rng.choice(cand)
            ws[k] = typo(ws[k], rng)
            parts[j] = " ".join(ws)
    digits_op = rng.random()
    cut = [r("digits_all_dropped"), r("digits_partly_dropped"), r("digit_one_changed"), r("digits_reordered")]
    acc = 0.0
    for kind, pr in zip(["all", "part", "one", "reorder"], cut):
        acc += pr
        if digits_op < acc:
            parts = [_digit_edit(p, kind, rng) for p in parts]
            break
    if len(parts) >= 3 and rng.random() < r("component_dropped"):
        del parts[rng.randrange(1, len(parts))]
    if len(parts) >= 2 and rng.random() < r("components_reordered"):
        rng.shuffle(parts)
    out = ", ".join(parts)
    if r.country == "France" and rng.random() < 0.3:  # records carry fewer accents than S1
        out = fold(out)
    if rng.random() < r("addr_upper"):
        out = out.upper()
    return out


def _digit_edit(part: str, kind: str, rng: random.Random) -> str:
    if kind == "all":
        return re.sub(r"\d", "", part).strip()
    nums = list(re.finditer(r"\d+", part))
    if not nums:
        return part
    m = rng.choice(nums)
    s = m.group(0)
    if kind == "part" and len(s) >= 2:
        i = rng.randrange(len(s))
        s = s[:i] + s[i + 1:]
    elif kind == "one":
        i = rng.randrange(len(s))
        s = s[:i] + str((int(s[i]) + rng.randint(1, 9)) % 10) + s[i + 1:]
    elif kind == "reorder" and len(s) >= 2:
        s = s[::-1]
    return part[:m.start()] + s + part[m.end():]


def decoy(row, peers_by_city, template_words, rng: random.Random):
    """One look-alike built the way train decoys differ from their anchor S1.

    Types and weights follow the anchored train decoys (same core name ≈ 3–5x same address):
      A: same core name, house number changed, legal form usually changed.
      B: same address, different distinctive name word (template swap).
      C: same name template, house number kept, street taken from another S1 in the city.
    """
    name, addr = row.business_name, row.business_address
    kind = rng.choices(["A", "B", "C"], weights=[0.6, 0.2, 0.2])[0]
    core, legal = split_legal(name)
    forms = LEGAL_FR if row.country == "France" else LEGAL_EN
    if kind in ("A", "C"):
        if rng.random() < 0.9:
            legal = [rng.choice(forms).upper()] if (rng.random() < 0.5 or not legal) else []
        name2 = " ".join(core + legal)
        if kind == "A":
            addr2 = re.sub(r"\d+", lambda m: str(max(1, int(m.group(0)) + rng.choice([-7, -3, -2, -1, 1, 2, 5, 11, 20]))), addr, count=1)
        else:
            city = addr.split(",")[-2].strip() if addr.count(",") >= 2 else ""
            other = peers_by_city.get(city)
            if not other:
                return None
            o = rng.choice(other)
            num = re.match(r"\s*\d+", addr)
            street = re.sub(r"^\s*\d+\s*", "", o.split(",")[0])
            addr2 = (num.group(0).strip() + " " if num else "") + street + "," + ",".join(addr.split(",")[1:])
    else:
        if len(core) < 2 or not template_words:
            return None
        j = rng.randrange(len(core))
        core = core[:j] + [rng.choice(template_words)] + core[j + 1:]
        name2, addr2 = " ".join(core + legal), addr
    return name2, addr2, kind


def sample_count(params, rng):
    d = params["counts"]["dist"]
    k = rng.choices(list(d), weights=list(d.values()))[0]
    return 6 + min(int(rng.expovariate(0.9)), 5) if k == "6+" else int(k)


def build(s1: pd.DataFrame, params: dict, seed: int):
    rng = random.Random(seed)
    country = s1.country.iloc[0]
    rates = {src: Rates(params, country, src) for src in ("S2", "S3")}
    peers_by_city, words = {}, {}
    for a in s1.business_address:
        if a.count(",") >= 2:
            peers_by_city.setdefault(a.split(",")[-2].strip(), []).append(a)
    for n in s1.business_name:
        for w in split_legal(n)[0][1:]:
            words[w] = words.get(w, 0) + 1
    template = [w for w, c in words.items() if c >= 50]
    rows = {"S2": [], "S3": []}
    gt, next_id = [], {"S2": 10**9, "S3": 10**9}
    decoys_per_s1 = params["counts"]["decoys_per_S1_France_implied"]
    for row in s1.itertuples(index=False):
        k = sample_count(params, rng)
        ids = []
        for _ in range(k):
            src = "S2" if rng.random() < 0.49 else "S3"
            rid = f"{src}-{next_id[src]}"; next_id[src] += 1
            rows[src].append((rid, noisy_name(row.business_name, rates[src], rng),
                              noisy_address(row.business_address, rates[src], rng), row.country, "copy", row.entity_id))
            ids.append(rid)
        gt.append((row.entity_id, ",".join(ids)))
        n_dec = int(decoys_per_s1) + (rng.random() < decoys_per_s1 % 1)
        for _ in range(n_dec):
            made = decoy(row, peers_by_city, template, rng)
            if made is None:
                continue
            src = "S2" if rng.random() < 0.49 else "S3"
            rid = f"{src}-{next_id[src]}"; next_id[src] += 1
            n2, a2, kind = made
            rows[src].append((rid, noisy_name(n2, rates[src], rng), noisy_address(a2, rates[src], rng),
                              row.country, f"decoy_{kind}", row.entity_id))
    cols = ["entity_id", "business_name", "business_address", "country", "_kind", "_anchor"]
    return ({src: pd.DataFrame(v, columns=cols).sample(frac=1, random_state=seed) for src, v in rows.items()},
            pd.DataFrame(gt, columns=["source1_entity_id", "matched_entity_ids"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True); ap.add_argument("--country", required=True)
    ap.add_argument("--params", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    params = json.loads(Path(a.params).read_text())
    s1 = pd.read_csv(a.s1, sep="\t", dtype=str, keep_default_na=False)
    s1 = s1[s1.country == a.country]
    if a.limit:
        s1 = s1.sample(a.limit, random_state=a.seed)
    pools, gt = build(s1, params, a.seed)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    s1.to_csv(out / "s1.tsv", sep="\t", index=False)
    for src, df in pools.items():
        df.iloc[:, :4].to_csv(out / f"{src.lower()}.tsv", sep="\t", index=False)
        df[["entity_id", "_kind", "_anchor"]].to_csv(out / f"{src.lower()}_provenance.tsv", sep="\t", index=False)
    gt.to_csv(out / "gt.tsv", sep="\t", index=False)
    print({"s1": len(s1), "s2": len(pools["S2"]), "s3": len(pools["S3"]),
           "true_links": int(gt.matched_entity_ids.str.count("S").sum())})


if __name__ == "__main__":
    main()
