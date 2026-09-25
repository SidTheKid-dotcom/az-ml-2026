"""Label-free diagnostics for any matching_results.tsv (negative-side manual section 3).

Per country: rows, empty rate, mean list length, longest list, lists >= 9, contested
record IDs (IDs in more than one list) and entities holding one, S2+S3 co-occurrence
among non-empty lists, exact-key agreement (share of unique-key links present, key
links whose record is given to a different S1, key links absent everywhere), and
for France the street-mismatch rate among accepted links: links whose names share a
template (equal normalized names, or equal length >= 2 tokens differing in one
token) but whose street tokens disagree (both non-empty, Jaccard 0 and sorted-token
ratio < 0.6; the looser Jaccard-0-only rate is reported too).

Key: normalized name + all address digits in order (``ber.forensic_features``); a
key link pairs a Source 1 whose key is unique in its country with every S2/S3
record carrying the same key. Uses only the supplied test TSVs; no labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv

from .forensic_features import (StreetVocabulary, default_vocabulary, exact_keys,
                                forensic_features, normalize_names, template_similar)

COLUMNS = ["entity_id", "business_name", "business_address", "country"]
COUNTRIES = ("France", "India", "US")
STREET_RATIO = 0.6


def id_to_int(ids: pa.Array) -> np.ndarray:
    """"S2-566025912" -> 2 * 10**12 + 566025912 (unique across sources)."""
    ids = pc.cast(ids, pa.string())
    source = pc.cast(pc.utf8_slice_codeunits(ids, 1, 2), pa.int64()).to_numpy()
    number = pc.cast(pc.utf8_slice_codeunits(ids, 3), pa.int64()).to_numpy()
    return source * 10**12 + number


def _key_hash(country: np.ndarray, keys: np.ndarray) -> np.ndarray:
    joined = pd.Series(country, dtype=object) + "\x1f" + pd.Series(keys, dtype=object)
    hashed = pd.util.hash_pandas_object(joined, index=False).to_numpy()
    hashed[keys == ""] = 0
    return hashed


def read_source(path: Path, keep_text_for: str | None = "France", block: int = 64 << 20) -> dict:
    """Stream a TSV; return int ids, country codes, key hashes, and text for one country."""
    reader = pacsv.open_csv(path, read_options=pacsv.ReadOptions(block_size=block),
                            parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                            convert_options=pacsv.ConvertOptions(
                                column_types={c: pa.string() for c in COLUMNS}, strings_can_be_null=False))
    ids, countries, hashes, text = [], [], [], []
    for batch in reader:
        table = pa.Table.from_batches([batch])
        country = table["country"].to_numpy()
        keys = exact_keys(table["business_name"], table["business_address"])
        ids.append(id_to_int(table["entity_id"].combine_chunks()))
        countries.append(country.astype(object))
        hashes.append(_key_hash(country, keys))
        if keep_text_for:
            mask = country == keep_text_for
            if mask.any():
                text.append(table.filter(pa.array(mask)).select(["entity_id", "business_name", "business_address"])
                            .to_pandas())
    out = {"id": np.concatenate(ids), "country": np.concatenate(countries), "key": np.concatenate(hashes)}
    out["text"] = pd.concat(text, ignore_index=True) if text else pd.DataFrame(columns=COLUMNS[:3])
    return out


def load_matching(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S1 int ids per row, link row index, linked record int ids)."""
    table = pacsv.read_csv(path, parse_options=pacsv.ParseOptions(delimiter="\t", quote_char=False),
                           convert_options=pacsv.ConvertOptions(column_types={"source1_entity_id": pa.string(),
                                                                              "matched_entity_ids": pa.string()},
                                                                strings_can_be_null=False))
    s1 = id_to_int(table["source1_entity_id"].combine_chunks())
    lists = pc.split_pattern(pc.utf8_rtrim(table["matched_entity_ids"].combine_chunks(), characters="\r"), ",")
    flat = pc.list_flatten(lists)
    parents = pc.list_parent_indices(lists).to_numpy()
    nonempty = pc.not_equal(flat, "").to_numpy(zero_copy_only=False)
    return s1, parents[nonempty], id_to_int(flat.filter(pa.array(nonempty)))


def diagnose(matching: Path, s1: dict, records: dict, vocab: StreetVocabulary) -> dict:
    started = time.monotonic()
    s1_ids, link_row, link_rec = load_matching(matching)
    order = np.argsort(s1["id"])
    pos = np.searchsorted(s1["id"], s1_ids, sorter=order)
    if np.any(pos >= len(order)) or np.any(s1["id"][order[np.minimum(pos, len(order) - 1)]] != s1_ids):
        raise ValueError("matching file contains Source 1 IDs not in test_source1")
    row_s1 = order[pos]                      # row -> index into s1 arrays
    if len(np.unique(row_s1)) != len(row_s1) or len(row_s1) != len(s1["id"]):
        raise ValueError("matching file must have exactly one row per Source 1")
    rec_order = np.argsort(records["id"])
    rpos = np.searchsorted(records["id"], link_rec, sorter=rec_order)
    if np.any(rpos >= len(rec_order)) or np.any(records["id"][rec_order[np.minimum(rpos, len(rec_order) - 1)]] != link_rec):
        raise ValueError("matching file links unknown S2/S3 IDs")
    link_s1 = row_s1[link_row]
    link_r = rec_order[rpos]
    pair_code = link_s1.astype(np.int64) * len(records["id"]) + link_r
    if len(np.unique(pair_code)) != len(pair_code):
        raise ValueError("matching file contains duplicate S1-record links")
    if np.any(records["country"][link_r] != s1["country"][link_s1]):
        cross = int(np.sum(records["country"][link_r] != s1["country"][link_s1]))
    else:
        cross = 0
    n_s1 = len(s1["id"])
    lengths = np.bincount(link_s1, minlength=n_s1)
    rec_claims = np.bincount(link_r, minlength=len(records["id"]))
    contested_link = rec_claims[link_r] > 1
    holds_contested = np.bincount(link_s1[contested_link], minlength=n_s1) > 0
    source = records["id"][link_r] // 10**12
    has_s2 = np.bincount(link_s1[source == 2], minlength=n_s1) > 0
    has_s3 = np.bincount(link_s1[source == 3], minlength=n_s1) > 0

    # Exact-key links: unique Source 1 key in its country (hash includes country).
    s1_key = s1["key"]
    valid = s1_key != 0
    uniq, inverse, counts = np.unique(s1_key[valid], return_inverse=True, return_counts=True)
    unique_s1 = np.flatnonzero(valid)[counts[inverse] == 1]
    rk = records["key"]
    rk_order = np.argsort(rk)
    rk_sorted = rk[rk_order]
    lo = np.searchsorted(rk_sorted, s1_key[unique_s1], side="left")
    hi = np.searchsorted(rk_sorted, s1_key[unique_s1], side="right")
    n_hits = hi - lo
    key_s1 = np.repeat(unique_s1, n_hits)
    key_r = rk_order[np.concatenate([np.arange(a, b) for a, b in zip(lo[n_hits > 0], hi[n_hits > 0])])
                     if (n_hits > 0).any() else np.empty(0, dtype=np.int64)]
    link_code = link_s1.astype(np.int64) * (1 << 32) + link_r
    key_code = key_s1.astype(np.int64) * (1 << 32) + key_r
    link_code.sort()
    present = np.isin(key_code, link_code, assume_unique=False)
    claimed = rec_claims[key_r] > 0
    elsewhere = ~present & claimed
    missing = ~present & ~claimed

    report = {"file": str(matching), "sha256": _sha256(matching), "rows": int(n_s1),
              "links": int(len(link_r)), "cross_country_links": cross, "countries": {}}
    for country in COUNTRIES:
        m = s1["country"] == country
        lm = s1["country"][link_s1] == country
        km = s1["country"][key_s1] == country
        nonempty = m & (lengths > 0)
        rec_c = link_r[lm]
        uniq_rec, rec_counts = np.unique(rec_c, return_counts=True)
        report["countries"][country] = {
            "rows": int(m.sum()),
            "empty_rate": float(np.mean(lengths[m] == 0)) if m.any() else None,
            "mean_list_length": float(lengths[m].mean()) if m.any() else None,
            "mean_list_length_nonempty": float(lengths[nonempty].mean()) if nonempty.any() else None,
            "longest_list": int(lengths[m].max()) if m.any() else None,
            "lists_ge_9": int(np.sum(lengths[m] >= 9)),
            "links": int(lm.sum()),
            "contested_record_ids": int(np.sum(rec_counts > 1)),
            "contested_extra_claims": int(np.sum(rec_counts[rec_counts > 1] - 1)),
            "entities_holding_contested_id": int(np.sum(holds_contested[m])),
            "entities_holding_contested_id_share": float(np.mean(holds_contested[m])) if m.any() else None,
            "s2_s3_cooccurrence_nonempty": float(np.mean((has_s2 & has_s3)[nonempty])) if nonempty.any() else None,
            "exact_key": {
                "unique_key_s1": int(np.sum(m[unique_s1])),
                "key_links": int(km.sum()),
                "entities_with_key_link": int(len(np.unique(key_s1[km]))),
                "present_share": float(np.mean(present[km])) if km.any() else None,
                "given_to_different_s1": int(np.sum(elsewhere[km])),
                "absent_everywhere": int(np.sum(missing[km])),
                "key_hits_on_empty_lists": int(np.sum(lengths[key_s1[km]] == 0)),
            },
        }
    report["countries"]["France"]["street_mismatch"] = france_street_mismatch(link_s1, link_r, s1, records, vocab)
    report["heuristic_reference_ranges"] = {
        "scope": "structural diagnostics only; not evidence of French accuracy",
        "contested_record_ids": 0, "longest_list": "<= 11", "empty_rate": "about 5.6-6.0%",
        "mean_list_length": "about 3.24-3.30 at v1 precision", "s2_s3_cooccurrence": "about 80%",
        "exact_key_present": ">= 97%"}
    report["elapsed_seconds"] = round(time.monotonic() - started, 1)
    return report


def france_street_mismatch(link_s1, link_r, s1: dict, records: dict, vocab: StreetVocabulary) -> dict:
    fr = s1["country"][link_s1] == "France"
    ls, lr = link_s1[fr], link_r[fr]
    a = _text_rows(s1["text"], s1["id"][ls])
    b = _text_rows(records["text"], records["id"][lr])
    an, bn = normalize_names(a.business_name), normalize_names(b.business_name)
    templ = np.fromiter((template_similar(x, y) for x, y in zip(an, bn)), dtype=bool, count=len(an))
    feats = forensic_features(a.business_name.to_numpy()[templ], a.business_address.to_numpy()[templ],
                              b.business_name.to_numpy()[templ], b.business_address.to_numpy()[templ],
                              "France", vocab)
    both = (feats.n_left_street_tokens > 0) & (feats.n_right_street_tokens > 0)
    j0 = both & (feats.street_token_jaccard == 0)
    mismatch = j0 & (feats.street_token_ratio < STREET_RATIO)
    return {
        "accepted_links": int(len(ls)),
        "template_name_links": int(templ.sum()),
        "template_links_with_both_streets": int(both.sum()),
        "street_mismatch_links": int(mismatch.sum()),
        "street_mismatch_rate_among_template_links": (float(mismatch.sum() / int(templ.sum()))
                                                        if templ.any() else None),
        "street_mismatch_rate_among_accepted": float(mismatch.sum() / len(ls)) if len(ls) else None,
        "street_jaccard0_links": int(j0.sum()),
        "street_jaccard0_rate_among_template_links": (float(j0.sum() / int(templ.sum()))
                                                       if templ.any() else None),
        "entities_with_street_mismatch_link": int(len(np.unique(ls[templ][mismatch.to_numpy()]))),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_population(data: Path) -> tuple[dict, dict]:
    s1 = read_source(data / "test_source1.tsv")
    parts = [read_source(data / f"test_source{i}.tsv") for i in (2, 3)]
    records = {k: np.concatenate([p[k] for p in parts]) for k in ("id", "country", "key")}
    records["text"] = pd.concat([p["text"] for p in parts], ignore_index=True)
    if len(np.unique(s1["id"])) != len(s1["id"]):
        raise ValueError("test_source1 contains duplicate entity IDs")
    if len(np.unique(records["id"])) != len(records["id"]):
        raise ValueError("test sources 2/3 contain duplicate entity IDs")
    return s1, records


def parse_report_names(raw: str | None, paths: list[Path]) -> list[str]:
    names = [p.parent.name or p.stem for p in paths] if raw is None else [n.strip() for n in raw.split(",")]
    if len(names) != len(paths) or any(not n or Path(n).name != n for n in names) or len(set(names)) != len(names):
        raise ValueError("--names must contain one unique, non-path name per matching file")
    return names


def _text_rows(text: pd.DataFrame, int_ids: np.ndarray) -> pd.DataFrame:
    keys = id_to_int(pa.array(text.entity_id.to_numpy(), type=pa.string()))
    order = np.argsort(keys)
    pos = np.searchsorted(keys, int_ids, sorter=order)
    rows = order[np.minimum(pos, len(order) - 1)]
    if np.any(keys[rows] != int_ids):
        raise ValueError("linked French entity has no text")
    return text.iloc[rows].reset_index(drop=True)


def main():
    root = Path("/home/sidvish/Documents/Work/Personal/az-ml-new")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("matching", type=Path, nargs="+")
    parser.add_argument("--data", type=Path, default=root / "data/raw/dataset/test")
    parser.add_argument("--vocab", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=root / "output/france_pseudo/diagnostics")
    parser.add_argument("--names", default=None, help="comma-separated report names, one per file")
    args = parser.parse_args()
    vocab = StreetVocabulary.load(args.vocab) if args.vocab else default_vocabulary()
    started = time.monotonic()
    s1, records = load_population(args.data)
    print(f"loaded {len(s1['id'])} S1 and {len(records['id'])} records in {time.monotonic() - started:.0f}s", flush=True)
    names = parse_report_names(args.names, args.matching)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, path in zip(names, args.matching):
        report = diagnose(path, s1, records, vocab)
        (args.output_dir / f"{name}.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"name": name, **{c: {k: v for k, v in r.items() if not isinstance(v, dict)}
                                            for c, r in report["countries"].items()}}), flush=True)


if __name__ == "__main__":
    main()
