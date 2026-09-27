"""Validate K rows files and compare them label-free with the scored S1 file (and optional others).

Checks per file: header, LF only, S1 order = reference order restricted to the file's countries, no
duplicate IDs, list <= 11, one owner, every link inside the test candidate pairs. Comparison: links,
mean list, empty rows per country, links shared with / added to / removed from the S1 file.

    python -m ber.k_compare FILE [FILE ...] --out report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.k3_test import DATA, sha256

HEADER = "source1_entity_id\tmatched_entity_ids"


def load(path, s1_index, rec_index):
    raw = Path(path).read_bytes()
    problems = []
    if b"\r" in raw:
        problems.append("CR bytes")
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    if lines[0] != HEADER:
        problems.append("bad header")
    s1_rows, ls, rs, longest = [], [], [], 0
    for line in lines[1:]:
        s1, ids = line.split("\t")
        i = s1_index[s1]
        s1_rows.append(i)
        items = [v for v in ids.split(",") if v]
        if len(items) != len(set(items)):
            problems.append(f"duplicate ids for {s1}")
        longest = max(longest, len(items))
        for v in items:
            ls.append(i)
            rs.append(rec_index.get(v, -1))
    return np.asarray(s1_rows), np.asarray(ls, dtype=np.int64), np.asarray(rs, dtype=np.int64), longest, problems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "country"])
    s1_ids = np.asarray(ref.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref.column("country").to_pylist(), dtype=object)
    rec_ids = pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    n_rec = len(rec_ids)
    s1_index = {v: i for i, v in enumerate(s1_ids)}
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    left = np.load(DATA / "test_prep/left.npy").astype(np.int64)
    right = np.load(DATA / "test_prep/right.npy").astype(np.int64)
    cand = np.sort(left * n_rec + right)
    _, bl, br, _, _ = load(DATA / "s1file/matching_results.tsv", s1_index, rec_index)
    base_keys = np.unique(bl * n_rec + br)
    out = {}
    for f in a.files:
        s1_rows, ls, rs, longest, problems = load(f, s1_index, rec_index)
        countries = np.unique(country[s1_rows])
        expect = np.flatnonzero(np.isin(country, countries))
        if len(s1_rows) != len(expect) or not np.all(s1_rows == expect):
            problems.append("S1 rows are not the reference order for these countries")
        if np.any(rs < 0):
            problems.append(f"{int((rs < 0).sum())} unknown record ids")
        contested = int((np.bincount(rs[rs >= 0], minlength=n_rec) > 1).sum())
        if contested:
            problems.append(f"{contested} records with 2+ owners")
        if longest > 11:
            problems.append(f"longest list {longest}")
        k = ls * n_rec + rs
        pos = np.minimum(np.searchsorted(cand, k), len(cand) - 1)
        outside = int((cand[pos] != k).sum())
        if outside:
            problems.append(f"{outside} links outside the candidates")
        in_base = np.isin(k, base_keys)
        base_here = base_keys[np.isin(base_keys // n_rec, s1_rows)]
        per_c = {}
        lengths = np.bincount(ls, minlength=len(s1_ids))
        base_len = np.bincount(bl, minlength=len(s1_ids))
        for c in countries:
            m = country[s1_rows] == c
            rows_c = s1_rows[m]
            lk = country[ls] == c
            per_c[str(c)] = {"s1": int(m.sum()), "links": int(lk.sum()), "mean_list": float(lengths[rows_c].mean()),
                             "empty_rows": int((lengths[rows_c] == 0).sum()),
                             "s1_file_links": int(base_len[rows_c].sum()),
                             "s1_file_empty_rows": int((base_len[rows_c] == 0).sum()),
                             "shared_with_s1_file": int((in_base & lk).sum()),
                             "added_vs_s1_file": int((~in_base & lk).sum())}
        out[f] = {"sha256": sha256(f), "rows": int(len(s1_rows)), "links": int(len(ls)), "longest": longest,
                  "passed": not problems, "problems": problems[:10], "by_country": per_c,
                  "removed_vs_s1_file": int(len(np.setdiff1d(base_here, k)))}
        print(json.dumps({f: out[f]}, default=float), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
