"""Verify a US/India rows file against the reference matching file's S1 order and diff it vs other files (pod).

    python verify_rows.py ROWS.tsv REFERENCE_FULL.tsv [COMPARE.tsv ...]

Checks: header, LF only, S1 IDs = the reference file's US/India S1s in the same order, lists <= 11, no duplicate
IDs, one owner. Diff: links only in ROWS / only in COMPARE, by country. Label-free.
"""
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

HEADER = "source1_entity_id\tmatched_entity_ids"
rows_path, ref_path = Path(sys.argv[1]), Path(sys.argv[2])
ref = pq.read_table("/opt/az-ml-new/test_inputs/dense_test/reference.parquet", columns=["entity_id", "country"])
country = dict(zip(ref.column("entity_id").to_pylist(), ref.column("country").to_pylist()))


def read(path):
    raw = Path(path).read_bytes()
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    return raw, lines[0], [ln.split("\t") for ln in lines[1:]]


raw, head, body = read(rows_path)
_, rhead, rbody = read(ref_path)
problems = []
if b"\r" in raw:
    problems.append("CR bytes")
if head != HEADER:
    problems.append("bad header")
ref_ui = [r[0] for r in rbody if country.get(r[0]) in ("US", "India")]
if [r[0] for r in body] != ref_ui:
    problems.append("S1 order/IDs differ from the reference file's US/India rows")
seen = {}
longest = 0
for s1, ids in body:
    lst = [v for v in ids.split(",") if v]
    longest = max(longest, len(lst))
    if len(lst) != len(set(lst)):
        problems.append(f"duplicate IDs in {s1}")
    for v in lst:
        if v in seen:
            problems.append(f"record {v} has two owners")
        seen[v] = s1
if longest > 11:
    problems.append("list > 11")
mine = {(s1, v) for s1, ids in body for v in ids.split(",") if v}
out = {"file": str(rows_path), "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(body), "links": len(mine),
       "longest": longest, "passed": not problems, "problems": problems[:20], "reference": str(ref_path), "diff": {}}
for other in sys.argv[3:]:
    _, _, ob = read(other)
    theirs = {(s1, v) for s1, ids in ob if country.get(s1) in ("US", "India") for v in ids.split(",") if v}
    d = {}
    for c in ("US", "India"):
        d[c] = {"only_here": sum(1 for p in mine - theirs if country[p[0]] == c),
                "only_there": sum(1 for p in theirs - mine if country[p[0]] == c),
                "common": sum(1 for p in mine & theirs if country[p[0]] == c)}
    out["diff"][other] = d
print(json.dumps(out, indent=1))
