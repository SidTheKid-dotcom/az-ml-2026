"""Idea 4: attach the L1 probabilities to the evaluation splits and write K's deliverables (label-free).

x_l1 columns: [p2_full, p2_nov1] (p2_full is NaN on route rows: no v1 pair scores there).
prep/{brest,test,brest_u8f,test_u8f}/x_l1.npy + names_l1.json; deliver/l1_{brest,test}_{full,nov1}.npy (K's row
orders; France test rows NaN) + l1_meta.json.
"""
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
L1 = P / "l1"
FT, NT = os.environ.get("L1_FULL_TAG", ""), os.environ.get("L1_NOV1_TAG", "")
BLK = os.environ.get("L1_BLOCK", "l1")
D = ROOT / os.environ.get("L1_DELIVER", "deliver_l1")
D.mkdir(exist_ok=True)
tj = json.loads((L1 / "targets.json").read_text())
n_cand, n_tp = tj["arena_cand"], tj["test_prep"]
al = np.load(L1 / "targets_arena_left.npy")
part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
bpos = np.flatnonzero(part[al[:n_cand]] == 1)
crow = np.load(P / "arena_cand/cand_rows.npy")
if not np.array_equal(crow[bpos], np.load(P / "brest/pair_rows.npy")):
    raise SystemExit("B_REST row order mismatch")
meta = json.loads((P / "test/meta.json").read_text())
fr = np.asarray(meta["countries"])[np.load(P / "test/country_code.npy")][np.load(P / "test/left.npy")] == "France"
cols = {}
for v in ("full", "nov1"):
    tg = FT if v == "full" else NT
    pa, pt = np.load(L1 / f"p2_{v}{tg}_arena.npy"), np.load(L1 / f"p2_{v}{tg}_test.npy")
    if v == "full":
        pa, pt = pa.copy(), pt.copy()
        pa[n_cand:] = np.nan
        pt[n_tp:] = np.nan
    cols[v] = (pa, pt)
xb = np.column_stack([cols[v][0][bpos] for v in ("full", "nov1")])
xt = np.column_stack([cols[v][1][:n_tp] for v in ("full", "nov1")])
xt[fr] = np.nan
np.save(P / f"brest/x_{BLK}.npy", xb)
np.save(P / f"test/x_{BLK}.npy", xt)
xbu = np.vstack([xb, np.column_stack([cols[v][0][n_cand:] for v in ("full", "nov1")])])
xtu = np.vstack([xt, np.column_stack([cols[v][1][n_tp:] for v in ("full", "nov1")])])
for s, x in (("brest_u8f", xbu), ("test_u8f", xtu)):
    if len(x) != len(np.load(P / s / "left.npy", mmap_mode="r")):
        raise SystemExit(f"{s}: row count mismatch")
    np.save(P / s / f"x_{BLK}.npy", x)
for s, x in (("brest_u8r", xbu), ("test_u8r", xtu)):
    if (P / s / "left.npy").exists():
        n = len(np.load(P / s / "left.npy", mmap_mode="r"))
        np.save(P / s / f"x_{BLK}.npy", np.vstack([x, np.full((n - len(x), 2), np.nan, np.float32)]))
for s in ("brest", "brest_u8f", "brest_u8r"):
    (P / s / f"names_{BLK}.json").write_text(json.dumps([f"{BLK}_p2_full", f"{BLK}_p2_nov1"]))
out = {}
for j, v in enumerate(("full", "nov1")):
    for s, x in (("brest", xb[:, j]), ("test", xt[:, j])):
        f = D / f"l1_{s}_{v}.npy"
        np.save(f, x.astype(np.float32))
        out[f.name] = {"rows": int(len(x)), "nan": int(np.isnan(x).sum()),
                       "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
(D / "l1_files.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
