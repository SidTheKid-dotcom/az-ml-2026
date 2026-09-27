"""U8n fix (27 Sep, Agent U): make route rows comparable between arena B_REST and test.

In the arena, only B_REST claimants carry C2/Qwen scores, so a route record's C2/Qwen rival features see no A
claimant. On test every claimant is scored (route-row adversarial AUC 0.72-0.77 on rival_c2). For route rows:
  * the C2 and Qwen blocks are set to NaN (route rows have no C2/Qwen score of their own anyway);
  * the v1 block gets the record/S1-level v1 context that exists identically for all S1s on both sides:
    s1_candidates_ge_0_5, rival_blend (the record's best v1 blend over all dense pairs), record_claimants_ge_0_5,
    log1p_name_mult, record_kind. Other v1 columns stay NaN.
Original candidate rows are unchanged.

    python u8n_fix.py brest|test
Writes prep/<split>_u8f/ (x_base, x_q replaced; everything else hard-linked from <split>_u8n).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.forensic_features import address_kinds
from ber.pair_tree_arena import top_two

ROOT = Path("/home/ubuntu/run27/U")
split = sys.argv[1]
src, dst = ROOT / "prep" / f"{split}_u8n", ROOT / "prep" / f"{split}_u8f"
dst.mkdir(parents=True, exist_ok=True)
for f in src.iterdir():
    if f.name not in ("x_base.npy", "x_q.npy") and not (dst / f.name).exists():
        os.link(f, dst / f.name)
L, R = np.load(src / "left.npy"), np.load(src / "right.npy")
route = np.load(src / "x_route.npy")[:, 0] == 1
xb, xq = np.load(src / "x_base.npy"), np.load(src / "x_q.npy")
if split == "brest":
    t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_idx"])
    pl, pr = t.column("left_idx").to_numpy().astype(np.int64), t.column("right_idx").to_numpy().astype(np.int64)
    blend = np.nan_to_num(np.load(ROOT / "data/arena/probabilities.npy").astype(np.float32), nan=0.0)
    pool_addr = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["business_address"]).column("business_address")
    rec_kind = address_kinds(pool_addr)
    n_s1 = len(np.load(ROOT / "prep/brest/true_count.npy"))
else:
    pl = np.load(ROOT / "data/test/index/left.npy").astype(np.int64)
    pr = np.load(ROOT / "data/test/index/right.npy").astype(np.int64)
    blend = np.nan_to_num(np.load(ROOT / "data/test/probabilities.npy").astype(np.float32), nan=0.0)
    pool_addr = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["business_address"]).column(
        "business_address")
    set_pool = pq.read_table(ROOT / "u8src/test/pool.parquet", columns=["business_address", "raw_row_idx"])
    dense_raw = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["raw_row_idx"]).column(
        "raw_row_idx").to_numpy()
    extra = ~np.isin(set_pool.column("raw_row_idx").to_numpy(), dense_raw)
    rec_kind = np.r_[address_kinds(pool_addr), address_kinds(set_pool.column("business_address").filter(extra))]
    n_s1 = pq.ParquetFile(ROOT / "data/test/dense_test/reference.parquet").metadata.num_rows
if len(pl) != len(blend):
    raise SystemExit("pairs and v1 blend are not aligned")
n_rec = max(int(R.max()) + 1, int(pr.max()) + 1, len(rec_kind))
rec_top1, _ = top_two(pr, blend, n_rec)
strong = blend >= 0.5
s1_strong = np.bincount(pl[strong], minlength=n_s1).astype(np.float32)
rec_strong = np.bincount(pr[strong], minlength=n_rec).astype(np.float32)
mult = np.load(src / "mult.npy") if (src / "mult.npy").exists() else None
rows = np.flatnonzero(route)
xb[rows, 12:] = np.nan
xq[rows] = np.nan
xb[rows, 4] = s1_strong[L[rows]]
xb[rows, 5] = rec_top1[R[rows]]
xb[rows, 6] = rec_strong[R[rows]]
if mult is not None:
    xb[rows, 8] = np.log1p(mult[rows]).astype(np.float32)
else:  # test: S1 multiplicity from the S1's original rows
    s1m = np.full(n_s1, np.nan, np.float32)
    s1m[L[~route]] = xb[~route, 8]
    xb[rows, 8] = s1m[L[rows]]
xb[rows, 9] = rec_kind[R[rows]].astype(np.float32)
np.save(dst / "x_base.npy", xb)
np.save(dst / "x_q.npy", xq)
print(split, "route rows", len(rows), "rival_blend>=0.5 share", float((xb[rows, 5] >= 0.5).mean()),
      "record kind counts", np.bincount(xb[rows, 9].astype(np.int64), minlength=3).tolist())
