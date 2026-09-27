"""U8r exports for K / integration (27 Sep, Agent U). Label-free on test.

    python u8r_export.py WORKTAG OUT_DIR
1. u8r_pairs_{brest,test}.parquet: every row of brest_u8r / test_u8r (dense + route + reverse; test US/India):
   s1_id, rec_id, row_type (0 dense, 1 route, 2 reverse), rec_outside_pool, p (B_REST: 5-fold OOF with U's frozen
   fold file; test: mean of the 5 fold models), and for route/reverse rows the ed / coll / v1ctx / route / rev blocks.
2. us_india_rows.tsv (from u4_output) + extra_candidate_pairs.tsv: accepted test links that are not dense test
   candidate pairs (route or reverse), header "source1_entity_id\\tentity_id".
"""
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.pair_tree_arena import decide

R = Path("/home/ubuntu/run27/U")
P = R / "prep"
tag, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
w = R / "u4" / tag
rep = json.loads((w / "report.json").read_text())
thr = float(rep["thresholds"]["full_oof_threshold"])


def ids(path):
    return np.asarray(pq.read_table(path, columns=["entity_id"]).column("entity_id").to_pylist(), dtype=object)


SPLITS = os.environ.get("U_EXPORT_SPLITS", "brest_u8r,test_u8r").split(",")


def names(split, b):
    return json.loads((P / SPLITS[0] / f"names_{b}.json").read_text())


for split in SPLITS:
    d = P / split
    left, right = np.load(d / "left.npy"), np.load(d / "right.npy")
    rt = np.zeros(len(left), np.int8)
    rt[np.load(d / "x_route.npy", mmap_mode="r")[:, 0] == 1] = 1
    rev_file = d / ("x_rev.npy" if (d / "x_rev.npy").exists() else "x_rev3.npy")
    rt[np.load(rev_file, mmap_mode="r")[:, 0] == 1] = 2
    if (d / "x_nat.npy").exists():
        rt[np.load(d / "x_nat.npy", mmap_mode="r")[:, 0] == 1] = 3
    extra = json.loads((d / "extra_rec_ids.json").read_text()) if (d / "extra_rec_ids.json").exists() else []
    if split.startswith("brest"):
        s1 = ids(R / "data/arena/reference.parquet")
        rec = np.concatenate([ids(R / "data/arena/pool.parquet"), np.asarray(extra, dtype=object)])
        rows = np.arange(len(left))
        p = np.load(w / "oof.npy")
    else:
        s1 = ids(R / "data/test/dense_test/reference.parquet")
        rec = np.concatenate([ids(R / "data/test/dense_test/pool.parquet"), np.asarray(extra, dtype=object)])
        rows = np.load(w / "test_rows.npy")
        p = np.load(w / "test_prob_mean.npy")
    n_pool = len(rec) - len(extra)
    L, Rr = left[rows], right[rows]
    cols = {"s1_id": pa.array(s1[L].tolist()), "rec_id": pa.array(rec[Rr].tolist()), "row_type": pa.array(rt[rows]),
            "rec_outside_pool": pa.array(Rr >= n_pool), "p": pa.array(p.astype(np.float32))}
    sel = rt[rows] > 0
    blist = [("ed", "x_ed"), ("coll", "x_coll"), ("route", "x_route"),
             ("rev", "x_rev") if (d / "x_rev.npy").exists() else ("rev3", "x_rev3")]
    if (d / "x_nat.npy").exists():
        blist.append(("nat", "x_nat"))
    for b, fn in blist:
        x = np.load(d / f"{fn}.npy", mmap_mode="r")
        nm = names(split, b)
        blk = np.full((len(rows), x.shape[1]), np.nan, np.float32)
        idx = rows[sel]
        blk[sel] = x[idx]
        for j, n in enumerate(nm):
            cols[(f"coll_{n}" if b == "coll" else n)] = pa.array(blk[:, j])
    xb = np.load(d / "x_base.npy", mmap_mode="r")
    v1n = ["s1_candidates_ge_0_5", "rival_blend", "record_claimants_ge_0_5", "log1p_name_mult", "record_kind"]
    for j, n in zip([4, 5, 6, 8, 9], v1n):
        c = np.full(len(rows), np.nan, np.float32)
        c[sel] = xb[rows[sel], j]
        cols[f"v1ctx_{n}"] = pa.array(c)
    t = pa.table(cols)
    pq.write_table(t, out / f"pairs_{split.split('_')[0]}.parquet", compression="zstd")
    print(split, t.num_rows, "rows;", {k: int((rt[rows] == k).sum()) for k in (0, 1, 2)}, flush=True)
    if split.startswith("test"):
        acc = decide(p, L, Rr, thr, None)
        ext = acc & (rt[rows] > 0)
        with open(out / "extra_candidate_pairs.tsv", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("source1_entity_id\tentity_id\n")
            for a, b in zip(s1[L[ext]], rec[Rr[ext]]):
                fh.write(f"{a}\t{b}\n")
        guard = ext & (Rr >= n_pool) & (rt[rows] == 1) & (p < 0.955)
        print("extra accepted links", int(ext.sum()), "route links to outside-pool records below 0.955:", int(guard.sum()))
shutil.copy(w / "us_india_rows.tsv", out / "us_india_rows.tsv")
for f in ("report.json", "output_report.json", "rowtype_diag.json"):
    if (w / f).exists():
        shutil.copy(w / f, out / f)
print("done")
