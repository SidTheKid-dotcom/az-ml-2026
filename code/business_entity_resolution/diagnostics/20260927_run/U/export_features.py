"""Export Agent U per-pair feature families for other decision models (27 Sep).

    python export_features.py brest|test OUT_DIR

One parquet per split, one row per candidate pair (the pair_tree_c2 candidate rows):
  brest: pair_row (row of arena v1/pairs.parquet), left_idx / right_idx (arena reference / pool rows),
         s1_id, rec_id, fold (frozen 5-fold file), and the blocks below.
  test:  cand_row (row of test_inputs/index/{left,right}.npy), left_idx / right_idx (dense_test reference /
         pool rows), s1_id, rec_id; US/India rows only.
Blocks: ed_* (U7/U9 edit/invented-name, 17), u1_* (EA name search at the natural pool density, 7),
u1d_* (brest only: the same at a test-like US S1 density; on test u1d == u1), coll_* (collective, 25),
and model probabilities p_* (brest: 5-fold component OOF; test: mean of the 5 fold models).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
split, out = sys.argv[1], Path(sys.argv[2])
ROUTE = split.endswith("_u8f")  # route rows only (new candidate pairs), with the U8f model probability
out.mkdir(parents=True, exist_ok=True)
d = P / split
MODELS = {"p_u4": "base_q_coll", "p_u4_ed": "base_q_coll_ed", "p_u4_u1_ed": "base_q_coll_u1_ed",
          "p_u4_u1d_ed": "base_q_coll_u1d_ed"}
left, right = np.load(d / "left.npy"), np.load(d / "right.npy")
if ROUTE:
    MODELS = {"p_u8f": "u8f__base_q_coll_ed_route"}
flag = np.load(d / "x_route.npy")[:, 0] == 1 if ROUTE else None
if split.startswith("brest"):
    rows = np.flatnonzero(flag) if ROUTE else np.arange(len(left))
    ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["entity_id"]).column("entity_id")
    pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["entity_id"]).column("entity_id")
    cols = {"left_idx": left[rows], "right_idx": right[rows],
            "fold": np.load(ROOT / "folds/brest_fold5_u4-20260927.npy")[left[rows]]}
    if not ROUTE:
        cols = {"pair_row": np.load(d / "pair_rows.npy"), **cols}
else:
    meta = json.loads((d / "meta.json").read_text())
    country = np.asarray(meta["countries"])[np.load(d / "country_code.npy")]
    rows = np.flatnonzero(np.isin(country[left], ["US", "India"]) & (flag if ROUTE else True))
    ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["entity_id"]).column("entity_id")
    pool = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["entity_id"]).column("entity_id")
    if (d / "extra_rec_ids.json").exists():
        pool = pa.chunked_array([pool.combine_chunks(), pa.array(json.loads((d / "extra_rec_ids.json").read_text()),
                                                                  type=pool.type)])
    cols = {"left_idx": left[rows], "right_idx": right[rows]}
    if not ROUTE:
        cols = {"cand_row": np.load(d / "cand_rows.npy")[rows], **cols}
table = {k: pa.array(v) for k, v in cols.items()}
table["s1_id"] = ref.take(pa.array(left[rows]))
table["rec_id"] = pool.take(pa.array(right[rows]))
for block in (("ed", "coll", "route", "base") if ROUTE else ("ed", "u1", "u1d", "coll")):
    f = d / f"x_{block}.npy"
    if not f.exists():
        continue
    x = np.load(f)[rows]
    names = json.loads(((P / "brest_u8f") if block == "route" else (P / "brest")).joinpath(f"names_{block}.json").read_text())
    if block == "base":  # route rows: only the v1 record/S1 context columns are filled (C2 block is NaN)
        keep = [4, 5, 6, 8, 9]
        x, names = x[:, keep], [f"v1ctx_{names[k]}" for k in keep]
    if block == "coll":
        names = [f"coll_{n}" for n in names]
    for j, n in enumerate(names):
        table[n] = pa.array(x[:, j].astype(np.float32))
for name, work in MODELS.items():
    w = ROOT / "u4" / work
    if split.startswith("brest") and (w / "oof.npy").exists():
        table[name] = pa.array(np.load(w / "oof.npy")[rows].astype(np.float32))
    elif split.startswith("test") and (w / "test_prob_mean.npy").exists():
        trows = np.load(w / "test_rows.npy")
        pm = np.load(w / "test_prob_mean.npy")
        if ROUTE:
            pos = np.searchsorted(trows, rows)
            if not np.array_equal(trows[pos], rows):
                raise SystemExit(f"{work}: route rows missing from test predictions")
            pm = pm[pos]
        elif not np.array_equal(trows, rows):
            raise SystemExit(f"{work}: test rows differ")
        table[name] = pa.array(pm.astype(np.float32))
t = pa.table(table)
path = out / (f"u_route_pairs_{split.split('_')[0]}.parquet" if ROUTE else f"u_features_{split}.parquet")
pq.write_table(t, path, compression="zstd")
print(path, t.num_rows, t.num_columns, t.column_names)
