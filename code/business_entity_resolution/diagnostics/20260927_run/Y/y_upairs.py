"""Agent Y: index arrays for Agent U's u8r_v2 US/India pairs with 0.02 <= p <= 0.98 (B_REST: arena v1 records;
test: dense test records). Records outside the dense/arena pool have no text here and are left unscored (NaN).
Usage: y_upairs.py prep SPLIT  -> Y/us/SPLIT/{left,right,tree}.npy + keys.parquet
       y_upairs.py export SPLIT SCORES_DIR OUT.parquet"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
Y = Path("/home/ubuntu/run27/Y")
SRC = {"brest": (Path("/home/ubuntu/run27/S/in/arena/v1"), Y / "in/U/u8r_pairs_brest.parquet"),
       "test": (Y / "in/test/dense_test", Y / "in/U/u8r_pairs_test.parquet")}
cmd, split = sys.argv[1], sys.argv[2]
recdir, pairs = SRC[split]
out = Y / "us" / split
if cmd == "prep":
    out.mkdir(parents=True, exist_ok=True)
    t = pq.read_table(pairs, columns=["s1_id", "rec_id", "row_type", "rec_outside_pool", "p"]).to_pandas()
    t = t[(t.p >= 0.02) & (t.p <= 0.98)].reset_index(drop=True)
    ref = pq.read_table(recdir / "reference.parquet", columns=["entity_id"]).column(0).to_pylist()
    pool = pq.read_table(recdir / "pool.parquet", columns=["entity_id"]).column(0).to_pylist()
    li = pd.Series(np.arange(len(ref)), index=ref).reindex(t.s1_id).to_numpy()
    ri = pd.Series(np.arange(len(pool)), index=pool).reindex(t.rec_id).to_numpy()
    ok = ~np.isnan(li) & ~np.isnan(ri)
    t["has_text"] = ok
    t.to_parquet(out / "keys.parquet", index=False)
    np.save(out / "left.npy", np.where(ok, li, 0).astype(np.int64))
    np.save(out / "right.npy", np.where(ok, ri, 0).astype(np.int64))
    np.save(out / "tree.npy", ok.astype(np.float32))  # score_external gate: only rows with text are scored
    print(json.dumps({"split": split, "uncertain": len(t), "with_text": int(ok.sum()),
                      "outside_pool_flag": int(t.rec_outside_pool.sum()), "s1_missing": int(np.isnan(li).sum())}))
else:
    sc, dest = Path(sys.argv[3]), Path(sys.argv[4])
    t = pd.read_parquet(out / "keys.parquet")
    s = np.load(sc / "scores.npy"); m = np.load(sc / "scored_mask.npy")
    assert len(s) == len(t) and np.array_equal(m, t.has_text.to_numpy())
    t["laya"] = np.where(m, s, np.nan).astype(np.float32)
    t = t.rename(columns={"p": "u8r_v2_p"})
    t.to_parquet(dest, index=False)
    print(json.dumps({"rows": len(t), "scored": int(m.sum()), "mean_laya": float(np.nanmean(t.laya)),
                      "pearson_vs_u8r": float(np.corrcoef(t.laya[m], t.u8r_v2_p[m])[0, 1])}))
