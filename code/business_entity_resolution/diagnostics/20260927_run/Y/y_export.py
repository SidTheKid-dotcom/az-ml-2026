"""Agent Y: keyed per-pair export of a score_external output over the full test pair index.
Usage: y_export.py SCORES_DIR OUT.parquet   (rows = scored_mask; columns s1_id, rec_id, test_row, tree, laya)."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
T = Path("/home/ubuntu/run27/Y/in/test")
sc, out = Path(sys.argv[1]), Path(sys.argv[2])
assert (sc / "complete.json").exists(), sc
s = np.load(sc / "scores.npy", mmap_mode="r"); m = np.load(sc / "scored_mask.npy")
left = np.load(T / "index/left.npy", mmap_mode="r"); right = np.load(T / "index/right.npy", mmap_mode="r")
base = np.load(T / "base_probabilities.npy", mmap_mode="r")
assert len(s) == len(m) == len(left) == len(right) == len(base)
rows = np.flatnonzero(m)
s1 = np.asarray(pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id"]).column(0).to_pylist(), dtype=object)
rec = np.asarray(pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id"]).column(0).to_pylist(), dtype=object)
l, r = np.asarray(left[rows], dtype=np.int64), np.asarray(right[rows], dtype=np.int64)
df = pd.DataFrame({"s1_id": s1[l], "rec_id": rec[r], "test_row": rows.astype(np.int64),
                   "tree": np.asarray(base[rows], dtype=np.float32), "laya": np.asarray(s[rows], dtype=np.float32)})
assert not df.duplicated(["s1_id", "rec_id"]).any()
df.to_parquet(out, index=False)
print(json.dumps({"rows": len(df), "s1": int(df.s1_id.nunique()), "mean_laya": float(df.laya.mean()),
                  "share_ge_0.5": float((df.laya >= .5).mean()), "out": str(out)}))
