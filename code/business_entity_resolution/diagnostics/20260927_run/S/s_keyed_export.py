"""Agent S: keyed parquet of a route/reverse score set: the set's keys.parquet (s1_id, rec_id, left_idx, right_idx
[, raw_row_idx]; right_idx as in the source file) + score column. Usage: s_keyed_export.py SET_DIR SCORES_DIR COLUMN OUT.parquet"""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
d, sc, col, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
k = pd.read_parquet(d / "keys.parquet")
s = np.load(sc / "scores.npy"); m = np.load(sc / "scored_mask.npy")
assert len(s) == len(k) and m.all(), "scores misaligned or incomplete"
k[col] = s.astype(np.float32)
k.to_parquet(out, index=False)
print(json.dumps({"rows": len(k), "mean": float(s.mean()), "ge_0.5": float((s >= .5).mean())}))
