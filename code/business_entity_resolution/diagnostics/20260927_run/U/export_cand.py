"""Export arena candidate rows (all partitions) as int32 arrays for the A-fitted feature tables (label-free)."""
import numpy as np
from pathlib import Path
out = Path("/workspace/az-ml-new/run27/U/prep/arena_cand")
out.mkdir(parents=True, exist_ok=True)
with np.load("/opt/az-ml-new/agents/pair_tree_c2/step005000/arena_v1_cache.npz") as c:
    for k in ("left", "right", "cand_rows"):
        np.save(out / (k + ".npy"), c[k].astype(np.int32 if k != "cand_rows" else np.int64))
print("ok")
