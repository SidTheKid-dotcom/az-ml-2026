"""Agent J: per-shard band masks over U's probability: core = 0.05 <= u_p <= 0.95, mid = the rest of [0.02, 0.98]."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
root = Path(sys.argv[1])
for s in sys.argv[2:]:
    k = pd.read_parquet(root / s / "pairs.parquet", columns=["u_p"])
    p = k.u_p.to_numpy()
    core = (p >= .05) & (p <= .95)
    np.save(root / s / "mask_core.npy", core)
    np.save(root / s / "mask_mid.npy", ~core)
    print(s, len(p), "core", int(core.sum()), "mid", int((~core).sum()))
