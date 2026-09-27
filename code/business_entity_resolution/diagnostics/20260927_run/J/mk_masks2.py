"""Agent J: split the mid band into route/reverse rows (midrr) and dense rows (middense)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
root = Path(sys.argv[1])
for s in sys.argv[2:]:
    k = pd.read_parquet(root / s / "pairs.parquet", columns=["u_p", "row_type"])
    p, rt = k.u_p.to_numpy(), k.row_type.to_numpy()
    mid = ~((p >= .05) & (p <= .95))
    np.save(root / s / "mask_midrr.npy", mid & (rt != 0))
    np.save(root / s / "mask_middense.npy", mid & (rt == 0))
    print(s, "midrr", int((mid & (rt != 0)).sum()), "middense", int((mid & (rt == 0)).sum()))
