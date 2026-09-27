"""Agent F follow-up: all French candidate rows (original, un-normalized inputs) for C2 step_010000 scoring."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fcommon import OUT, load
d = load(need_accept=False)
rows = np.flatnonzero(d["france"]); o = OUT / "g4"; o.mkdir(parents=True, exist_ok=True)
np.save(o / "rows.npy", rows); np.save(o / "left.npy", d["left"][rows]); np.save(o / "right.npy", d["right"][rows])
np.save(o / "tree.npy", d["base"][rows].astype(np.float32))
print("french rows", len(rows), "gated", int((d["base"][rows] >= 0.01).sum()))
