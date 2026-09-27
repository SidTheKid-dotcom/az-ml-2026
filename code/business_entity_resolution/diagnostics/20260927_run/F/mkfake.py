"""Dry-run helper: fake G2 output holding the OLD C2 scores (rebuild must then reproduce the S1 file's French rows)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fcommon import OUT, load
d = load(need_accept=False)
rows = np.load(OUT / "g2/rows.npy"); o = OUT / "g2/dryrun_oldscores"; o.mkdir(parents=True, exist_ok=True)
np.save(o / "scores.npy", np.nan_to_num(d["c2"][rows], nan=0.0).astype(np.float32)); (o / "complete.json").write_text("{}\n")
