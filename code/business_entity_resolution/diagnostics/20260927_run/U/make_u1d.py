"""U1 at test-like density (27 Sep, Agent U): rebuild the B_REST u1 block from the name search over the
density-matched arena S1 set (every B_REST and India S1 + a seeded 53% of the other US S1s; density_check.py)
so the model is trained at the S1 density it meets on test. Test keeps its natural-density block (x_u1).
Writes prep/brest/x_u1d.npy (+ names) and prep/test/x_u1d.npy (copy of x_u1). Label-free."""
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
z = np.load(ROOT / "u2" / "density_u1_search.npz")
import sys
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "brest"
x = np.load(P / SPLIT / "x_u1.npy").copy()
left, right = np.load(P / SPLIT / "left.npy"), np.load(P / SPLIT / "right.npy")
rows = np.flatnonzero(~np.isnan(x[:, 0]))
r, s = right[rows], left[rows]
this = x[rows, 4]
x[rows, 0], x[rows, 1] = z["bsim"][r], z["ssim"][r]
x[rows, 2] = z["bsim"][r] - z["ssim"][r]
x[rows, 3] = z["nties"][r]
x[rows, 5] = (z["best"][r] == s).astype(np.float32)
x[rows, 6] = z["bsim"][r] - this
np.save(P / SPLIT / "x_u1d.npy", x)
names = json.loads((P / "brest/names_u1.json").read_text())
(P / SPLIT / "names_u1d.json").write_text(json.dumps([n.replace("u1_", "u1d_") for n in names]))
test_split = SPLIT.replace("brest", "test")
if (P / test_split / "x_u1.npy").exists():
    shutil.copy(P / test_split / "x_u1.npy", P / test_split / "x_u1d.npy")
print("rows", len(rows))
