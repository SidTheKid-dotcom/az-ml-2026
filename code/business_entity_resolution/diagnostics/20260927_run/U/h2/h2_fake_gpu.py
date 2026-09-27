"""DRY RUN ONLY: fake GPU step A outputs (random same-country records / S1s) so every downstream HOLDOUT-2 stage can
be exercised end-to-end before the real GPU run. Writes OUT/gpu_out/*.npy in the real layout. Label-free.

    python h2_fake_gpu.py OUTDIR
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/ubuntu/run27/U")
out = Path(sys.argv[1]) / "gpu_out"
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
pcc = np.load(ROOT / "rtokens/train_pool.cc.npy")
scc = np.load(ROOT / "h2/tokens/h2_s1.cc.npy")
n_arena = 2_072_821
h_cc = scc[n_arena:]
by_c = {c: np.flatnonzero(pcc == c) for c in (0, 1)}
s_by_c = {c: np.flatnonzero(scc == c) for c in (0, 1)}
fwd = np.empty((len(h_cc), 50), np.int32)
for c in (0, 1):
    rows = np.flatnonzero(h_cc == c)
    fwd[rows] = rng.choice(by_c[c], (len(rows), 50))
    for r in rows:  # no duplicate record within one S1's list
        while len(np.unique(fwd[r])) < 50:
            fwd[r] = rng.choice(by_c[c], 50, replace=False)
fs = np.sort(rng.uniform(0.3, 0.95, (len(h_cc), 50)).astype(np.float32), axis=1)[:, ::-1]
np.save(out / "h2_fwd_idx.npy", fwd)
np.save(out / "h2_fwd_score.npy", np.ascontiguousarray(fs))
rev = np.empty((len(pcc), 20), np.int32)
for c in (0, 1):
    rows = np.flatnonzero(pcc == c)
    rev[rows] = rng.choice(s_by_c[c], (len(rows), 20))
rs = np.sort(rng.uniform(0.2, 0.8, (len(pcc), 20)).astype(np.float16), axis=1)[:, ::-1]
np.save(out / "h2_rev_idx.npy", rev)
np.save(out / "h2_rev_score.npy", np.ascontiguousarray(rs))
chk = np.arange(0, n_arena, n_arena // 20000)[:20000]
np.save(out / "check_arena_rows.npy", chk)
np.save(out / "check_arena_fwd_idx.npy", rng.choice(len(pcc), (len(chk), 50)).astype(np.int32))
print("fake gpu_out written (DRY RUN)")
