"""The 20,000 B_REST monitor S1s of the 26 Sep france_mix arena (seed 20260926 draw) as an arena v1 S1 mask."""
import numpy as np, sys
part = np.load("/dev/shm/c2v2/in/s1_partition.npy")
r = np.random.default_rng(20260926); r.choice(np.flatnonzero(part == 0), 254_312, replace=False)
mon = r.choice(np.flatnonzero(part == 1), 20_000, replace=False)
m = np.zeros(len(part), bool); m[mon] = True; np.save(sys.argv[1], m)
b = part == 1; np.save(sys.argv[2], b); print(int(m.sum()), int(b.sum()))
