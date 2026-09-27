"""Agent Y copy of Agent S's s_probe_rows.py (paths only). Original: cache-row indices of the probe-A (drop_T2r) and probe-C' links (lead's rebuilt TSVs) -> Y/g1/{rows,set}.npy.
Probe truth rates (~64-69% / ~92-96%) are pricing-model inferences: a stress test, not French accuracy. No labels read."""
import sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from y_fcommon import load
d = load(need_accept=False)
left, right, s1_ids, rec_ids = d["left"].astype(np.int64), d["right"].astype(np.int64), d["s1_ids"], d["rec_ids"]
n_rec = len(rec_ids)
s1_pos = {v: i for i, v in enumerate(s1_ids)}; rec_pos = {v: i for i, v in enumerate(rec_ids)}
keys = left * n_rec + right; o = np.argsort(keys); ks = keys[o]
out = Path("/home/ubuntu/run27/Y/g1"); out.mkdir(parents=True, exist_ok=True)
rows, sets, R = [], [], {}
for k, f in enumerate(("/home/ubuntu/run27/Y/in/probe_links/probeA_T2r.tsv", "/home/ubuntu/run27/Y/in/probe_links/probeCprime.tsv")):
    t = pd.read_csv(f, sep="\t", dtype=str)
    q = np.array([s1_pos[s] * n_rec + rec_pos[r] for s, r in zip(t.iloc[:, 0], t.iloc[:, 1])], dtype=np.int64)
    pos = np.minimum(np.searchsorted(ks, q), len(ks) - 1)
    assert (ks[pos] == q).all(), f"probe link not in candidate rows: {f}"
    rows.append(o[pos]); sets.append(np.full(len(q), k, np.int8)); R[f] = int(len(q))
np.save(out / "rows.npy", np.concatenate(rows)); np.save(out / "set.npy", np.concatenate(sets))
print(json.dumps(R))
