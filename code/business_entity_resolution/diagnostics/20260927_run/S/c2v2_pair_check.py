"""Agent S: dense C2 v2 vs continuation on B_REST route (U) and reverse (R) pairs, with B_REST labels only.
Route labels: arena rec_owner[right] == left (right in the compact arena pool). Reverse labels: R's `label` column.
Usage: c2v2_pair_check.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
S = Path("/dev/shm/c2v2/score"); I = Path("/dev/shm/c2v2/in")
part = np.load(I / "s1_partition.npy"); owner = np.load(I / "labels/rec_owner.npy")
def auc(y, s):
    o = np.argsort(s, kind="stable"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))
def ap(y, s):
    o = np.argsort(-s, kind="stable"); yy = y[o]; tp = np.cumsum(yy)
    return float((tp[yy] / (np.flatnonzero(yy) + 1)).sum() / max(1, yy.sum()))
R = {}
for name in ("brest_routes", "brest_reverse"):
    k = pd.read_parquet(S / name / "keys.parquet")
    assert (part[k.left_idx.to_numpy()] == 1).all()
    if name == "brest_routes":
        y = owner[k.right_idx.to_numpy()] == k.left_idx.to_numpy()
    else:
        y = pd.read_parquet("/dev/shm/c2v2/rev/r_reverse_pairs_b_rest.parquet", columns=["label"]).label.to_numpy().astype(bool)
    R[name] = {"pairs": int(len(y)), "true": int(y.sum())}
    for model, d in (("dense", S / f"out_{name}"), ("cont", S / f"cont_{name}")):
        if not (d / "complete.json").exists():
            continue
        s = np.load(d / "scores.npy")
        e = {"auc": auc(y, s), "average_precision": ap(y, s)}
        for t in (.111, .5, .9):
            p = s >= t
            e[f"c2_ge_{t}"] = {"accepted": int(p.sum()), "precision": float(y[p].mean()) if p.any() else None,
                               "recall": float(p[y].mean()) if y.any() else None}
        R[name][model] = e
Path(sys.argv[1]).write_text(json.dumps(R, indent=1) + "\n"); print(json.dumps(R, indent=1))
