"""Label-free acceptance by row type (dense / route / reverse) on B_REST OOF vs test mean, per country.

    python rowtype_diag.py WORKDIR BREST_SPLIT TEST_SPLIT [THRESHOLD]
Also reports B_REST precision of accepted rows by type (B_REST labels only).
"""
import json
import sys
from pathlib import Path

import numpy as np

from ber.pair_tree_arena import decide

R = Path("/home/ubuntu/run27/U")
P = R / "prep"
w, bs, ts = R / "u4" / sys.argv[1], sys.argv[2], sys.argv[3]
rep = json.loads((w / "report.json").read_text())
thr = float(sys.argv[4]) if len(sys.argv) > 4 else float(rep["thresholds"]["full_oof_threshold"])


def rowtype(split, n):
    t = np.zeros(n, np.int8)
    if (P / split / "x_route.npy").exists():
        t[np.load(P / split / "x_route.npy", mmap_mode="r")[:, 0] == 1] = 1
    if (P / split / "x_rev.npy").exists():
        t[np.load(P / split / "x_rev.npy", mmap_mode="r")[:, 0] == 1] = 2
    if (P / split / "x_rev3.npy").exists() and not (P / split / "x_rev.npy").exists():
        t[np.load(P / split / "x_rev3.npy", mmap_mode="r")[:, 0] == 1] = 2
    if (P / split / "x_nat.npy").exists():
        t[np.load(P / split / "x_nat.npy", mmap_mode="r")[:, 0] == 1] = 3
    return t


out = {"threshold": thr}
bl, br = np.load(P / bs / "left.npy"), np.load(P / bs / "right.npy")
bm = json.loads((P / bs / "meta.json").read_text())
bc = np.asarray(bm["countries"])[np.load(P / bs / "country_code.npy")]
oof = np.load(w / "oof.npy")
acc = decide(oof, bl, br, thr, None)
lab = np.load(P / bs / "label.npy")
pop = np.load(P / bs / "population.npy")
bt = rowtype(bs, len(bl))
tl, tr = np.load(P / ts / "left.npy"), np.load(P / ts / "right.npy")
tm = json.loads((P / ts / "meta.json").read_text())
tc = np.asarray(tm["countries"])[np.load(P / ts / "country_code.npy")]
rows = np.load(w / "test_rows.npy")
p = np.load(w / "test_prob_mean.npy")
L, Rr = tl[rows], tr[rows]
acc_t = decide(p, L, Rr, thr, None)
tt = rowtype(ts, len(tl))[rows]
for c in ("US", "India"):
    nb, nt = int((pop & (bc == c)).sum()), int((tc == c).sum())
    for k, name in ((0, "dense"), (1, "route"), (2, "reverse"), (3, "native_dict")):
        mb = (bt == k) & (bc[bl] == c)
        mt = (tt == k) & (tc[L] == c)
        if not mb.any() and not mt.any():
            continue
        out[f"{c}|{name}"] = {
            "brest_rows_per_s1": round(float(mb.sum() / nb), 4), "test_rows_per_s1": round(float(mt.sum() / nt), 4),
            "brest_accepted_per_1k_rows": round(float((acc & mb).sum() / max(mb.sum(), 1) * 1000), 3),
            "test_accepted_per_1k_rows": round(float((acc_t & mt).sum() / max(mt.sum(), 1) * 1000), 3),
            "brest_accepted_per_1k_s1": round(float((acc & mb).sum() / nb * 1000), 3),
            "test_accepted_per_1k_s1": round(float((acc_t & mt).sum() / nt * 1000), 3),
            "brest_precision": round(float(lab[acc & mb].mean()), 4) if (acc & mb).any() else None}
print(json.dumps(out, indent=1))
(w / "rowtype_diag.json").write_text(json.dumps(out, indent=1) + "\n")
