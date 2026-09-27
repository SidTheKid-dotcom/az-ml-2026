"""Adversarial validation (27 Sep, Agent U): B_REST candidate rows vs US/India TEST candidate rows,
within country x record source (S2/S3). Per-feature AUC (max(AUC, 1-AUC)) and a per-block CatBoost domain
classifier AUC (70/30 split). AUC > 0.6 flags a feature for investigation. Label-free.

    python adv_val.py BLOCK [BLOCK ...]      e.g. base q coll
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

ROOT = Path("/home/ubuntu/run27/U")
PREP = ROOT / "prep"
N = 200_000
BS, TS = os.environ.get("U_BREST", "brest"), os.environ.get("U_TEST", "test")
ROUTE_ONLY = os.environ.get("ADV_ROUTE_ONLY", "") == "1"
REV_ONLY = os.environ.get("ADV_REV_ONLY", "") == "1"
NAT_ONLY = os.environ.get("ADV_NAT_ONLY", "") == "1"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def source(path, idx, extra=None):
    ids = pq.read_table(path, columns=["entity_id"]).column("entity_id").to_pylist() + (extra or [])
    pre = np.asarray([s[:2] for s in ids], dtype=object)
    return pre[idx]


def names(block):
    f = PREP / BS / f"names_{block}.json"
    return json.loads(f.read_text()) if f.exists() else None


blocks = sys.argv[1:] or ["base", "q", "coll"]
bm = json.loads((PREP / BS / "meta.json").read_text())
tm = json.loads((PREP / TS / "meta.json").read_text())
b_country = np.asarray(bm["countries"])[np.load(PREP / BS / "country_code.npy")][np.load(PREP / BS / "left.npy")]
t_left = np.load(PREP / TS / "left.npy")
t_country = np.asarray(tm["countries"])[np.load(PREP / TS / "country_code.npy")][t_left]
bextra = json.loads((PREP / BS / "extra_rec_ids.json").read_text()) if (PREP / BS / "extra_rec_ids.json").exists() else None
b_src = source(ROOT / "data/arena/pool.parquet", np.load(PREP / BS / "right.npy"), bextra)
extra = json.loads((PREP / TS / "extra_rec_ids.json").read_text()) if (PREP / TS / "extra_rec_ids.json").exists() else None
t_src = source(ROOT / "data/test/dense_test/pool.parquet", np.load(PREP / TS / "right.npy"), extra)
if ROUTE_ONLY:
    b_country = np.where(np.load(PREP / BS / "x_route.npy")[:, 0] == 1, b_country, "none")
    t_country = np.where(np.load(PREP / TS / "x_route.npy")[:, 0] == 1, t_country, "none")
if REV_ONLY:
    b_country = np.where(np.load(PREP / BS / "x_rev.npy")[:, 0] == 1, b_country, "none")
    t_country = np.where(np.load(PREP / TS / "x_rev.npy")[:, 0] == 1, t_country, "none")
if NAT_ONLY:
    b_country = np.where(np.load(PREP / BS / "x_nat.npy")[:, 0] == 1, b_country, "none")
    t_country = np.where(np.load(PREP / TS / "x_nat.npy")[:, 0] == 1, t_country, "none")
log(f"sources brest {dict(zip(*np.unique(b_src, return_counts=True)))} test {dict(zip(*np.unique(t_src, return_counts=True)))}")
rng = np.random.default_rng(20260927)
out = {}
for block in blocks:
    fn = {"base": "x_base", "q": "x_q", "coll": "x_coll"}.get(block, f"x_{block}")
    xb = np.load(PREP / BS / f"{fn}.npy", mmap_mode="r")
    xt = np.load(PREP / TS / f"{fn}.npy", mmap_mode="r")
    nm = names(block) or [f"{block}_{i}" for i in range(xb.shape[1])]
    res = {}
    for c in ("US", "India"):
        for s in sorted(set(b_src)):
            bi = np.flatnonzero((b_country == c) & (b_src == s))
            ti = np.flatnonzero((t_country == c) & (t_src == s))
            if len(bi) < 1000 or len(ti) < 1000:
                continue
            bi = np.sort(rng.choice(bi, min(N, len(bi)), replace=False))
            ti = np.sort(rng.choice(ti, min(N, len(ti)), replace=False))
            X = np.nan_to_num(np.vstack([np.asarray(xb[bi]), np.asarray(xt[ti])]), nan=-2.0)
            y = np.r_[np.zeros(len(bi)), np.ones(len(ti))]
            per = {}
            for j, n in enumerate(nm):
                a = roc_auc_score(y, X[:, j]) if np.ptp(X[:, j]) > 0 else 0.5
                per[n] = round(float(max(a, 1 - a)), 4)
            from catboost import CatBoostClassifier
            idx = rng.permutation(len(y))
            cut = int(0.7 * len(y))
            m = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, thread_count=8,
                                   allow_writing_files=False, random_seed=0)
            try:
                m.fit(X[idx[:cut]], y[idx[:cut]])
                auc = roc_auc_score(y[idx[cut:]], m.predict_proba(X[idx[cut:]])[:, 1])
                imp = dict(sorted(zip(nm, map(float, m.get_feature_importance())), key=lambda kv: -kv[1])[:6])
            except Exception as exc:  # all-constant block in this stratum
                auc, imp = 0.5, {"skipped": str(exc)[:80]}
            res[f"{c}|{s}"] = {"n_brest": int(len(bi)), "n_test": int(len(ti)), "block_auc": round(float(auc), 4),
                               "block_top_importance": imp,
                               "flagged_gt_0.6": {k: v for k, v in per.items() if v > 0.6}, "feature_auc": per}
            log(f"{block} {c}|{s}: block AUC {auc:.4f}; flagged {res[f'{c}|{s}']['flagged_gt_0.6']}")
    out[block] = res
(ROOT / "u4" / f"adv_val_{BS}_{'route_' if ROUTE_ONLY else ''}{'rev_' if REV_ONLY else ''}{'nat_' if NAT_ONLY else ''}{'_'.join(blocks)}.json").write_text(json.dumps(out, indent=1) + "\n")
