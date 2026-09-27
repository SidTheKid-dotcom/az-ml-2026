"""Adversarial validation of the K pair features: B_REST rows vs test rows, within country (label-free).

For each country, samples up to 400k rows per side, fits a small LightGBM (test = 1) on the K-only
feature columns (K3 probability + claimant-set features), and reports held-out AUC plus the top
features by gain; AUC > 0.6 flags a shift (plan E5 #10). Also reports per-feature means on both sides.

    python -m ber.k_adversarial --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ber import record_assign as ra


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    import lightgbm as lgb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B, x, half, grp_key, partition, component, n_s1 = ra.build_brest()
    cols = [ra.NAMES.index(n) for n in ra.CLAIM_NAMES if n != "is_none"]
    names = [n for n in ra.CLAIM_NAMES if n != "is_none"]
    xb_arena = x[:, cols]
    arena_code = np.load(ra.DATA / "brest_prep/country_code.npy")[B["left"]]  # 0 India, 1 US
    del x
    # test claimant features (as in record_assign.test_cmd)
    d = ra.DATA / "test_prep"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")  # 0 France, 1 India, 2 US
    xbt = np.load(d / "x_base.npy", mmap_mode="r")
    xqt = np.load(d / "x_q.npy", mmap_mode="r")
    k3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
    import pyarrow.parquet as pq

    rec_ids = pq.read_table(ra.DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    rows = np.flatnonzero(code[left] != 0)
    base = np.asarray(xbt[:, ra.BASE_NAMES.index("base")], dtype=np.float32)
    claim = ra.claimant_features(right, left, base, k3, np.zeros(len(left), bool), len(rec_ids), len(code), rows,
                                 rec_s3[right[rows]],
                                 np.nan_to_num(np.asarray(xbt[rows, ra.BASE_NAMES.index("c2")]), nan=-1.0),
                                 np.nan_to_num(np.asarray(xqt[rows, 0]), nan=-1.0), right[rows])
    xt = claim[:, :len(names)]
    test_code = code[left[rows]]
    rng = np.random.default_rng(0)
    report = {}
    for cname, ac, tc in (("India", 0, 1), ("US", 1, 2)):
        ia = np.flatnonzero(arena_code == ac)
        it = np.flatnonzero(test_code == tc)
        ia = rng.choice(ia, min(400_000, len(ia)), replace=False)
        it = rng.choice(it, min(400_000, len(it)), replace=False)
        X0 = np.vstack([xb_arena[ia], xt[it]])
        y0 = np.r_[np.zeros(len(ia)), np.ones(len(it))]
        perm = rng.permutation(len(y0))
        X0, y0 = X0[perm], y0[perm]
        n_tr = int(0.7 * len(y0))
        structural = {"rec_out_best_base", "rec_out_best_k3", "rec_n_claim_grp"}
        dense = structural | {"rec_n_claim_all", "rec_best_c2_grp", "rec_best_q_grp", "rec_k3_mass", "rank_base_in_rec",
                              "rank_k3_in_rec", "rec_base_ge_0_5", "rec_k3_ge_0_5"}
        robust = [n for n in names if n not in dense]
        res = {}
        for tag, keep in (("all", names), ("minus_structural", [n for n in names if n not in structural]),
                          ("robust", robust), ("robust_minus_rivals", [n for n in robust if "rival" not in n]),
                          ("k3_only", ["k3"]), ("s1_side", ["s1_n_rows", "s1_k3_mass", "s1_k3_ge_0_5", "k3_rank_in_s1",
                                                              "k3_gap_to_s1_top"])):
            j = [names.index(n) for n in keep]
            m = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "verbose": -1,
                           "num_threads": 30}, lgb.Dataset(X0[:n_tr][:, j], y0[:n_tr], feature_name=keep), 150)
            pr = m.predict(X0[n_tr:][:, j])
            yt = y0[n_tr:]
            order = np.argsort(pr)
            ranks = np.empty(len(pr))
            ranks[order] = np.arange(1, len(pr) + 1)
            npos, nneg = (yt == 1).sum(), (yt == 0).sum()
            auc = (ranks[yt == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)
            gain = dict(zip(keep, map(float, m.feature_importance("gain"))))
            top = sorted(gain.items(), key=lambda kv: -kv[1])[:8]
            res[tag] = {"auc": float(auc), "top_gain": top}
            print(cname, tag, round(float(auc), 4), [t[0] for t in top[:5]], flush=True)
        means = {n: [float(np.nanmean(xb_arena[ia][:, j])), float(np.nanmean(xt[it][:, j]))] for j, n in enumerate(names)}
        report[cname] = dict(res, means_arena_vs_test=means)
    (out / "adversarial.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
