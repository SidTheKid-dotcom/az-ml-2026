"""K3: generator-likelihood GBDT on edit-operation features only (``ber.edit_ops``).

Trained on arena partition A (two models on the two component-parity halves of A,
each predicting the other half, so every A row has an out-of-sample score), the
threshold is chosen on that A out-of-sample prediction (one owner + cap 11 among
A rows), and B_REST is scored with the mean of the two models: the arena B_REST
labels are read only for the report. Leave-one-country-out: a model fit on A-US
(A-India) is scored on B_REST India (US) with its threshold picked on an A holdout
of the training country. SLICE / HOLDOUT-2 labels are never read.

    python -m ber.k3_model train --data DIR --feat DIR/feat.npy --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, one_owner
from ber.edit_ops import FEATURES

T0 = time.time()
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.05, 0.951, 0.025), 3))
PARAMS = {"objective": "binary", "learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 200,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
          "max_bin": 255, "verbose": -1, "seed": 0}
ROUNDS = 700
SAMPLE_S1 = 350_000
CAT = ["legal_rel", "hn_rel", "name_digits_rel"]


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rank_within(group, score):
    order = np.lexsort((np.arange(len(score)), -score, group))
    g = group[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    idx = np.arange(len(g))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    rank = np.empty(len(g), dtype=np.int32)
    rank[order] = (idx - first).astype(np.int32)
    return rank


def decide(prob, left, right, threshold, cap=11):
    accept = one_owner(prob >= threshold, left, right, prob)
    rows = np.flatnonzero(accept)
    if rows.size:
        accept[rows[rank_within(left[rows], prob[rows]) >= cap]] = False
    return accept


def sweep(prob, left, right, label, true_count, population, thresholds=THRESHOLDS):
    grid = {}
    for t in thresholds:
        grid[t] = evaluate(decide(prob, left, right, t), left, label, true_count, population=population)["macro_f05"]
    best = max(grid, key=lambda t: (grid[t], -abs(t - 0.5)))
    return best, grid


def fit(x, y, rounds=ROUNDS, threads=30, valid=None):
    import lightgbm as lgb

    names = FEATURES + ["rec_s3"]
    cat_idx = [names.index(c) for c in CAT]
    params = dict(PARAMS, num_threads=threads)
    ds = lgb.Dataset(x, y.astype(np.float32), feature_name=names, categorical_feature=cat_idx, free_raw_data=True)
    return lgb.train(params, ds, num_boost_round=rounds)


def load(data, feat_path):
    d = Path(data)
    left = np.load(d / "arena_cand/left.npy").astype(np.int64)
    right = np.load(d / "arena_cand/right.npy").astype(np.int64)
    cand_rows = np.load(d / "arena_cand/cand_rows.npy")
    label = np.asarray(np.load(d / "labels/label.npy", mmap_mode="r")[cand_rows], dtype=bool)
    partition = np.load(d / "arena/s1_partition.npy")
    component = np.load(d / "arena/component_ids.npy")
    true_count = np.load(d / "labels/true_count.npy")
    country = np.asarray(pq.read_table(d / "arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    ids = pq.read_table(d / "arena/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    rec_s3 = np.fromiter((i.startswith("S3") for i in ids), dtype=bool, count=len(ids))
    del ids
    feat = np.load(feat_path, mmap_mode="r")
    if len(feat) != len(left):
        raise SystemExit("feature rows do not match arena candidate rows")
    return {"left": left, "right": right, "cand_rows": cand_rows, "label": label, "partition": partition,
            "component": component, "true_count": true_count, "country": country, "rec_s3": rec_s3, "feat": feat}


def xmat(D, rows):
    return np.hstack([np.asarray(D["feat"][rows], dtype=np.float32),
                      D["rec_s3"][D["right"][rows]].astype(np.float32)[:, None]])


def sample_s1(mask, n, seed):
    s1 = np.flatnonzero(mask)
    if len(s1) > n:
        s1 = np.sort(np.random.default_rng(seed).choice(s1, n, replace=False))
    out = np.zeros(len(mask), dtype=bool)
    out[s1] = True
    return out


def predict(model, D, rows, chunk=2_000_000):
    out = np.empty(len(rows), dtype=np.float32)
    for s in range(0, len(rows), chunk):
        out[s:s + chunk] = model.predict(xmat(D, rows[s:s + chunk]), num_threads=30)
    return out


def slim(r):
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy",
            "empty_predictions")
    out = {k: r[k] for k in keys if k in r}
    if "by_country" in r:
        out["by_country"] = {c: {k: v[k] for k in ("s1", "macro_f05", "precision", "recall", "false_links")}
                             for c, v in r["by_country"].items()}
    return out


def ea_stats(accept, label, rec_empty):
    m = rec_empty
    acc, lab = accept & m, label & m
    return {"ea_candidate_true": int(lab.sum()), "ea_accepted": int(acc.sum()),
            "ea_true_accepted": int((acc & label).sum()),
            "ea_recall_in_candidates": float((acc & label).sum() / max(1, lab.sum())),
            "ea_precision": float((acc & label).sum() / max(1, acc.sum()))}


def train_cmd(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    D = load(a.data, a.feat)
    left, right, label, part, comp = D["left"], D["right"], D["label"], D["partition"], D["component"]
    tc, country = D["true_count"], D["country"]
    n_s1 = len(part)
    log(f"loaded {len(left)} arena candidate rows; A S1 {int((part == 0).sum())}, B_REST {int((part == 1).sum())}")
    row_part = part[left]
    a_rows_all = np.flatnonzero(row_part == 0)
    b_rows = np.flatnonzero(row_part == 1)
    half = (comp % 2).astype(np.int8)
    prob_all = np.full(len(left), np.nan, dtype=np.float32)
    models = []
    for h in (0, 1):
        s1_mask = sample_s1((part == 0) & (half == h), SAMPLE_S1, seed=h)
        rows = np.flatnonzero(s1_mask[left])
        log(f"half {h}: fit on {int(s1_mask.sum())} A S1s, {len(rows)} rows, {int(label[rows].sum())} true")
        model = fit(xmat(D, rows), label[rows], threads=a.threads)
        model.save_model(str(out / f"k3_half{h}.txt"))
        models.append(model)
        other = a_rows_all[half[left[a_rows_all]] != h]
        prob_all[other] = predict(model, D, other)
        log(f"half {h}: predicted {len(other)} A rows of the other half")
    rest = np.flatnonzero(row_part != 0)
    prob_all[rest] = 0.5 * (predict(models[0], D, rest) + predict(models[1], D, rest))
    np.save(out / "k3_prob_arena_cand.npy", prob_all)
    log("scored every arena candidate row (A: other-half model; B_REST/SLICE: mean of both)")

    # threshold on the A out-of-sample prediction
    pa = part == 0
    t_a, grid_a = sweep(prob_all[a_rows_all], left[a_rows_all], right[a_rows_all], label[a_rows_all], tc, pa)
    log(f"A OOF threshold {t_a}: {grid_a[t_a]:.6f}")
    # B_REST
    pb = part == 1
    lb, rb, yb, pbr = left[b_rows], right[b_rows], label[b_rows], prob_all[b_rows]
    acc = decide(pbr, lb, rb, t_a)
    res = evaluate(acc, lb, yb, tc, population=pb, country=country, return_per_s1=True)
    per_k3 = res.pop("per_s1")
    rec_empty = np.asarray(D["feat"][b_rows, FEATURES.index("rec_addr_empty")], dtype=np.float32) > 0.5
    t_b, grid_b = sweep(pbr, lb, rb, yb, tc, pb)
    report = {"question": "K3 edit-operation GBDT (no reranker scores): A-trained, B_REST evaluated",
              "features": FEATURES + ["rec_s3"], "params": PARAMS, "rounds": ROUNDS, "sample_s1_per_half": SAMPLE_S1,
              "a_oof_threshold": t_a, "a_oof_grid": {str(k): v for k, v in grid_a.items()},
              "b_rest": slim(res), "b_rest_ea": ea_stats(acc, yb, rec_empty),
              "b_rest_oracle_threshold_info_only": {"threshold": t_b, "f05": grid_b[t_b]},
              "b_rest_grid_info_only": {str(k): v for k, v in grid_b.items()}}
    log(f"B_REST at A threshold: {json.dumps(slim(res))}")
    np.save(out / "per_s1_k3_brest.npy", per_k3)
    imp = {n: float(v) for n, v in zip(FEATURES + ["rec_s3"], models[0].feature_importance("gain"))}
    report["importance_gain_half0"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))

    # LOCO
    loco = {}
    for train_c, test_c in (("US", "India"), ("India", "US")):
        s1_mask = sample_s1((part == 0) & (country == train_c) & (comp % 5 != 0), SAMPLE_S1, seed=7)
        rows = np.flatnonzero(s1_mask[left])
        model = fit(xmat(D, rows), label[rows], threads=a.threads)
        model.save_model(str(out / f"k3_loco_{train_c}.txt"))
        hold_s1 = (part == 0) & (country == train_c) & (comp % 5 == 0)
        hr = np.flatnonzero(hold_s1[left])
        ph = predict(model, D, hr)
        t_h, _ = sweep(ph, left[hr], right[hr], label[hr], tc, hold_s1)
        pop = pb & (country == test_c)
        tr = b_rows[pop[lb]]
        pt = predict(model, D, tr)
        r = evaluate(decide(pt, left[tr], right[tr], t_h), left[tr], label[tr], tc, population=pop)
        pooled = res["by_country"][test_c]
        t_o, g_o = sweep(pt, left[tr], right[tr], label[tr], tc, pop)
        loco[f"{train_c}_to_{test_c}"] = {"threshold_from_train_country_holdout": t_h, "result": slim(r),
                                          "pooled_model_same_s1": {k: pooled[k] for k in ("macro_f05", "precision",
                                                                                           "recall")},
                                          "oracle_threshold_info_only": {"threshold": t_o, "f05": g_o[t_o]}}
        log(f"LOCO {train_c}->{test_c}: {r['macro_f05']:.6f} (pooled {pooled['macro_f05']:.6f}) at {t_h}")
    report["loco"] = loco
    report["elapsed_seconds"] = round(time.time() - T0, 1)
    report["labels_read"] = "A and B_REST only; SLICE/HOLDOUT-2 never read"
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log("done")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train"])
    p.add_argument("--data", default="/home/ubuntu/k/data")
    p.add_argument("--feat", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=30)
    a = p.parse_args()
    train_cmd(a)


if __name__ == "__main__":
    main()
