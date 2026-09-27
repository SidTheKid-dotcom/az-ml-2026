"""K2: record-centric assignment. For every record, its claimant S1s plus a "none" option compete in one
softmax (LightGBM with a grouped softmax cross-entropy objective, one group per record); S1 lists are
built from P(owner = s | record) (threshold or per-S1 expected-F0.5 decoding), one owner by construction.

Arena protocol (B_REST only, labels of A/SLICE/HOLDOUT-2 never read):
* rows = B_REST candidate rows of the C2 pair tree (U prep), grouped by (record, component half);
* pair features = v1 + C2 (x_base), Qwen (x_q), K3 edit-op features + K3 probability (A-trained);
* claimant-set features over every arena candidate row of the record (any partition, features only):
  counts, best/second v1 base and K3 prob, the best claimant outside B_REST, record K3 mass;
  S1-side features: list size, K3 mass, rank of the row, gap to the S1 top;
* 2-fold cross-fit by component parity (as the pair tree): the model fit on one half predicts the other;
  each half's decision rule is chosen on the other half's out-of-fold probabilities.

    python -m ber.record_assign evaluate --out DIR
    python -m ber.record_assign test --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, iterated_best_response, one_owner
from ber.edit_ops import FEATURES as K3F

T0 = time.time()
DATA = Path("/home/ubuntu/k/data")
K3DIR = Path("/home/ubuntu/k/out/k3")
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.2, 0.951, 0.025), 3))
LAMS = (0.02, 0.055, 0.1)
PARAMS = {"learning_rate": 0.08, "num_leaves": 255, "min_data_in_leaf": 200, "feature_fraction": 0.8,
          "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0, "max_bin": 255, "verbose": -1, "seed": 0}
ROUNDS = 500
BASE_NAMES = ["blend", "base", "blend_rank_in_s1", "blend_gap_to_s1_top", "s1_candidates_ge_0_5", "rival_blend",
              "record_claimants_ge_0_5", "name_exact", "log1p_name_mult", "record_kind", "key_self", "key_other",
              "c2", "c2_blend", "c2_rank_in_s1", "c2_gap_to_s1_top", "s1_c2_ge_0_5", "rival_c2",
              "record_claimants_c2_ge_0_5", "c2_minus_rival"]
Q_NAMES = ["q", "q_blend", "q_rank_in_s1", "q_gap_to_s1_top", "s1_q_ge_0_5", "rival_q", "record_claimants_q_ge_0_5",
           "q_minus_rival", "c2q_mean", "c2q_min", "c2_minus_q"]
CLAIM_NAMES = ["k3", "rec_s3", "rec_n_claim_all", "rec_n_claim_grp", "rec_best_base", "rec_best_k3", "rec_k3_mass",
               "rec_base_ge_0_5", "rec_k3_ge_0_5", "rec_out_best_base", "rec_out_best_k3", "rec_best_c2_grp",
               "rec_best_q_grp", "rival_base_all", "rival_k3_all", "base_minus_rival_all", "k3_minus_rival_all",
               "rank_base_in_rec", "rank_k3_in_rec", "s1_n_rows", "s1_k3_mass", "s1_k3_ge_0_5", "k3_rank_in_s1",
               "k3_gap_to_s1_top", "is_none"]
REC_LEVEL = ["rec_s3", "rec_n_claim_all", "rec_n_claim_grp", "rec_best_base", "rec_best_k3", "rec_k3_mass",
             "rec_base_ge_0_5", "rec_k3_ge_0_5", "rec_out_best_base", "rec_out_best_k3", "rec_best_c2_grp",
             "rec_best_q_grp"]
NAMES = BASE_NAMES + Q_NAMES + K3F + CLAIM_NAMES


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


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


def top_two(group, score, n):
    order = np.lexsort((-score, group))
    g, s = group[order], score[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    top1 = np.full(n, -1.0, dtype=np.float32)
    top2 = np.full(n, -1.0, dtype=np.float32)
    top1[g[start]] = s[start]
    second = np.zeros(len(g), dtype=bool)
    second[1:] = start[:-1] & ~start[1:]
    top2[g[second]] = s[second]
    return top1, top2


def group_max(group, value, n, fill=-1.0):
    out = np.full(n, fill, dtype=np.float32)
    np.maximum.at(out, group, value.astype(np.float32))
    return out


def claimant_features(all_right, all_left, all_base, all_k3, all_outside, n_rec, n_s1, rows, grp_rec, grp_c2,
                      grp_q, grp_key):
    """Claimant-set features for candidate rows ``rows`` (indices into the all-claimant arrays)."""
    b = np.nan_to_num(all_base, nan=0.0).astype(np.float32)
    k = np.nan_to_num(all_k3, nan=0.0).astype(np.float32)
    n_claim = np.bincount(all_right, minlength=n_rec).astype(np.float32)
    b1, b2 = top_two(all_right, b, n_rec)
    k1, k2 = top_two(all_right, k, n_rec)
    kmass = np.bincount(all_right, weights=k, minlength=n_rec).astype(np.float32)
    b05 = np.bincount(all_right[b >= 0.5], minlength=n_rec).astype(np.float32)
    k05 = np.bincount(all_right[k >= 0.5], minlength=n_rec).astype(np.float32)
    ob = group_max(all_right[all_outside], b[all_outside], n_rec)
    ok = group_max(all_right[all_outside], k[all_outside], n_rec)
    rank_b = rank_within(all_right, b)
    rank_k = rank_within(all_right, k)
    s1_n = np.bincount(all_left, minlength=n_s1).astype(np.float32)
    s1_km = np.bincount(all_left, weights=k, minlength=n_s1).astype(np.float32)
    s1_k05 = np.bincount(all_left[k >= 0.5], minlength=n_s1).astype(np.float32)
    s1_top, _ = top_two(all_left, k, n_s1)
    rank_ks1 = rank_within(all_left, k)
    r, l = all_right[rows], all_left[rows]
    br, kr = b[rows], k[rows]
    rival_b = np.where(br >= b1[r], b2[r], b1[r])
    rival_k = np.where(kr >= k1[r], k2[r], k1[r])
    # group-level (record, half) counts among the modelled rows
    _, ginv, gcount = np.unique(grp_key, return_inverse=True, return_counts=True)
    n_grp = gcount[ginv].astype(np.float32)
    c2g = group_max(ginv, grp_c2, len(gcount))[ginv]
    qg = group_max(ginv, grp_q, len(gcount))[ginv]
    cols = [kr, grp_rec, n_claim[r], n_grp, b1[r], k1[r], kmass[r], b05[r], k05[r], ob[r], ok[r], c2g, qg,
            rival_b, rival_k, br - rival_b, kr - rival_k, np.minimum(rank_b[rows], 50), np.minimum(rank_k[rows], 50),
            s1_n[l], s1_km[l], s1_k05[l], np.minimum(rank_ks1[rows], 50), kr - s1_top[l], np.zeros(len(rows))]
    return np.column_stack(cols).astype(np.float32)


def add_none_rows(x, grp_key, label=None):
    """Append one "none" row per group (record-level columns copied, pair columns NaN); sort by group."""
    uniq, first = np.unique(grp_key, return_index=True)
    none = np.full((len(uniq), x.shape[1]), np.nan, dtype=np.float32)
    for name in REC_LEVEL:
        j = NAMES.index(name)
        none[:, j] = x[first, j]
    none[:, NAMES.index("is_none")] = 1.0
    X = np.vstack([x, none])
    key = np.concatenate([grp_key, uniq])
    is_pair = np.concatenate([np.ones(len(x), bool), np.zeros(len(uniq), bool)])
    src = np.concatenate([np.arange(len(x)), np.full(len(uniq), -1)])
    order = np.argsort(key, kind="stable")
    X, key, is_pair, src = X[order], key[order], is_pair[order], src[order]
    y = None
    if label is not None:
        yl = np.concatenate([label.astype(np.float32), np.zeros(len(uniq), np.float32)])[order]
        _, ginv = np.unique(key, return_inverse=True)
        has_true = np.bincount(ginv, weights=yl) > 0
        y = np.where(is_pair, yl, (~has_true[ginv]).astype(np.float32))
    _, ginv = np.unique(key, return_inverse=True)
    return X, ginv.astype(np.int64), is_pair, src, y


def softmax_groups(score, g, n_groups=None):
    """Softmax of ``score`` within groups ``g`` (``g`` non-decreasing: rows sorted by group)."""
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    m = np.maximum.reduceat(score, starts)
    e = np.exp(score - m[g])
    z = np.add.reduceat(e, starts)
    return e / z[g]


def make_objective(g, n_groups):
    def obj(preds, data):
        y = data.get_label()
        p = softmax_groups(preds.astype(np.float64), g, n_groups)
        grad = p - y
        hess = np.maximum(p * (1.0 - p), 1e-6)
        return grad, hess
    return obj


def fit(X, y, g, threads, rounds=ROUNDS):
    import lightgbm as lgb

    n_groups = int(g.max()) + 1
    params = dict(PARAMS, objective=make_objective(g, n_groups), num_threads=threads)
    ds = lgb.Dataset(X, y, feature_name=NAMES, categorical_feature=[NAMES.index(c) for c in ("legal_rel", "hn_rel")],
                     free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def fit_binary(X, y, threads, rounds=ROUNDS):
    import lightgbm as lgb

    params = dict(PARAMS, objective="binary", num_threads=threads)
    ds = lgb.Dataset(X, y.astype(np.float32), feature_name=NAMES,
                     categorical_feature=[NAMES.index(c) for c in ("legal_rel", "hn_rel")], free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=rounds)


def raw_predict(model, X, chunk=2_000_000):
    out = np.empty(len(X), dtype=np.float64)
    for s in range(0, len(X), chunk):
        out[s:s + chunk] = model.predict(X[s:s + chunk], raw_score=True, num_threads=30)
    return out


def cap(accept, p, left, k=11):
    rows = np.flatnonzero(accept)
    if rows.size:
        accept = accept.copy()
        accept[rows[rank_within(left[rows], p[rows]) >= k]] = False
    return accept


def decode(p, left, right, rule, n_s1):
    kind, val = rule
    if kind == "thr":
        return cap(one_owner(p >= val, left, right, p), p, left)
    acc, _ = iterated_best_response(p, left, right, lam=val, max_k=11, n_s1=n_s1)
    return cap(one_owner(acc, left, right, p), p, left)


# ----------------------------------------------------------------------------- data

def load_brest():
    d = DATA / "brest_prep"
    B = {k: np.load(d / f"{k}.npy") for k in ("left", "right", "pair_rows", "label", "rec_empty", "oof_c2tree")}
    B["x_base"] = np.load(d / "x_base.npy", mmap_mode="r")
    B["x_q"] = np.load(d / "x_q.npy", mmap_mode="r")
    return B


def arena_claimants():
    left = np.load(DATA / "arena_cand/left.npy").astype(np.int64)
    right = np.load(DATA / "arena_cand/right.npy").astype(np.int64)
    cand_rows = np.load(DATA / "arena_cand/cand_rows.npy")
    base = np.asarray(np.load(DATA / "arena/base_probabilities.npy", mmap_mode="r")[cand_rows], dtype=np.float32)
    k3 = np.load(K3DIR / "k3_prob_arena_cand.npy")
    return left, right, cand_rows, base, k3


def build_brest():
    B = load_brest()
    left_all, right_all, cand_rows, base_all, k3_all = arena_claimants()
    partition = np.load(DATA / "arena/s1_partition.npy")
    component = np.load(DATA / "arena/component_ids.npy")
    n_s1, n_rec = len(partition), int(max(right_all.max(), B["right"].max())) + 1
    pos = np.searchsorted(cand_rows, B["pair_rows"])
    if not np.all(cand_rows[pos] == B["pair_rows"]):
        raise SystemExit("B_REST rows are not a subset of the arena candidate rows")
    ids = pq.read_table(DATA / "arena/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    rec_s3 = np.fromiter((i.startswith("S3") for i in ids), dtype=bool, count=len(ids)).astype(np.float32)
    del ids
    left, right = B["left"], B["right"]
    half = (component[left] % 2).astype(np.int64)
    grp_key = right * 2 + half
    xb = np.asarray(B["x_base"], dtype=np.float32)
    xq = np.asarray(B["x_q"], dtype=np.float32)
    feat = np.asarray(np.load(K3DIR.parent / "k3feat_arena/feat.npy", mmap_mode="r")[pos], dtype=np.float32)
    outside = partition[left_all] != 1
    claim = claimant_features(right_all, left_all, base_all, k3_all, outside, n_rec, n_s1, pos, rec_s3[right],
                              np.nan_to_num(xb[:, BASE_NAMES.index("c2")], nan=-1.0),
                              np.nan_to_num(xq[:, 0], nan=-1.0), grp_key)
    x = np.hstack([xb, xq, feat, claim])
    del xb, xq, feat, claim
    log(f"B_REST feature matrix {x.shape}")
    return B, x, half, grp_key, partition, component, n_s1


def apply_drop(x, drop):
    """Ablation: named feature groups become all-missing (never split on)."""
    if not drop:
        return
    groups = {"k3": K3F + ["k3", "rec_best_k3", "rec_k3_mass", "rec_k3_ge_0_5", "rec_out_best_k3", "rival_k3_all",
                           "k3_minus_rival_all", "rank_k3_in_rec", "s1_k3_mass", "s1_k3_ge_0_5", "k3_rank_in_s1",
                           "k3_gap_to_s1_top"],
              "q": Q_NAMES + ["rec_best_q_grp"],
              "claim": [n for n in CLAIM_NAMES if n not in ("k3", "rec_s3", "is_none")],
              "out": ["rec_out_best_base", "rec_out_best_k3", "rec_n_claim_grp"],
              # density/partition-sensitive record features (adversarial AUC B_REST vs test 0.88-0.90)
              "dense": ["rec_out_best_base", "rec_out_best_k3", "rec_n_claim_grp", "rec_n_claim_all",
                        "rec_best_c2_grp", "rec_best_q_grp", "rec_k3_mass", "rank_base_in_rec", "rank_k3_in_rec",
                        "rec_base_ge_0_5", "rec_k3_ge_0_5"]}
    for gname in drop.split(","):
        for n in groups[gname]:
            x[:, NAMES.index(n)] = np.nan
    log(f"dropped feature groups {drop}")


def evaluate_cmd(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B, x, half, grp_key, partition, component, n_s1 = build_brest()
    left, right, label = B["left"], B["right"], B["label"]
    if a.with_coll:  # B_REST-only variant: append U's 25 collective features (no test counterpart yet)
        coll = np.asarray(np.load(DATA / "brest_prep/x_coll.npy", mmap_mode="r"), dtype=np.float32)
        x = np.hstack([x, coll])
        NAMES.extend(f"coll_{i}" for i in range(coll.shape[1]))
        del coll
        log(f"appended x_coll: {x.shape}")
    apply_drop(x, a.drop)
    true_count = np.load(DATA / "labels/true_count.npy")
    country = np.asarray(pq.read_table(DATA / "arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    pop = partition == 1
    s1_half = (component % 2).astype(np.int64)
    # baseline: the C2 pair tree OOF with its per-half thresholds (0.985511)
    thr = np.where(half == 1, 0.75, 0.725)
    base_acc = cap(one_owner(B["oof_c2tree"] >= thr, left, right, B["oof_c2tree"]), B["oof_c2tree"], left)
    base_res = evaluate(base_acc, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per_base = base_res.pop("per_s1")
    log(f"pair tree OOF baseline: {base_res['macro_f05']:.6f}")
    p = np.zeros(len(left), dtype=np.float64)
    p_none = {}
    models = {}
    for h in (0, 1):
        tr = np.flatnonzero(half == h)
        te = np.flatnonzero(half != h)
        if a.objective == "binary":  # ablation: same features, per-pair logloss, no record grouping / none option
            model = fit_binary(x[tr], label[tr], a.threads, rounds=a.rounds)
            model.save_model(str(out / f"k2_half{h}.txt"))
            models[h] = model
            p[te] = model.predict(x[te], num_threads=30)
            p_none[h] = None
            log(f"half {h}: binary model predicted {len(te)} rows of the other half")
            continue
        X, g, is_pair, src, y = add_none_rows(x[tr], grp_key[tr], label[tr])
        log(f"half {h}: {len(X)} rows ({int(is_pair.sum())} pairs, {int((~is_pair).sum())} none), "
            f"none-true {int(y[~is_pair].sum())}")
        model = fit(X, y, g, a.threads, rounds=a.rounds)
        model.save_model(str(out / f"k2_half{h}.txt"))
        models[h] = model
        del X, y
        X, g, is_pair, src, _ = add_none_rows(x[te], grp_key[te])
        prob = softmax_groups(raw_predict(model, X), g, int(g.max()) + 1)
        p[te[src[is_pair]]] = prob[is_pair]
        p_none[h] = float(prob[~is_pair].mean())
        del X
        log(f"half {h}: predicted {len(te)} rows of the other half")
    np.save(out / "oof_k2.npy", p.astype(np.float32))
    # rule per half: chosen on the other half's OOF
    rules = [("thr", t) for t in THRESHOLDS] + [("ef", lam) for lam in LAMS]
    grid = {h: {} for h in (0, 1)}
    for rule in rules:
        acc = decode(p, left, right, rule, n_s1)
        per = evaluate(acc, left, label, true_count, return_per_s1=True)["per_s1"]
        for h in (0, 1):
            grid[h][rule] = float(per[pop & (s1_half == h)].mean())
        log(f"rule {rule}: half0 {grid[0][rule]:.6f} half1 {grid[1][rule]:.6f}")
    pick = {h: max(grid[h], key=lambda r: grid[h][r]) for h in (0, 1)}
    # half h rows use the rule picked on half 1-h
    final = np.zeros(len(left), dtype=bool)
    for h in (0, 1):
        acc = decode(p, left, right, pick[1 - h], n_s1)
        final |= acc & (half == h)
    final = cap(one_owner(final, left, right, p), p, left)
    res = evaluate(final, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per_k2 = res.pop("per_s1")
    boot = component_bootstrap(per_base, per_k2, component, n=a.bootstrap, seed=20260927, population=pop)
    ea = B["rec_empty"]

    def ea_stats(acc):
        lab = label & ea
        return {"ea_accepted": int((acc & ea).sum()), "ea_true_accepted": int((acc & lab).sum()),
                "ea_candidate_true": int(lab.sum()),
                "ea_recall_in_candidates": float((acc & lab).sum() / max(1, lab.sum())),
                "ea_precision": float((acc & lab).sum() / max(1, (acc & ea).sum()))}

    full_pick = max(rules, key=lambda r: grid[0][r] + grid[1][r])
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy",
            "empty_predictions")
    slim = lambda r: dict({k: r[k] for k in keys}, by_country={c: {k: v[k] for k in ("macro_f05", "precision",
                                                                                          "recall", "false_links")}
                                                                for c, v in r["by_country"].items()})
    report = {"question": "K2 record-centric softmax assignment vs C2 pair tree OOF (B_REST, 2-fold component cross-fit)",
              "features": NAMES, "objective": a.objective, "dropped_groups": a.drop, "with_coll": a.with_coll, "params": {k: v for k, v in PARAMS.items()}, "rounds": a.rounds,
              "pair_tree_baseline": slim(base_res), "pair_tree_ea": ea_stats(base_acc),
              "k2": slim(res), "k2_ea": ea_stats(final), "k2_minus_pair_tree": boot,
              "picks": {str(h): list(pick[h]) for h in (0, 1)}, "full_pick_for_test": list(full_pick),
              "grid": {str(h): {f"{r[0]}:{r[1]}": v for r, v in grid[h].items()} for h in (0, 1)},
              "mean_p_none_other_half": p_none, "labels_read": "B_REST only",
              "elapsed_seconds": round(time.time() - T0, 1)}
    imp = dict(zip(NAMES, map(float, models[0].feature_importance("gain"))))
    report["importance_gain_half0_top40"] = dict(sorted(imp.items(), key=lambda kv: -kv[1])[:40])
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    np.save(out / "per_s1_k2.npy", per_k2)
    np.save(out / "per_s1_pair_tree.npy", per_base)
    log(f"K2: {res['macro_f05']:.6f} vs pair tree {base_res['macro_f05']:.6f}; boot {boot}")


def evaluate_folds_cmd(a):
    """Binary pair model, F-fold component cross-fit (fold = component % F); each fold's rule is chosen on the
    OOF of the other folds; test uses the mean of the F fold models."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B, x, half, grp_key, partition, component, n_s1 = build_brest()
    left, right, label = B["left"], B["right"], B["label"]
    apply_drop(x, a.drop)
    F = a.folds
    true_count = np.load(DATA / "labels/true_count.npy")
    country = np.asarray(pq.read_table(DATA / "arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    pop = partition == 1
    fold_row = (component[left] % F).astype(np.int64)
    s1_fold = (component % F).astype(np.int64)
    thr = np.where(half == 1, 0.75, 0.725)
    base_acc = cap(one_owner(B["oof_c2tree"] >= thr, left, right, B["oof_c2tree"]), B["oof_c2tree"], left)
    base_res = evaluate(base_acc, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per_base = base_res.pop("per_s1")
    p = np.zeros(len(left), dtype=np.float64)
    models = []
    for f in range(F):
        tr, te = np.flatnonzero(fold_row != f), np.flatnonzero(fold_row == f)
        model = fit_binary(x[tr], label[tr], a.threads, rounds=a.rounds)
        model.save_model(str(out / f"k2_fold{f}.txt"))
        models.append(model)
        p[te] = model.predict(x[te], num_threads=30)
        log(f"fold {f}: fit {len(tr)} rows, predicted {len(te)}")
    np.save(out / "oof_k2.npy", p.astype(np.float32))
    rules = [("thr", t) for t in THRESHOLDS]
    per_rule = {}
    for rule in rules:
        per_rule[rule] = evaluate(decode(p, left, right, rule, n_s1), left, label, true_count,
                                  return_per_s1=True)["per_s1"]
    pick = {}
    for f in range(F):
        m = pop & (s1_fold != f)
        pick[f] = max(rules, key=lambda r: per_rule[r][m].mean())
    final = np.zeros(len(left), dtype=bool)
    for f in range(F):
        final |= decode(p, left, right, pick[f], n_s1) & (fold_row == f)
    final = cap(one_owner(final, left, right, p), p, left)
    res = evaluate(final, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per_k2 = res.pop("per_s1")
    boot = component_bootstrap(per_base, per_k2, component, n=a.bootstrap, seed=20260927, population=pop)
    full_pick = max(rules, key=lambda r: per_rule[r][pop].mean())
    ea = B["rec_empty"]
    lab = label & ea
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy",
            "empty_predictions")
    report = {"question": f"K binary pair model, {F}-fold component cross-fit, vs C2 pair tree OOF (B_REST)",
              "features": NAMES, "objective": "binary", "folds": F, "dropped_groups": a.drop, "rounds": a.rounds,
              "params": PARAMS, "pair_tree_baseline": {k: base_res[k] for k in keys},
              "k2": dict({k: res[k] for k in keys}, by_country={c: {k: v[k] for k in ("macro_f05", "precision",
                                                                                          "recall", "false_links")}
                                                                for c, v in res["by_country"].items()}),
              "k2_ea": {"ea_recall_in_candidates": float((final & lab).sum() / max(1, lab.sum())),
                        "ea_precision": float((final & lab).sum() / max(1, (final & ea).sum()))},
              "k2_minus_pair_tree": boot, "picks": {str(f): list(pick[f]) for f in range(F)},
              "full_pick_for_test": list(full_pick),
              "full_oof_grid": {str(r[1]): float(per_rule[r][pop].mean()) for r in rules},
              "labels_read": "B_REST only", "elapsed_seconds": round(time.time() - T0, 1)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    np.save(out / "per_s1_k2.npy", per_k2)
    log(f"K {F}-fold: {res['macro_f05']:.6f} vs pair tree {base_res['macro_f05']:.6f}; boot {boot}; picks {pick}")


def test_cmd(a):
    """US/India rows on the test candidates: both half models (mean raw score), one group per record."""
    import lightgbm as lgb

    out = Path(a.out)
    rep = json.loads((out / "report.json").read_text())
    kind, val = rep["full_pick_for_test"]
    fold_files = sorted(out.glob("k2_fold*.txt"))
    models = [lgb.Booster(model_file=str(f)) for f in fold_files] if fold_files else \
        [lgb.Booster(model_file=str(out / f"k2_half{h}.txt")) for h in (0, 1)]
    d = DATA / "test_prep"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")
    xb = np.asarray(np.load(d / "x_base.npy", mmap_mode="r"), dtype=np.float32)
    xq = np.asarray(np.load(d / "x_q.npy", mmap_mode="r"), dtype=np.float32)
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    k3 = np.load(Path(a.k3_test) / "k3_prob_test_cand.npy")
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"])
    s1_ids = np.asarray(ref.column("entity_id").to_pylist(), dtype=object)
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    n_s1, n_rec = len(s1_ids), len(rec_ids)
    base = xb[:, BASE_NAMES.index("base")]
    outside = np.zeros(len(left), dtype=bool)
    rows = np.flatnonzero(code[left] != 0)  # US/India rows only; claimant sets still use every test row
    claim = claimant_features(right, left, base, k3, outside, n_rec, n_s1, rows, rec_s3[right[rows]],
                              np.nan_to_num(xb[rows, BASE_NAMES.index("c2")], nan=-1.0),
                              np.nan_to_num(xq[rows, 0], nan=-1.0), right[rows])
    x = np.hstack([xb[rows], xq[rows], feat[rows], claim])
    del xb, xq, feat, claim
    log(f"test US/India feature matrix {x.shape}")
    p = np.zeros(len(left), dtype=np.float64)
    if rep.get("objective") == "binary":  # mean of the two half models' probabilities
        p[rows] = np.mean([m.predict(x, num_threads=30) for m in models], axis=0)
        del x
    else:
        X, g, is_pair, src, _ = add_none_rows(x, right[rows])
        del x
        raw = 0.5 * (raw_predict(models[0], X) + raw_predict(models[1], X))
        prob = softmax_groups(raw, g)
        p[rows[src[is_pair]]] = prob[is_pair]
    np.save(out / "k2_prob_test_cand.npy", p.astype(np.float32))
    acc = decode(p, left, right, (kind, val), n_s1)
    keep = code != 0
    rows_sel = np.flatnonzero(acc)
    rows_sel = rows_sel[np.lexsort((-p[rows_sel], left[rows_sel]))]
    starts = np.searchsorted(left[rows_sel], np.arange(n_s1), side="left")
    stops = np.searchsorted(left[rows_sel], np.arange(n_s1), side="right")
    path = out / "k2_us_india_rows.tsv"
    with open(path.with_suffix(".tmp"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i in np.flatnonzero(keep):
            f.write(f"{s1_ids[i]}\t{','.join(rec_ids[right[rows_sel[starts[i]:stops[i]]]])}\n")
    path.with_suffix(".tmp").replace(path)
    lengths = (stops - starts)[keep]
    oof = np.load(out / "oof_k2.npy")
    result = {"objective": rep.get("objective", "softmax"), "rule": [kind, val], "file": str(path),
              "share_p_ge_rule_test_us_india": float((p[rows] >= val).mean()) if kind == "thr" else None,
              "share_p_ge_rule_b_rest_oof": float((oof >= val).mean()) if kind == "thr" else None, "sha256": sha256(path), "rows": int(keep.sum()),
              "links": int(lengths.sum()), "longest": int(lengths.max()), "empty_rows": int((lengths == 0).sum()),
              "mean_list": float(lengths.mean()),
              "known_shift": "test claimant sets have no outside-partition claimants (arena: 24% of gated B_REST "
                             "rows had one); rec_n_claim_grp equals rec_n_claim_all on test",
              "labels_read": False}
    (out / "test_report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    log(json.dumps(result, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["evaluate", "evaluate_folds", "test"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--k3-test", default="/home/ubuntu/k/out/k3_test")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=30)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--objective", choices=["softmax", "binary"], default="softmax")
    p.add_argument("--drop", default="", help="ablation: comma list of k3,q,claim,out")
    p.add_argument("--with-coll", action="store_true", help="append U's x_coll (B_REST only)")
    a = p.parse_args()
    {"evaluate": evaluate_cmd, "evaluate_folds": evaluate_folds_cmd, "test": test_cmd}[a.command](a)


if __name__ == "__main__":
    main()
