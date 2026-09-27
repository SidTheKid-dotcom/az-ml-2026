"""Two-stage string-feature LightGBM on the arena: train on partition A, evaluate on B_REST.

    python -m ber.string_gbdt_train train --feat O/arena --data D --work W
    python -m ber.string_gbdt_train apply --feat O/test_France --work W --out DIR
    python -m ber.string_gbdt_train loco --feat O/arena --data D --work W

Stage 1: pairwise string features only (no country). Two LightGBMs on
component-disjoint halves of an A-fit S1 sample give out-of-fold P1 on the
sample; every other pair gets the mean of both. Stage 2: stage-1 features plus
listwise/competition features of P1 (rank, gap to top/next, near-top count,
best rival S1 for the record, claimants, exact-core-name claimants) on rows
with P1 >= GATE or S1 rank < 3. Early stopping and the threshold/variant are
chosen on an A-tune sample (component-disjoint from A-fit); B_REST is scored
once. SLICE labels are never read (SLICE S1s only appear as unlabeled rivals).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule
from ber.pair_tree_arena import decide, rank_within, top_two
from ber.string_gbdt import CATEGORICAL, FEATURES, log

L2 = ["p1", "p1_rank", "p1_gap_top", "p1_gap_next", "s1_n_near", "s1_n_ge05", "s1_sum", "rival", "p1_minus_rival",
      "rec_rank", "rec_n_ge05", "rec_ncand", "s1_ncand", "rec_core_eq_claimants", "s1_core_eq_cands"]
GATE = 0.001
SEED = 0
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.30, 0.951, 0.025), 3))
PARAMS = dict(objective="binary", learning_rate=0.1, num_leaves=255, min_data_in_leaf=200, feature_fraction=0.85,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255, num_threads=16, verbose=-1,
              seed=SEED)


def load_feat(feat):
    feat = Path(feat)
    with np.load(feat / "pairs.npz") as z:
        pairs = {k: z[k] for k in z.files}
    X = np.load(feat / "X.npy", mmap_mode="r")
    s1 = pd.read_parquet(feat / "s1.parquet")
    rec = pd.read_parquet(feat / "rec.parquet")
    return pairs, X, s1, rec


def take(X, rows, chunk=4_000_000):
    out = np.empty((len(rows), X.shape[1]), dtype=np.float32)
    for i in range(0, len(rows), chunk):
        out[i:i + chunk] = X[rows[i:i + chunk]]
    return out


def fit(Xtr, ytr, Xva, yva, feature_names, cats, rounds=1500):
    import lightgbm as lgb
    dtr = lgb.Dataset(Xtr, ytr, feature_name=feature_names, categorical_feature=cats, free_raw_data=True)
    dva = lgb.Dataset(Xva, yva, reference=dtr, feature_name=feature_names, categorical_feature=cats)
    model = lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)])
    log(f"best iteration {model.best_iteration}, valid {dict(model.best_score['valid_0'])}")
    return model


def predict_rows(models, X, rows, chunk=4_000_000):
    out = np.empty(len(rows), dtype=np.float32)
    for i in range(0, len(rows), chunk):
        x = X[rows[i:i + chunk]] if rows is not None else X[i:i + chunk]
        out[i:i + chunk] = np.mean([m.predict(x, num_threads=16) for m in models], axis=0)
    return out


def listwise(left, right, p1, core_eq, n_s1, n_rec):
    """Stage-2 listwise/competition features over every pair of the set (float32 matrix, L2 order)."""
    rank = rank_within(left, p1).astype(np.float32)
    s1_top, _ = top_two(left, p1, n_s1)
    order = np.lexsort((np.arange(len(p1)), -p1, left))
    ps, ls = p1[order], left[order]
    nxt = np.full(len(p1), -1.0, dtype=np.float32)
    nxt[:-1] = np.where(ls[1:] == ls[:-1], ps[1:], -1.0)
    gap_next = np.empty(len(p1), dtype=np.float32)
    gap_next[order] = ps - nxt
    near = p1 >= (s1_top[left] - 0.05)
    s1_near = np.bincount(left[near], minlength=n_s1)
    strong = p1 >= 0.5
    s1_strong = np.bincount(left[strong], minlength=n_s1)
    s1_sum = np.bincount(left, weights=p1, minlength=n_s1)
    r1, r2 = top_two(right, p1, n_rec)
    rival = np.where(p1 >= r1[right], r2[right], r1[right]).astype(np.float32)
    rec_rank = rank_within(right, p1).astype(np.float32)
    rec_strong = np.bincount(right[strong], minlength=n_rec)
    rec_n = np.bincount(right, minlength=n_rec)
    s1_n = np.bincount(left, minlength=n_s1)
    ce = core_eq > 0.5
    rec_ce = np.bincount(right[ce], minlength=n_rec)
    s1_ce = np.bincount(left[ce], minlength=n_s1)
    cols = [p1, np.minimum(rank, 99), p1 - s1_top[left], gap_next, s1_near[left], s1_strong[left], s1_sum[left],
            rival, p1 - rival, np.minimum(rec_rank, 99), rec_strong[right], rec_n[right], s1_n[left], rec_ce[right],
            s1_ce[left]]
    return np.column_stack(cols).astype(np.float32)


def splits(data, n_s1):
    part = np.load(Path(data) / "arena/finalize_v1/s1_partition.npy")[:n_s1]
    comp = np.load(Path(data) / "arena/finalize_v1/component_ids.npy")[:n_s1]
    _, comp = np.unique(comp, return_inverse=True)
    rng = np.random.default_rng(SEED)
    n_comp = int(comp.max()) + 1
    u = rng.random(n_comp)
    a_comp = np.zeros(n_comp, bool)
    a_comp[comp[part == 0]] = True
    return part, comp, u, a_comp


def summary(res):
    keys = ("macro_f05", "precision", "recall", "false_links", "s1", "singleton_accuracy", "predicted_links")
    out = {k: res[k] for k in keys}
    if "by_country" in res:
        out["by_country"] = {c: {k: v[k] for k in keys} for c, v in res["by_country"].items()}
    return out


def veto_mask(X, rows=None):
    """House-number shift signature: both have a number, S1 number absent from the record, record number differs
    by a shift (generator +delta, other positive, or negative). Copies only truncate/pad/append a letter."""
    h = np.asarray(X[:, FEATURES.index("h_rel")]) if rows is None else X[rows, FEATURES.index("h_rel")]
    hin = np.asarray(X[:, FEATURES.index("h1_in_rec_digits")]) if rows is None else X[rows, FEATURES.index("h1_in_rec_digits")]
    return np.isin(h, (7, 8, 9)) & (hin == 0)


def decode(prob, thr, variant, left, right, rec_empty, core_eq, mult, veto=None):
    if variant == "model_veto":
        prob = np.where(veto, 0.0, prob).astype(np.float32)
        variant = "model"
    acc = decide(prob, left, right, thr, None)
    if variant == "model_then_name_only":
        acc, _ = name_only_rule(acc, left, right, rec_empty, core_eq, mult)
        rows = np.flatnonzero(acc)
        if rows.size:
            acc[rows[rank_within(left[rows], prob[rows]) >= 11]] = False
    return acc


def sweep(prob, rows, left, right, label, true_count, pop, rec_empty, core_eq, mult,
          variants=("model", "model_then_name_only", "model_veto"), veto=None):
    best, table = None, []
    l, r, lab = left[rows], right[rows], label[rows]
    for thr in THRESHOLDS:
        for v in variants:
            acc = decode(prob[rows], thr, v, l, r, rec_empty[rows], core_eq[rows], mult[rows],
                         None if veto is None else veto[rows])
            f = evaluate(acc, l, lab, true_count, population=pop)["macro_f05"]
            table.append({"threshold": thr, "variant": v, "f05": f})
            if best is None or f > best["f05"]:
                best = {"threshold": thr, "variant": v, "f05": f}
    return best, table


def train(args):
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    pairs, X, s1, rec = load_feat(args.feat)
    left, right, label, route = pairs["left"].astype(np.int64), pairs["right"].astype(np.int64), pairs["label"], pairs["route"]
    n_s1, n_rec = len(s1), len(rec)
    true_count = np.load(Path(args.data) / "arena/labels/true_count.npy")[:n_s1].astype(np.int64)
    part, comp, u, a_comp = splits(args.data, n_s1)
    # A-tune: components with u < 0.2; A-fit sample: u in [0.2, 0.2 + frac)
    s1_u = u[comp]
    is_a = part == 0
    tune_s1 = is_a & (s1_u < 0.2)
    rng = np.random.default_rng(SEED)
    tune_keep = rng.random(n_s1) < args.tune_frac  # subsample tune S1s (keeps whole-list evaluation per S1)
    tune_s1 &= tune_keep
    fit_s1 = is_a & (s1_u >= 0.2) & (s1_u < 0.2 + args.fit_frac)
    fold = (np.floor(s1_u * 1e6).astype(np.int64) % 2)  # component-level fold
    if args.train_country:
        only = s1.country.to_numpy() == args.train_country
        fit_s1 &= only
        tune_s1 &= only
        log(f"train country {args.train_country} only")
    brest = part == 1
    log(f"S1: A-fit {int(fit_s1.sum())} (fold0 {int((fit_s1 & (fold == 0)).sum())}), A-tune {int(tune_s1.sum())}, B_REST {int(brest.sum())}")
    row_fit = fit_s1[left]
    row_tune = tune_s1[left]
    cats = [FEATURES.index(c) for c in CATEGORICAL]
    tune_rows = np.flatnonzero(row_tune)
    Xtune = take(X, tune_rows)
    ytune = label[tune_rows].astype(np.float32)
    import lightgbm as lgb
    reuse = args.reuse and (work / "p1.npy").exists() and (work / "stage2.txt").exists()
    models1 = []
    p1 = np.zeros(len(left), dtype=np.float32)
    for f in ((0, 1) if not reuse else ()):
        tr = np.flatnonzero(row_fit & (fold[left] == f))
        log(f"stage1 fold {f}: {len(tr)} rows, {int(label[tr].sum())} positive")
        m = fit(take(X, tr), label[tr].astype(np.float32), Xtune, ytune, FEATURES, cats)
        m.save_model(str(work / f"stage1_fold{f}.txt"))
        models1.append(m)
        oof = np.flatnonzero(row_fit & (fold[left] == 1 - f))
        p1[oof] = predict_rows([m], X, oof)
    if reuse:
        models1 = [lgb.Booster(model_file=str(work / f"stage1_fold{f}.txt")) for f in (0, 1)]
        p1 = np.load(work / "p1.npy")
        log("reused stage-1 models and p1")
    else:
        rest = np.flatnonzero(~row_fit)
        log(f"stage1 predict {len(rest)} other rows (mean of both folds)")
        p1[rest] = predict_rows(models1, X, rest)
        np.save(work / "p1.npy", p1)
    imp = {n: float(v) for n, v in zip(FEATURES, models1[0].feature_importance("gain"))}
    core_eq = np.asarray(X[:, FEATURES.index("n_core_eq")])
    Z = listwise(left, right, p1, core_eq, n_s1, n_rec)
    s1_rank = Z[:, 1]
    cand = (p1 >= GATE) | (s1_rank < 3)
    log(f"stage2 candidates {int(cand.sum())} of {len(left)}; true kept {int(label[cand].sum())} of {int(label.sum())}")
    names2 = FEATURES + L2
    tr = np.flatnonzero(cand & row_fit)
    va = np.flatnonzero(cand & row_tune)
    X2tr = np.hstack([take(X, tr), Z[tr]])
    X2va = np.hstack([take(X, va), Z[va]])
    log(f"stage2 fit rows {len(tr)}, tune rows {len(va)}")
    if reuse:
        m2 = lgb.Booster(model_file=str(work / "stage2.txt"))
    else:
        m2 = fit(X2tr, label[tr].astype(np.float32), X2va, label[va].astype(np.float32), names2, cats)
        m2.save_model(str(work / "stage2.txt"))
    del X2tr
    imp2 = {n: float(v) for n, v in zip(names2, m2.feature_importance("gain"))}
    prob = np.zeros(len(left), dtype=np.float32)
    ev = np.flatnonzero(cand & (row_tune | brest[left]))
    for i in range(0, len(ev), 4_000_000):
        rr = ev[i:i + 4_000_000]
        prob[rr] = m2.predict(np.hstack([X[rr], Z[rr]]), num_threads=16)
    np.save(work / "prob.npy", prob)
    rec_empty = rec["empty"].to_numpy()[right]
    mult = s1["mult"].to_numpy()[left]
    veto = veto_mask(X)
    report = {"features": names2, "gate": GATE, "fit_s1": int(fit_s1.sum()), "tune_s1": int(tune_s1.sum()),
              "stage1_iterations": [m.current_iteration() for m in models1], "stage2_iterations": m2.current_iteration(),
              "stage1_importance_gain": imp, "stage2_importance_gain": imp2}
    country = s1.country.to_numpy()
    for name, pr in (("stage1", p1), ("stage2", prob)):
        best, table = sweep(pr, np.flatnonzero(row_tune), left, right, label, true_count, tune_s1, rec_empty, core_eq, mult,
                            veto=veto)
        br = np.flatnonzero(brest[left])
        acc = decode(pr[br], best["threshold"], best["variant"], left[br], right[br], rec_empty[br], core_eq[br], mult[br], veto[br])
        res = evaluate(acc, left[br], label[br], true_count, population=brest, country=country, return_per_s1=True)
        by_variant = {}
        for v in ("model", "model_then_name_only", "model_veto"):
            thr_v = max((t for t in table if t["variant"] == v), key=lambda t: t["f05"])["threshold"]
            a_v = decode(pr[br], thr_v, v, left[br], right[br], rec_empty[br], core_eq[br], mult[br], veto[br])
            by_variant[v] = {"threshold": thr_v, **summary(evaluate(a_v, left[br], label[br], true_count, population=brest))}
        # routes-off variant (dense only) for reference
        brd = br[~route[br]]
        acc_d = decode(pr[brd], best["threshold"], best["variant"], left[brd], right[brd], rec_empty[brd], core_eq[brd], mult[brd], veto[brd])
        res_d = evaluate(acc_d, left[brd], label[brd], true_count, population=brest, country=country)
        comp_full = np.load(Path(args.data) / "arena/finalize_v1/component_ids.npy")[:n_s1]
        boot = component_bootstrap(np.zeros(n_s1), res["per_s1"], comp_full, n=300, seed=0, population=brest)
        report[name] = {"a_tune_best": best, "a_tune_sweep": table, "b_rest": summary(res),
                        "b_rest_dense_only": summary(res_d), "b_rest_ci": [boot["ci_low"], boot["ci_high"]],
                        "b_rest_by_variant_at_own_tune_threshold": by_variant}
        np.save(work / f"b_rest_per_s1_{name}.npy", res["per_s1"].astype(np.float32))
        log(f"{name}: tune {best}; B_REST {summary(res)}; dense-only {res_d['macro_f05']:.6f}; CI {boot['ci_low']:.6f}-{boot['ci_high']:.6f}")
    (work / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    log("done")


def loco(args):
    """Stage-1-only transfer check: train on one country's A-fit rows, evaluate B_REST per country."""
    work = Path(args.work)
    pairs, X, s1, rec = load_feat(args.feat)
    left, right, label = pairs["left"].astype(np.int64), pairs["right"].astype(np.int64), pairs["label"]
    n_s1 = len(s1)
    true_count = np.load(Path(args.data) / "arena/labels/true_count.npy")[:n_s1].astype(np.int64)
    part, comp, u, a_comp = splits(args.data, n_s1)
    s1_u = u[comp]
    country = s1.country.to_numpy()
    is_a = part == 0
    rng = np.random.default_rng(SEED)
    tune_s1 = is_a & (s1_u < 0.2) & (rng.random(n_s1) < args.tune_frac)
    fold = (np.floor(s1_u * 1e6).astype(np.int64) % 2)
    fit_s1 = is_a & (s1_u >= 0.2) & (s1_u < 0.2 + args.fit_frac) & (fold == 0)
    brest = part == 1
    rec_empty = rec["empty"].to_numpy()[right]
    mult = s1["mult"].to_numpy()[left]
    core_eq = np.asarray(X[:, FEATURES.index("n_core_eq")])
    cats = [FEATURES.index(c) for c in CATEGORICAL]
    out = {}
    br = np.flatnonzero(brest[left])
    Xbr = take(X, br)
    for train_c in ("US", "India", "both"):
        cmask = np.ones(n_s1, bool) if train_c == "both" else country == train_c
        s1_tr = fit_s1 & cmask
        if train_c == "both":  # same S1 budget as a single country: half of each
            s1_tr &= rng.random(n_s1) < 0.5
        tr = np.flatnonzero(s1_tr[left])
        tu_pop = tune_s1 & cmask
        tu = np.flatnonzero(tu_pop[left])
        log(f"LOCO train {train_c}: {int(s1_tr.sum())} S1, {len(tr)} rows")
        m = fit(take(X, tr), label[tr].astype(np.float32), take(X, tu), label[tu].astype(np.float32), FEATURES, cats)
        p = np.zeros(len(left), dtype=np.float32)
        p[tu] = m.predict(take(X, tu), num_threads=16)
        p[br] = m.predict(Xbr, num_threads=16)
        best, _ = sweep(p, tu, left, right, label, true_count, tu_pop, rec_empty, core_eq, mult, variants=("model",))
        acc = decode(p[br], best["threshold"], "model", left[br], right[br], rec_empty[br], core_eq[br], mult[br])
        res = evaluate(acc, left[br], label[br], true_count, population=brest, country=country)
        out[train_c] = {"s1": int(s1_tr.sum()), "threshold": best, "b_rest": summary(res)}
        log(f"LOCO {train_c}: {json.dumps(out[train_c]['b_rest']['by_country'])}")
    (work / "loco.json").write_text(json.dumps(out, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "loco"])
    ap.add_argument("--feat", required=True)
    ap.add_argument("--data")
    ap.add_argument("--work", required=True)
    ap.add_argument("--fit-frac", type=float, default=0.2)
    ap.add_argument("--tune-frac", type=float, default=0.35)
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--train-country")
    args = ap.parse_args()
    {"train": train, "loco": loco}[args.what](args)


if __name__ == "__main__":
    main()
