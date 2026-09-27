"""Agent J: B_REST evaluation of the 4B judge on U's uncertain pairs + K-lite stack A/B (labels: B_REST only).

Usage: eval_brest.py IN_DIR LLM_BREST.parquet OUT.json
  IN_DIR: u/u8r_pairs_brest.parquet, gt/train_ground_truth.tsv, brest_fold5_u4-20260927.npy,
          arena reference.parquet / s1_partition.npy / component_ids.npy (paths below).
  LLM_BREST.parquet: s1_id, rec_id, llm_logit (uncertain rows).

1. Uncertain rows: AUC of U's OOF p vs llm_p (all, by row type, by country); precision/recall of llm at the
   threshold matching U's accepted count at 0.70.
2. K-lite: on uncertain rows, logistic stack on [logit u_p, llm_logit, product, row-type dummies], 5-fold OOF
   on U's frozen fold file; other rows keep U's p. Decision = per-fold threshold chosen on the other folds
   (grid 0.50..0.95), one owner, cap 11; macro F0.5 over all B_REST S1s incl. missed truth and singletons;
   baseline = the same protocol with U's p alone. Paired component bootstrap of the delta.
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()
ARENA = "/opt/az-ml-new/artifacts/arena"
IN, LLM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
GRID = np.round(np.arange(0.5, 0.951, 0.025), 3)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def auc(score, y):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    ranks = np.empty(len(s))
    # average ranks for ties
    i = 0
    idx = np.arange(1, len(s) + 1, dtype=np.float64)
    uniq, start, counts = np.unique(s, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    ranks_sorted = np.repeat(avg, counts)
    ranks[order] = ranks_sorted
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logreg(x, y, iters=50, l2=1e-4):
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        z = x @ w
        p = 1 / (1 + np.exp(-z))
        g = x.T @ (p - y) + l2 * w
        h = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(w))
        step = np.linalg.solve(h, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logreg(w, x):
    x = np.hstack([x, np.ones((len(x), 1))])
    return 1 / (1 + np.exp(-(x @ w)))


def decide(p, s1, rec, thr_per_row):
    acc = p >= thr_per_row
    idx = np.flatnonzero(acc)
    idx = idx[np.argsort(-p[idx], kind="stable")]
    _, first = np.unique(rec[idx], return_index=True)  # one owner: best p per record
    keep = idx[np.sort(first)]
    keep = keep[np.lexsort((-p[keep], s1[keep]))]  # by S1, p desc
    s = s1[keep]
    start = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    rank = np.arange(len(s)) - np.repeat(start, np.diff(np.r_[start, len(s)]))
    keep = keep[rank < 11]
    out = np.zeros(len(p), dtype=bool)
    out[keep] = True
    return out


def per_s1_f05(sel, s1, y, true_count, n_s1):
    pred = np.bincount(s1[sel], minlength=n_s1)
    tp = np.bincount(s1[sel & y], minlength=n_s1)
    den = 4 * pred + true_count
    f = np.divide(5.0 * tp, den, out=np.zeros(n_s1), where=den > 0)
    f[(true_count == 0) & (pred == 0)] = 1.0
    return f, pred, tp


def crossfit(p, s1, rec, y, true_count, n_s1, s1_fold, pop):
    row_fold = s1_fold[s1]
    per = {}
    for t in GRID:
        sel = decide(p, s1, rec, np.full(len(p), t))
        per[t] = per_s1_f05(sel, s1, y, true_count, n_s1)[0]
    pick = {}
    for f in range(5):
        other = pop & (s1_fold != f) & (s1_fold >= 0)
        pick[f] = max(GRID, key=lambda t: per[t][other].mean())
    thr = np.array([pick[f] for f in range(5)])[np.clip(row_fold, 0, 4)]
    sel = decide(p, s1, rec, thr)
    f05, pred, tp = per_s1_f05(sel, s1, y, true_count, n_s1)
    full_grid = {float(t): float(per[t][pop].mean()) for t in GRID}
    return f05, sel, {int(k): float(v) for k, v in pick.items()}, full_grid


def bootstrap(delta, comp, pop, n=1000, seed=20260927):
    d = delta[pop]
    c = comp[pop]
    cu, ci = np.unique(c, return_inverse=True)
    sums = np.bincount(ci, weights=d)
    cnt = np.bincount(ci)
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(n):
        w = rng.poisson(1.0, len(cu))
        reps.append((w * sums).sum() / max(1, (w * cnt).sum()))
    reps = np.asarray(reps)
    return {"delta": float(d.mean()), "ci_low": float(np.percentile(reps, 2.5)),
            "ci_high": float(np.percentile(reps, 97.5)), "reps": n}


def main():
    ref = pq.read_table(f"{ARENA}/v1/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    part = np.load(f"{ARENA}/finalize_v1/s1_partition.npy")
    comp = np.load(f"{ARENA}/finalize_v1/component_ids.npy")
    s1_fold = np.load(f"{IN}/brest_fold5_u4-20260927.npy").astype(np.int64)
    n_s1 = len(ref)
    pop = part == 1
    s1_index = pd.Series(np.arange(n_s1), index=ref.entity_id.to_numpy())
    u = pq.read_table(f"{IN}/u/u8r_pairs_brest.parquet", columns=["s1_id", "rec_id", "row_type", "rec_outside_pool", "p"]).to_pandas()
    s1 = s1_index.reindex(u.s1_id.to_numpy()).to_numpy()
    if np.isnan(s1).any():
        raise SystemExit("U rows with unknown S1")
    s1 = s1.astype(np.int64)
    if not pop[s1].all():
        raise SystemExit("U B_REST rows outside B_REST")
    rec_codes, rec_uniq = pd.factorize(u.rec_id)
    gt = pd.read_csv(f"{IN}/gt/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    gt = gt[gt.source1_entity_id.isin(set(ref.entity_id[pop]))]
    tp_pairs = gt.assign(rec=gt.matched_entity_ids.str.split(",")).explode("rec")
    tp_pairs = tp_pairs[tp_pairs.rec.fillna("").str.len() > 0]
    true_count = np.zeros(n_s1, dtype=np.int64)
    tc = tp_pairs.groupby("source1_entity_id").size()
    true_count[s1_index.reindex(tc.index).to_numpy().astype(np.int64)] = tc.to_numpy()
    truth = set(zip(tp_pairs.source1_entity_id, tp_pairs.rec))
    y = np.fromiter(((a, b) in truth for a, b in zip(u.s1_id, u.rec_id)), dtype=bool, count=len(u))
    log(f"U B_REST rows {len(u):,}; true rows {int(y.sum()):,}; B_REST S1s {int(pop.sum()):,}; true links {int(true_count[pop].sum()):,}")
    llm = pd.read_parquet(LLM)
    m = u[["s1_id", "rec_id"]].reset_index().merge(llm, on=["s1_id", "rec_id"], how="inner")
    unc_rows = m["index"].to_numpy()
    zl = np.full(len(u), np.nan)
    zl[unc_rows] = m.llm_logit.to_numpy()
    band = (u.p >= .02) & (u.p <= .98)
    log(f"LLM rows matched {len(m):,}; uncertain-band rows {int(band.sum()):,}")
    rep = {"u_rows": len(u), "uncertain_rows": int(band.sum()), "llm_rows": len(m)}
    has = ~np.isnan(zl)
    up = u.p.to_numpy().astype(np.float64)
    cc = ref.country.to_numpy()[s1]
    rt = u.row_type.to_numpy()
    blocks = {"all": has}
    for t, name in ((0, "dense"), (1, "route"), (2, "reverse")):
        blocks[name] = has & (rt == t)
    for c in ("US", "India"):
        blocks[c] = has & (cc == c)
    rep["auc"] = {}
    for name, mask in blocks.items():
        rep["auc"][name] = {"rows": int(mask.sum()), "true": int(y[mask].sum()),
                            "u_p": auc(up[mask], y[mask]), "llm": auc(zl[mask], y[mask]),
                            "avg_logit": auc(logit(up[mask]) + zl[mask], y[mask])}
    log(json.dumps(rep["auc"]))
    # matched accepted count at U's 0.70
    acc_u = has & (up >= .70)
    k = int(acc_u.sum())
    order = np.argsort(-zl[has])
    thr_l = zl[has][order[k - 1]] if k > 0 else np.inf
    acc_l = has & (zl >= thr_l)
    yy = y[has].sum()
    rep["matched_threshold"] = {
        "u_threshold": .70, "accepted": k, "llm_logit_threshold": float(thr_l),
        "u_precision": float(y[acc_u].mean()), "u_recall_in_band": float(y[acc_u].sum() / max(1, yy)),
        "llm_precision": float(y[acc_l].mean()), "llm_recall_in_band": float(y[acc_l].sum() / max(1, yy))}
    log(json.dumps(rep["matched_threshold"]))
    # K-lite stack
    rows = np.flatnonzero(has)
    L, Z = logit(up[rows]), zl[rows]
    r1, r2, ind = (rt[rows] == 1).astype(float), (rt[rows] == 2).astype(float), (cc[rows] == "India").astype(float)
    # U-only block (incl. nonlinear recalibration of U's p) then the llm block
    x = np.column_stack([L, L * L, r1, r2, r1 * L, r2 * L, ind, ind * L,
                         Z, Z * L, r1 * Z, r2 * Z, ind * Z, Z * Z])
    rf = s1_fold[s1[rows]]
    u_cols = list(range(8))  # U-only block: logit u_p, square, row-type/India dummies and interactions

    def oof(cols):
        out = np.empty(len(rows))
        for f in range(5):
            tr, te = rf != f, rf == f
            w = fit_logreg(x[tr][:, cols], y[rows][tr].astype(float))
            out[te] = predict_logreg(w, x[te][:, cols])
        return out

    ps = oof(list(range(x.shape[1])))
    pu = oof(u_cols)
    rep["stack_auc_uncertain"] = {"u_p": auc(up[rows], y[rows]), "stack_u_only": auc(pu, y[rows]), "stack_u_llm": auc(ps, y[rows])}
    log(json.dumps(rep["stack_auc_uncertain"]))
    p_new = up.copy()
    p_new[rows] = ps
    p_uo = up.copy()
    p_uo[rows] = pu
    f_base, sel_b, pick_b, grid_b = crossfit(up, s1, rec_codes, y, true_count, n_s1, s1_fold, pop)
    f_uo, sel_uo, pick_uo, grid_uo = crossfit(p_uo, s1, rec_codes, y, true_count, n_s1, s1_fold, pop)
    f_new, sel_n, pick_n, grid_n = crossfit(p_new, s1, rec_codes, y, true_count, n_s1, s1_fold, pop)

    def summary(f05, sel):
        tp = int((sel & y).sum())
        out = {"macro_f05": float(f05[pop].mean()), "precision": tp / max(1, int(sel.sum())),
               "recall": tp / max(1, int(true_count[pop].sum())), "false_links": int((sel & ~y).sum()),
               "links": int(sel.sum())}
        for c in ("US", "India"):
            cm = pop & (ref.country.to_numpy() == c)
            out[f"f05_{c}"] = float(f05[cm].mean())
        sm = pop & (true_count == 0)
        pred = np.bincount(s1[sel], minlength=n_s1)
        out["singleton_accuracy"] = float((pred[sm] == 0).mean())
        return out

    rep["k_lite"] = {"baseline_u": dict(summary(f_base, sel_b), picks=pick_b),
                     "stack_u_only": dict(summary(f_uo, sel_uo), picks=pick_uo),
                     "stack_u_llm": dict(summary(f_new, sel_n), picks=pick_n),
                     "bootstrap_llm_vs_u_only": bootstrap(f_new - f_uo, comp, pop),
                     "bootstrap_llm_vs_baseline_u": bootstrap(f_new - f_base, comp, pop),
                     "bootstrap_u_only_vs_baseline_u": bootstrap(f_uo - f_base, comp, pop),
                     "full_grid_baseline": grid_b, "full_grid_stack": grid_n, "full_grid_u_only": grid_uo,
                     "protocol": "uncertain rows restacked (logistic, 5-fold U folds); other rows U p; per-fold threshold on other folds; one owner; cap 11"}
    log(json.dumps({k: v for k, v in rep["k_lite"].items() if not k.startswith("full_grid")}))
    np.save(OUT.replace(".json", "_stack_oof.npy"), ps.astype(np.float32))
    json.dump(rep, open(OUT, "w"), indent=1, default=float)
    log("EVAL DONE")


if __name__ == "__main__":
    main()
