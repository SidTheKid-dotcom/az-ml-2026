"""K x J llm overlay on B_REST (labels: B_REST only). cpu2 (K's workspace) was lost, so K-final F OOF p is gone;
kf_C OOF (S3) is re-keyed from U's brest_prep / U8f route / R reverse row orders and verified against K's report.

Overlay (test-applicable, because test K probabilities are lost too): on J's llm band rows, a cross-fitted logistic
model on [K accept bit, logit u_p, llm_logit, row-type/India one-hots, interactions]; per-fold threshold chosen on
the other folds; non-band rows keep K's decision; one owner (K links first), cap 11, per-source S2<=5 / S3<=6.
Control = same without the llm columns. Paired component bootstrap (1000 reps, seed 20260927).
"""
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()
H = "/home/ubuntu/run27"
W = "/home/ubuntu/kllm"
OUT = f"{W}/out"
REPORT = sys.argv[1] if len(sys.argv) > 1 else "brest_report.json"
GRID = np.round(np.arange(0.5, 0.951, 0.025), 3)


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_logreg(x, y, iters=50, l2=1e-4):
    x = np.hstack([x, np.ones((len(x), 1))])
    w = np.zeros(x.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(x @ w)))
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


def decide(p, s1, rec, thr, is_s2=None, cap=11):
    """threshold -> one owner (best p per record) -> per S1 by p desc: cap 11 (+ S2<=5 / S3<=6 if is_s2 given)."""
    acc = p >= thr
    idx = np.flatnonzero(acc)
    idx = idx[np.argsort(-p[idx], kind="stable")]
    _, first = np.unique(rec[idx], return_index=True)
    keep = idx[np.sort(first)]
    keep = keep[np.lexsort((-p[keep], s1[keep]))]
    s = s1[keep]
    start = np.r_[0, np.flatnonzero(np.diff(s)) + 1]
    rank = np.arange(len(s)) - np.repeat(start, np.diff(np.r_[start, len(s)]))
    ok = rank < cap
    if is_s2 is not None:
        for flag, lim in ((True, 5), (False, 6)):
            m = is_s2[keep] == flag
            sub = keep[m]
            ss = s1[sub]
            st = np.r_[0, np.flatnonzero(np.diff(ss)) + 1] if len(ss) else np.array([], dtype=np.int64)
            rk = np.arange(len(ss)) - np.repeat(st, np.diff(np.r_[st, len(ss)])) if len(ss) else np.array([])
            ok_sub = rk < lim
            ok[np.flatnonzero(m)[~ok_sub]] = False
    keep = keep[ok]
    out = np.zeros(len(p), dtype=bool)
    out[keep] = True
    return out


def per_s1(sel, s1, y, tc, n_s1):
    pred = np.bincount(s1[sel], minlength=n_s1)
    tp = np.bincount(s1[sel & y], minlength=n_s1)
    den = 4 * pred + tc
    f = np.divide(5.0 * tp, den, out=np.zeros(n_s1), where=den > 0)
    f[(tc == 0) & (pred == 0)] = 1.0
    return f


def bootstrap(delta, comp, pop, n=1000, seed=20260927):
    d, c = delta[pop], comp[pop]
    cu, ci = np.unique(c, return_inverse=True)
    sums, cnt = np.bincount(ci, weights=d), np.bincount(ci)
    rng = np.random.default_rng(seed)
    reps = np.empty(n)
    for i in range(n):
        w = rng.poisson(1.0, len(cu))
        reps[i] = (w * sums).sum() / max(1, (w * cnt).sum())
    return {"delta": float(d.mean()), "ci_low": float(np.percentile(reps, 2.5)),
            "ci_high": float(np.percentile(reps, 97.5)), "p_le_zero": float((reps <= 0).mean()), "reps": n}


def auc(score, y):
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    order = np.argsort(score, kind="mergesort")
    s = score[order]
    _, start, counts = np.unique(s, return_index=True, return_counts=True)
    ranks = np.empty(len(s))
    ranks[order] = np.repeat(start + (counts + 1) / 2.0, counts)
    npos, nneg = y.sum(), (~y).sum()
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def main():
    rep = {"labels_read": "B_REST only", "base": "kf_C OOF (S3 run27/K/kf_C/oof.npy); kf_F OOF lost with cpu2"}
    ref = pq.read_table(f"{H}/U/data/arena/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    n_s1 = len(ref)
    s1_index = pd.Index(ref.entity_id.to_numpy())
    pool_ids = np.asarray(pq.read_table(f"{H}/U/data/arena/pool.parquet", columns=["entity_id"]).column(0).to_pylist(),
                          dtype=object)
    pool_index = pd.Index(pool_ids)
    part = np.load(f"{H}/U/data/arena/finalize_v1/s1_partition.npy")
    comp = np.load(f"{H}/U/data/arena/finalize_v1/component_ids.npy")
    tc = np.load(f"{H}/U/data/arena/labels/true_count.npy")
    owner = np.load(f"{H}/U/data/arena/labels/rec_owner.npy")
    s1_fold = np.load(f"{H}/U/folds/brest_fold5_u4-20260927.npy").astype(np.int64)
    pop = part == 1
    country = ref.country.to_numpy()
    # dense rows: U brest_prep order
    d = f"{H}/U/prep/brest"
    dl, dr, dy = np.load(f"{d}/left.npy").astype(np.int64), np.load(f"{d}/right.npy").astype(np.int64), np.load(f"{d}/label.npy").astype(bool)
    # route rows: U8f file order
    R = pq.read_table(f"{H}/R/data/routes/u_route_pairs_brest.parquet", columns=["left_idx", "rec_id"]).to_pandas()
    rl = R.left_idx.to_numpy().astype(np.int64)
    rpi = pool_index.get_indexer(R.rec_id.to_numpy())
    ry = np.where(rpi >= 0, owner[np.clip(rpi, 0, None)] == rl, False)
    # reverse rows: R file order
    V = pq.read_table(f"{H}/R/out/r_reverse_pairs_b_rest.parquet", columns=["s1_id", "rec_id", "label"]).to_pandas()
    vl = s1_index.get_indexer(V.s1_id.to_numpy()).astype(np.int64)
    vy = V.label.to_numpy().astype(bool)
    log(f"dense {len(dl)} (true {int(dy.sum())}); routes {len(rl)} (true {int(ry.sum())}); rev {len(vl)} (true {int(vy.sum())})")
    assert len(dl) == 3895977 and len(rl) == 689474 and len(vl) == 452167 and int(ry.sum()) == 4897 and int(vy.sum()) == 6480
    assert (vl >= 0).all()
    s1 = np.concatenate([dl, rl, vl])
    rec_s = np.concatenate([pool_ids[dr], R.rec_id.to_numpy().astype(object), V.rec_id.to_numpy().astype(object)])
    y = np.concatenate([dy, ry, vy])
    src = np.concatenate([np.zeros(len(dl), np.int8), np.ones(len(rl), np.int8), np.full(len(vl), 2, np.int8)])
    pK = np.load(f"{W}/in/kfC_oof.npy").astype(np.float64)
    assert len(pK) == len(s1)
    rec, rec_u = pd.factorize(rec_s)
    is_s2 = np.asarray([r.startswith("S2") for r in rec_u], dtype=bool)[rec]
    # 1) verify kf_C: thr 0.775 (all fold picks), one owner, cap 11 -> K report 0.989254 (2,304 false links)
    selK0 = decide(pK, s1, rec, np.full(len(pK), 0.775))
    fK0 = per_s1(selK0, s1, y, tc, n_s1)
    rep["verify_kfC"] = {"macro_f05": float(fK0[pop].mean()), "false_links": int((selK0 & ~y).sum()),
                         "links": int(selK0.sum()), "k_report": 0.9892535396448391, "k_false_links": 2304}
    log(f"VERIFY kf_C {rep['verify_kfC']}")
    if abs(rep["verify_kfC"]["macro_f05"] - 0.9892535396448391) > 2e-6:
        log("VERIFY FAILED: row keys do not reproduce K's kf_C number")
        json.dump(rep, open(f"{OUT}/brest_report.json", "w"), indent=1)
        sys.exit(2)
    # 2) K baseline as on test: route rows >= 0.955 (guard), per-source caps
    thrK = np.where(src == 1, 0.955, 0.775)
    selKg = decide(pK, s1, rec, thrK)
    rep["kfC_guarded_no_srccap"] = float(per_s1(selKg, s1, y, tc, n_s1)[pop].mean())
    selK = decide(pK, s1, rec, thrK, is_s2)
    fK = per_s1(selK, s1, y, tc, n_s1)
    log(f"kf_C guarded {rep['kfC_guarded_no_srccap']:.6f}; + per-source caps {fK[pop].mean():.6f}")
    # pair-level K accept bit (any source row), as on test
    pair = pd.Series(s1.astype(np.int64) * 20_000_000 + rec)
    acc_pair = set(pair[selK].to_numpy().tolist())
    a_pair = pair.isin(acc_pair).to_numpy()
    # 3) band rows = J's llm rows joined by (s1_id, rec_id, row_type)
    L = pd.read_parquet(f"{W}/in/llm_brest.parquet")
    key_rows = pd.DataFrame({"s1_id": ref.entity_id.to_numpy()[s1], "rec_id": rec_s, "row_type": src, "i": np.arange(len(s1))})
    m = L.merge(key_rows, on=["s1_id", "rec_id", "row_type"], how="left")
    rep["band_join"] = {"llm_rows": len(L), "matched": int(m.i.notna().sum()),
                        "by_type": {int(t): [int((L.row_type == t).sum()), int(m[m.row_type == t].i.notna().sum())] for t in (0, 1, 2)}}
    log(f"band join {rep['band_join']}")
    m = m[m.i.notna()].drop_duplicates("i")
    rows = m.i.to_numpy().astype(np.int64)
    up, zl = m.u_p.to_numpy().astype(np.float64), m.llm_logit.to_numpy().astype(np.float64)
    rt = src[rows]
    ind = (country[s1[rows]] == "India").astype(float)
    a = a_pair[rows].astype(float)
    Lu = logit(up)
    r1, r2 = (rt == 1).astype(float), (rt == 2).astype(float)
    X = np.column_stack([a, Lu, a * Lu, r1, r2, ind, r1 * a, r2 * a, ind * a, r1 * Lu, r2 * Lu,
                         zl, zl * a, r1 * zl, r2 * zl, ind * zl, zl * Lu])
    ctrl = list(range(11))
    full = list(range(X.shape[1]))
    yb = y[rows].astype(float)
    rf = s1_fold[s1[rows]]
    rep["band_auc"] = {"u_p": auc(up, yb), "llm": auc(zl, yb), "kfC_p": auc(pK[rows], yb), "K_acc": auc(a, yb)}
    log(f"band AUC {rep['band_auc']}")
    variants = {"all_band": np.ones(len(rows), bool),
                "no_us_dense": ~((rt == 0) & (ind == 0))}  # PRIMARY (pre-registered per J's adversarial flag)
    if REPORT.endswith("_v2.json"):  # route rows excluded: test route acceptance 5.7x B_REST (label-free flag)
        variants = {"no_us_dense_no_route": ~((rt == 0) & (ind == 0)) & (rt != 1)}
    base_pf = selK.astype(np.float64)  # K links priority 1
    row_fold = s1_fold[s1]
    res = {}
    for vname, vm in variants.items():
        rr, Xv, yv, fv = rows[vm], X[vm], yb[vm], rf[vm]
        for arm, cols in (("control", ctrl), ("llm", full)):
            q = np.empty(len(rr))
            for f in range(5):
                tr, te = fv != f, fv == f
                w = fit_logreg(Xv[tr][:, cols], yv[tr])
                q[te] = predict_logreg(w, Xv[te][:, cols])
            pf = base_pf.copy()
            pf[rr] = q
            per = {}
            for t in GRID:
                thr = np.full(len(pf), 0.5)
                thr[rr] = t
                per[t] = per_s1(decide(pf, s1, rec, thr, is_s2), s1, y, tc, n_s1)
            pick = {f: float(max(GRID, key=lambda t: per[t][pop & (s1_fold != f) & (s1_fold >= 0)].mean())) for f in range(5)}
            thr = np.full(len(pf), 0.5)
            thr[rr] = np.array([pick[f] for f in range(5)])[np.clip(row_fold[rr], 0, 4)]
            sel = decide(pf, s1, rec, thr, is_s2)
            fo = per_s1(sel, s1, y, tc, n_s1)
            full_t = float(max(GRID, key=lambda t: per[t][pop].mean()))
            wfull = fit_logreg(Xv[:, cols], yv)
            tp = int((sel & y).sum())
            out = {"macro_f05": float(fo[pop].mean()), "precision": tp / max(1, int(sel.sum())),
                   "recall": tp / max(1, int(tc[pop].sum())), "false_links": int((sel & ~y).sum()), "links": int(sel.sum()),
                   "US": float(fo[pop & (country == "US")].mean()), "India": float(fo[pop & (country == "India")].mean()),
                   "singleton_accuracy": float((np.bincount(s1[sel], minlength=n_s1)[pop & (tc == 0)] == 0).mean()),
                   "picks": pick, "full_threshold": full_t, "weights_full": wfull.tolist(), "cols": cols,
                   "band_rows": int(len(rr)), "changed_vs_K": {"added": int((sel & ~selK).sum()), "removed": int((selK & ~sel).sum())},
                   "added_by_type": {int(t): int((sel & ~selK & (src == t)).sum()) for t in (0, 1, 2)},
                   "removed_by_type": {int(t): int((selK & ~sel & (src == t)).sum()) for t in (0, 1, 2)},
                   "band_auc_q": auc(q, yv), "vs_K": bootstrap(fo - fK, comp, pop)}
            res[(vname, arm)] = (fo, out)
            np.save(f"{OUT}/q_{vname}_{arm}.npy", q.astype(np.float32))
            log(f"{vname}/{arm}: {out['macro_f05']:.6f} P {out['precision']:.5f} R {out['recall']:.5f} FL {out['false_links']} "
                f"vs K {out['vs_K']['delta']:+.6f} [{out['vs_K']['ci_low']:+.6f},{out['vs_K']['ci_high']:+.6f}] picks {pick} full_t {full_t}")
        fo_l, out_l = res[(vname, "llm")]
        fo_c, _ = res[(vname, "control")]
        out_l["vs_control"] = bootstrap(fo_l - fo_c, comp, pop)
        log(f"{vname}: llm vs control {out_l['vs_control']}")
    tpK = int((selK & y).sum())
    rep["K_baseline"] = {"macro_f05": float(fK[pop].mean()), "precision": tpK / int(selK.sum()), "recall": tpK / int(tc[pop].sum()),
                         "false_links": int((selK & ~y).sum()), "links": int(selK.sum()),
                         "US": float(fK[pop & (country == "US")].mean()), "India": float(fK[pop & (country == "India")].mean()),
                         "protocol": "kf_C OOF, thr 0.775 (routes 0.955), one owner, cap 11, S2<=5/S3<=6"}
    rep["variants"] = {f"{v}/{a}": o for (v, a), (_, o) in res.items()}
    rep["primary"] = "no_us_dense/llm (pre-registered: J's adversarial flag on US dense rows)"
    rep["feature_names"] = ["K_acc", "logit_u_p", "K_acc*logit_u_p", "route", "reverse", "India", "route*K_acc", "reverse*K_acc",
                            "India*K_acc", "route*logit_u_p", "reverse*logit_u_p", "llm_logit", "llm*K_acc", "route*llm",
                            "reverse*llm", "India*llm", "llm*logit_u_p"]
    json.dump(rep, open(f"{OUT}/{REPORT}", "w"), indent=1, default=float)
    log("BREST DONE")


if __name__ == "__main__":
    main()
