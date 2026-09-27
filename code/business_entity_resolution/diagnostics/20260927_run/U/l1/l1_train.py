"""Idea 4 (27 Sep, Agent U): level-1 pair model trained on arena partition A with out-of-sample-on-A features only.

Features (no C2/Qwen): Agent G string block (60), v1 pair block (12; variant nov1 keeps only the v1 record/S1
context), collective (25), edit-op (17, word tables out-of-fold on A), and in stage 2 listwise/claimant features of
the stage-1 probability p1.

Stage 1: 2 folds over A components (hash "l1a" groups, fold = group % 2); each fold model is fitted on the other
fold (minus the early-stopping holdout H = 10% of A components, hash "l1h") and predicts its fold (A rows: OOF);
non-A rows get the mean of the two fold models. Stage 2: listwise features of p1 over the dense candidate rows of
each split (claimants from every partition, as on test); route rows get theirs over dense + route rows. Fitted on
A \\ H, early stopping on H. Non-A predictions only are delivered. Labels: partition A only.

    python l1_train.py full|nov1
Writes prep/l1/p1_<v>_{arena,test}.npy, p2_<v>_{arena,test}.npy, models and l1_<v>_report.json.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier

from ber.pair_tree_arena import rank_within, top_two

ROOT = Path("/home/ubuntu/run27/U")
L1 = ROOT / "prep" / "l1"
V = sys.argv[1]
CTX = [4, 5, 6, 8, 9]  # v1 context columns kept by nov1
PARAMS = dict(depth=8, learning_rate=0.1, iterations=1500, loss_function="Logloss", od_type="Iter", od_wait=80,
              thread_count=32, random_seed=0, verbose=250, allow_writing_files=False)
MAX_ROWS = int(os.environ.get("L1_MAXROWS", "4500000")) or 10**12
FRAC = float(os.environ.get("L1_FRAC", "1.0"))  # share of A components used for fitting
TAG = os.environ.get("L1_TAG", "")
THREADS = int(os.environ.get("L1_THREADS", "32"))
PARAMS.update(json.loads(os.environ.get("L1_CB", "{}")))
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def block(split):
    g = np.load(L1 / "g" / f"g_{split}.npy", mmap_mode="r")
    v1 = np.load(L1 / f"{split}_v1.npy")
    if V == "nov1":
        v1 = v1[:, CTX]
    return np.hstack([np.asarray(g, np.float32), v1, np.load(L1 / f"{split}_coll.npy"), np.load(L1 / f"{split}_ed.npy")])


def listwise(p, left, right, n_s1, n_rec):
    """rank, gap to S1 top, S1 count >= .5, S1 sum, S1 candidates, best other claimant, p - rival, record count >= .5,
    record candidates."""
    s = np.nan_to_num(p, nan=-1.0).astype(np.float32)
    rank = np.minimum(rank_within(left, s), 99).astype(np.float32)
    s1_top, _ = top_two(left, s, n_s1)
    t1, t2 = top_two(right, s, n_rec)
    rival = np.where(s >= t1[right], t2[right], t1[right]).astype(np.float32)
    ge = s >= 0.5
    cols = [rank, s - s1_top[left], np.bincount(left[ge], minlength=n_s1)[left],
            np.bincount(left, weights=np.maximum(s, 0), minlength=n_s1)[left], np.bincount(left, minlength=n_s1)[left],
            rival, s - rival, np.bincount(right[ge], minlength=n_rec)[right], np.bincount(right, minlength=n_rec)[right]]
    return np.column_stack(cols).astype(np.float32)


def listwise_split(split, p, n_dense):
    left = np.load(L1 / f"targets_{split}_left.npy")
    right = np.load(L1 / f"targets_{split}_right.npy")
    n_s1, n_rec = int(left.max()) + 1, int(right.max()) + 1
    out = np.empty((len(p), 9), np.float32)
    out[:n_dense] = listwise(p[:n_dense], left[:n_dense], right[:n_dense], n_s1, n_rec)
    if len(p) > n_dense:
        out[n_dense:] = listwise(p, left, right, n_s1, n_rec)[n_dense:]
    return out


def fit(x, y, xe, ye):
    m = CatBoostClassifier(**PARAMS)
    m.fit(x, y, eval_set=(xe, ye), use_best_model=True)
    return m


def proba(m, x, chunk=3_000_000):
    out = np.empty(len(x), np.float32)
    for s in range(0, len(x), chunk):
        out[s:s + chunk] = m.predict_proba(x[s:s + chunk])[:, 1]
    return out


def main():
    tj = json.loads((L1 / "targets.json").read_text())
    n_cand = tj["arena_cand"]
    al = np.load(L1 / "targets_arena_left.npy")
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    comp = np.load(ROOT / "data/arena/finalize_v1/component_ids.npy").astype(np.int64)
    uc, inv = np.unique(comp, return_inverse=True)
    hold_c = np.asarray([int(hashlib.sha256(f"l1h:{int(c)}".encode()).hexdigest()[:8], 16) % 10 == 0 for c in uc])
    hold_s1 = hold_c[inv] & (part == 0)
    use_c = np.asarray([int(hashlib.sha256(f"l1s:{int(c)}".encode()).hexdigest()[:8], 16) % 1000 < FRAC * 1000
                        for c in uc])
    use_s1 = use_c[inv] & (part == 0)
    grp = np.load(L1 / "a_group.npy")
    is_a = np.zeros(len(al), bool)
    is_a[:n_cand] = part[al[:n_cand]] == 0
    a_rows = np.flatnonzero(is_a)
    crow = np.load(ROOT / "prep/arena_cand/cand_rows.npy")
    label = np.load(ROOT / "data/arena/labels/label.npy", mmap_mode="r")
    y = np.zeros(len(al), bool)
    y[a_rows] = np.asarray(label[crow[a_rows]]).astype(bool)
    fold = np.where(is_a, grp[al] % 2, -1)
    hold = np.zeros(len(al), bool)
    hold[a_rows] = hold_s1[al[a_rows]]
    use = np.zeros(len(al), bool)
    use[a_rows] = use_s1[al[a_rows]]
    xa = block("arena")
    xt = block("test")
    log(f"{V}: arena X {xa.shape}, test X {xt.shape}; A rows {len(a_rows)} (true {int(y[a_rows].sum())}), "
        f"holdout {int(hold.sum())}")
    rng = np.random.default_rng(20260927)
    PARAMS["thread_count"] = THREADS
    rep = {"variant": V, "tag": TAG, "frac_a_components": FRAC, "params": PARAMS, "max_rows": MAX_ROWS,
           "a_rows": int(len(a_rows)), "a_rows_used": int((use & ~hold).sum()), "stage1": {}}
    p1a = np.zeros(len(al), np.float32)
    p1t = np.zeros(len(xt), np.float32)
    nonA = np.flatnonzero(~is_a)
    for f in (0, 1):
        tr = np.flatnonzero(is_a & (fold != f) & ~hold & use)
        ev = np.flatnonzero(is_a & (fold != f) & hold)
        if len(tr) > MAX_ROWS:
            tr = np.sort(rng.choice(tr, MAX_ROWS, replace=False))
        m = fit(xa[tr], y[tr], xa[ev], y[ev])
        m.save_model(str(L1 / f"l1_{V}{TAG}_s1_fold{f}.cbm"))
        te = np.flatnonzero(is_a & (fold == f))
        p1a[te] = proba(m, xa[te])
        p1a[nonA] += proba(m, xa[nonA]) / 2
        p1t += proba(m, xt) / 2
        rep["stage1"][str(f)] = {"fit_rows": int(len(tr)), "eval_rows": int(len(ev)), "best_iteration": m.get_best_iteration(),
                                 "best_eval_logloss": m.get_best_score().get("validation", {})}
        log(f"stage1 fold {f}: best iter {m.get_best_iteration()}")
    np.save(L1 / f"p1_{V}{TAG}_arena.npy", p1a)
    np.save(L1 / f"p1_{V}{TAG}_test.npy", p1t)
    lwa = listwise_split("arena", p1a, n_cand)
    lwt = listwise_split("test", p1t, tj["test_prep"])
    x2a = np.hstack([xa, p1a[:, None], lwa])
    x2t = np.hstack([xt, p1t[:, None], lwt])
    del xa, xt
    tr = np.flatnonzero(is_a & ~hold & use)
    ev = np.flatnonzero(is_a & hold)
    if len(tr) > MAX_ROWS:
        tr = np.sort(rng.choice(tr, MAX_ROWS, replace=False))
    m = fit(x2a[tr], y[tr], x2a[ev], y[ev])
    m.save_model(str(L1 / f"l1_{V}{TAG}_s2.cbm"))
    p2a = np.full(len(al), np.nan, np.float32)
    p2a[nonA] = proba(m, x2a[nonA])
    p2t = proba(m, x2t)
    np.save(L1 / f"p2_{V}{TAG}_arena.npy", p2a)
    np.save(L1 / f"p2_{V}{TAG}_test.npy", p2t)
    rep["stage2"] = {"fit_rows": int(len(tr)), "eval_rows": int(len(ev)), "best_iteration": m.get_best_iteration(),
                     "best_eval_logloss": m.get_best_score().get("validation", {})}
    names = json.loads((L1 / "g" / "g_meta.json").read_text())["features"]
    v1n = ["blend", "base", "blend_rank_in_s1", "blend_gap_to_s1_top", "s1_candidates_ge_0_5", "rival_blend",
           "record_claimants_ge_0_5", "name_exact", "log1p_name_mult", "record_kind", "key_self", "key_other"]
    v1n = [v1n[i] for i in CTX] if V == "nov1" else v1n
    coll = json.loads((ROOT / "prep/brest/names_coll.json").read_text())
    ed = json.loads((ROOT / "prep/brest/names_ed.json").read_text())
    lw = ["p1", "p1_rank", "p1_gap_top", "s1_n_ge05", "s1_sum", "s1_ncand", "p1_rival", "p1_minus_rival",
          "rec_n_ge05", "rec_ncand"]
    fn = [f"g_{n}" for n in names] + [f"v1_{n}" for n in v1n] + [f"coll_{n}" for n in coll] + ed + lw
    rep["features"] = fn
    rep["importance_stage2"] = dict(sorted(zip(fn, map(float, m.get_feature_importance())), key=lambda kv: -kv[1])[:30])
    # A-holdout diagnostics (A labels): logloss/AUC of p1 (OOF) and p2 on H
    from sklearn.metrics import roc_auc_score, log_loss
    h = np.flatnonzero(hold)
    rep["a_holdout"] = {"p1_auc": float(roc_auc_score(y[h], p1a[h])), "p1_logloss": float(log_loss(y[h], np.clip(p1a[h], 1e-6, 1 - 1e-6))),
                        "p2_auc": float(roc_auc_score(y[h], proba(m, x2a[h]))),
                        "p2_logloss": float(log_loss(y[h], np.clip(proba(m, x2a[h]), 1e-6, 1 - 1e-6)))}
    (L1 / f"l1_{V}{TAG}_report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    log(json.dumps({k: rep[k] for k in ("stage1", "stage2", "a_holdout")}, default=float))


if __name__ == "__main__":
    main()
