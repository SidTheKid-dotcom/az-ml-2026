"""U4 (27 Sep, Agent U): 5-fold component-grouped OOF pair CatBoost on B_REST with feature blocks.

Commands::

    python u4_tree.py fit  --variant full [--threads 22]      # 5 fold models + OOF + report
    python u4_tree.py test --variant full                     # mean of the 5 fold models on test rows

Blocks: base = 12 v1 + 8 C2 (pair_tree_c2); q = 11 Qwen (qwen_ensemble.q_features);
coll = 25 collective (collective_features); extra blocks from prep/{brest,test}/x_<name>.npy.
A variant is "+"-joined blocks, e.g. base+q+coll. Hyperparameters, row cap and seed as pair_tree_c2.
Fold = frozen fold file (hash(salt, claim-graph component) % 5). Threshold per fold is chosen on the
other four folds' OOF (one owner + cap 11, macro F0.5 over their S1s); the OOF score decodes every row at
its fold's threshold jointly. The test threshold is the full-OOF best (reported separately).
B_REST labels only; SLICE/HOLDOUT-2 never read; C2/Qwen A-partition scores never used.
"""
import argparse
import hashlib
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from ber.arena_decision import component_bootstrap
from ber.pair_tree_arena import decide, false_merge
from ber.pair_tree_c2 import CATBOOST, FIT_ROWS, SEED, Arena, accept_per_row, fit_catboost, proba, slim

ROOT = Path(os.environ.get("U_ROOT", "/home/ubuntu/run27/U"))
PREP = ROOT / "prep"
FOLDS = ROOT / "folds" / os.environ.get("U_FOLDS", "brest_fold5_u4-20260927.npy")
BREST = os.environ.get("U_BREST", "brest")
TEST = os.environ.get("U_TEST", "test")
TAG = os.environ.get("U_TAG", "")
PARAMS = {**CATBOOST, **json.loads(os.environ.get("U_CB", "{}"))}
PER_COUNTRY = os.environ.get("U_PER_COUNTRY", "") == "1"  # one model per country and fold
COUNTRIES = ("US", "India")


def workdir(variant):
    return ROOT / "u4" / (((TAG + "__") if TAG else "") + variant.replace("+", "_"))
GRID = tuple(float(t) for t in np.round(np.arange(0.45, 0.901, 0.025), 3))
T0 = time.time()
_G = {}


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def blocks(split, variant, rows=None):
    parts = []
    for b in variant.split("+"):
        name = {"base": "x_base", "q": "x_q", "coll": "x_coll"}.get(b, f"x_{b}")
        sd = {"brest": BREST, "test": TEST}.get(split, split)
        arr = np.load(PREP / sd / f"{name}.npy", mmap_mode="r")
        parts.append(np.asarray(arr if rows is None else arr[rows], dtype=np.float32))
    return np.hstack(parts)


def load_arena(split=None):
    d = PREP / (split or BREST)
    ld = lambda n: np.load(d / f"{n}.npy")  # noqa: E731
    meta = json.loads((d / "meta.json").read_text())
    arena = Arena(ld("left"), ld("right"), ld("label"), ld("true_count"), ld("rec_empty"), ld("name_exact"),
                  ld("mult"), ld("population"), np.asarray(meta["countries"])[ld("country_code")])
    return arena, ld("component")


def _score_t(t):
    arena, prob, rows, pop = _G["arena"], _G["prob"], _G["rows"], _G["pop"]
    acc = np.zeros(len(arena.left), bool)
    acc[rows] = decide(prob[rows], arena.left[rows], arena.right[rows], t, None)
    return t, arena.score(acc, pop)["macro_f05"]


def sweep(arena, prob, rows, pop, workers=8):
    _G.update(arena=arena, prob=prob, rows=rows, pop=pop)
    with get_context("fork").Pool(workers) as p:
        res = dict(p.map(_score_t, GRID))
    best = max(res, key=lambda t: (res[t], -abs(t - 0.7)))
    return res, best


def evaluate_oof(arena, component, fold_s1, oof, tag, workers=8):
    """Cross-fitted thresholds (pick on the other 4 folds) + full-OOF best. Returns result, per_s1, picks."""
    fold_row = fold_s1[arena.left]
    picks = {}
    for k in range(5):
        other = np.flatnonzero(fold_row != k)
        _, best = sweep(arena, oof, other, arena.population & (fold_s1 != k) & (fold_s1 >= 0), workers)
        picks[k] = best
    thr = np.asarray([picks[k] for k in range(5)], np.float32)[fold_row]
    res = arena.score(accept_per_row(arena, oof, thr, "model"), full=True)
    per = res.pop("per_s1")
    grid, best = sweep(arena, oof, np.arange(len(oof)), arena.population, workers)
    log(f"{tag}: cross-fitted {res['macro_f05']:.6f} picks {picks}; full-OOF best t={best} {grid[best]:.6f}")
    return slim(res), per, {"per_fold_threshold": {str(k): v for k, v in picks.items()}, "full_oof_threshold": best,
                            "full_oof_f05": grid[best], "full_oof_grid": {str(t): s for t, s in grid.items()}}


def fit_cmd(args):
    work = workdir(args.variant)
    work.mkdir(parents=True, exist_ok=True)
    arena, component = load_arena()
    fold_s1 = np.load(FOLDS)
    fold_row = fold_s1[arena.left]
    if np.any(fold_row < 0):
        raise SystemExit("B_REST row outside the fold file")
    x = blocks("brest", args.variant)
    y = arena.label
    log(f"{args.variant}: X {x.shape}, positives {int(y.sum())}")
    oof_path = work / "oof.npy"
    if oof_path.exists():
        oof = np.load(oof_path)
        log("reuse saved OOF")
    elif PER_COUNTRY:
        oof = np.zeros(len(y), np.float32)
        crow = arena.country[arena.left]
        for k in range(5):
            for c in COUNTRIES:
                tr, te = (fold_row != k) & (crow == c), (fold_row == k) & (crow == c)
                model, n = fit_catboost(x[tr], y[tr], args.threads, params=PARAMS)
                model.save_model(str(work / f"model_fold{k}_{c}.cbm"))
                oof[te] = proba(model, x[te])
                log(f"fold {k} {c}: fit {n} rows, predicted {int(te.sum())}")
        np.save(oof_path, oof)
    else:
        oof = np.zeros(len(y), np.float32)
        for k in range(5):
            tr, te = fold_row != k, fold_row == k
            model, n = fit_catboost(x[tr], y[tr], args.threads, params=PARAMS)
            model.save_model(str(work / f"model_fold{k}.cbm"))
            oof[te] = proba(model, x[te])
            log(f"fold {k}: fit {n} rows, predicted {int(te.sum())}")
        np.save(oof_path, oof)
    res, per, picks = evaluate_oof(arena, component, fold_s1, oof, args.variant, args.workers)
    np.save(work / "per_s1.npy", per)
    boot = {}
    for other in args.compare:
        p = ROOT / "u4" / other.replace("+", "_") / "per_s1.npy"
        if p.exists():
            boot[f"minus_{other}"] = component_bootstrap(np.load(p), per, component, n=1000, seed=20260927,
                                                         population=arena.population)
    # paired vs the saved 2-fold trees (their own saved OOF + saved per-half picks, parity-checked in prep)
    arena0 = load_arena("brest")[0] if BREST != "brest" else arena
    half_row = (component[arena0.left] % 2).astype(np.int8)
    for name, thr_pair in (("oof_c2tree", (0.725, 0.725)), ("oof_qtree", (0.725, 0.725)), ("oof_coll", None)):
        if thr_pair is None:
            rep = json.loads((ROOT / "prep_reports" / "coll_report.json").read_text()) \
                if (ROOT / "prep_reports" / "coll_report.json").exists() else None
            if rep is None:
                continue
            thr_pair = (rep["crossfit_picks"]["0"]["model"]["threshold"], rep["crossfit_picks"]["1"]["model"]["threshold"])
        thr = np.where(half_row == 1, thr_pair[0], thr_pair[1]).astype(np.float32)
        saved = arena0.score(accept_per_row(arena0, np.load(PREP / "brest" / f"{name}.npy"), thr, "model"),
                            full=True)["per_s1"]
        boot[f"minus_saved_{name}"] = component_bootstrap(saved, per, component, n=1000, seed=20260927,
                                                          population=arena.population)
        boot[f"minus_saved_{name}"]["saved_f05"] = float(saved[arena.population].mean())
    names = []
    for b in args.variant.split("+"):
        f = PREP / BREST / f"names_{b}.json"
        names += json.loads(f.read_text()) if f.exists() else [f"{b}_{i}" for i in range(np.load(
            PREP / BREST / ({"base": "x_base", "q": "x_q", "coll": "x_coll"}.get(b, f"x_{b}") + ".npy"),
            mmap_mode="r").shape[1])]
    from catboost import CatBoostClassifier
    imp = {}
    for k in range(5):
        for suffix in ([f"_{c}" for c in COUNTRIES] if PER_COUNTRY else [""]):
            m = CatBoostClassifier()
            m.load_model(str(work / f"model_fold{k}{suffix}.cbm"))
            for n_, v in zip(names, m.get_feature_importance()):
                imp[n_] = imp.get(n_, 0.0) + float(v) / (5 * (2 if PER_COUNTRY else 1))
    report = {"variant": args.variant, "features": names, "catboost": PARAMS, "tag": TAG, "fit_rows_cap": FIT_ROWS,
              "seed": SEED, "fold_file": str(FOLDS), "fold_file_sha256": sha(FOLDS),
              "protocol": "5-fold component-grouped OOF on B_REST; per-fold threshold on the other 4 folds; "
                          "one owner + cap 11", "result": res, "thresholds": picks, "bootstrap": boot,
              "importance_mean": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
              "oof_sha256": sha(oof_path), "slice_read": False, "holdout2_rows_read": 0,
              "elapsed_s": round(time.time() - T0, 1)}
    (work / "report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    log(json.dumps({"result": res, "boot": {k: (round(v["delta"], 6), round(v["ci_low"], 6), round(v["ci_high"], 6))
                                            for k, v in boot.items()}}, default=float))


def test_cmd(args):
    from catboost import CatBoostClassifier
    work = workdir(args.variant)
    d = PREP / TEST
    meta = json.loads((d / "meta.json").read_text())
    countries = np.asarray(meta["countries"])
    s1_country = countries[np.load(d / "country_code.npy")]
    left = np.load(d / "left.npy")
    rows = np.flatnonzero(np.isin(s1_country[left], ["US", "India"]))
    x = blocks("test", args.variant, rows)
    preds = []
    rc = s1_country[left[rows]]
    for k in range(5):
        if PER_COUNTRY:
            pk = np.zeros(len(rows), np.float32)
            for c in COUNTRIES:
                m = CatBoostClassifier()
                m.load_model(str(work / f"model_fold{k}_{c}.cbm"))
                sel = rc == c
                pk[sel] = proba(m, x[sel])
            preds.append(pk)
        else:
            m = CatBoostClassifier()
            m.load_model(str(work / f"model_fold{k}.cbm"))
            preds.append(proba(m, x))
        log(f"test fold {k} predicted {len(rows)} rows")
    preds = np.stack(preds)
    np.save(work / "test_rows.npy", rows)
    np.save(work / "test_pred_folds.npy", preds.astype(np.float16))
    np.save(work / "test_prob_mean.npy", preds.mean(0).astype(np.float32))
    log("saved test predictions")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "test"])
    ap.add_argument("--variant", required=True)
    ap.add_argument("--threads", type=int, default=22)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compare", nargs="*", default=["base"])
    a = ap.parse_args()
    (fit_cmd if a.cmd == "fit" else test_cmd)(a)


if __name__ == "__main__":
    main()
