"""P5 (27 Sep, Agent U): per-country isotonic calibration + count posterior (ber.v2_decision) on a saved B_REST OOF.

Calibration for the S1s of fold k is fitted on the other four folds of the frozen component fold file.
Count prior P(T) and discovery (share of true links inside the candidate rows) come from partition A labels.
Decode per S1: expected-F0.5-optimal prefix of its candidates by calibrated p (CountPosterior.best_prefix),
then one owner (highest calibrated p wins) and cap 11. Baseline: the same OOF, threshold 0.725, one owner,
cap 11. Paired component bootstrap. B_REST labels + A labels only.

    python p5_decode.py OOF_NAME [GATE]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

from ber.arena_decision import component_bootstrap
from ber.pair_tree_arena import rank_within
from ber.pair_tree_c2 import Arena, accept_per_row, slim
from ber.arena_decision import one_owner
from ber.v2_decision import CountPosterior, fit_calibration, predict_calibration

PREP = Path("/workspace/az-ml-new/run27/U/prep/brest")
ARENA = Path("/workspace/az-ml-new/arena")
CACHE = Path("/opt/az-ml-new/agents/pair_tree_c2/step005000/arena_v1_cache.npz")
FOLDS = Path("/workspace/az-ml-new/run27/U/folds/brest_fold5_u4-20260927.npy")
OUT = Path("/workspace/az-ml-new/run27/U/audits")
OOF = sys.argv[1] if len(sys.argv) > 1 else "oof_qtree"
GATE = float(sys.argv[2]) if len(sys.argv) > 2 else 0.005
MAXLIST = 15
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


ld = lambda n: np.load(PREP / f"{n}.npy")  # noqa: E731
meta = json.loads((PREP / "meta.json").read_text())
country = np.asarray(meta["countries"])[ld("country_code")]
arena = Arena(ld("left"), ld("right"), ld("label"), ld("true_count"), ld("rec_empty"), ld("name_exact"), ld("mult"),
              ld("population"), country)
L, R, Y = arena.left, arena.right, arena.label
part = np.load(ARENA / "finalize_v1/s1_partition.npy")
comp = ld("component")
fold = np.load(FOLDS)
raw = ld(OOF).astype(np.float64)

# ---- A-partition count prior and discovery (training labels)
with np.load(CACHE) as c:
    cl, cr = c["left"].astype(np.int64), c["cand_rows"]
a_rows = part[cl] == 0
label_all = np.load(ARENA / "labels/label.npy", mmap_mode="r")
a_pos = int(np.asarray(label_all[cr[a_rows]]).sum())
a_prev = a_pos / int(a_rows.sum())
tc = arena.true_count.astype(np.int64)
a_true = int(tc[part == 0].sum())
hist = np.bincount(tc[part == 0])
prior = (hist + 0.5) / (hist.sum() + 0.5 * len(hist))
prior = prior / prior.sum()
discovery = a_pos / a_true
log(f"A prior over T=0..{len(hist) - 1}; discovery {discovery:.4f}; candidate prevalence {a_prev:.4f}")
post = CountPosterior(prior, discovery, a_prev)

gate = raw >= GATE
calibrated = np.zeros(len(raw))
cal_models = {}
for k in range(5):
    fit_refs = (part == 1) & (fold != k) & (fold >= 0)
    sel = fit_refs[L]
    tp_ = {"gated_pair_prevalence": float(Y[sel & gate].mean()), "ungated_pair_prevalence": float(Y[sel & ~gate].mean()),
           "country_pair_prior": {}}
    models = fit_calibration(raw, gate, Y, L, country, fit_refs, tp_)
    held = fold[L] == k
    pred = predict_calibration(models, raw[held], gate[held], L[held], country)
    calibrated[held] = pred
    cal_models[k] = {c: {"p_ungated": m["p_ungated"], "gated_pairs": m["gated_pairs"]} for c, m in models.items()}
    log(f"fold {k} calibrated")

# ---- per-S1 expected-F0.5 prefix
order = np.lexsort((R, -calibrated, L))
Ls = L[order]
starts = np.flatnonzero(np.r_[True, Ls[1:] != Ls[:-1]])
ends = np.r_[starts[1:], len(Ls)]
choose = np.zeros(len(L), bool)
dist = {}
for a, b in zip(starts, ends):
    rows = order[a:b]
    g = gate[rows]
    p = calibrated[rows]
    gp = p[g][:MAXLIST]
    n_total = b - a
    ung = float(p[~g].mean()) if (~g).any() else 1e-6
    if len(gp) == 0:
        dist[0] = dist.get(0, 0) + 1
        continue
    kbest, _ = post.best_prefix(gp, ung, max(n_total, len(gp)))
    choose[rows[g][:kbest]] = True
    dist[kbest] = dist.get(kbest, 0) + 1
log(f"prefix choices {dict(sorted(dist.items()))}")
acc = one_owner(choose, L, R, calibrated)
rows = np.flatnonzero(acc)
acc[rows[rank_within(L[rows], calibrated[rows]) >= 11]] = False
res = arena.score(acc, full=True)
per = res.pop("per_s1")
base = arena.score(accept_per_row(arena, raw.astype(np.float32), np.full(len(L), 0.725, np.float32), "model"),
                   full=True)
per_b = base.pop("per_s1")
# calibrated + single threshold (cross-fitted: pick on the other folds)
thr_grid = np.round(np.arange(0.4, 0.9, 0.025), 3)
res_t = {}
boot = component_bootstrap(per_b, per, comp, n=1000, seed=20260927, population=arena.population)
out = {"oof": OOF, "gate": GATE, "fold_file": str(FOLDS), "a_discovery": discovery, "a_candidate_prevalence": a_prev,
       "a_prior": prior.tolist(), "baseline_0.725": slim(base), "p5": slim(res), "p5_minus_baseline": boot,
       "prefix_distribution": {str(k): v for k, v in sorted(dist.items())}, "calibration": cal_models,
       "keep_rule": "keep only if delta >= +0.0003 with CI > 0"}
(OUT / f"p5_{OOF}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
log(json.dumps({"baseline": base["macro_f05"], "p5": res["macro_f05"], "boot": boot}, default=float))
