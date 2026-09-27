"""Stage-1 string GBDT trained on Agent S's Frenchified arena (arena_fr_v3, option A, train-derived only).

    python -m ber.string_gbdt_fr --feat O/arena_fr --data D --work W [--stage2-from Wv2]

Fit on partition-0 pairs (French-operation world + A replay), early stopping and threshold on the
Frenchified tuning world (partition 2 = tune_s1), monitor on the 20k B_REST monitor S1s (partition 1).
Candidate lists in the French worlds are pruned (tree-gated negatives + synthetic), so only a pairwise
(stage-1) model is fit here. The same booster is saved as stage1_fold0/1 so ber.string_gbdt_apply can
use it either alone (--stage1-only) or under the arena-trained stage 2 copied from --stage2-from.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from ber.arena_decision import evaluate
from ber.string_gbdt import CATEGORICAL, FEATURES, log
from ber.string_gbdt_train import THRESHOLDS, decode, fit, load_feat, summary, take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--stage2-from")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    d = Path(args.data) / "arena_fr_v3"
    pairs, X, s1, rec = load_feat(args.feat)
    left, right, label = pairs["left"].astype(np.int64), pairs["right"].astype(np.int64), pairs["label"]
    part = np.load(d / "s1_partition.npy")
    true_count = np.load(d / "labels/true_count.npy").astype(np.int64)
    tr = np.flatnonzero(part[left] == 0)
    tu = np.flatnonzero(part[left] == 2)
    mo = np.flatnonzero(part[left] == 1)
    cats = [FEATURES.index(c) for c in CATEGORICAL]
    log(f"fit rows {len(tr)} (pos {int(label[tr].sum())}), tune rows {len(tu)}, monitor rows {len(mo)}")
    m = fit(take(X, tr), label[tr].astype(np.float32), take(X, tu), label[tu].astype(np.float32), FEATURES, cats)
    for f in (0, 1):
        m.save_model(str(work / f"stage1_fold{f}.txt"))
    p = np.zeros(len(left), dtype=np.float32)
    ev = np.concatenate([tu, mo])
    p[ev] = m.predict(take(X, ev), num_threads=16)
    rec_empty = rec["empty"].to_numpy()[right]
    mult = s1["mult"].to_numpy()[left]
    core_eq = np.asarray(X[:, FEATURES.index("n_core_eq")])
    report = {"fit_rows": int(len(tr)), "iterations": m.best_iteration,
              "importance_gain": dict(zip(FEATURES, map(float, m.feature_importance("gain"))))}
    for name, rows, pop in (("tune_fr_world", tu, part == 2), ("b_rest_monitor", mo, part == 1)):
        sweep = []
        for thr in THRESHOLDS:
            acc = decode(p[rows], thr, "model", left[rows], right[rows], rec_empty[rows], core_eq[rows], mult[rows])
            sweep.append((float(evaluate(acc, left[rows], label[rows], true_count, population=pop)["macro_f05"]), thr))
        report[name + "_sweep"] = sweep
    best_thr = max(report["tune_fr_world_sweep"])[1]
    for name, rows, pop in (("tune_fr_world", tu, part == 2), ("b_rest_monitor", mo, part == 1)):
        acc = decode(p[rows], best_thr, "model", left[rows], right[rows], rec_empty[rows], core_eq[rows], mult[rows])
        report[name] = summary(evaluate(acc, left[rows], label[rows], true_count, population=pop))
    report["stage1"] = {"a_tune_best": {"threshold": best_thr, "variant": "model"}}
    if args.stage2_from:
        shutil.copy(Path(args.stage2_from) / "stage2.txt", work / "stage2.txt")
        v2 = json.loads((Path(args.stage2_from) / "report.json").read_text())
        report["stage2"] = {"a_tune_best": v2["stage2"]["a_tune_best"], "from": args.stage2_from}
    else:
        report["stage2"] = report["stage1"]
    (work / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    log(json.dumps({k: report[k] for k in ("tune_fr_world", "b_rest_monitor", "stage1")}, indent=1))


if __name__ == "__main__":
    main()
