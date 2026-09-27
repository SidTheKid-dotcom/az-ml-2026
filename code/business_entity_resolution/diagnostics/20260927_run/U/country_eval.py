"""Per-country models vs the pooled model (27 Sep, Agent U). B_REST labels only.

    U_BREST=brest_u8r python country_eval.py POOLED_WORK PERCOUNTRY_WORK [FOLDFILE]

Decoders compared (all one owner + cap 11, 5-fold cross-fitted thresholds):
  a pooled model, one threshold per fold (picked on the other 4 folds);
  b pooled model, one threshold per fold AND country (picked on the other 4 folds' S1s of that country);
  c per-country model, per-country thresholds;  d per-country model, one pooled threshold.
Paired component bootstrap, overall and restricted to each country's S1s.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ber.arena_decision import component_bootstrap  # noqa: E402
from ber.pair_tree_c2 import accept_per_row, slim  # noqa: E402
from u4_tree import ROOT, load_arena, sweep  # noqa: E402

pooled, perc = ROOT / "u4" / sys.argv[1], ROOT / "u4" / sys.argv[2]
fold_s1 = np.load(ROOT / "folds" / (sys.argv[3] if len(sys.argv) > 3 else "brest_fold5_u4-20260927.npy"))
arena, component = load_arena()
fold_row = fold_s1[arena.left]
s1c = arena.country
crow = s1c[arena.left]
pop = arena.population
C = ("US", "India")


def picks(oof, per_country):
    thr = np.zeros(len(oof), np.float32)
    out = {}
    for k in range(5):
        groups = C if per_country else (None,)
        for c in groups:
            rows = np.flatnonzero((fold_row != k) & ((crow == c) if c else True))
            p = pop & (fold_s1 != k) & (fold_s1 >= 0) & ((s1c == c) if c else True)
            _, best = sweep(arena, oof, rows, p, workers=8)
            sel = (fold_row == k) & ((crow == c) if c else True)
            thr[sel] = best
            out[f"{k}|{c}"] = best
    return thr, out


res, per = {}, {}
for name, work, pc in (("a_pooled_thr", pooled, False), ("b_pooled_model_country_thr", pooled, True),
                       ("c_country_model_country_thr", perc, True), ("d_country_model_pooled_thr", perc, False)):
    oof = np.load(work / "oof.npy")
    thr, pk = picks(oof, pc)
    r = arena.score(accept_per_row(arena, oof, thr, "model"), full=True)
    per[name] = r.pop("per_s1")
    res[name] = {"result": slim(r), "thresholds": pk}
    print(name, round(r["macro_f05"], 6), {c: round(v["macro_f05"], 6) for c, v in r["by_country"].items()}, flush=True)
boot = {}
for x, y in (("a_pooled_thr", "b_pooled_model_country_thr"), ("a_pooled_thr", "c_country_model_country_thr"),
             ("b_pooled_model_country_thr", "c_country_model_country_thr"), ("a_pooled_thr", "d_country_model_pooled_thr")):
    for c in (None,) + C:
        p = pop & ((s1c == c) if c else True)
        b = component_bootstrap(per[x], per[y], component, n=1000, seed=20260927, population=p)
        boot[f"{y}_minus_{x}|{c or 'all'}"] = b
        print(f"{y} - {x} [{c or 'all'}]: {b['delta']:+.6f} [{b['ci_low']:+.6f}, {b['ci_high']:+.6f}]", flush=True)
out = {"pooled": str(pooled), "per_country": str(perc), "fold_file": sys.argv[3] if len(sys.argv) > 3 else "main",
       "results": res, "bootstrap": boot, "note": "per-country bootstrap deltas are macro deltas over that country's S1s"}
(perc / "country_eval.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
