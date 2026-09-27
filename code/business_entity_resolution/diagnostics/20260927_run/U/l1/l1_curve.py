"""Idea 4 learning curve: standalone B_REST score of each A-trained L1 (stage-2 p2) and a single-column block for
the tree. Standalone decode: one owner + cap 11, threshold picked per fold on the other 4 folds (frozen fold file).

    python l1_curve.py VARIANT TAG [TAG ...]      (TAG "" = the v1 delivery model; use "base" for it)
Writes prep/{brest,test}/x_l1<v><tag>.npy (+ names) and prep/l1/curve_<v><tag>.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from u4_tree import evaluate_oof, load_arena  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
L1 = P / "l1"
v = sys.argv[1]
tj = json.loads((L1 / "targets.json").read_text())
n_cand, n_tp = tj["arena_cand"], tj["test_prep"]
al = np.load(L1 / "targets_arena_left.npy")
part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
bpos = np.flatnonzero(part[al[:n_cand]] == 1)
meta = json.loads((P / "test/meta.json").read_text())
fr = np.asarray(meta["countries"])[np.load(P / "test/country_code.npy")][np.load(P / "test/left.npy")] == "France"
arena, component = load_arena("brest")
fold_s1 = np.load(ROOT / "folds/brest_fold5_u4-20260927.npy")
for tag in sys.argv[2:]:
    t = "" if tag == "base" else tag
    pa = np.load(L1 / f"p2_{v}{t}_arena.npy")[bpos]
    pt = np.load(L1 / f"p2_{v}{t}_test.npy")[:n_tp].copy()
    pt[fr] = np.nan
    name = f"l1{v}{tag}"
    np.save(P / "brest" / f"x_{name}.npy", pa[:, None].astype(np.float32))
    np.save(P / "test" / f"x_{name}.npy", pt[:, None].astype(np.float32))
    (P / "brest" / f"names_{name}.json").write_text(json.dumps([name]))
    res, per, picks = evaluate_oof(arena, component, fold_s1, pa.astype(np.float32), name, workers=8)
    rep = json.loads((L1 / f"l1_{v}{t}_report.json").read_text())
    out = {"variant": v, "tag": tag, "frac": rep.get("frac_a_components", "4.5M-row sample"),
           "a_rows_used": rep.get("a_rows_used"), "stage2_fit_rows": rep["stage2"]["fit_rows"],
           "a_holdout": rep["a_holdout"], "standalone_brest": res, "thresholds": picks["per_fold_threshold"]}
    (L1 / f"curve_{v}{tag}.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print(json.dumps({"tag": tag, "f05": res["macro_f05"], "P": res["precision"], "R": res["recall"],
                      "a_holdout": rep["a_holdout"]}, default=float), flush=True)
