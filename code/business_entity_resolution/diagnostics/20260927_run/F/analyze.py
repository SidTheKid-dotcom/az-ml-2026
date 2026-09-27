"""Agent F: F7 (probe links under C2 10k / base / 40k) and F12 (Qwen pilot vs C2) tables. No labels read."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fcommon import OUT, load

d = load(need_accept=False)
cand = d["cache"]["cand_rows"].astype(np.int64); c2, france, base = d["c2"], d["france"], d["base"]
n_fr = int((d["country"] == "France").sum())
g1rows = np.load(OUT / "g1/rows.npy"); g1set = np.load(OUT / "g1/set.npy")
def stats(x):
    x = np.asarray(x, np.float64); return {"n": int(len(x)), "mean": round(float(np.nanmean(x)), 4), "share_ge_0.5": round(float((x >= .5).mean()), 4)}
R = {"F7": {}, "F12": {}}
models = {"c2_step040000 (stored)": c2[g1rows]}
for nm, sub in (("c2_step010000", "scores_step010000"), ("bge_base", "scores_base")):
    p = OUT / "g1" / sub
    if (p / "complete.json").exists():
        models[nm] = np.load(p / "scores.npy")
for nm, v in models.items():
    R["F7"][nm] = {"probe_A_T2r(64-69% true)": stats(v[g1set == 0]), "probe_Cprime(92-96% true)": stats(v[g1set == 1])}
QD = Path("/workspace/az-ml-new/qwen/test_step_003250_c2min0.01"); QP = OUT / "g3/qwen_pilot"
if (QP / "complete.json").exists():
    qfull = np.asarray(np.load(QD / "scores.npy", mmap_mode="r")[cand]); qm = np.asarray(np.load(QD / "scored_mask.npy", mmap_mode="r")[cand])
    pfull = np.asarray(np.load(QP / "scores.npy", mmap_mode="r")[cand]); pm = np.asarray(np.load(QP / "scored_mask.npy", mmap_mode="r")[cand])
    q = np.where(pm, pfull, np.where(qm, qfull, np.nan)); has = pm | qm
    c2z = np.nan_to_num(c2, nan=0.0)
    sa = np.load(OUT / "g3/sel_rows_a.npy"); sb = np.load(OUT / "g3/sel_rows_b.npy")
    th = np.flatnonzero(france & (base >= .9) & (c2z < .05))
    R["F12"]["tree_high_c2_low_lacking_qwen(a)"] = {"qwen": stats(q[sa]), "c2": stats(c2z[sa])}
    R["F12"]["tree_high_c2_low_all(141k)"] = {"qwen": stats(q[th]), "c2": stats(c2z[th]), "qwen_coverage": float(has[th].mean())}
    R["F12"]["probe_links_c2_lt_0.01(b)"] = {"qwen": stats(q[sb]), "c2": stats(c2z[sb])}
    for nm, k in (("probe_A_T2r", 0), ("probe_Cprime", 1)):
        r = g1rows[g1set == k]
        R["F12"][nm + "_all_links"] = {"qwen": stats(q[r]), "c2": stats(c2z[r]), "qwen_coverage": float(has[r].mean())}
    ex = np.flatnonzero(france & qm)
    fr_s1_ex = len(np.unique(d["left"][ex]))
    R["F12"]["existing_french_c2_ge_0.01_pairs"] = {
        "pairs": int(len(ex)), "s1s_with_pairs": int(fr_s1_ex),
        "qwen": {**stats(qfull[ex]), "band_0.05_0.5_per_french_s1": round(float(((qfull[ex] >= .05) & (qfull[ex] < .5)).sum() / n_fr), 4)},
        "c2": {**stats(c2z[ex]), "band_0.05_0.5_per_french_s1": round(float(((c2z[ex] >= .05) & (c2z[ex] < .5)).sum() / n_fr), 4)},
        "agree_at_0.5": round(float(((qfull[ex] >= .5) == (c2z[ex] >= .5)).mean()), 4),
        "qwen_yes_c2_no": int(((qfull[ex] >= .5) & (c2z[ex] < .5)).sum()), "qwen_no_c2_yes": int(((qfull[ex] < .5) & (c2z[ex] >= .5)).sum())}
    usi = np.flatnonzero(~france & qm)
    R["F12"]["reference_us_india_c2_ge_0.01_pairs"] = {"qwen": stats(qfull[usi]), "c2": stats(c2z[usi]),
        "agree_at_0.5": round(float(((qfull[usi] >= .5) == (c2z[usi] >= .5)).mean()), 4)}
(OUT / "analyze.json").write_text(json.dumps(R, indent=1) + "\n")
print(json.dumps(R, indent=1))
