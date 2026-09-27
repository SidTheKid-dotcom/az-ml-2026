"""Agent S task 4 (CPU, no labels): pre-count and write the French rescoring set for C2-FR.

Set = French cache candidate rows where C2 40k >= 0.01 or v1 tree >= 0.01 or Qwen >= 0.01, or
(tree >= 0.9 and C2 < 0.05). Above --max-pairs, prune to the uncertain band (plain blend 0.05-0.5)
+ disagreement set (tree >= 0.9 & C2 < 0.05, or |C2 - Qwen| >= 0.5) + currently accepted rows.
Writes /workspace/az-ml-new/run27/S/fr_rescore/{left,right,tree,rows}.npy and precount.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "F"))
from fcommon import OUT as FOUT, load, plain_of

MAX_PAIRS = 5_000_000
S = Path("/workspace/az-ml-new/run27/S/fr_rescore")
QD = Path("/workspace/az-ml-new/qwen/test_step_003250_c2min0.01")
t0 = time.time()
S.mkdir(parents=True, exist_ok=True)
d = load(need_accept=False)
left, right, base, c2, france = d["left"], d["right"], d["base"], d["c2"], d["france"]
cand = d["cache"]["cand_rows"].astype(np.int64)
c2z = np.nan_to_num(c2, nan=0.0)
q = np.full(len(cand), np.nan, np.float32)
if (QD / "scores.npy").exists():
    qs = np.asarray(np.load(QD / "scores.npy", mmap_mode="r")[cand]); qm = np.asarray(np.load(QD / "scored_mask.npy", mmap_mode="r")[cand])
    q = np.where(qm, qs, np.nan).astype(np.float32)
qz = np.nan_to_num(q, nan=0.0)
comp = {"c2_ge_0.01": france & (c2z >= .01), "tree_ge_0.01": france & (base >= .01), "qwen_ge_0.01": france & (qz >= .01),
        "tree_ge_0.9_c2_lt_0.05": france & (base >= .9) & (c2z < .05)}
sel = np.zeros(len(left), bool)
for v in comp.values():
    sel |= v
R = {"french_cache_rows": int(france.sum()), "components": {k: int(v.sum()) for k, v in comp.items()},
     "set_rows": int(sel.sum()), "qwen_scores_found": bool((QD / "scores.npy").exists()),
     "c2_unscored_in_set": int((sel & np.isnan(c2)).sum())}
pp, pl = plain_of(d, c2)
if sel.sum() > MAX_PAIRS:
    band = france & (pp >= .05) & (pp < .5)
    dis = comp["tree_ge_0.9_c2_lt_0.05"] | (france & ~np.isnan(q) & (np.abs(c2z - qz) >= .5))
    sel = band | dis | (pl & france)
    R["pruned"] = {"rule": "uncertain band 0.05-0.5 | disagreement | currently accepted", "rows": int(sel.sum())}
rows = np.flatnonzero(sel)
np.save(S / "rows.npy", rows); np.save(S / "left.npy", left[rows]); np.save(S / "right.npy", right[rows])
np.save(S / "tree.npy", np.maximum(base[rows], 0.01).astype(np.float32))
R["written_rows"] = int(len(rows)); R["french_s1_touched"] = int(len(np.unique(left[rows])))
R["seconds"] = round(time.time() - t0, 1)
(S / "precount.json").write_text(json.dumps(R, indent=1) + "\n")
print(json.dumps(R, indent=1))
