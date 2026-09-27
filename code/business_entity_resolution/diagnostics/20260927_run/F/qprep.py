"""Agent F G3a prep (F12 pilot): select French pairs lacking Qwen scores in (a) tree-high/C2-low (tree >= 0.9 &
old C2 < 0.05) and (b) probe links (A, C') with old C2 < 0.01. Writes a selector dir usable as ber.qwen_train score
--c2-scores DIR --min-c2 0.5 (selected pairs = 1.0, others 0.0; mask all True). No labels read."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from fcommon import OUT, load
from ber.pair_tree_c2 import log

d = load(need_accept=False)
cand = d["cache"]["cand_rows"].astype(np.int64); base, c2, france = d["base"], d["c2"], d["france"]
QD = Path("/workspace/az-ml-new/qwen/test_step_003250_c2min0.01")
qmask = np.asarray(np.load(QD / "scored_mask.npy", mmap_mode="r")[cand])
pidx = np.load("/workspace/az-ml-new/qwen/score_test/pair_index.npy")
in_data = np.zeros(d["n_pairs"], bool); in_data[pidx] = True
c2z = np.nan_to_num(c2, nan=0.0)
th = france & (base >= .9) & (c2z < .05)
g1rows = np.load(OUT / "g1/rows.npy"); g1set = np.load(OUT / "g1/set.npy")
pr = np.zeros(len(cand), bool); pr[g1rows] = True
pb = pr & (c2z < .01)
sel_a = th & ~qmask; sel_b = pb & ~qmask
sel = sel_a | sel_b
cover = in_data[cand[sel]]
rep = {"tree_high_c2_low_french_rows": int(th.sum()), "of_which_have_qwen": int((th & qmask).sum()), "to_score_a": int(sel_a.sum()),
       "probe_rows_c2_lt_0.01": int(pb.sum()), "probe_A_c2_lt_0.01": int(pb[g1rows[g1set == 0]].sum()),
       "probe_C_c2_lt_0.01": int(pb[g1rows[g1set == 1]].sum()), "to_score_b": int(sel_b.sum()),
       "to_score_total": int(sel.sum()), "in_qwen_tokenized_data": int(cover.sum())}
log(json.dumps(rep))
out = OUT / "g3/selector"; out.mkdir(parents=True, exist_ok=True)
s = np.lib.format.open_memmap(out / "scores.npy", mode="w+", dtype=np.float32, shape=(d["n_pairs"],)); s[:] = 0; s[cand[sel]] = 1.0; s.flush(); del s
np.save(out / "scored_mask.npy", np.ones(d["n_pairs"], bool))
(out / "complete.json").write_text(json.dumps({"state": "complete", "note": "selector for Agent F F12 pilot, not C2 scores", **rep}) + "\n")
np.save(OUT / "g3/sel_rows_a.npy", np.flatnonzero(sel_a)); np.save(OUT / "g3/sel_rows_b.npy", np.flatnonzero(sel_b))
(OUT / "g3/qprep.json").write_text(json.dumps(rep, indent=1) + "\n")
