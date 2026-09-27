"""Agent S: filter array for Qwen-FR French scoring (full test index): max(C2 40k, C2-FR, 1[tree>=0.5]) on French
tree-gated pairs; qwen_train score --min-c2 0.01 then skips pairs that no C2 variant or the tree finds plausible.
Usage: s_qwen_filter.py C2FR_SCORES_DIR OUT_DIR"""
import json, sys
from pathlib import Path
import numpy as np
c2 = np.load("/opt/az-ml-new/runs/c2_step040000/test_scores/scores.npy", mmap_mode="r")
fr = np.load(Path(sys.argv[1]) / "scores.npy", mmap_mode="r"); frm = np.load(Path(sys.argv[1]) / "scored_mask.npy")
tree = np.load("/opt/az-ml-new/test_inputs/base_probabilities.npy", mmap_mode="r")
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
f = np.maximum(np.maximum(np.nan_to_num(np.asarray(c2, np.float32)), np.asarray(fr, np.float32)), (np.asarray(tree) >= .5).astype(np.float32))
f[~frm] = 0
np.save(out / "scores.npy", f.astype(np.float32)); np.save(out / "scored_mask.npy", frm)
R = {"french_gated": int(frm.sum()), "kept_ge_0.01": int((frm & (f >= .01)).sum()),
     "rule": "max(C2 40k, C2-FR 18k, 1[tree>=0.5]) >= 0.01 on French tree>=0.01 pairs"}
(out / "complete.json").write_text(json.dumps(R) + "\n"); print(json.dumps(R))
