"""Score the first 300 B_REST gated pairs on CPU with the 2-step smoke checkpoint; compare with direct forward."""
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ber import qwen_train as qt

torch.set_num_threads(6)
BREST = Path("/workspace/az-ml-new/qwen/score_brest")
WORK = Path("/workspace/az-ml-new/qwen/smoke_cpu")
ck = WORK / "run" / "checkpoints" / "step_000002"
bm = json.loads((BREST / "manifest.json").read_text())
sub = WORK / "brest_small"
if sub.exists():
    shutil.rmtree(sub)
sub.mkdir()
b_off = np.load(BREST / "offsets.npy")
tokens = np.asarray(np.load(BREST / "tokens.npy", mmap_mode="r")[:b_off[300]])
np.save(sub / "tokens.npy", tokens)
np.save(sub / "offsets.npy", b_off[:301])
pair_index = np.load(BREST / "pair_index.npy")[:300]
np.save(sub / "pair_index.npy", pair_index)
(sub / "manifest.json").write_text(json.dumps({**bm, "scored_pairs": 300}))
if (WORK / "score").exists():
    shutil.rmtree(WORK / "score")
args = SimpleNamespace(data=str(sub), checkpoint=str(ck), output=str(WORK / "score"), device="cpu", amp="fp32",
                       batch_size=64, token_budget=20_000, flush_every=2, threads=6, c2_scores=None, min_c2=None)
out = qt.score(args)
s = np.load(WORK / "score" / "scores.npy", mmap_mode="r")
m = np.load(WORK / "score" / "scored_mask.npy", mmap_mode="r")
model = qt.load_model(ck, "cpu", torch)
model.eval()
scorer = qt.YesNoScorer(model, bm["yes_id"], bm["no_id"])
with torch.no_grad():
    direct = [float(scorer(torch, torch.from_numpy(tokens[b_off[k]:b_off[k + 1]].astype(np.int64))[None],
                           torch.tensor([b_off[k + 1] - b_off[k] - 1]))[0]) for k in (0, 150, 299)]
print(json.dumps({"scored": int(np.asarray(m).sum()), "mask_ok": bool(np.array_equal(np.flatnonzero(np.asarray(m)), pair_index)),
                  "prob_mean": float(np.asarray(s)[pair_index].mean()),
                  "direct_vs_batched": [[1 / (1 + np.exp(-d)), float(s[pair_index[k]])] for d, k in zip(direct, (0, 150, 299))],
                  "runtime": out["runtime"]}), flush=True)
