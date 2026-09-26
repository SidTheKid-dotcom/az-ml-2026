"""CPU smoke test of ber.qwen_train with the real Qwen3-Reranker-0.6B (no GPU).

1. zero-shot scores on the first 2 training steps (256 partition-A pairs): AUC by label;
2. qwen_train.train for 2 steps on CPU (fp32), checkpoint written and reloaded;
3. qwen_train.score on the first 300 B_REST gated pairs with that checkpoint.
"""
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ber import qwen_train as qt

MODEL = "/workspace/az-ml-new/models/Qwen3-Reranker-0.6B"
DATA = Path("/workspace/az-ml-new/qwen/train_data")
BREST = Path("/workspace/az-ml-new/qwen/score_brest")
WORK = Path("/workspace/az-ml-new/qwen/smoke_cpu")
torch.set_num_threads(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
if WORK.exists():
    shutil.rmtree(WORK)
WORK.mkdir(parents=True)
manifest = json.loads((DATA / "manifest.json").read_text())
tokens, offsets = np.load(DATA / "tokens.npy", mmap_mode="r"), np.load(DATA / "offsets.npy")
labels = np.load(DATA / "labels.npy")
t0 = time.time()
SKIP = len(sys.argv) > 2 and sys.argv[2] == 'skip-zero-shot'
model = None if SKIP else qt.load_model(MODEL, "cpu", torch)
if not SKIP:
    model.eval()
    scorer = qt.YesNoScorer(model, manifest["yes_id"], manifest["no_id"])
    rows = np.arange(256)
    scores = np.empty(256, dtype=np.float32)
    with torch.no_grad():
        for part in np.array_split(rows, 8):
            ids, last = qt.pad_right(tokens, offsets, part, manifest["pad_id"])
            scores[part] = scorer(torch, torch.from_numpy(ids), torch.from_numpy(last)).numpy()
    pos, neg = scores[labels[:256] == 1], scores[labels[:256] == 0]
    auc = float((pos[:, None] > neg[None, :]).mean() + .5 * (pos[:, None] == neg[None, :]).mean())
    print(json.dumps({"zero_shot_auc_first_256": auc, "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
                      "seconds": round(time.time() - t0, 1)}), flush=True)

model = scorer = None

# 2 training steps on CPU through the real train() path, from a 2-step copy of the data
small = WORK / "train_small"
small.mkdir()
n = 2 * manifest["batch_size"]
np.save(small / "tokens.npy", np.asarray(tokens[:offsets[n]]))
np.save(small / "offsets.npy", offsets[:n + 1])
np.save(small / "labels.npy", labels[:n])
np.save(small / "weights.npy", np.load(DATA / "weights.npy")[:n])
(small / "manifest.json").write_text(json.dumps({**manifest, "steps": 2}))
args = SimpleNamespace(data=str(small), model=MODEL, output=str(WORK / "run"), device="cpu", amp="fp32", steps=2,
                       time_budget_min=None, reserve_min=0.0, calibrate_steps=220, microbatches=8,
                       learning_rate=2e-5, weight_decay=.01, warmup=400, schedule="cosine", lr_floor=.1,
                       checkpoint_every=1, log_every=1, resume=False, compile=False, threads=4, seed=1)
t1 = time.time()
result = qt.train(args)
print(json.dumps({"train_result": result, "seconds": round(time.time() - t1, 1)}), flush=True)
ck = Path(result["final_checkpoint"])
print(sorted(p.name for p in ck.iterdir()), flush=True)

# score the first 300 B_REST gated pairs with the 2-step checkpoint
bm = json.loads((BREST / "manifest.json").read_text())
sub = WORK / "brest_small"
sub.mkdir()
b_off = np.load(BREST / "offsets.npy")
np.save(sub / "tokens.npy", np.asarray(np.load(BREST / "tokens.npy", mmap_mode="r")[:b_off[300]]))
np.save(sub / "offsets.npy", b_off[:301])
np.save(sub / "pair_index.npy", np.load(BREST / "pair_index.npy")[:300])
(sub / "manifest.json").write_text(json.dumps({**bm, "scored_pairs": 300}))
sargs = SimpleNamespace(data=str(sub), checkpoint=str(ck), output=str(WORK / "score"), device="cpu", amp="fp32",
                        batch_size=64, token_budget=20_000, flush_every=2, threads=4)
t2 = time.time()
out = qt.score(sargs)
s = np.load(WORK / "score" / "scores.npy", mmap_mode="r")
m = np.load(WORK / "score" / "scored_mask.npy", mmap_mode="r")
print(json.dumps({"score_scored": int(np.asarray(m).sum()), "score_mean": float(np.asarray(s)[np.asarray(m)].mean()),
                  "seconds": round(time.time() - t2, 1)}), flush=True)
