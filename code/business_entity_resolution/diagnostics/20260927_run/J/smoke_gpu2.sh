#!/bin/bash
# Agent J: 4B LoRA smoke test on gpu2 (L40S) once no other GPU process runs: throughput, memory, loss, merge check.
set -u
cd ~/J
export PYTHONPATH=$HOME/J/code
PY=$HOME/s-venv/bin/python
M=$HOME/J/models/Qwen3-Reranker-4B
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 15; done
echo "GPU free $(date)"
$PY -u -m ber.qwen4b_train train --data $HOME/J/train_v1 --model $M --output $HOME/J/smoke_run --steps 40 \
  --microbatches 8 --lora-dropout 0 --warmup 10 --log-every 5 --checkpoint-every 40 --threads 4 > logs/smoke_train.log 2>&1
echo "train exit $? $(date)"
CK=$HOME/J/smoke_run/checkpoints/step_000040
$PY -u -m ber.qwen4b_train score --data $HOME/J/score_inputs/mini_brest --checkpoint $CK --model $M \
  --output $HOME/J/smoke_score_merged --batch-size 512 --token-budget 32768 > logs/smoke_score_merged.log 2>&1
echo "score merged exit $? $(date)"
$PY -u -m ber.qwen4b_train score --data $HOME/J/score_inputs/mini_brest --checkpoint $CK --model $M --no-merge \
  --output $HOME/J/smoke_score_unmerged --batch-size 512 --token-budget 32768 > logs/smoke_score_unmerged.log 2>&1
echo "score unmerged exit $? $(date)"
$PY - <<'PY'
import numpy as np
a = np.load("/home/ubuntu/J/smoke_score_merged/logits_gated.npy"); b = np.load("/home/ubuntu/J/smoke_score_unmerged/logits_gated.npy")
d = np.abs(a - b)
print("merge check: n", len(a), "max |dlogit|", float(d.max()), "mean", float(d.mean()), "p99", float(np.percentile(d, 99)), "corr", float(np.corrcoef(a, b)[0, 1]))
PY
echo "SMOKE DONE $(date)"
