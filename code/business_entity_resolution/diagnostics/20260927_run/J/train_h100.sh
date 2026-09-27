#!/bin/bash
# Agent J: full 4B LoRA training on the H100 after Agent S's C2 v2 chain ("ALL DONE") and an idle GPU.
set -u
cd /root/J
export PYTHONPATH=/root/J/code
PY=/opt/az-venv/bin/python
M=/dev/shm/J/models/Qwen3-Reranker-4B
until grep -q "ALL DONE" /root/c2v2/after.log; do sleep 15; done
while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 10; done
[ -e /root/J/HOLD_TRAIN ] && { echo "held $(date)"; exit 0; }
echo "GPU free $(date)"
$PY -u -m ber.qwen4b_train train --data /dev/shm/J/train_v1 --model $M --output /dev/shm/J/run1 --steps 3463 \
  --time-budget-min ${BUDGET:-60} --microbatches ${MB:-4} --lora-dropout 0 --learning-rate 1e-4 --warmup 100 \
  --checkpoint-every 500 --log-every 25 --threads 8 --stop-file /root/J/STOP_TRAIN > logs/train_run1.log 2>&1
echo "train exit $? $(date)"
