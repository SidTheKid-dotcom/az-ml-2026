#!/bin/bash
# Agent J: H100 4B LoRA run2 = fixed 2,250 steps (288k pairs), cosine over the fixed steps (run1 was stopped: its
# time-budget calibration would have measured the rate while Agent S's cont scoring shared the GPU).
set -u
cd /root/J
export PYTHONPATH=/root/J/code
PY=/opt/az-venv/bin/python
M=/dev/shm/J/models/Qwen3-Reranker-4B
echo "start $(date)"
$PY -u -m ber.qwen4b_train train --data /dev/shm/J/train_v1 --model $M --output /dev/shm/J/run2 --steps ${STEPS:-2250} \
  --microbatches ${MB:-4} --lora-dropout 0 --learning-rate 1e-4 --warmup 100 \
  --checkpoint-every 250 --log-every 25 --threads 8 --stop-file /root/J/STOP_TRAIN > logs/train_run2.log 2>&1
echo "train exit $? $(date)"
