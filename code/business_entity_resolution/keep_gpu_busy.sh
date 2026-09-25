#!/bin/bash
# On the pod. Starts experiments/queue.json when the A40 has no compute process.
# One instance. Does not touch a job that is already running.
# /workspace does not allow chmod. Start with: bash keep_gpu_busy.sh
set -u
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p /workspace/az-ml-new/experiments /tmp/az-ml-v2/logs
exec 9>/tmp/az-ml-v2/logs/keep_gpu_busy.lock
flock -n 9 || exit 0
LOG=/workspace/az-ml-new/experiments/keep_gpu_busy.log
QUEUE=/workspace/az-ml-new/experiments/queue.json
while true; do
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q '[0-9]'; then
    sleep 30
    continue
  fi
  if pgrep -f 'run_gpu_queue.py|run_v2_queue.py' >/dev/null 2>&1; then
    sleep 30
    continue
  fi
  echo "$(date -u +%FT%TZ) gpu idle, starting queue" >> "$LOG"
  /opt/amlc-venv/bin/python -u /workspace/az-ml-new/experiments/run_gpu_queue.py \
    --queue "$QUEUE" --logs /tmp/az-ml-v2/logs >> "$LOG" 2>&1 || true
  sleep 60
done
