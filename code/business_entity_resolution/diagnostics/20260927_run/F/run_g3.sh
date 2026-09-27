#!/bin/bash
# Agent F G3a: wait for the G2/G1 queue, run the Qwen F12 pilot on the selected pairs, then hand the GPU to Q.
set -u
F=/workspace/az-ml-new/run27/F
QPID=$(cat $F/logs/gpu_queue.pid)
while kill -0 $QPID 2>/dev/null; do sleep 5; done
echo "G3a start $(date -u)"
PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src /opt/az-venv/bin/python -m ber.qwen_train score \
  --data /workspace/az-ml-new/qwen/score_test --checkpoint /workspace/az-ml-new/qwen/run1/checkpoints/step_003250 \
  --output $F/g3/qwen_pilot --c2-scores $F/g3/selector --min-c2 0.5 --batch-size 1024 --token-budget 131072 --threads 8 \
  > $F/logs/g3a.log 2>&1
echo "G3a exit $? $(date -u)"
if [ "$(cat /workspace/az-ml-new/run27/gpu.lock)" = "F" ]; then echo Q > /workspace/az-ml-new/run27/gpu.lock; echo "lock -> Q $(date -u)"; fi
