#!/bin/bash
# Agent F follow-up: C2 step_010000 on all French candidate rows (original inputs), concurrent with Q's training.
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src
F=/workspace/az-ml-new/run27/F; T=/opt/az-ml-new/test_inputs/dense_test
echo "G4 start $(date -u)"
/opt/az-venv/bin/python -m ber.score_external --checkpoint /workspace/az-ml-new/c2_plain/ckpt/step_010000 \
  --reference $T/reference.parquet --pool $T/pool.parquet --left $F/g4/left.npy --right $F/g4/right.npy --tree $F/g4/tree.npy \
  --output $F/g4/scores_step010000 --gate 0.01 --amp bf16 --inference-batch 1024 --threads 6 --encode-workers 3 > $F/logs/g4.log 2>&1
echo "G4 exit $? $(date -u)"
