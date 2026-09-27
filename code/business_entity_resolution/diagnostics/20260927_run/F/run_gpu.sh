#!/bin/bash
# Agent F GPU queue: G2 (F1 rescoring, critical) then G1 (F7 probes: step_010000, base model).
set -u
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src
PY=/opt/az-venv/bin/python
F=/workspace/az-ml-new/run27/F
CK=/workspace/az-ml-new/c2_plain/ckpt
COMMON="--gate 0.01 --amp bf16 --inference-batch 1024 --threads 8 --encode-workers 4"
echo "G2 start $(date -u)"
$PY -m ber.score_external --checkpoint $CK/step_040000 --reference $F/inputs/reference.parquet --pool $F/inputs/pool.parquet \
  --left $F/g2/left.npy --right $F/g2/right.npy --tree $F/g2/tree.npy --output $F/g2/scores_step040000 $COMMON > $F/logs/g2.log 2>&1
echo "G2 exit $? $(date -u)"
T=/opt/az-ml-new/test_inputs/dense_test
$PY -m ber.score_external --checkpoint $CK/step_010000 --reference $T/reference.parquet --pool $T/pool.parquet \
  --left $F/g1/left.npy --right $F/g1/right.npy --tree $F/g1/tree.npy --output $F/g1/scores_step010000 $COMMON > $F/logs/g1_10k.log 2>&1
echo "G1 10k exit $? $(date -u)"
$PY /opt/az-ml-new/repo/code/business_entity_resolution/diagnostics/20260927_run/F/base_score.py --checkpoint /workspace/az-ml-new/models/bge-reranker-v2-m3 --reference $T/reference.parquet --pool $T/pool.parquet \
  --left $F/g1/left.npy --right $F/g1/right.npy --tree $F/g1/tree.npy --output $F/g1/scores_base $COMMON > $F/logs/g1_base.log 2>&1
echo "G1 base exit $? $(date -u)"
