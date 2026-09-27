#!/bin/bash
# Agent S: score C2-FR on the French rescoring set (raw test text). GPU, ~15 min.
# Usage: s_score_france.sh CHECKPOINT_DIR NAME [REFERENCE POOL]
set -u
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
T=/opt/az-ml-new/test_inputs/dense_test; R=/workspace/az-ml-new/run27/S/fr_rescore
REF=${3:-$T/reference.parquet}; POOL=${4:-$T/pool.parquet}
/opt/az-venv/bin/python -u -m ber.score_external --checkpoint $1 --reference $REF --pool $POOL \
  --left $R/left.npy --right $R/right.npy --tree $R/tree.npy --gate 0.01 \
  --output $R/$2 --inference-batch 1024 --threads 8 --encode-workers 6 > $R/$2.log 2>&1
echo "exit $?"
