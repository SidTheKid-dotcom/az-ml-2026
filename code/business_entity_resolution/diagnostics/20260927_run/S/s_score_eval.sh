#!/bin/bash
# Agent S: score one checkpoint on the arena_fr evaluation S1s (B_REST monitor + Frenchified tuning world). ~2-4 min GPU.
# Usage: s_score_eval.sh CHECKPOINT_DIR NAME
set -u
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
M=/workspace/az-ml-new/run27/S/arena_fr_v3; O=/workspace/az-ml-new/run27/S/eval
mkdir -p $O
/opt/az-venv/bin/python -u -m ber.score_external --checkpoint $1 --reference $M/arena/reference.parquet --pool $M/arena/pool.parquet \
  --pairs $M/arena/pairs.parquet --tree $M/arena/v1_tree.npy --gate 0.01 --population $M/eval_s1.npy \
  --output $O/$2 --inference-batch 1024 --threads 3 --encode-workers 3 > $O/$2.log 2>&1
echo "exit $?"
