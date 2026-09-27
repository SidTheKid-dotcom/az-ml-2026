#!/bin/bash
# Agent S: C2-FR training (GPU). Run only when /workspace/az-ml-new/run27/gpu.lock contains S.
set -u
L=/workspace/az-ml-new/run27/gpu.lock
[ "$(tr -d '[:space:]' < $L)" = "S" ] || { echo "lock is not S: $(cat $L)"; exit 3; }
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
INIT=/workspace/az-ml-new/c2_plain/ckpt/step_040000
M=/workspace/az-ml-new/run27/S/arena_fr_v3
O=/workspace/az-ml-new/run27/S/c2fr
mkdir -p $O
exec /opt/az-venv/bin/python -u -m ber.fast_trainer --command train --mode c2 --text-view plain \
  --steps ${STEPS:-18000} --batch-size 128 --microbatch 128 --fused --no-freeze-embeddings --amp bf16 \
  --length-buckets 48,64,80,96,128,160,256 --save-every 2000 --defer-checkpoint-hash --compile --prefetch 8 \
  --learning-rate 1e-5 --schedule cosine --weight-decay 0.01 --warmup-fraction 0.05 --seed 20260925 \
  --arena $M/arena --labels $M/labels --partition $M/s1_partition.npy \
  --sealed-reservation /workspace/az-ml-new/v2_holdout2 \
  --model $INIT --allowlist /opt/az-ml-new/repo/problem-statement/models.yaml \
  --output $O/out --checkpoint-root $O/ckpt --device cuda --threads 8
