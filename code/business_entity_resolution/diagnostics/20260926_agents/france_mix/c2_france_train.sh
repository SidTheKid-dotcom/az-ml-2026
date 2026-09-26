#!/bin/bash
# C2 French fine-tune on the mixed arena. Waits for the C2 step-40000 checkpoint. Uses the GPU.
set -u
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
INIT=/workspace/az-ml-new/c2_plain/ckpt/step_040000
M=/workspace/az-ml-new/france_mix
O=/workspace/az-ml-new/c2_france
mkdir -p $O
until [ -f $INIT/complete.json ]; do sleep 20; done
exec /opt/az-venv/bin/python -u -m ber.fast_trainer --command train --mode c2 --text-view plain \
  --steps 6000 --batch-size 128 --microbatch 128 --fused --no-freeze-embeddings --amp bf16 \
  --length-buckets 64,128,256 --save-every 2000 --defer-checkpoint-hash --compile --prefetch 8 \
  --learning-rate 1e-5 --schedule cosine \
  --arena $M/arena --labels $M/labels --partition $M/s1_partition.npy \
  --sealed-reservation /workspace/az-ml-new/v2_holdout2 \
  --model $INIT --allowlist /opt/az-ml-new/repo/problem-statement/models.yaml \
  --output $O/out --checkpoint-root $O/ckpt --device cuda --threads 16
