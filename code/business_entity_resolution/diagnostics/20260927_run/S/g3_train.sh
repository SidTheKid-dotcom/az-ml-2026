#!/bin/bash
# Agent S gpu3 (A10G 24 GB): C2-FR variant from the BASE BAAI/bge-reranker-v2-m3 on arena_fr_v3 (rebuilt here).
# C2's original base-init learning rate (2e-5), cosine over 10k steps, batch 128 as 4 x 32 microbatches.
set -u
R=/home/ubuntu/run27/S
export PYTHONPATH=$R/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
M=$R/arena_fr_v3; O=$R/c2fr_base
mkdir -p $O
exec /opt/pytorch/bin/python -u -m ber.fast_trainer --command train --mode c2 --text-view plain \
  --steps ${STEPS:-10000} --batch-size 128 --microbatch ${MICRO:-32} --fused --no-freeze-embeddings --amp bf16 \
  --length-buckets 48,64,80,96,128,160,256 --save-every 2000 --defer-checkpoint-hash ${COMPILE---compile} --prefetch 8 \
  --learning-rate 2e-5 --schedule cosine --weight-decay 0.01 --warmup-fraction 0.05 --seed 20260925 \
  --arena $M/arena --labels $M/labels --partition $M/s1_partition.npy \
  --sealed-reservation $R/v2_holdout2 \
  --model $R/models/bge-reranker-v2-m3 --allowlist $R/repo/problem-statement/models.yaml \
  --output $O/out --checkpoint-root $O/ckpt --device cuda --threads 6
