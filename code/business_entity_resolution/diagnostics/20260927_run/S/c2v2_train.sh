#!/bin/bash
# Agent S: "C2 v2" = bge-reranker-v2-m3 from BASE on the standard arena (US/India, partition A, TRAIN labels, dense top-50
# candidates + C2's typed hard negatives). No A-side route/reverse pairs exist, so this is C2's recipe re-run with a cosine
# schedule and fine length buckets. 40k steps, batch 128 (microbatch 128), lr 2e-5.
set -u
export PYTHONPATH=/root/s_stage/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
I=/dev/shm/c2v2/in; O=/dev/shm/c2v2
exec /opt/az-venv/bin/python -u -m ber.fast_trainer --command train --mode c2 --text-view plain \
  --steps ${STEPS:-40000} --batch-size 128 --microbatch 128 --fused --no-freeze-embeddings --amp bf16 \
  --length-buckets 48,64,80,96,128,160,256 --save-every 4000 --defer-checkpoint-hash --compile --prefetch 8 \
  --learning-rate 2e-5 --schedule cosine --weight-decay 0.01 --warmup-fraction 0.05 --seed 20260925 \
  --arena $I/arena --labels $I/labels --partition $I/s1_partition.npy --sealed-reservation /root/s_stage/v2_holdout2 \
  --model /root/s_stage/models/bge-reranker-v2-m3 --allowlist /root/s_stage/repo/problem-statement/models.yaml \
  --output $O/out --checkpoint-root $O/ckpt --device cuda --threads 8
