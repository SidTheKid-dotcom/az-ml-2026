#!/bin/bash
# Agent Y gpu3 (A10G 24 GB): Agent S's C2-FR(base) recipe (S/g3_train.sh: arena_fr_v3 partition A, plain text,
# batch 128, lr 2e-5 cosine, 5% warmup, seed 20260925) with the Laya-multilingual encoder (mmBERT-base/ModernBERT,
# fresh 1-logit BCE head) instead of bge-reranker-v2-m3. Only differences: model/allowlist, microbatch, tokenizer workers.
set -u
Y=/home/ubuntu/run27/Y; S=/home/ubuntu/run27/S; A=$S/arena_fr_v3
export PYTHONPATH=$Y/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
export BER_TOKENIZER_WORKERS=${WORKERS:-2}
O=$Y/laya_c2fr; mkdir -p $O
exec /opt/pytorch/bin/python -u -m ber.laya_c2 train --mode c2 --text-view plain \
  --steps ${STEPS:-10000} --batch-size 128 --microbatch ${MICRO:-64} --fused --no-freeze-embeddings --amp bf16 \
  --length-buckets 48,64,80,96,128,160,256 --save-every 2000 --defer-checkpoint-hash ${COMPILE---compile} --prefetch 8 \
  --learning-rate ${LR:-2e-5} --schedule cosine --weight-decay 0.01 --warmup-fraction 0.05 --seed 20260925 \
  --arena $A/arena --labels $A/labels --partition $A/s1_partition.npy --sealed-reservation $S/v2_holdout2 \
  --model $Y/models/laya-multilingual-seqcls --allowlist $Y/repo/problem-statement/models_laya.yaml \
  --output $O/out --checkpoint-root $O/ckpt --device cuda --threads 6
