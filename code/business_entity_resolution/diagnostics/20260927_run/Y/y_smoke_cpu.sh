#!/bin/bash
# Agent Y gpu3 CPU smoke (no GPU): 2 training steps of the unchanged fast_trainer via ber.laya_c2 on arena_fr_v3,
# checkpoint commit, then score_external through ber.laya_c2 on a tiny population. Output under Y/smoke (deleted later).
set -u
Y=/home/ubuntu/run27/Y; S=/home/ubuntu/run27/S; A=$S/arena_fr_v3
export PYTHONPATH=$Y/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=3 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
PY=/opt/pytorch/bin/python
rm -rf $Y/smoke; mkdir -p $Y/smoke
nice -n 10 $PY -u -m ber.laya_c2 train --mode c2 --text-view plain --steps 2 --batch-size 8 --microbatch 4 --no-fused \
  --no-freeze-embeddings --amp fp32 --length-buckets 48,64,80,96,128,160,256 --save-every 2 --prefetch 2 \
  --learning-rate 2e-5 --schedule cosine --weight-decay 0.01 --warmup-fraction 0.05 --seed 20260925 \
  --arena $A/arena --labels $A/labels --partition $A/s1_partition.npy --sealed-reservation $S/v2_holdout2 \
  --model $Y/models/laya-multilingual-seqcls --allowlist $Y/repo/problem-statement/models_laya.yaml \
  --output $Y/smoke/out --checkpoint-root $Y/smoke/ckpt --device cpu --threads 3
echo "train exit $?"
$PY -c "import numpy as np; p=np.load('$A/eval_s1.npy'); m=np.zeros(len(p),bool); m[np.flatnonzero(p)[:50]]=True; np.save('$Y/smoke/pop50.npy', m)"
nice -n 10 $PY -u -m ber.laya_c2 score --checkpoint $Y/smoke/ckpt/step_000002 --reference $A/arena/reference.parquet \
  --pool $A/arena/pool.parquet --pairs $A/arena/pairs.parquet --tree $A/arena/v1_tree.npy --gate 0.01 \
  --population $Y/smoke/pop50.npy --output $Y/smoke/score --device cpu --amp fp32 --inference-batch 64 --threads 3 --encode-workers 1
echo "score exit $?"
