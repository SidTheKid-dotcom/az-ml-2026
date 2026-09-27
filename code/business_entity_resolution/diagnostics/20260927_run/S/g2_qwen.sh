#!/bin/bash
# Agent S gpu2 (L40S 46 GB): Qwen3-Reranker-0.6B French variant ("Qwen-FR") from the HF base on arena_fr_v3.
# prepare-train materializes the same ArenaBatchSampler world (1:3, typed/random, partition A) C2 uses.
set -u
R=/home/ubuntu/run27/S; PY=/home/ubuntu/s-venv/bin/python
export PYTHONPATH=$R/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6 PYTHONDONTWRITEBYTECODE=1
M=$R/arena_fr_v3; Q=$R/models/Qwen3-Reranker-0.6B; D=$R/qwen_fr_data; O=$R/qwen_fr
if [ ! -f $D/manifest.json ]; then
  $PY -u -m ber.qwen_train prepare-train --model $Q --out $D --arena $M/arena --labels $M/labels \
    --partition $M/s1_partition.npy --sealed-reservation $R/v2_holdout2 --steps ${STEPS:-12000} || exit 1
fi
exec $PY -u -m ber.qwen_train train --data $D --model $Q --output $O --time-budget-min ${BUDGET:-110} \
  --learning-rate 2e-5 --schedule cosine --checkpoint-every 2000 ${COMPILE---compile}
