#!/bin/bash
# Agent S gpu3: after C2-FR(base) training -> score eval S1s (B_REST monitor + French tuning world) and the French
# rescoring set (French S1s, v1 tree >= 0.01; only with G3_FRANCE=1 -- default: French scoring runs on the H100) with the final checkpoint.
set -u
R=/home/ubuntu/run27/S; PY=/opt/pytorch/bin/python
export PYTHONPATH=$R/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
while kill -0 $(cat $R/train.pid) 2>/dev/null; do sleep 30; done
CK=$(ls -d $R/c2fr_base/ckpt/step_* | sort | tail -1); N=c2fr_base_$(basename $CK)
[ -f $CK/complete.json ] || { echo "no complete checkpoint"; exit 1; }
mkdir -p $R/eval $R/fr
[ -f $R/france_s1.npy ] || $PY $R/scripts/s_france_mask.py $R/in/test $R/france_s1.npy
$PY -u -m ber.score_external --checkpoint $CK --reference $R/arena_fr_v3/arena/reference.parquet --pool $R/arena_fr_v3/arena/pool.parquet \
  --pairs $R/arena_fr_v3/arena/pairs.parquet --tree $R/arena_fr_v3/arena/v1_tree.npy --gate 0.01 --population $R/arena_fr_v3/eval_s1.npy \
  --output $R/eval/$N --inference-batch 512 --threads 6 --encode-workers 4 > $R/eval/$N.log 2>&1; echo "eval exit $? $(date -u)"
[ -n "${G3_FRANCE:-}" ] && $PY -u -m ber.score_external --checkpoint $CK --reference $R/in/test/dense_test/reference.parquet --pool $R/in/test/dense_test/pool.parquet \
  --left $R/in/test/index/left.npy --right $R/in/test/index/right.npy --tree $R/in/test/base_probabilities.npy --gate 0.01 \
  --population $R/france_s1.npy --output $R/fr/$N --inference-batch 512 --threads 6 --encode-workers 4 > $R/fr/$N.log 2>&1; echo "france exit $? $(date -u)"
