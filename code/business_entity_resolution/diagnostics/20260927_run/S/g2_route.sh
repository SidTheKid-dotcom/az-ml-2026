#!/bin/bash
# Agent S gpu2: Qwen-FR (step 11,250) scores for U route pairs and R reverse pairs (B_REST + test), after French scoring.
set -u
R=/home/ubuntu/run27/S; PY=/home/ubuntu/s-venv/bin/python
export PYTHONPATH=$R/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
cd $R
SETS_REV=$R/rev SETS_OUT=$R/rsets SETS_ARENA_POOL=$R/in/arena/v1/pool.parquet SETS_TEST_POOL=$R/in/test/dense_test/pool.parquet \
  $PY scripts/c2v2_score_sets.py || exit 1
for SET in brest_routes test_routes brest_reverse test_reverse; do
  case $SET in brest_*) QREF=$R/in/arena/v1/reference.parquet; QPOOL=$R/rsets/brest_ext/pool.parquet;;
                *) QREF=$R/in/test/dense_test/reference.parquet; QPOOL=$R/rsets/test_ext/pool.parquet;; esac
  QREF=$QREF QPOOL=$QPOOL $PY scripts/s_qwen_prep_score.py pairs $R/rsets/$SET models/Qwen3-Reranker-0.6B $R/rsets/q_$SET > $R/rsets/q_$SET.prep.log 2>&1 || exit 1
done
echo "prep done $(date -u)"
while pgrep -f "[q]wen_train score" > /dev/null; do sleep 20; done
for SET in brest_routes test_routes brest_reverse test_reverse; do
  $PY -u -m ber.qwen_train score --data $R/rsets/q_$SET --checkpoint $R/qwen_fr/checkpoints/step_011250 --output $R/rsets/out_$SET --weights-bf16 \
    > $R/rsets/out_$SET.log 2>&1; echo "$SET $? $(date -u)"
done
echo "ALL DONE $(date -u)"
