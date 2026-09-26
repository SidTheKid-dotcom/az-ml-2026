#!/bin/bash
# Usage: c2_france_score.sh STEP  -> French test pairs + mixed-arena monitor (candidate vs C2 step 40000). Uses the GPU.
set -u
STEP=$(printf "%06d" ${1:-6000})
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
PY=/opt/az-venv/bin/python; T=/opt/az-ml-new/test_inputs; M=/workspace/az-ml-new/france_mix; O=/workspace/az-ml-new/c2_france
CK=$O/ckpt/step_$STEP; BASE=/workspace/az-ml-new/c2_plain/ckpt/step_040000
until [ -f $CK/complete.json ]; do sleep 20; done
# 1) French test pairs only (259,452 French S1s; 3,121,523 pairs with tree >= 0.01). Non-French rows stay 0.
$PY -u -m ber.score_external --checkpoint $CK --reference $T/dense_test/reference.parquet --pool $T/dense_test/pool.parquet \
  --left $T/index/left.npy --right $T/index/right.npy --tree $T/base_probabilities.npy --gate 0.01 \
  --population $M/test_france_s1.npy --output $O/test_france_scores_step$STEP --inference-batch 1024 \
  > $O/test_france_scores_step$STEP.log 2>&1 || { echo "FRENCH TEST SCORE FAILED"; exit 1; }
# 2) Monitor: B_REST 20k US/India sample (real labels) + French SLICE (pseudo), candidate and baseline on the same pairs.
for NAME in cand_step$STEP base_c2_step040000; do
  C=$CK; [ $NAME = base_c2_step040000 ] && C=$BASE
  [ -f $O/mix_eval_$NAME/complete.json ] && continue
  $PY -u -m ber.score_external --checkpoint $C --reference $M/arena/reference.parquet --pool $M/arena/pool.parquet \
    --pairs $M/arena/pairs.parquet --tree $M/arena/v1_tree.npy --gate 0.01 --population $M/eval_s1.npy \
    --output $O/mix_eval_$NAME --inference-batch 1024 > $O/mix_eval_$NAME.log 2>&1 || { echo "MIX SCORE FAILED $NAME"; exit 1; }
done
# W/T: the C2 chain's choice for step 40000 (chosen_on_full_b_rest), else pure CE at 0.5.
R=/opt/az-ml-new/runs/c2_step040000/eval/report.json
W=$($PY -c "import json;print(json.load(open('$R'))['chosen_on_full_b_rest']['weight'])" 2>/dev/null || echo 1.0)
TH=$($PY -c "import json;print(json.load(open('$R'))['chosen_on_full_b_rest']['threshold'])" 2>/dev/null || echo 0.5)
$PY /opt/az-ml-new/agents/france_mix/france_mix_arena.py evaluate --output $M --scores $O/mix_eval_cand_step$STEP \
  --baseline-scores $O/mix_eval_base_c2_step040000 --weight $W --threshold $TH --report $O/mix_eval_step$STEP.json
