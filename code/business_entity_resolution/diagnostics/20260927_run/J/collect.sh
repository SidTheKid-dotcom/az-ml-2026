#!/bin/bash
# Agent J (EC2): pull scored passes, assemble keyed parquets, evaluate on B_REST, adversarial check.
# Usage: collect.sh OUTNAME PASS [PASS ...]   (PASS = scores_raw tag: core, mid, fr, early1000 ...); EVAL=1 to evaluate
set -u
NAME=$1; shift
cd ~/run27/J
PY=/opt/az-ml-new/venv/bin/python
RAWS=""
for P in "$@"; do
  aws s3 sync s3://az-ml-new-20260926-533267274451/run27/J/scores_raw/$P/ raw/$P/ --only-show-errors
  RAWS="$RAWS raw/$P"
done
$PY code/assemble.py keys deliver_$NAME $RAWS
if [ "${EVAL:-0}" = 1 ] && [ -e deliver_$NAME/llm_brest.parquet ]; then
  $PY -u code/eval_brest.py in deliver_$NAME/llm_brest.parquet eval/${NAME}_eval.json > logs/eval_$NAME.log 2>&1
  tail -3 logs/eval_$NAME.log | cut -c1-3500
fi
if [ "${ADV:-0}" = 1 ] && [ -e deliver_$NAME/llm_brest.parquet ] && [ -e deliver_$NAME/llm_test.parquet ]; then
  $PY code/adv_check.py in/texts deliver_$NAME/llm_brest.parquet deliver_$NAME/llm_test.parquet eval/${NAME}_adv.json > logs/adv_$NAME.log 2>&1
  echo "adv exit $?"
fi
echo "COLLECT DONE $(date)"
