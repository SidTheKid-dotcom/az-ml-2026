#!/bin/bash
# Agent S gpu2: after Qwen-FR training -> score eval S1s and the French rescoring set with the final checkpoint.
set -u
R=/home/ubuntu/run27/S; PY=/home/ubuntu/s-venv/bin/python
export PYTHONPATH=$R/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
while pgrep -f "ber.qwen_train train" >/dev/null; do sleep 30; done
until grep -q PREPDONE $R/qprep.log; do sleep 30; done
CK=$(ls -d $R/qwen_fr/checkpoints/step_* | sort | tail -1); N=qwen_fr_$(basename $CK)
[ -f $CK/complete.json ] || { echo "no complete checkpoint"; exit 1; }
mkdir -p $R/eval $R/fr
$PY -u -m ber.qwen_train score --data $R/qwen_score_eval --checkpoint $CK --output $R/eval/$N --weights-bf16 > $R/eval/$N.log 2>&1; echo "eval exit $? $(date -u)"
$PY -u -m ber.qwen_train score --data $R/qwen_score_france --checkpoint $CK --output $R/fr/$N --weights-bf16 > $R/fr/$N.log 2>&1; echo "france exit $? $(date -u)"
