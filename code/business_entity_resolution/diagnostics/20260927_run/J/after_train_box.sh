#!/bin/bash
# Agent J (gpu2/gpu3): wait for the shipped final adapter, fetch it, score this box's shard queue (uploads each shard).
# Env: PY, MODEL, IN, OUT, TOK, BS, MERGEFLAG, SHARDS
set -u
cd ~/J
export PYTHONPATH=$HOME/J/code
RUN=$HOME/J/run1
until [ -s $RUN/checkpoints/step_final/adapter_model.safetensors ] && [ -s $RUN/checkpoints/step_final/complete.json ]; do
  bash fetch_ck.sh get_ck.txt final $RUN > logs/fetch_final.log 2>&1
  [ -s $RUN/checkpoints/step_final/complete.json ] || sleep 30
done
echo "fetched final $(date)"
TABLE=$HOME/J/put_scores_run1.txt bash score_queue.sh $RUN/checkpoints/step_final $MODEL $IN $OUT $TOK $BS "${MERGEFLAG:-}" $SHARDS
echo "AFTER DONE $(date)"
