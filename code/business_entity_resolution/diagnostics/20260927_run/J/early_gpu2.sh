#!/bin/bash
# Agent J (gpu2): early read — fetch intermediate adapter TAG, score SHARDS (merged), upload under the early table.
set -u
cd ~/J
export PYTHONPATH=$HOME/J/code
TAG=$1; shift
RUN=$HOME/J/run2_$TAG
until [ -s $RUN/checkpoints/step_$TAG/complete.json ]; do
  bash fetch_ck.sh get_ck.txt $TAG $RUN > logs/fetch_$TAG.log 2>&1
  [ -s $RUN/checkpoints/step_$TAG/complete.json ] || sleep 30
done
echo "fetched $TAG $(date)"
PY=$HOME/s-venv/bin/python TABLE=$HOME/J/put_scores_early1000.txt bash score_queue.sh $RUN/checkpoints/step_$TAG \
  $HOME/J/models/Qwen3-Reranker-4B $HOME/J/score_inputs $HOME/J/early_$TAG 65536 2048 "" "$@"
echo "EARLY DONE $(date)"
