#!/bin/bash
# Agent J (H100): after training -> ship final adapter (tag final) -> score H100 shard queue (uploads each shard).
set -u
cd /root/J
export PYTHONPATH=/root/J/code
RUN=/dev/shm/J/run1
until [ -e $RUN/result.json ]; do sleep 20; done
CK=$(/opt/az-venv/bin/python -c "import json; print(json.load(open('$RUN/result.json'))['final_checkpoint'])")
echo "final checkpoint $CK $(date)"
bash ship_ck.sh put_ck.txt final $CK $RUN
PY=/opt/az-venv/bin/python TABLE=/root/J/put_scores_run1.txt bash score_queue.sh $CK /dev/shm/J/models/Qwen3-Reranker-4B \
  /dev/shm/J/score_inputs /dev/shm/J/scores_run1 ${TOK:-131072} ${BS:-4096} "${MERGEFLAG:-}" ${SHARDS:-brest_0 brest_1 brestconf_0 test_0 test_1 fr1_0 fr1_1 fr2_0 fr2_1}
echo "AFTER DONE $(date)"
