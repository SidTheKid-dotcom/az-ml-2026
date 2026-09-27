#!/bin/bash
# Agent J (workstation): dispatch "band shard" scoring items to whichever GPU box is idle (H100, gpu2).
# Waits for the H100 run's result.json, ships the final adapter, has gpu2 fetch it, then dispatches items in
# priority order: B_REST/test core band, B_REST/test mid band, French tier 1, French tier 2.
set -u
KEY=/home/sidvish/.ssh/az-ml-new-ec2
SSHO="-F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15 -i $KEY"
H100="ssh $SSHO -p <redacted> root@<redacted>"
GPU2="ssh $SSHO ubuntu@<redacted>"
RUN=/dev/shm/J/run2
ITEMS=${ITEMS:-"core:brest_0 core:brest_1 core:test_3 core:test_0 core:test_1 core:test_2 mid:brest_0 mid:brest_1 mid:test_0 mid:test_1 mid:test_2 mid:test_3 fr:fr1_0 fr:fr1_1 fr:fr1_2 fr:fr1_3 fr:fr2_0 fr:fr2_1 fr:fr2_2 fr:fr2_3"}
log() { echo "[$(date +%H:%M:%S)] $*"; }

until timeout 25 $H100 "test -e $RUN/result.json" 2>/dev/null; do sleep 20; done
CK=$(timeout 25 $H100 "/opt/az-venv/bin/python -c \"import json; print(json.load(open('$RUN/result.json'))['final_checkpoint'])\"")
log "final checkpoint $CK"
if [ "${SKIP_SETUP:-0}" != 1 ]; then
  timeout 300 $H100 "cd /root/J && bash ship_ck.sh put_ck.txt final $CK $RUN" | tr '\n' ' '; echo
  timeout 300 $GPU2 "cd ~/J && bash fetch_ck.sh get_ck.txt final /home/ubuntu/J/run1 > logs/fetch_final.log 2>&1; ls /home/ubuntu/J/run1/checkpoints/step_final" | tr '\n' ' '; echo
fi
GCK=/home/ubuntu/J/run1/checkpoints/step_final

idle() {  # $1 = box name
  if [ "$1" = h100 ]; then timeout 25 $H100 'pgrep -f "qwen4b_trai[n] score" > /dev/null && echo busy || echo idle' 2>/dev/null
  else timeout 25 $GPU2 'pgrep -f "qwen4b_trai[n] score" > /dev/null && echo busy || echo idle' 2>/dev/null; fi
}
dispatch() {  # $1 box, $2 band, $3 shard
  local band=$2 s=$3
  if [ "$1" = h100 ]; then
    timeout 25 $H100 "cd /root/J && (MASK=$band PY=/opt/az-venv/bin/python TABLE=/root/J/put_$band.txt PYTHONPATH=/root/J/code nohup bash score_queue.sh $CK /dev/shm/J/models/Qwen3-Reranker-4B /dev/shm/J/score_inputs /dev/shm/J/scores_$band 131072 4096 '' $s > logs/q_${band}_$s.log 2>&1 < /dev/null &)"
  else
    timeout 25 $GPU2 "cd ~/J && (MASK=$band PY=/home/ubuntu/s-venv/bin/python TABLE=/home/ubuntu/J/put_$band.txt PYTHONPATH=/home/ubuntu/J/code nohup bash score_queue.sh $GCK /home/ubuntu/J/models/Qwen3-Reranker-4B /home/ubuntu/J/score_inputs /home/ubuntu/J/scores_$band 65536 2048 '' $s > logs/q_${band}_$s.log 2>&1 < /dev/null &)"
  fi
  log "dispatched $band:$s -> $1"
}
set -- $ITEMS
while [ $# -gt 0 ]; do
  [ -e /tmp/claude-1000/J_COORD_STOP ] && { log "stop file"; break; }
  for box in h100 gpu2; do
    [ $# -gt 0 ] || break
    if [ "$(idle $box)" = idle ]; then
      item=$1; shift
      dispatch $box ${item%%:*} ${item#*:}
      sleep 25  # let the new process appear before the next idle check
    fi
  done
  sleep 15
done
log "all items dispatched"
