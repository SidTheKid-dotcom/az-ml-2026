#!/bin/bash
# Agent S: after the C2 v2 chain, score B_REST route/reverse pairs with the continuation model and compare (labels: B_REST).
set -u
PY=/opt/az-venv/bin/python; SRC=/root/s_stage/repo/src; D=/opt/az-ml-new/repo/code/business_entity_resolution/diagnostics/20260927_run/S
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8; C=/dev/shm/c2v2; S=$C/score
until grep -q "ALL DONE" /root/c2v2/after.log; do sleep 30; done
for SET in brest_routes brest_reverse; do
  PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $C/cont/ckpt/step_006000 --reference $C/in/arena/reference.parquet --pool $S/brest_ext/pool.parquet \
    --left $S/$SET/left.npy --right $S/$SET/right.npy --tree $S/$SET/tree.npy --gate 0.01 --output $S/cont_$SET \
    --inference-batch 1024 --threads 8 --encode-workers 6 > $S/cont_$SET.log 2>&1; echo "cont $SET $? $(date -u)"; done
PYTHONPATH=$SRC $PY $D/c2v2_pair_check.py /root/c2v2/pair_check.json > /root/c2v2/pair_check.log 2>&1; echo "check $? $(date -u)"
