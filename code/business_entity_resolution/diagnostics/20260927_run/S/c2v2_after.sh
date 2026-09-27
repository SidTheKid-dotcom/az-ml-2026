#!/bin/bash
# Agent S: C2 v2 after the dense 40k stage -> continuation on arena + R reverse pairs (6k steps, lr 1e-5) ->
# monitor comparison (dense-40k vs continuation vs stored C2 40k) -> score B_REST/test dense candidates and U routes +
# R reverse pairs with the chosen model.
set -u
PY=/opt/az-venv/bin/python; SRC=/root/s_stage/repo/src; D=/opt/az-ml-new/repo/code/business_entity_resolution/diagnostics/20260927_run/S
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
C=/dev/shm/c2v2; W=/root/c2v2; T=/opt/az-ml-new/test_inputs
while kill -0 42634 2>/dev/null; do sleep 20; done; echo "dense done $(date -u)"
DENSE=$C/ckpt/step_040000; [ -f $DENSE/complete.json ] || { echo "no dense final"; exit 1; }
PYTHONPATH=$SRC $PY -u -m ber.fast_trainer --command train --mode c2 --text-view plain --steps 6000 --batch-size 128 --microbatch 128 \
  --fused --no-freeze-embeddings --amp bf16 --length-buckets 48,64,80,96,128,160,256 --save-every 2000 --defer-checkpoint-hash --compile \
  --prefetch 8 --learning-rate 1e-5 --schedule cosine --weight-decay 0.01 --warmup-fraction 0.05 --seed 20260925 \
  --arena $C/aug/arena --labels $C/aug/labels --partition $C/aug/s1_partition.npy --sealed-reservation /root/s_stage/v2_holdout2 \
  --model $DENSE --allowlist /root/s_stage/repo/problem-statement/models.yaml --output $C/cont/out --checkpoint-root $C/cont/ckpt \
  --device cuda --threads 8 > $W/cont.log 2>&1; echo "cont exit $? $(date -u)"
CONT=$C/cont/ckpt/step_006000
$PY $D/c2v2_monitor_mask.py $W/monitor_s1.npy $W/brest_s1.npy
for M in dense:$DENSE cont:$CONT; do N=${M%%:*}; K=${M#*:}; [ -f $K/complete.json ] || continue
  PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $K --reference $C/in/arena/reference.parquet --pool $C/in/arena/pool.parquet \
    --pairs $C/in/arena/pairs.parquet --tree $C/in/arena/base_probabilities.npy --gate 0.01 --population $W/monitor_s1.npy \
    --output $W/mon_$N --inference-batch 1024 --threads 8 --encode-workers 6 > $W/mon_$N.log 2>&1
  PYTHONPATH=$SRC $PY $D/c2v2_eval.py $W/mon_$N $W/monitor_s1.npy $W/mon_$N.json
done
CH=$($PY -c "
import json,os
best=('dense',-9)
for n in ('dense','cont'):
    p='$W/mon_%s.json'%n
    if os.path.exists(p):
        v=json.load(open(p))['candidate']['macro_f05']
        if v>best[1]+1e-9: best=(n,v)
print(best[0])")
K=$DENSE; [ "$CH" = "cont" ] && K=$CONT; echo "chosen $CH $K $(date -u)"; echo $K > $W/chosen.txt
S=$C/score
PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $K --reference $C/in/arena/reference.parquet --pool $C/in/arena/pool.parquet \
  --pairs $C/in/arena/pairs.parquet --tree $C/in/arena/base_probabilities.npy --gate 0.01 --population $W/brest_s1.npy \
  --output $S/out_brest_dense --inference-batch 1024 --threads 8 --encode-workers 6 > $S/out_brest_dense.log 2>&1; echo "brest dense $? $(date -u)"
PYTHONPATH=$SRC $PY $D/c2v2_eval.py $S/out_brest_dense $W/brest_s1.npy $W/brest_full_eval.json
for SET in brest_routes brest_reverse; do
  PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $K --reference $C/in/arena/reference.parquet --pool $S/brest_ext/pool.parquet \
    --left $S/$SET/left.npy --right $S/$SET/right.npy --tree $S/$SET/tree.npy --gate 0.01 --output $S/out_$SET \
    --inference-batch 1024 --threads 8 --encode-workers 6 > $S/out_$SET.log 2>&1; echo "$SET $? $(date -u)"; done
for SET in test_routes test_reverse; do
  PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $K --reference $T/dense_test/reference.parquet --pool $S/test_ext/pool.parquet \
    --left $S/$SET/left.npy --right $S/$SET/right.npy --tree $S/$SET/tree.npy --gate 0.01 --output $S/out_$SET \
    --inference-batch 1024 --threads 8 --encode-workers 6 > $S/out_$SET.log 2>&1; echo "$SET $? $(date -u)"; done
PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $K --reference $T/dense_test/reference.parquet --pool $T/dense_test/pool.parquet \
  --left $T/index/left.npy --right $T/index/right.npy --tree $T/base_probabilities.npy --gate 0.01 \
  --output $S/out_test_dense --inference-batch 1024 --threads 8 --encode-workers 6 > $S/out_test_dense.log 2>&1; echo "test dense $? $(date -u)"
echo "ALL DONE $(date -u)"
