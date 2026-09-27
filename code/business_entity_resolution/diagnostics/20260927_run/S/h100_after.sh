#!/bin/bash
# Agent S, new pod (never touches /workspace): after the H100 C2-FR(base) run -> eval scoring (B_REST monitor + French
# tuning world) -> gate 2 vs stored C2 40k (walk back through checkpoints if needed) -> French rescoring set scoring
# with the chosen checkpoint -> French rows rebuild + diagnostics.
set -u
A=/root/s_stage/arena_fr_v3; CK=/dev/shm/c2fr_h100/ckpt; S=/root/S; PY=/opt/az-venv/bin/python
SRC=/root/s_stage/repo/src; OPT=/opt/az-ml-new/repo/code/business_entity_resolution
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
mkdir -p $S/eval $S/fr
while kill -0 $(cat /root/c2fr_h100/train.pid) 2>/dev/null; do sleep 20; done
echo "train done $(date -u)"
LAST=$(ls -d $CK/step_* | sort | tail -1 | sed 's/.*step_0*//')
CHOSEN=""
for ST in $LAST $((LAST-2000)) $((LAST-4000)) $((LAST-6000)); do
  C=$CK/step_$(printf %06d $ST); [ -f $C/complete.json ] || continue
  N=c2fr_h100_step$(printf %06d $ST)
  PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $C --reference $A/arena/reference.parquet --pool $A/arena/pool.parquet \
    --pairs $A/arena/pairs.parquet --tree $A/arena/v1_tree.npy --gate 0.01 --population $A/eval_s1.npy \
    --output $S/eval/$N --inference-batch 1024 --threads 8 --encode-workers 6 > $S/eval/$N.log 2>&1
  PYTHONPATH=$SRC $PY -m ber.frenchify_arena evaluate --output $A --scores $S/eval/$N --baseline-scores $S/eval/base_c2_40k_mapped \
    --report $S/eval/$N.json > $S/eval/$N.eval.log 2>&1
  PASS=$($PY -c "import json;print(json.load(open('$S/eval/$N.json'))['B_REST']['gate2_pass'])")
  echo "gate2 $N $PASS $(date -u)"
  if [ "$PASS" = "True" ]; then CHOSEN=$C; break; fi
done
[ -n "$CHOSEN" ] || { CHOSEN=$CK/step_$(printf %06d $LAST); echo "NO checkpoint passed gate 2; scoring the final one for diagnostics only"; }
N=c2fr_h100_$(basename $CHOSEN); echo $CHOSEN > $S/chosen_checkpoint.txt
[ -f $S/france_s1.npy ] || $PY $OPT/diagnostics/20260927_run/S/s_france_mask.py /opt/az-ml-new/test_inputs $S/france_s1.npy
T=/opt/az-ml-new/test_inputs
PYTHONPATH=$SRC $PY -u -m ber.score_external --checkpoint $CHOSEN --reference $T/dense_test/reference.parquet --pool $T/dense_test/pool.parquet \
  --left $T/index/left.npy --right $T/index/right.npy --tree $T/base_probabilities.npy --gate 0.01 --population $S/france_s1.npy \
  --output $S/fr/$N --inference-batch 1024 --threads 8 --encode-workers 6 > $S/fr/$N.log 2>&1
echo "france scored $? $(date -u)"
cd $S && S_OUT=$S PYTHONPATH=$OPT/src OMP_NUM_THREADS=4 $PY -u $OPT/diagnostics/20260927_run/S/s_rebuild.py $S/fr/$N h100_ > $S/rebuild_h100.log 2>&1
echo "rebuild $? $(date -u)"
