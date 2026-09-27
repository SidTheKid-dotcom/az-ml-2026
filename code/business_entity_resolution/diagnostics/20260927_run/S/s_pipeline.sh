#!/bin/bash
# Agent S GPU pipeline: wait for lock=S -> train C2-FR -> score eval (C2-FR final + C2 40k baseline) -> gate 2
# (walk back through checkpoints if the B_REST drop exceeds 0.0005) -> score the French rescoring set with the
# chosen checkpoint -> write Q into the lock.
set -u
D=/opt/az-ml-new/repo/code/business_entity_resolution/diagnostics/20260927_run/S
S=/workspace/az-ml-new/run27/S; L=/workspace/az-ml-new/run27/gpu.lock
M=$S/arena_fr_v3; E=$S/eval; CK=$S/c2fr/ckpt; STEPS=${STEPS:-18000}
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src
PY=/opt/az-venv/bin/python
mkdir -p $E
echo "wait for lock S $(date -u)"
until [ "$(tr -d '[:space:]' < $L)" = "S" ]; do sleep 15; done
echo "lock S $(date -u)"
if [ ! -f $CK/step_$(printf %06d $STEPS)/complete.json ]; then
  STEPS=$STEPS bash $D/s_train.sh > $S/train.log 2>&1
  echo "train exit $? $(date -u)"
fi
FINAL=$CK/step_$(printf %06d $STEPS)
[ -f $FINAL/complete.json ] || { echo "no final checkpoint"; exit 1; }
bash $D/s_score_eval.sh /workspace/az-ml-new/c2_plain/ckpt/step_040000 base_c2_40k
CHOSEN=""
for ST in $STEPS $((STEPS-2000)) $((STEPS-4000)) $((STEPS-6000)) $((STEPS-8000)) $((STEPS-10000)); do
  C=$CK/step_$(printf %06d $ST); [ -f $C/complete.json ] || continue
  N=c2fr_step$(printf %06d $ST)
  bash $D/s_score_eval.sh $C $N
  $PY -m ber.frenchify_arena evaluate --output $M --scores $E/$N --baseline-scores $E/base_c2_40k --report $E/$N.json > $E/$N.eval.log 2>&1
  PASS=$($PY -c "import json;print(json.load(open('$E/$N.json'))['B_REST']['gate2_pass'])")
  echo "gate2 $N $PASS $(date -u)"
  if [ "$PASS" = "True" ]; then CHOSEN=$C; break; fi
done
[ -n "$CHOSEN" ] || { echo "no checkpoint passed gate 2; releasing GPU"; echo Q > $L; exit 2; }
echo "chosen $CHOSEN"; echo $CHOSEN > $S/chosen_checkpoint.txt
bash $D/s_score_france.sh $CHOSEN c2fr_$(basename $CHOSEN)_raw
echo "france scoring done $(date -u)"
echo Q > $L
echo "lock -> Q $(date -u)"
