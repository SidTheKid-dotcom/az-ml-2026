#!/bin/bash
# Agent Y gpu3 (lead 12:40 IST: French deliverables on S3 by 14:20 IST, French first, US/India second).
# Waits for checkpoint CK (default step_010000) to be complete, stops the trainer there (own process), then:
# (1) French rescoring set (test index, French S1s, v1 tree >= 0.01) -> keyed export, French rows at 0.10 and 0.70
#     with label-free diagnostics, family agreement -> upload French pieces to S3 immediately;
# (2) arena_fr_v3 eval (B_REST monitor + Frenchified TUNE) -> frenchify_arena evaluate vs stored C2 40k (report only);
# (3) Agent U's u8r_v2 US/India pairs with 0.02 <= p <= 0.98 (B_REST + test).
set -u
Y=/home/ubuntu/run27/Y; S=/home/ubuntu/run27/S; A=$S/arena_fr_v3; PY=/opt/pytorch/bin/python
D=$Y/repo/diagnostics/20260927_run; B=s3://az-ml-new-20260926-533267274451/run27/Y
export PYTHONPATH=$Y/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
CK=${CK:-$Y/laya_c2fr/ckpt/step_010000}; N=laya_$(basename $CK)
until [ -f $CK/complete.json ] || ! kill -0 $(cat $Y/train.pid) 2>/dev/null; do sleep 10; done
[ -f $CK/complete.json ] || { echo "no complete checkpoint $CK"; exit 1; }
sleep 5
if kill -0 $(cat $Y/train.pid) 2>/dev/null && [ -z "${KEEP_TRAINING:-}" ]; then
  kill $(cat $Y/train.pid); echo "stopped trainer at $(basename $CK) $(date -u)"
  while kill -0 $(cat $Y/train.pid) 2>/dev/null; do sleep 2; done
fi
echo "checkpoint $CK $(date -u)"
mkdir -p $Y/eval $Y/fr $Y/us $Y/results
[ -f $Y/france_s1.npy ] || $PY $D/S/s_france_mask.py $Y/in/test $Y/france_s1.npy
SC="$PY -u -m ber.laya_c2 score --checkpoint $CK --inference-batch ${IB:-512} --threads 6 --encode-workers 4"
[ -f $Y/fr/$N/complete.json ] || $SC --reference $Y/in/test/dense_test/reference.parquet \
  --pool $Y/in/test/dense_test/pool.parquet --left $Y/in/test/index/left.npy --right $Y/in/test/index/right.npy \
  --tree $Y/in/test/base_probabilities.npy --gate 0.01 --population $Y/france_s1.npy --output $Y/fr/$N > $Y/fr/$N.log 2>&1
echo "france scored $? $(date -u)"
( [ -f $Y/eval/$N/complete.json ] || $SC --reference $A/arena/reference.parquet --pool $A/arena/pool.parquet \
    --pairs $A/arena/pairs.parquet --tree $A/arena/v1_tree.npy --gate 0.01 --population $A/eval_s1.npy \
    --output $Y/eval/$N > $Y/eval/$N.log 2>&1
  echo "eval scored $? $(date -u)"
  $PY -m ber.frenchify_arena evaluate --output $A --scores $Y/eval/$N --baseline-scores $Y/eval/base_c2_40k_mapped \
    --report $Y/results/eval_$N.json > $Y/eval/$N.eval.log 2>&1; echo "evaluate $? $(date -u)"
  if [ -z "${SKIP_US:-}" ]; then
    for SP in brest test; do
      R=$( [ $SP = brest ] && echo $S/in/arena/v1 || echo $Y/in/test/dense_test )
      [ -f $Y/us/${SP}_$N/complete.json ] || $SC --reference $R/reference.parquet --pool $R/pool.parquet \
        --left $Y/us/$SP/left.npy --right $Y/us/$SP/right.npy --tree $Y/us/$SP/tree.npy --gate 0.5 \
        --output $Y/us/${SP}_$N > $Y/us/${SP}_$N.log 2>&1
      echo "us $SP scored $? $(date -u)"
      $PY $Y/scripts/y_upairs.py export $SP $Y/us/${SP}_$N $Y/results/laya_us_india_uncertain_${SP}.parquet; echo "us export $SP $? $(date -u)"
    done
  fi ) > $Y/logs/after2_gpu_tail.log 2>&1 &
$PY $Y/scripts/y_export.py $Y/fr/$N $Y/results/laya_french_pair_scores.parquet; echo "export $? $(date -u)"
cd $Y && S_OUT=$Y/results S_G1=$Y/g1 OMP_NUM_THREADS=4 $PY -u $Y/scripts/y_rebuild.py $Y/fr/$N laya_t010_ > $Y/logs/rebuild_t010.log 2>&1
echo "rebuild 0.10 $? $(date -u)"
cd $Y && S_THRESHOLD=0.7 S_OUT=$Y/results S_G1=$Y/g1 OMP_NUM_THREADS=4 $PY -u $Y/scripts/y_rebuild.py $Y/fr/$N laya_t070_ > $Y/logs/rebuild_t070.log 2>&1
echo "rebuild 0.70 $? $(date -u)"
$PY $Y/scripts/y_agree.py $Y/results/agreement.json laya_t010=$Y/results/laya_t010_french_rows.tsv \
  laya_t070=$Y/results/laya_t070_french_rows.tsv > $Y/logs/agree.log 2>&1; echo "agree $? $(date -u)"
mkdir -p $Y/deliver_fr && cd $Y/deliver_fr && rm -f ./* && \
  for f in laya_french_pair_scores.parquet laya_t010_french_rows.tsv laya_t010_diagnostics.json laya_t070_french_rows.tsv \
           laya_t070_diagnostics.json agreement.json; do cp $Y/results/$f .; done && \
  sha256sum * > SHA256SUMS_french && aws s3 cp --recursive --only-show-errors . $B/french/ && echo "french upload ok $(date -u)"
wait
echo "after2 done $(date -u)"
