#!/bin/bash
# Agent J: score shards in order with one checkpoint, uploading each finished shard when TABLE is set.
# Usage: [PY=..] [TABLE=put_table] [MASK=core|mid] score_queue.sh CKPT MODEL INROOT OUTROOT TOKENS BATCH MERGEFLAG SHARD...
set -u
CK=$1; M=$2; IN=$3; OUT=$4; TOK=$5; BS=$6; MF=$7; shift 7
PY=${PY:-/opt/az-venv/bin/python}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p $OUT
for s in "$@"; do
  [ -e "$OUT/STOP" ] && { echo "stop requested"; break; }
  KM=""
  [ -n "${MASK:-}" ] && [ -e "$IN/$s/mask_${MASK}.npy" ] && KM="--keep-mask $IN/$s/mask_${MASK}.npy"
  if [ ! -e "$OUT/$s/complete.json" ]; then
    echo "start $s $KM $(date)"
    $PY -u -m ber.qwen4b_train score --data $IN/$s --checkpoint $CK --model $M --output $OUT/$s \
      --batch-size $BS --token-budget $TOK --flush-every 50 $MF $KM > $OUT/log_$s.txt 2>&1
    echo "done $s exit $? $(date)"
  fi
  if [ -n "${TABLE:-}" ] && [ -e "$OUT/$s/complete.json" ] && [ ! -e "$OUT/$s/.uploaded" ]; then
    bash $HERE/upload_shard.sh $TABLE $OUT $s && touch $OUT/$s/.uploaded
  fi
done
echo "QUEUE DONE $(date)"
