#!/bin/bash
# Agent J (H100): upload one adapter checkpoint under TAG. Usage: ship_ck.sh TABLE TAG CKDIR RUNDIR
TABLE=$1; TAG=$2; CK=$3; RUN=$4
cp $RUN/train_config.json $CK/train_config.json 2>/dev/null
[ -e $RUN/result.json ] && cp $RUN/result.json $CK/result.json
LOGF=${LOGF:-/root/J/logs/train_run2.log}; [ -e $LOGF ] && cp $LOGF $CK/train_log.txt
grep "^$TAG " $TABLE | while read -r tag name url; do
  [ -e "$CK/$name" ] || continue
  code=$(curl -sS --retry 3 -o /dev/null -w '%{http_code}' -X PUT -T "$CK/$name" "$url")
  echo "ship $TAG/$name $code"
done
