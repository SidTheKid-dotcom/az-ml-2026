#!/bin/bash
# Agent J (scoring boxes): fetch adapter checkpoint TAG into RUNROOT/checkpoints/step_TAG (+ RUNROOT/train_config.json).
# Usage: fetch_ck.sh TABLE TAG RUNROOT
TABLE=$1; TAG=$2; RUN=$3
CK=$RUN/checkpoints/step_$TAG
mkdir -p $CK
grep "^$TAG " $TABLE | while read -r tag name url; do
  code=$(curl -sS --retry 3 -o $CK/$name -w '%{http_code}' "$url")
  [ "$code" = "200" ] || rm -f $CK/$name
  echo "fetch $TAG/$name $code"
done
cp $CK/train_config.json $RUN/train_config.json
ls -la $CK
