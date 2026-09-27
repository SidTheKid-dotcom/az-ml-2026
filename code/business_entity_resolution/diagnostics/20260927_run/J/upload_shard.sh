#!/bin/bash
# Agent J: upload one scored shard (logits + complete.json) with presigned PUT URLs. Usage: upload_shard.sh TABLE OUTROOT SHARD
TABLE=$1; OUT=$2; S=$3
grep "^$S " $TABLE | while read -r shard name url; do
  code=$(curl -sS --retry 3 -o /dev/null -w '%{http_code}' -X PUT -T "$OUT/$S/$name" "$url")
  echo "upload $S/$name $code"
done
