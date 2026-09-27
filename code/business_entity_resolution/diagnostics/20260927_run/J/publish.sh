#!/bin/bash
# Agent J (EC2): upload assembled deliverables of deliver_NAME to s3 run27/J/ (+ eval/adv JSONs).
# Usage: publish.sh NAME FILE...   (FILE in deliver_NAME, e.g. llm_brest.parquet llm_test.parquet)
set -u
NAME=$1; shift
cd ~/run27/J
B=s3://az-ml-new-20260926-533267274451/run27/J
for f in "$@"; do
  aws s3 cp deliver_$NAME/$f $B/$f --only-show-errors && echo "uploaded $f $(sha256sum deliver_$NAME/$f | cut -c1-64)"
done
aws s3 cp deliver_$NAME/assemble_report.json $B/assemble_report_$NAME.json --only-show-errors
for j in eval/${NAME}_eval.json eval/${NAME}_adv.json; do [ -e $j ] && aws s3 cp $j $B/$(basename $j) --only-show-errors && echo "uploaded $j"; done
echo "PUBLISH DONE $(date)"
