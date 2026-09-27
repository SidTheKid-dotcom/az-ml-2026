#!/bin/bash
# Agent Y gpu3: upload deliverables to s3://.../run27/Y/ and write SHA256SUMS (uploaded alongside).
# Usage: y_upload.sh N   (N = laya_step_XXXXXX, the scored checkpoint name)
set -u
Y=/home/ubuntu/run27/Y; N=$1; B=s3://az-ml-new-20260926-533267274451/run27/Y
CK=$Y/laya_c2fr/ckpt/${N#laya_}
cd $Y
rm -rf deliver; mkdir -p deliver/model deliver/train deliver/code deliver/results deliver/fr_full_index
for f in config.json model.safetensors tokenizer.json tokenizer_config.json training_config.json complete.json state.json; do
  ln -s $CK/$f deliver/model/$f; done
ln -s $Y/models/laya-multilingual-seqcls.provenance.json deliver/model/initial_model_provenance.json
for f in results.json training_config.json latest_checkpoint.json; do [ -f laya_c2fr/out/$f ] && ln -s $Y/laya_c2fr/out/$f deliver/train/$f; done
ln -s $Y/logs/train.log deliver/train/train.log
for f in $Y/results/*; do [ -f "$f" ] && ln -s $f deliver/results/$(basename $f); done
[ -f parity/parity_diagnostics.json ] && ln -s $Y/parity/parity_diagnostics.json deliver/results/parity_check_c2_40k_diagnostics.json
for f in scores.npy scored_mask.npy complete.json; do ln -s $Y/fr/$N/$f deliver/fr_full_index/$f; done
for f in laya_convert.py laya_c2.py; do ln -s $Y/repo/src/ber/$f deliver/code/$f; done
for f in $Y/scripts/*; do ln -s $f deliver/code/$(basename $f); done
ln -s $Y/repo/problem-statement/models_laya.yaml deliver/code/models_laya.yaml
(cd deliver && find -L . -type f ! -name SHA256SUMS | sort | xargs sha256sum > SHA256SUMS)
[ -f $Y/README_head.md ] && /opt/pytorch/bin/python $Y/scripts/y_readme.py $N $Y/README_head.md $Y/deliver $Y/deliver/README.md
aws s3 cp --recursive --only-show-errors --follow-symlinks deliver/ $B/ && echo "upload ok $(date -u)"
