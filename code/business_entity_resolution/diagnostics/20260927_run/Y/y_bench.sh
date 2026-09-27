#!/bin/bash
# Agent Y: isolated inference-throughput probe for the French scoring plan (trainer paused with SIGSTOP, resumed after).
# Scores ~5,000 French S1s' gated pairs with the step_002000 checkpoint into Y/bench (throwaway).
set -u
Y=/home/ubuntu/run27/Y; PY=/opt/pytorch/bin/python
export PYTHONPATH=$Y/repo/src HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1
$PY -c "import numpy as np; f=np.load('$Y/france_s1.npy'); m=np.zeros(len(f),bool); m[np.flatnonzero(f)[:5000]]=True; np.save('$Y/bench_pop.npy', m)"
rm -rf $Y/bench
kill -STOP $(cat $Y/train.pid); echo "trainer paused $(date -u +%T)"
$PY -u -m ber.laya_c2 score --checkpoint $Y/laya_c2fr/ckpt/step_002000 --inference-batch 512 --threads 6 --encode-workers 4 \
  --reference $Y/in/test/dense_test/reference.parquet --pool $Y/in/test/dense_test/pool.parquet \
  --left $Y/in/test/index/left.npy --right $Y/in/test/index/right.npy --tree $Y/in/test/base_probabilities.npy --gate 0.01 \
  --population $Y/bench_pop.npy --output $Y/bench --score-chunk 30000 2>&1 | grep -E '"done"|seconds|Error|Traceback'
kill -CONT $(cat $Y/train.pid); echo "trainer resumed $(date -u +%T)"
