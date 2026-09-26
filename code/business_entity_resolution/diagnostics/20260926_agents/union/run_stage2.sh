#!/bin/bash
cd /opt/az-ml-new/repo/code/business_entity_resolution/src
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
U=/workspace/az-ml-new/union
L=/opt/az-ml-new/agents/union
nice -n 10 /opt/az-venv/bin/python -u -m ber.union_rank fit --mined $U/full_a --raw $U/raw --threads 4 --output $U/ranker > $L/ranker_fit.log 2>&1 &
nice -n 10 /opt/az-venv/bin/python -u -m ber.union_mine mine --set b_rest --features --raw $U/raw --workers 4 --keep 0 --output $U/full_b_rest > $L/full_b_rest.log 2>&1
nice -n 10 /opt/az-venv/bin/python -u -m ber.union_mine mine --set test --features --raw $U/raw --workers 4 --keep 0 --output $U/full_test > $L/full_test.log 2>&1
wait
echo STAGE2DONE
