#!/bin/bash
cd /opt/az-ml-new/repo/code/business_entity_resolution/src
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
for S in a b_rest test; do
  nice -n 10 /opt/az-venv/bin/python -u -m ber.union_mine mine --set $S --features --raw /workspace/az-ml-new/union/raw --workers 8 --keep 0 --output /workspace/az-ml-new/union/full_$S > /opt/az-ml-new/agents/union/full_$S.log 2>&1 || echo FAIL $S >> /opt/az-ml-new/agents/union/full_$S.log
done
