#!/usr/bin/env bash
set -euo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while [[ ! -f artifacts/full_dense_medium/complete.json ]]; do sleep 20; done
/opt/amlc-venv/bin/python -u -m ber.tabular_sweep \
  --data artifacts/full_dense_medium --pairs artifacts/full_dense_medium/pairs.parquet \
  --output artifacts/medium_dense_sweep --device GPU --iterations 650 --threads 2 \
  --depths 6 8 10 --negative-weights 1 2 --gpu-ram-part .22
/opt/amlc-venv/bin/python -u -m ber.residual_experiment \
  --data artifacts/full_dense_medium --base-dir artifacts/medium_dense_sweep \
  --output artifacts/medium_dense_residual_sweep
