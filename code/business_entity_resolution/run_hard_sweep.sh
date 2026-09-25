#!/usr/bin/env bash
set -euo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while [[ ! -f data/hard/metadata.json ]]; do
  sleep 20
done
exec /opt/amlc-venv/bin/python -u -m ber.tabular_sweep \
  --data data/hard --pairs data/hard/candidate_pairs.parquet \
  --output artifacts/hard_tabular_sweep --device GPU \
  --iterations 650 --threads 2 --depths 6 8 --negative-weights 1 2 4 \
  --gpu-ram-part .22 --baseline-checkpoints artifacts/quick_v1/catboost_deep.joblib
