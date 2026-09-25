#!/usr/bin/env bash
set -euo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
while [[ ! -f artifacts/full_dense_medium/complete.json || ! -f data/medium_hard/metadata.json ]]; do sleep 20; done
/opt/amlc-venv/bin/python -u -m ber.merge_candidates \
  --lexical-data data/medium_hard --dense-data artifacts/full_dense_medium \
  --embeddings artifacts/full_train_e5_ft200/pool_embeddings.npy \
  --output data/medium_union --durable-output /workspace/az-ml-new/artifacts/medium_union
/opt/amlc-venv/bin/python -u -m ber.tabular_sweep \
  --data data/medium_union --pairs data/medium_union/candidate_pairs.parquet \
  --output artifacts/medium_union_sweep --device GPU --iterations 650 --threads 2 \
  --depths 6 8 10 --negative-weights 1 2 --gpu-ram-part .22
/opt/amlc-venv/bin/python -u -m ber.residual_experiment \
  --data data/medium_union --base-dir artifacts/medium_union_sweep \
  --output artifacts/medium_union_residual_sweep
