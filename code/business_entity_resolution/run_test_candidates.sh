#!/usr/bin/env bash
# Label-free preparation only. A successful run does not constitute a submission.
set -euo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=/opt/amlc-venv/bin/python
"$python_bin" -u -m ber.full_embeddings \
  --model artifacts/e5_small_ft200/finetuned_model \
  --inputs data/raw/dataset/test/test_source2.tsv data/raw/dataset/test/test_source3.tsv \
  --expected-rows 4887273 5082316 \
  --output artifacts/full_test_e5_ft200 \
  --durable-output /workspace/az-ml-new/artifacts/full_test_e5_ft200 \
  --batch-size 512 --chunk-size 50000 --max-length 128 --threads 2
"$python_bin" -u -m ber.full_dense_candidates \
  --query-input data/raw/dataset/test/test_source1.tsv --expected-query-rows 1732544 \
  --embeddings artifacts/full_test_e5_ft200/pool_embeddings.npy \
  --model artifacts/e5_small_ft200/finetuned_model \
  --output artifacts/full_dense_test \
  --durable-output /workspace/az-ml-new/artifacts/full_dense_test \
  --top-k 50 --batch-size 512 --query-batch-size 512 --pool-batch-size 50000 \
  --threads 2 --device cuda --backend torch
