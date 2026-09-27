#!/bin/bash
# Agent F follow-up: rebuild French rows for the three C2-10k variants (after g4 scoring completes), in parallel.
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /opt/az-ml-new/repo/code/business_entity_resolution/diagnostics/20260927_run/F
F=/workspace/az-ml-new/run27/F
for m in 10k mean strict10k; do
  F_MODE=$m F_TAG=v10k_$m/ /opt/az-venv/bin/python rebuild.py > $F/logs/rebuild_v10k_$m.log 2>&1 &
done
wait
echo "variants done $(date -u)"
