#!/bin/bash
# Sequential CPU queue for agent pair_tree_c2 (keeps <= 24 threads): dry-run test, then step-20000 evaluate.
cd /opt/az-ml-new/agents/pair_tree_c2
export PYTHONPATH=/opt/az-ml-new/repo/code/business_entity_resolution/src PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=20 OPENBLAS_NUM_THREADS=20 MKL_NUM_THREADS=20
PY=/opt/az-venv/bin/python
while pgrep -f "prep_test.py" >/dev/null || pgrep -f "step010000_ext --cache" >/dev/null; do sleep 15; done
echo "$(date -u) dryrun start"
$PY -u -m ber.pair_tree_c2 test --scores /opt/az-ml-new/runs/c2_step005000/b_rest_scores \
  --work step005000 --cache step005000 --test-scores dryrun_fake_test_scores \
  --france-scores dryrun_fake_test_scores --out dryrun_output_NOT_FOR_SUBMISSION --threads 20 > dryrun_test.log 2>&1
echo "$(date -u) dryrun exit $?"
while [ ! -f /opt/az-ml-new/runs/c2_step020000/b_rest_scores/complete.json ]; do sleep 30; done
echo "$(date -u) step20000 evaluate start"
$PY -u -m ber.pair_tree_c2 evaluate --scores /opt/az-ml-new/runs/c2_step020000/b_rest_scores \
  --work step020000 --cache step005000 --threads 20 > evaluate_step020000.log 2>&1
echo "$(date -u) step20000 evaluate exit $?"
