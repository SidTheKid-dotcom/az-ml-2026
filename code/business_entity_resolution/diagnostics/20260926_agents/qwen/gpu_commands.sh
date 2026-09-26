#!/usr/bin/env bash
# Qwen3-Reranker-0.6B lane (agent Q). Run from /opt/az-ml-new/agents/qwen on the H100 pod.
# Agent Q never starts GPU jobs; the controller launches them when the GPU is free.
#   ./gpu_commands.sh train                 # GPU, ~70 min wall clock (auto-sized steps)
#   ./gpu_commands.sh resume                # GPU, after a crash: continues from the newest checkpoint
#   CK=... ./gpu_commands.sh score-brest    # GPU, 2,576,129 gated B_REST pairs (resumable)
#   CK=... ./gpu_commands.sh score-test     # GPU, 10,499,994 gated test pairs (resumable)
#   QB=... ./gpu_commands.sh ensemble       # CPU, B_REST cross-fit report
#   QB=... QT=... ./gpu_commands.sh test-files   # CPU, validated test files
# Optional C2 filter (must be identical for B_REST and test): MIN_C2=0.01 scores only gated
# pairs with C2(step 40000) >= 0.01 (B_REST keeps ~72%, test ~57%); unscored rows get Qwen = -1.
set -euo pipefail
cd /opt/az-ml-new/agents/qwen
. ./env.sh            # PYTHONPATH, HF_HUB_OFFLINE=1, PY, MODEL (=/workspace/az-ml-new/models/Qwen3-Reranker-0.6B)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 RAYON_NUM_THREADS=4
Q=/workspace/az-ml-new/qwen
C2=/opt/az-ml-new/runs/c2_step040000
TRAIN="$PY -u -m ber.qwen_train train --data $Q/train_data --model $MODEL --output $Q/run1 \
  --time-budget-min 70 --microbatches 2 --checkpoint-every 2000 --log-every 50 --device cuda --threads 4"
filter() { if [ -n "${MIN_C2:-}" ]; then echo "--min-c2 $MIN_C2 --c2-scores $1"; fi; }

case "${1:-}" in
train)  exec $TRAIN ;;
resume) exec $TRAIN --resume ;;
score-brest)
  exec $PY -u -m ber.qwen_train score --data $Q/score_brest --checkpoint "${CK:?set CK=$Q/run1/checkpoints/step_XXXXXX}" \
       --output $Q/brest_$(basename "$CK")${MIN_C2:+_c2min$MIN_C2} --batch-size 1024 --token-budget 131072 \
       --device cuda --threads 4 $(filter $C2/b_rest_scores) ;;
score-test)
  exec $PY -u -m ber.qwen_train score --data $Q/score_test --checkpoint "${CK:?set CK}" \
       --output $Q/test_$(basename "$CK")${MIN_C2:+_c2min$MIN_C2} --batch-size 1024 --token-budget 131072 \
       --device cuda --threads 4 $(filter $C2/test_scores) ;;
ensemble)   # reuses agent P's C2-only cross-fit if it was built from the same C2 scores
  exec $PY -u -m ber.qwen_ensemble evaluate --scores $C2/b_rest_scores --qwen-scores "${QB:?set QB=$Q/brest_step_XXXXXX}" \
       --baseline-work /opt/az-ml-new/agents/pair_tree_c2/step040000 --work $Q/ens_$(basename "$QB") \
       --threads ${THREADS:-8} --grid-workers ${GRID_WORKERS:-6} --bootstrap 1000 ;;
test-files)
  exec $PY -u -m ber.qwen_ensemble test --scores $C2/b_rest_scores --qwen-scores "${QB:?set QB}" --work $Q/ens_$(basename "$QB") \
       --test-scores $C2/test_scores --qwen-test-scores "${QT:?set QT=$Q/test_step_XXXXXX}" --out $Q/files_$(basename "$QT") \
       --compare-file /opt/az-ml-new/runs/s1_pair_tree_40k/matching_results.tsv \
       ${FRANCE_FILE:+--france-file $FRANCE_FILE} --threads ${THREADS:-8} ;;
*) echo "usage: $0 train|resume|score-brest|score-test|ensemble|test-files" >&2; exit 2 ;;
esac
