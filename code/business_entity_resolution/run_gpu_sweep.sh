#!/usr/bin/env bash
set -uo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
python_bin=/opt/amlc-venv/bin/python
base_pairs=artifacts/cache/7fa3a032c1070bc6/pairs.parquet
mkdir -p logs
run_experiment() {
    local label="$1"
    local model="$2"
    local steps="$3"
    printf '%s Starting %s\n' "$(date -Is)" "$label"
    "$python_bin" -u -m ber.hf_experiment \
        --data data/quick --base-pairs "$base_pairs" \
        --model "$model" --model-id "$model" --model-label "$label" \
        --allowlist problem-statement/models.yaml \
        --output "artifacts/$label" --batch-size 256 --max-length 128 \
        --device cuda --top-k 30 --iterations 400 --depth 6 --threads 6 \
        --finetune-steps "$steps" > "logs/$label.log" 2>&1
    local status=$?
    printf '%s Finished %s exit=%s\n' "$(date -Is)" "$label" "$status"
    mkdir -p "/workspace/az-ml-new/artifacts/$label"
    cp "logs/$label.log" "/workspace/az-ml-new/artifacts/$label/experiment.log"
    for filename in results.json retrieval.json thresholds.json model.joblib embedding_manifest.json; do
        if [ -f "artifacts/$label/$filename" ]; then
            cp "artifacts/$label/$filename" "/workspace/az-ml-new/artifacts/$label/$filename"
        fi
    done
    return "$status"
}
run_experiment e5_small intfloat/multilingual-e5-small 0
run_experiment bge_m3 BAAI/bge-m3 0
run_experiment e5_small_ft200 intfloat/multilingual-e5-small 200
printf '%s Sweep completed\n' "$(date -Is)"
