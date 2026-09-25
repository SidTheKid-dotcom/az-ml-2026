#!/usr/bin/env bash
# One fixed evaluation, after model selection. Never tune from its results.
set -euo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=/opt/amlc-venv/bin/python
pipeline=/workspace/az-ml-new/models/submission_v1_pipeline.json
tree=/workspace/az-ml-new/models/medium_dense_bge_ce1800/original_gate_and_tree.joblib
tree_sha=b5c7ca0876adbd15ee9cb193098e437d5970025f1afb7ba5eec50bec534063cb
expected_pipeline_sha=${1:?Pass the selected pipeline SHA256 explicitly}

# Record the whole procedure before retrieval's first possible truth access.
"$python_bin" - "$pipeline" "$expected_pipeline_sha" <<'PY'
import hashlib, json, shutil, sys
from pathlib import Path
import pandas as pd
from ber.full_embeddings import atomic_json, file_digest
source, expected = Path(sys.argv[1]), sys.argv[2]
config = json.loads(source.read_text())
assert file_digest(source) == expected, 'Pipeline differs from the selected digest'
assert config.get('inference_batch_size_provisional') is False, 'Finalize runtime first'
for name, digest in config['source_sha256'].items():
    assert file_digest(Path(name)) == digest, f'Frozen source changed: {name}'
split = pd.read_parquet('data/medium/reference.parquet', columns=['entity_id', 'split'])
ids = sorted(split.loc[split.split.eq('eval'), 'entity_id'])
assert len(ids) == len(set(ids)) == 12001
assert hashlib.sha256('\n'.join(ids).encode()).hexdigest() == 'a777ea1b07cfa375119233991ce3f36e16e01ab2c72d2493fcbc82208daf14b9', 'Holdout split changed'
out = Path('artifacts/final_holdout_dense_v1')
out.mkdir(parents=True, exist_ok=True)
frozen = out / 'selected_pipeline.json'
if frozen.exists():
    assert file_digest(frozen) == expected, 'Existing holdout identifies another procedure'
else:
    shutil.copyfile(source, frozen)
    atomic_json(out / 'selected_pipeline_identity.json', {
        'pipeline_sha256': expected, 'labels_accessed': False,
        'purpose': 'One fixed evaluation of the selected first-submission procedure'})
PY
pipeline=artifacts/final_holdout_dense_v1/selected_pipeline.json

if [[ ! -f artifacts/final_holdout_dense_v1/complete.json ]]; then
  "$python_bin" -u -m ber.full_dense_candidates \
    --data data/medium --splits eval --final-evaluation \
    --frozen-model-bundle "$tree" --frozen-model-sha256 "$tree_sha" \
    --embeddings artifacts/full_train_e5_ft200/pool_embeddings.npy \
    --model artifacts/e5_small_ft200/finetuned_model \
    --output artifacts/final_holdout_dense_v1 \
    --durable-output /workspace/az-ml-new/artifacts/final_holdout_dense_v1 \
    --top-k 50 --batch-size 512 --query-batch-size 512 --pool-batch-size 50000 \
    --threads 2 --device cuda --backend torch
fi
if [[ ! -f output/holdout_v1/complete.json ]]; then
  "$python_bin" -u -m ber.predict_streaming \
    --data artifacts/final_holdout_dense_v1 \
    --pairs artifacts/final_holdout_dense_v1/pairs.parquet \
    --checkpoint "$tree" --pipeline "$pipeline" \
    --output output/holdout_v1 --workers 4 --reference-batch-size 1000 \
    --expected-references 12001 --ce-threads 2
fi

"$python_bin" - "$expected_pipeline_sha" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
from ber.full_embeddings import atomic_json, file_digest
from ber.rerank_frozen import score_fixed_outputs
out = Path('output/holdout_v1')
complete = json.loads((out / 'complete.json').read_text())
report = complete['report']
assert complete['state'] == 'complete' and report['reference_count'] == 12001
assert report['pipeline_sha256'] == sys.argv[1] and not report['benchmark_only']
assert report['scorer'] == 'frozen_tree_ce'
fixed = {'pipeline_sha256': sys.argv[1], 'labels_accessed': False,
         'output_sha256': {name: file_digest(out / name) for name in
                          ('matching_results.tsv', 'candidate_pairs.tsv', 'probabilities.npy')}}
marker = out / 'predictions_fixed.json'
if marker.exists():
    assert json.loads(marker.read_text()) == fixed, 'Fixed predictions changed'
else:
    atomic_json(marker, fixed)
result_path = out / 'holdout_score.json'
if not result_path.exists():
    reference = pd.read_parquet('artifacts/final_holdout_dense_v1/reference.parquet')
    score = score_fixed_outputs(reference, out / 'matching_results.tsv',
                                'artifacts/final_holdout_dense_v1/truth.json', out)
    atomic_json(result_path, {'scope': 'Untouched 12,001 S1; all 10,320,219 training targets',
                             'pipeline_sha256': sys.argv[1], 'score': score})
print(result_path.read_text())
PY
