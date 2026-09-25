#!/usr/bin/env bash
# One submission build from the already frozen and evaluated procedure.
set -euo pipefail
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python_bin=/opt/amlc-venv/bin/python
pipeline=/workspace/az-ml-new/models/submission_v1_pipeline.json
tree=/workspace/az-ml-new/models/medium_dense_bge_ce1800/original_gate_and_tree.joblib
expected_pipeline_sha=${1:?Pass the selected pipeline SHA256 explicitly}
"$python_bin" - "$pipeline" "$expected_pipeline_sha" <<'PY'
import json, sys
from pathlib import Path
from ber.full_embeddings import file_digest
assert file_digest(Path(sys.argv[1])) == sys.argv[2], 'Selected pipeline changed'
config = json.loads(Path(sys.argv[1]).read_text())
for name, digest in config['source_sha256'].items():
    assert file_digest(Path(name)) == digest, f'Frozen source changed: {name}'
holdout = json.loads(Path('output/holdout_v1/holdout_score.json').read_text())
assert holdout['pipeline_sha256'] == sys.argv[2], 'Evaluate this exact pipeline first'
assert holdout['score']['reference_count'] == 12001, 'Holdout coverage is incomplete'
print('Complete fixed holdout report verified; waiting for official test candidates.')
PY

# This is a bounded dependency wait within one authorized build, not a scheduler.
for attempt in $(seq 1 1080); do
  [[ -f artifacts/full_dense_test/complete.json ]] && break
  sleep 20
done
[[ -f artifacts/full_dense_test/complete.json ]]
if [[ ! -f output/submission_v1/complete.json ]]; then
  "$python_bin" -u -m ber.predict_streaming \
    --data artifacts/full_dense_test --pairs artifacts/full_dense_test/pairs.parquet \
    --checkpoint "$tree" --pipeline "$pipeline" \
    --output output/submission_v1 --workers 4 --reference-batch-size 1000 \
    --expected-references 1732544 --reuse-sorted-pairs --ce-threads 2
fi
"$python_bin" - "$expected_pipeline_sha" <<'PY'
import json, sys
from pathlib import Path
r = json.loads(Path('output/submission_v1/complete.json').read_text())
assert r['state'] == 'complete' and not r['report']['benchmark_only']
assert r['report']['reference_count'] == 1732544
assert r['report']['pipeline_sha256'] == sys.argv[1], 'Completed predictions use another pipeline'
PY
"$python_bin" -u -m ber.validate_release \
  --matching output/submission_v1/matching_results.tsv \
  --candidate output/submission_v1/candidate_pairs.tsv \
  --test-dir data/raw/dataset/test --report output/submission_v1/validation.json

# Preserve only after validation passes; verify durable bytes before readiness.
"$python_bin" - "$expected_pipeline_sha" <<'PY'
import json, shutil, sys
from pathlib import Path
from ber.full_embeddings import atomic_json, file_digest
source = Path('output/submission_v1')
target = Path('/workspace/az-ml-new/releases/submission_v1')
target.mkdir(parents=True, exist_ok=True)
validation = json.loads((source / 'validation.json').read_text())
assert validation['passed'], 'Release failed official or strict validation'
files = ('matching_results.tsv', 'candidate_pairs.tsv', 'validation.json',
         'validation.validator.log', 'frozen_pipeline.json', 'report.json', 'complete.json')
hashes = {}
for name in files:
    expected = file_digest(source / name)
    shutil.copyfile(source / name, target / name)
    assert file_digest(target / name) == expected, f'Durable copy differs: {name}'
    hashes[name] = expected
shutil.copyfile('output/holdout_v1/holdout_score.json', target / 'holdout_score.json')
ready = {'ready_for_submission': True, 'pipeline_sha256': sys.argv[1],
         'source1_rows': 1732544, 'file_sha256': hashes,
         'leaderboard_file': 'matching_results.tsv',
         'validation_kind': 'format_and_integrity_only',
         'official_test_score': None, 'platform_submission_performed': False}
atomic_json(target / 'ready.json', ready)
atomic_json(source / 'ready.json', ready)
print(json.dumps(ready))
PY
