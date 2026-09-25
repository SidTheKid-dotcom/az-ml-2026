# Reproduce and experiment

Business entity resolution: each reference S1 can match zero or several S2/S3
records using noisy names, addresses and country. Score is mean **per-S1 F0.5**;
true singleton + empty prediction scores 1. Test labels are hidden. See the root
`AGENTS.md` and `learnings/CURRENT_STATE.md` before using the live GPU.

## Environment

The existing pod already has the tested runtime; do not reinstall it. For a new
Linux CUDA machine use Python 3.12, a compatible NVIDIA driver, and a separate env:

```bash
cd code/business_entity_resolution
python3.12 -m venv .venv
. .venv/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
export PYTHONPATH="$PWD/src"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
python -m unittest discover -s tests -v
```

`requirements.txt` preserves the measured local CPU environment. Use
`requirements-gpu.txt` for training/replay; do not combine the two NumPy pins.
The selected recipe was run on A40 / 48 GB with ~7.65 CPU / 50 GB RAM. Keep active memory
maps on fast local disk. Replay needs at least 30 GB free; more room is needed for
training, multiple caches and the final ZIP. Dependencies must be available;
installation on a clean machine is not yet independently reproduced.

## Exact selected-model inference

Use the supplied `dataset/` (containing `train/` and `test/`). The final package
contains the exact promoted weights under this code folder's `models/`. In the
working repository they are external at GPU `/workspace/az-ml-new/models`:

```bash
python src/reproduce.py --dataset /absolute/path/to/dataset \
  --models-root /workspace/az-ml-new/models --work-dir /fast-disk/replay-v1
```

Use an empty work directory, distinct from active jobs. `--dry-run` prints the
four commands without opening data or weights. The wrapper verifies frozen source,
encoder/tree/reranker, supplied test data and official-validator hashes; only checkpoint locations
change in a separately recorded relocated config. It encodes all 9,969,589 test
S2/S3 records, retrieves 50 per each of 1,732,544 S1s, scores all 86,627,200 candidate
pairs with the tree and eligible pairs with BGE, then validates both final files.
It preserves country labels including France. Outputs are
`<work-dir>/output/{matching_results.tsv,candidate_pairs.tsv,validation.json}`.

Validation runs the **unchanged supplied validator with `--check-ids`**, then
checks exact all-S1 coverage, duplicates and prediction-subset consistency.
A PASS establishes format/coverage, not accuracy. Replay on a different GPU may
have floating-point differences; compare actual files/accepted sets, not only
hashes of weights. The original selected run and its output hashes remain the
reference. No training or threshold search occurs in this command.

## Training recipe and new experiments

The following commands recreate the training procedure, rather than promise
bit-identical GPU retraining. Run in a separate working directory with the above
`PYTHONPATH`. Set `competition_data` to the supplied dataset directory and
`model_inventory` to this repository's `problem-statement/models.yaml` (or the
final package's `resources/models.yaml`). Keep saved TRAIN/TUNE splits for fair
comparisons. Never overwrite the selected weights or an active output directory.

```bash
python -m ber.prepare --dataset "$competition_data" --output data/quick \
  --references 12000 --distractors 160000 --seed 20260925
python -m ber.prepare --dataset "$competition_data" --output data/medium \
  --references 60000 --distractors 300000 --seed 20260926 \
  --exclude-reference-file data/quick/reference.parquet
python -m ber.experiment --data data/quick --output artifacts/quick_base \
  --models heuristic --top-k 20 --threads 2
```

Training loads pretrained models from the HF cache, without remote model code.
If absent, fetch only the selected repositories first:

```bash
python -c 'from huggingface_hub import snapshot_download; snapshot_download("intfloat/multilingual-e5-small"); snapshot_download("BAAI/bge-reranker-v2-m3")'
python -m ber.hf_experiment --data data/quick \
  --base-pairs artifacts/quick_base/pairs.parquet \
  --model intfloat/multilingual-e5-small --model-label e5_small_ft200 \
  --allowlist "$model_inventory" --output artifacts/e5_small_ft200 \
  --finetune-steps 200 --finetune-batch-size 32 --finetune-learning-rate 2e-5 \
  --temperature .05 --max-length 128 --seed 20260925 --threads 2
python -m ber.full_embeddings --model artifacts/e5_small_ft200/finetuned_model \
  --inputs "$competition_data/train/train_source2.tsv" "$competition_data/train/train_source3.tsv" \
  --expected-rows 5034616 5285603 --output artifacts/full_train_e5_ft200 \
  --batch-size 512 --chunk-size 50000 --max-length 128 --threads 2
python -m ber.full_dense_candidates --data data/medium --splits train,tune \
  --embeddings artifacts/full_train_e5_ft200/pool_embeddings.npy \
  --model artifacts/e5_small_ft200/finetuned_model --output artifacts/full_dense_medium \
  --top-k 50 --batch-size 512 --query-batch-size 512 --pool-batch-size 50000 \
  --threads 2 --device cuda --backend torch
python -m ber.tabular_sweep --data artifacts/full_dense_medium \
  --pairs artifacts/full_dense_medium/pairs.parquet --output artifacts/medium_dense_sweep \
  --device GPU --iterations 650 --threads 2 --depths 6 8 10 --negative-weights 1 2 \
  --gpu-ram-part .22 --seed 20260925
python -m ber.cross_encoder_experiment --data artifacts/full_dense_medium \
  --base-dir artifacts/medium_dense_sweep --base-format tabular --require-hard-pool \
  --model BAAI/bge-reranker-v2-m3 --model-id BAAI/bge-reranker-v2-m3 \
  --allowlist "$model_inventory" --output artifacts/medium_dense_bge_ce1800 \
  --steps 1800 --batch-size 32 --inference-batch-size 64 --max-length 256 \
  --negative-ratio 3 --conflict-fraction .5 --positive-fraction .5 \
  --learning-rate 2e-5 --weight-decay .01 --rerank-cutoff .01 \
  --tune-only-inference --threads 2 --gpu-memory-fraction .4 --seed 20260925
```

Historical selection: numeric tree depth 10, negative weight 2, and BGE weight .75
at threshold .775. E5 used 7,199 quick TRAIN businesses; tree/BGE used 35,999 medium
TRAIN businesses, not every supplied labeled reference. A rerun can select other
weights/thresholds; treat it as a new model, freeze its identities, then evaluate
on a new untouched holdout before promotion. Do not replace frozen v1 hashes to
force new weights through its replay gate. The original pod-specific wrappers
`run_frozen_holdout.sh`, `run_test_candidates.sh`, `run_first_submission.sh` are
for the existing v1 release; inspect completion markers before running them.

For a quick next experiment, shorten the CE budget to 200–600 steps or compare a
single tree/model/negative sampler, retain the full competing target pool, and
write to a new output directory. Use `--help` for supported adapters. There is no
requirement to keep E5/BGE: verify licenses and parameter count for other models,
add an adapter where necessary, and measure a controlled comparison first.

## Final submission ZIP

The live release wrapper creates `ready.json` only after complete prediction,
validation and durable-copy hash checks. From the **repository root**, build:

```bash
PYTHONPATH=code/business_entity_resolution/src python \
  code/business_entity_resolution/src/package_submission.py \
  --release /workspace/az-ml-new/releases/submission_v1 \
  --models-root /workspace/az-ml-new/models --destination releases/v1 \
  --team-name techquilla-shots
```

The supplied team name is `techquilla-shots`. Add `--team-members` only with real
names when available; an unspecified member line is omitted, never invented. The builder refuses pending files,
failed/incomplete validation, changed hashes or overwritten archives. It includes:

- `output/`: both validated TSVs, unchanged.
- `code/business_entity_resolution/`: source, tests, pinned environments, replay
  README, exact trained weights and compact provenance/official validator resources.
- `Documentation_template.md`: filled methodology with team details and verified
  release status; `package_manifest.json`: SHA256 for every packaged file.

Every archived file is reread and verified before the ZIP becomes final; a ZIP
SHA256 sidecar is saved. Large binaries/ZIPs remain outside Git. Upload only
`matching_results.tsv` to the live leaderboard; preserve the same bytes in the
final ZIP. Record the actual platform result; no script uploads automatically.

## Source map

| Modules | Purpose |
| --- | --- |
| `prepare`, `normalization`, `features`, `numeric_features`, `residual_features` | Data splits and candidate-group features |
| `candidates`, `hard_pool`, `full_embeddings`, `full_dense_candidates`, `full_dense_inference`, `merge_candidates` | Lexical/dense retrieval and full-pool candidate provenance |
| `models`, `experiment`, `hf_experiment`, `tabular_sweep`, `cross_encoder_experiment`, `residual_experiment` | Fit and compare models |
| `retrieval_ablation`, `phonetic_ablation`, `pruned_retrieval`, `decision_sweep`, `transfer_experiment`, `stress`, `cross_encoder_compare` | Focused ablations and failure diagnosis |
| `predict`, `predict_streaming`, `rerank_frozen`, `rerank_inference`, `validate_release` | Frozen inference and submission validation |
| `metrics`, `uncertainty`, `saved_report`, `scoreboard` | Exact scoring, comparisons and result ledger |
| `reproduce.py`, `package_submission.py` | Portable selected-model replay and final ZIP |
