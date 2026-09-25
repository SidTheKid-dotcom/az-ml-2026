# Operating runbook

## GPU connection and runtime

Verified connection from this workstation:

```bash
ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes \
  -i /home/sidvish/amlc-private/ssh/id_ed25519 \
  -p 22141 root@69.30.85.107
```

`~/.ssh/id_ed25519` was a different key and was rejected. The existing key above
works; do not make key regeneration a blocker. `-F /dev/null` avoids a malformed
system SSH config. SCP uses the same options with uppercase `-P 22141`.
RunPod pod: `spl4sjzfwibvz3` (`pleased_amber_armadillo`); MCP was reauthorized.
Never print pod environment secrets or copy private-key contents into this repo.

- GPU: one NVIDIA A40, 46,068 MiB usable VRAM.
- Actual container limits: about 7.65 CPU cores and 50,000,000,000 bytes RAM.
  Host `free`/`lscpu` show much larger resources and are misleading.
- `/tmp/az-ml-new`: fast 30 GB overlay, active data/code/jobs; scratch is ephemeral.
- `/workspace/az-ml-new`: durable project reports/models on object-backed storage.
  Use ordinary `cp`, not `cp -a` (timestamp preservation fails). Its `df` capacity
  is synthetic. Keep large active memory maps on local scratch.
- `/workspace/amlc`: another project, unchanged. Its HF cache/runtime are reused.

```bash
cd /tmp/az-ml-new
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-new/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
/opt/amlc-venv/bin/python -m ber.tabular_sweep --help
nvidia-smi
```

Runtime: Python 3.12.3, torch 2.8 + CUDA 12.8, transformers 5.17,
sentence-transformers 6.1. Extra packages are isolated in `/opt/az-ml-new-extra`:
CatBoost 1.2.10, Unidecode 1.4, sparse-dot-topn 1.2, faiss-cpu 1.15.1.
Local `.venv`/pinned requirements support CPU preparation, analysis and tests.
`gpu_runtime.json` records the GPU environment's exact measured package versions;
its NumPy version differs from the local environment. Do not overwrite the shared runtime.

## File map

| Location relative to GPU scratch | Contents |
| --- | --- |
| `data/raw/dataset/{train,test}` | Original supplied TSVs, extracted from `student_resource.zip` |
| `data/quick` | 12,000 reference businesses; train 7,199 / tune 2,400 / old eval 2,401 |
| `data/medium` | Disjoint 60,000 refs; train 35,999 / tune 12,000 / sealed eval 12,001 |
| `data/hard` | Quick TRAIN+TUNE, mined against full raw pool; reuse its candidate edges |
| `data/medium_hard` | Larger TRAIN+TUNE full-pool lexical/phonetic mining output |
| `artifacts/full_train_e5_ft200` | Full raw S2-then-S3 embeddings + progress/completion manifests |
| `artifacts/full_test_e5_ft200`, `artifacts/full_dense_test` | Label-free official test encoding/retrieval, currently preparing |
| `artifacts/full_dense_{quick,medium}` | Full-pool dense candidates and compact records |
| `data/{hard_union,medium_union}` | Exact lexical/phonetic/dense candidate unions |
| `artifacts/*sweep`, `artifacts/*ce*` | Configs, scores, selected checkpoints/probabilities |
| `logs/` | Background job stdout and supervisor state |

The active fine-tuned encoder is
`artifacts/e5_small_ft200/finetuned_model`; its verified durable copy is
`/workspace/az-ml-new/models/e5_small_ft200` (466 MB). Compact reports are copied
back to local `artifacts/` and summarized under Git-visible `learnings/`.
Large checkpoints/candidates are intentionally not in Git: copy every promoted
checkpoint/config to the durable project directory and record its path/hash.
The completed 7.93 GB training embedding file was copied and SHA-256 verified at
`/workspace/az-ml-new/artifacts/full_train_e5_ft200/pool_embeddings.npy`; the scratch
path is a symlink to it. This frees space for test vectors without changing raw data.
Preserved MiniLM/BGE checkpoints live under `/workspace/az-ml-new/models/`;
their preservation manifests record exact files and hashes.

## Data and exact score

| Population | S1 references | S2 + S3 targets |
| --- | ---: | ---: |
| Original train | 2,206,821 | 10,320,219 |
| Official test | 1,732,544 | 9,969,589 |

Fields: `entity_id`, `business_name`, `business_address`, `country`.
S1 is deduplicated; each S1 can match zero or many S2/S3 records, including several
within one source. True links share country. About 5.6% of training S1 are true
singletons. Train is US/India; test additionally has 259,452 French S1. IDs' numeric
parts are not features. The quick 201,536-target pool is substantially too easy.

For a reference with nonempty truth, `F0.5 = 5*TP / (4*predicted_count + true_count)`.
True singleton + empty prediction scores 1; true singleton + any prediction scores
0. Average over **all S1 references**, including failed retrieval. F1 is a separate
diagnostic. `ber.metrics` implements this; metric and alignment tests must pass.

## Resume and submission

Read `learnings/CURRENT_STATE.md`; check `ps`, logs and completion markers before
rerunning scripts. `metadata.json`/`complete.json` commit data only after output
files finish. A tree sweep writes `pairs.parquet` at the end; its `results.json`
updates incrementally and is not a completion marker. Commands are in module
`--help` and the `run_*sweep.sh` scripts; the latter use this pod's paths.

Freeze models, retrieval and thresholds; run `ber.predict --score --split eval`
only after preparing the final holdout's full-pool candidates. It writes per-S1
scores, bootstrap intervals, hashes and TSVs. A cross-encoder blend also requires
the saved CE checkpoint and blend/gating configuration; plain `ber.predict` is
only the tree scorer and must not be mislabeled as the ensemble.

Official outputs: `output/matching_results.tsv` (`source1_entity_id`,
`matched_entity_ids`) and `output/candidate_pairs.tsv` (`source1_entity_id`,
`candidate_entity_ids`), comma-separated ID lists inside tab-separated rows.
Include all 1,732,544 test S1 exactly once; every predicted ID must be in the
actual candidate set. Use `data/raw/utils/validate_submission.py --help`, then
validate with **`--check-ids`**, both files, and the supplied test sources. Independently
assert candidate subset/coverage: the supplied validator emits warnings for some
violations. No submission file is ready yet. At most five public submissions/day;
save hashes, configuration, reported score and exact file for each. Final packaging
also needs reproducible code and the supplied methodology template.

`ber.validate_release --matching output/matching_results.tsv --candidate
output/candidate_pairs.tsv --test-dir data/raw/dataset/test --report
artifacts/release_validation.json` runs the unchanged official script with ID
checks and applies the stricter streaming gate. Its report includes output/input
hashes; it validates format and integrity, **not score**. Run on actual final files.

`docs/source_integrity.json` verifies all ten extracted resource files against the
original ZIP's byte counts and CRC32, and inventories the six original statement
files. The ZIP, statement and raw dataset are preserved; experiment caches are separate.
