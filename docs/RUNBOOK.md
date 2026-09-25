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
| `data/medium` | Disjoint 60,000 refs; train 35,999 / tune 12,000 / evaluated holdout 12,001 |
| `data/hard` | Quick TRAIN+TUNE, mined against full raw pool; reuse its candidate edges |
| `data/medium_hard` | Larger TRAIN+TUNE full-pool lexical/phonetic mining output |
| `artifacts/full_train_e5_ft200` | Full raw S2-then-S3 embeddings + progress/completion manifests |
| `artifacts/full_test_e5_ft200`, `artifacts/full_dense_test` | Complete label-free test encoding/retrieval; 86,627,200 pairs |
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

The selected first dense procedure is **medium BGE1800**, 75% CE, threshold
0.775, original tree gate 0.01: **F0.5 0.968992 on 12,000 TUNE S1s**, precision
0.987471, recall 0.938760, 494 false links. The earlier union score 0.969812 used
2,400 different quick-reference TUNE businesses and is not its validation score.
`docs/checkpoints.json` records all promoted paths/digests. The active frozen JSON
is `docs/frozen_dense_pipeline.json`, preserved as
`/workspace/az-ml-new/models/submission_v1_pipeline.json`. Runtime is frozen:
CE batch 128, length 256, reference batches 1,000, length sorting and pad8. Verified
holdout F0.5 is 0.968292 on all 12,001 eval S1s against the full training pool.
That holdout is now consumed for evaluation, not available for future tuning.
No official test score exists.

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

Saved split reproduction: quick reservoir seed **20260925**, 12,000 S1s and
160,000 sampled distractors; medium seed **20260926**, 60,000 S1s and 300,000
distractors, excluding `data/quick/reference.parquet`. `ber.prepare` shuffles
each country's sampled S1s with `seed+1`, then assigns 60% TRAIN / 20% TUNE /
remainder eval. E5/CE training seed is **20260925**. Reuse saved splits instead
of resampling during release. Quick/medium ID overlap was independently verified
as zero without opening eval truth.

These digests are SHA256 of newline-joined, sorted IDs, without a trailing newline:

| Population | S1 count | ID SHA256 |
| --- | ---: | --- |
| Quick all | 12,000 | `8980d7ae3c65c38da6c9b9fddb7dfe23eec0c1659099d707af86a9f959f9464e` |
| Medium all | 60,000 | `451f7e94bc76080b0be4de8dfd1dd3d3ff452d66aa899f77eb29520281f80042` |
| Medium TRAIN | 35,999 | `47ec6f4736aaa6069c9dbddde67354637b3046d414165bea0c54ab0dcd5bc0dc` |
| Medium TUNE | 12,000 | `5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2` |
| Medium v1 holdout | 12,001 | `a777ea1b07cfa375119233991ce3f36e16e01ab2c72d2493fcbc82208daf14b9` |

The 600/1800 comparison additionally verifies identical text/row-order reference,
pool, pair, base-probability and training-truth digests in their
`experiment_config.json`; `artifacts/medium_dense_bge_budget_comparison.json`
collects them. Eval has 4,787 India / 7,214 US S1s. France remains unmeasured.

## Resume and submission

Read `learnings/CURRENT_STATE.md`; check `ps`, logs and completion markers before
rerunning scripts. `metadata.json`/`complete.json` commit data only after output
files finish. A tree sweep writes `pairs.parquet` at the end; its `results.json`
updates incrementally and is not a completion marker. Commands are in module
`--help` and the `run_*sweep.sh` scripts; the latter use this pod's paths.

The finalized runtime is `/workspace/az-ml-new/models/submission_v1_pipeline.json`,
identical to `docs/frozen_dense_pipeline.json`. Its SHA256 is
`9ac4050a72c64b05c032a17ec26f2a13b339c1b8c135791bf07b737534371f5e`.
Set `selected_pipeline_sha` to that digest; inspect live jobs before running:

```bash
bash code/business_entity_resolution/run_frozen_holdout.sh "$selected_pipeline_sha"
bash code/business_entity_resolution/run_first_submission.sh "$selected_pipeline_sha"
```

The holdout script verifies the exact 12,001 eval IDs and finalized pipeline,
saves an immutable pipeline copy before retrieval, and invokes the miner's
explicit `--splits eval --final-evaluation --frozen-model-bundle
--frozen-model-sha256` guard. Raw candidates are saved before selected truth is
read for diagnostics/compaction; truth records never become injected edges.
Integrated inference reads no labels, requires complete coverage, and hashes
fixed predictions before its one score. Artifacts:
`artifacts/final_holdout_dense_v1`, `output/holdout_v1/holdout_score.json`.
The underlying miner guard verifies only the tree; the script also freezes CE,
retrieval identity, blend, gate, threshold and runtime. Do not bypass this wrapper.

The submission script requires that same pipeline's complete holdout report,
waits for label-free test candidates, predicts every official S1, validates both
TSVs, and verifies durable copies under
`/workspace/az-ml-new/releases/submission_v1`. Only its successful `ready.json`
marks the files ready for the user to upload. It does not upload to the platform.
No benchmark prefix is allowed in release outputs. Runtime benchmarks are complete;
v1 test scoring and validation finished successfully. Inspect
`output/submission_v1/ready.json` before using either file; do not relaunch the
finished supervisor 46635. The frozen config flags describe selection time; later
evaluation/readiness is recorded separately and must not rewrite its digest.

Official outputs: `output/matching_results.tsv` (`source1_entity_id`,
`matched_entity_ids`) and `output/candidate_pairs.tsv` (`source1_entity_id`,
`candidate_entity_ids`), comma-separated ID lists inside tab-separated rows.
Include all 1,732,544 test S1 exactly once; every predicted ID must be in the
actual candidate set. Use `data/raw/utils/validate_submission.py --help`, then
validate with **`--check-ids`**, both files, and the supplied test sources. Independently
assert candidate subset/coverage: the supplied validator emits warnings for some
violations. The first submission files passed official and strict validation. Their local
paths and verified hashes are in `docs/submission_v1_release.json`. At most five public submissions/day;
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

## Fresh checkout and final package

Git contains code, evidence, rules and provenance; it intentionally does not
contain credentials, raw data, caches, model binaries or generated TSVs. On this
workstation the ZIP and `data/raw/` are at the repository root. On another machine
bring the supplied archive, extract it to `data/raw/`, and verify its identities
against `docs/source_integrity.json`. GPU access requires an authorized SSH key;
private GitHub access alone does not grant GPU access. If the endpoint changes,
read the existing RunPod pod's Connect details; do not recreate or delete it.

For exact prediction replay, use the three promoted model paths in
`docs/checkpoints.json`, not a newly trained approximation. For new experiments,
copy only required code/configs to a separate GPU work directory and reuse the
data/cache read-only. Never sync `.git`, secrets or the whole cache to the pod.
The portable recipe, pinned GPU dependencies, portal ZIP and archival bundle
commands are in `code/business_entity_resolution/README.md`. The under-500 MB
portal ZIP excludes weights; the larger archival bundle includes them. Git stays
small. Build only after the actual TSVs validate.

Record each platform attempt in `learnings/submissions.json`: release ID, file
SHA256, public score, platform status and whether the submission was actually
performed. A validator PASS is not a leaderboard score. Historical artifacts
retain their paths and hashes even when a newer model wins.

The first v1 upload is user-reported FAILED, with no error reason or platform
score yet. The release manifest is the historical as-built snapshot; the mutable
attempt record is `learnings/submissions.json`. See `learnings/CURRENT_STATE.md`
before choosing any diagnostic variant or submitting again.

## V2 compute migration: SageMaker AI

The user has $10,000 in AWS credits and wants to move **new experiments** to
SageMaker instead of paying for RunPod GPU time. This is a compute/storage move,
not a change to the matching objective or the frozen v1 release. The workstation's
AWS CLI currently reports an expired session (`aws login` is required); account
region, credit eligibility/expiry, SageMaker GPU training quota, and actual
regional price have not yet been verified. Do not submit a billable job until
these are checked in that account.

`docs/PLAN_v2.md` is a hypothesis backlog; this migration and the experiment
loop in `AGENTS.md` govern execution. The v1 production commands above describe
historical RunPod use, not a mandate to rerun them for v2.

1. Use short-lived **SageMaker training jobs** for GPU experiments and large
   offline prediction, not a continuously running inference endpoint. Start with
   one `ml.g6e.2xlarge` (one 48-GB L40S, 4 vCPUs, 64 GiB host RAM) for a
   bounded CE/TUNE parity and throughput pilot; move to `ml.g6e.4xlarge` if
   CPU or memory is the bottleneck. A 24-GB G5/G6 GPU is not a drop-in
   replacement for the A40's roughly 46 GiB usable VRAM. Confirm actual memory
   and availability on the job.
2. Stage only the supplied dataset, selected checkpoints, saved splits, and
   needed complete embeddings/candidate artifacts into a private S3 bucket in
   the job's region. The verified training pool embedding alone is 7.93 GB.
   Record object/file hashes; copy to the job's fast local storage and retain
   selected checkpoints and finished reports in S3. Do not re-encode the full
   pool or copy `/workspace/amlc`; install the pinned Python/CUDA dependencies in
   a separate image/environment and fetch only licensed models as needed.
3. The embedding manifests bind **absolute raw-input paths and mtimes** as well
   as checkpoint hashes (`ber.full_embeddings`, `ber.full_dense_candidates`).
   Plain S3 relocation will fail identity checks. In separate v2 code, verify
   the copied raw bytes/checkpoint, preserve their input mtimes or record a
   verified relocation, and rewrite only the relocated manifest/cache paths;
   never modify the historical v1 manifests or frozen source hashes.
4. Run a bounded pilot on existing TUNE IDs against the **full** training target
   pool, compare candidate identity, retrieval recall, macro F0.5, false links,
   singleton accuracy, runtime and memory against saved RunPod results. Hardware
   floating-point differences require a decision-level parity check. The v1
   holdout is already consumed and must not be reused for selection. Only after
   the pilot passes should larger v2 mining/training move; reserve a fresh
   entity-disjoint holdout first.
5. For long jobs, use checkpoint-to-S3 and test interruption/resume before using
   Managed Spot Training. Meter job duration, S3 storage/requests and data transfer
   against the credits; choose a per-job runtime limit and budget. Preserve and
   validate full-test TSVs through the existing official/strict validator before
   packaging. The v2 plan's BF16 **inference** item is already implemented in
   `ber.rerank_inference` on supported CUDA GPUs; benchmark any further change.

For scale only, [public third-party listings](https://cloudprice.net/aws/sagemaker/instances/ml.g6e.2xlarge)
put `ml.g6e.2xlarge` near **$2.80 per hour in us-east-1** (~$280 per 100 running hours), before storage and
transfer. This is not an account-specific SageMaker training quote or a guarantee
that these credits cover it: confirm both in the account's pricing and billing
views before estimating the usable GPU-hour budget.
