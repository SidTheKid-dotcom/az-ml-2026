# Claude Code takeover handoff — 26 Sep 2026 10:30 IST

You are taking over as controller of team **Techquilla Shots** for the Amazon
ML Challenge 2026 entity-resolution task. The deadline is **27 Sep 23:59 IST**
and the **last leaderboard upload counts**. The public leaderboard is only a
subset of test. Do not claim a rank or hidden-test score from local evidence.

This file is the live takeover document. Read it first, then read `AGENTS.md`,
`CONTEXT.md`, `learnings/CURRENT_STATE.md`, `docs/PLAN_v2.md`,
`learnings/EXPERIMENT_DECISIONS.md`, and
`learnings/negative_side_analysis_manual.md`. Several older status sections are
stale; this handoff records the state at takeover.

## Executive decision

There is enough evidence to make the next training bet. The primary path is:

1. Verify the new fast trainer with the short historical **G0b parity run**.
2. Run the six 300-step G0 throughput variants.
3. Train an arena-scale BGE C2 screen from the current plain-14,400 checkpoint,
   using arena A, 1:3 typed/random negatives, and entity-value stake weights.
4. Select only on the component-disjoint 100,000-S1 SLICE, then report once on
   B-rest. Do not use HOLDOUT-2.
5. Add ClaimNet, the resolver, and Qwen only as measured secondary features.

The main uncertainty is now model quality, not retrieval. The strongest
incremental CPU retrieval branch fails its predeclared gate once compared with
the P2 full-key union. Do not spend time generating those test candidates.

## Hard rules

- HOLDOUT-2 is sealed: 50,000 S1s, seed 20260929, sorted-ID SHA256
  `7c98dcaef35d5a392a051e0525f854a2a6ceb55c366616ff2184abd98aa4ae63`.
- Never read or tune on HOLDOUT-2 labels before the final freeze.
- Medium eval/v1 holdout was opened once at F0.5 0.968292. Never tune on it.
- Split and bootstrap by S1 or claim component, never by pair.
- Do not inject true edges into retrieval. Missed truth stays in denominators.
- Use the full 10,320,219-target training pool for promoted comparisons.
- France has no labels. Pseudo-label metrics are not French accuracy.
- No external business lookup, geocoding, or task labels.
- Final matching TSV must use LF endings, include every S1 exactly once, contain
  no contested record IDs, and be validated with the official validator
  `--check-ids` plus `ber.validate_release`.
- Do not upload U5 or another intermediate file merely because it exists. The
  user explicitly wants the strongest final candidate, with the current public
  best retained as fallback.
- Do not edit `/tmp/az-ml-new`; it is the frozen v1 code/data tree.
- Use at most two OpenMP/BLAS threads. Pod cgroup limits are about 7.65 CPU cores
  and 50 GB RAM despite the host reporting more.
- An unbounded CatBoost fit is forbidden. 750k pairs fit; 1.5M swapped; the
  prior full fit died above the 50 GB limit.

## Git state

- Repository: `/home/sidvish/Documents/Work/Personal/az-ml-new`
- Integration branch: `codex/experiment-loop`
- Current HEAD: `9aac672` (`Reject empty normalized name matches`)
- The branch is 21 commits ahead of `origin/codex/experiment-loop` at takeover.
- `docs/briefs/HANDOFF_controller.md` is an existing untracked user-provided
  handoff. Do not delete or overwrite it.
- No commit has been pushed by this controller session.

Recent integrated commits:

| Commit | Contents |
| --- | --- |
| `9aac672` | Reject normalized-empty/native-script names in the name-only rule |
| `a208857` | Historical G0b pair-order compatibility |
| `58f8be6` | Qwen3-Reranker-0.6B yes/no adapter and tests |
| `d26a801` | Arena development-ID exclusion handling |
| `8be54ef` | Fast C2 trainer, scorer, evaluator, G0/G0b adapter, canonical text |
| `e3ece73` | Arena components, A/B/SLICE partitioning, fidelity report |
| `45c2c7a` | R0 and C7 retrieval decisions |
| `7e4a31b` | Forensic features, France pseudo-labels and diagnostics |

The pod's active v3 code symlink was deployed from `9aac672`:
`/tmp/az-ml-v3/code -> /tmp/az-ml-v3/code-9aac672`.

## GPU access and runtime

Verified SSH:

```bash
ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes \
  -i /home/sidvish/amlc-private/ssh/id_ed25519 \
  -p 22141 root@69.30.85.107
```

SCP uses uppercase `-P 22141`. Do not print or copy private-key contents.

- Pod: `spl4sjzfwibvz3` (`pleased_amber_armadillo`)
- GPU: NVIDIA A40, about 46 GB usable VRAM
- Runtime: `/opt/amlc-venv/bin/python`
- Extra packages: `/opt/az-ml-new-extra`
- HF cache: `/workspace/amlc/cache/hf`, offline only
- Scratch: `/tmp/az-ml-v3` for new code/jobs
- Durable: `/workspace/az-ml-new`
- At takeover, **no GPU process is running** and GPU memory usage is zero.

Standard environment:

```bash
export PYTHONPATH=/opt/az-ml-new-extra:/tmp/az-ml-v3/code/code/business_entity_resolution/src
export HF_HOME=/workspace/amlc/cache/hf
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
```

Launch long jobs detached with `nohup setsid ... < /dev/null &`. An earlier
interactive SSH exit killed a useful 11,240-step run.

## Public submissions

| # | File | Public F0.5 |
| --- | --- | ---: |
| 1 | original v1 | 0.956390 |
| 2 | v1 + one-owner + blanket removals/additions | 0.964232 |
| 3 | v1 + one-owner only | **0.964563** |
| 4 | France from #3, US/India from #2 | 0.964258 |

The best confirmed public file is:
`/home/sidvish/Downloads/assignment-only-lf/matching_results.tsv`, SHA256
`3e0c6206c03ba06f8cb6d3d8e5a1e916230dff5c025061af3eab38120750bfbd`.

U5 exists at `output/u5_name_only/` and under Downloads, but the user chose not
to upload it. Preserve it as a diagnostic/fallback only. The old U5 was built
before commit `9aac672`; rebuild before any use because normalized-empty names
are now explicitly excluded. The corrected 12k TUNE result remains exactly
0.972953849, delta +0.003961714 versus v1, so the published TUNE gain did not
change.

## Completed GPU results

### Frozen v1

- Medium TUNE: F0.5 0.968992, precision 0.987471, recall 0.938760,
  494 false links.
- Opened v1 holdout: F0.5 0.968292. Do not reuse it for selection.
- Public v1 score: 0.956390.

### R0 reranker follow-ups

Both use the same 12,000 medium TUNE S1s and 600,000 dense candidates from the
full target pool.

| Model | TUNE F0.5 | Precision | Recall | False links |
| --- | ---: | ---: | ---: | ---: |
| v1 BGE1800 | 0.968992 | 0.987471 | 0.938760 | 494 |
| field-tagged BGE 7,200 | 0.973686 | 0.989378 | 0.947705 | 422 |
| plain `name | address` BGE 14,400 | **0.974288** | 0.990133 | 0.948380 | 392 |

Plain-14,400 is the current checkpoint leader. It is +0.005296 over v1 and
+0.000602 over field-7,200. Format and step count are confounded; do not claim
plain text itself is better.

Durable outputs:

- `/workspace/az-ml-new/experiments/r0_fields7200_20260926b/`
- `/workspace/az-ml-new/experiments/r0_pipe14400_20260926b/`

Local aligned score copies for CPU ensemble analysis:

- `/home/sidvish/Documents/Work/Personal/ext-experiments/model-scores/field7200/`
- `/home/sidvish/Documents/Work/Personal/ext-experiments/model-scores/plain14400/`

An exploratory 0.8 plain + 0.2 field blend with one-owner and name-only gained
+0.000453 over plain alone with a positive reused-TUNE interval, but this is
below the +0.0005 promotion gate. Entity-disjoint cross-fitting gained only
+0.000281 and its interval crossed zero. Keep plain-14,400; do not promote the
ensemble unless the independent Cursor analysis under `ext-experiments/F/`
later provides stronger evidence.

## Competitor-complete arena

Arena scoring is complete and durable.

- References: 2,072,821 eligible S1s
- Candidate pairs: 103,641,050 dense K50 pairs
- Compact target records: 10,170,471
- Accepted v1 pairs before one-owner: 6,821,703
- Score runtime: 8,409 seconds
- Arena labels: 7,055,003 retrieved true pairs / 7,173,489 total true links
- Dense link recall: 0.983483
- HOLDOUT-2 rows read: 0

Durable score artifacts:

- Root: `/workspace/az-ml-new/arena/v1/`
- `complete.json` SHA256 `5d5dbc9c79c3a10498e4efba69f98bab509da18788d679168b5a608c59bf2d00`
- `probabilities.npy` SHA256 `342b9600d9c31f0e7093cf0265ffe26eac61619a0db1258473bcd3db6ef3f721`
- `base_probabilities.npy` SHA256 `3bc8e165ec8dab185e0eb1659af074515733fc5cf5c84f4b405f34eeff1a9a38`
- `scored_pairs.parquet` SHA256 `16b301bf2f59a7212c58e85fce1ffadd3fb26cdeacbfda5a153703f6794a12f7`
- Labels: `/workspace/az-ml-new/arena/labels/`

### Arena partitions

The first completed partition/fidelity report is under
`/workspace/az-ml-new/arena/finalize_v1/`. It used the pre-`9aac672` name-only
caller, so preserve it as historical evidence. A corrected rerun is active at:

- output: `/workspace/az-ml-new/arena/finalize_v2/`
- log: `/tmp/az-ml-v3/logs/arena_finalize_v2.log`

The component/partition assignment itself is unaffected by the name-only fix:

- selected claim threshold: 0.05
- components: 1,835,789
- largest component: 2,397 S1s (0.116%)
- A: 1,450,975 S1s
- B total: 621,846 S1s
- SLICE: exactly 100,000 S1s
- B-rest: 521,846 S1s
- A fraction: 0.700000145

Saved quick/medium development IDs are absent from the eligible arena by design:
7,199 quick TRAIN, 35,999 medium TRAIN and 24,001 medium development IDs were
recorded as excluded, not treated as partition errors. A/B therefore partition
the unused eligible population independently of those old development sets.

Historical `finalize_v1` hashes:

- manifest `ef6f799056205f669388b23ba1c1bb1fb60f65a61893ca09d867da23c0dbc2b8`
- fidelity `68c52a8df0f43f7b0e41120b514393c2a1ccb4357dc9a2e04e708700b5880e55`
- component IDs `76a339aa6cb1b37391b4402ece9976d5e8e8eb2fbefd420f4f13f864486ba44a`
- partition `00b6edcb04b5ea9bbbf68fb8f0d16e90c3798dcb8705642083fd8c4f2335112a`

### Historical arena metrics before the normalized-empty fix

These are useful baselines; read `finalize_v2/fidelity.json` for the corrected
numbers before training.

| Decision | Arena macro F0.5 | Precision | Recall | False links |
| --- | ---: | ---: | ---: | ---: |
| v1 + one-owner | 0.972382 | 0.991436 | 0.937619 | 58,100 |
| v1 + name-only | 0.975361 | 0.992057 | 0.945250 | 54,292 |

Name-only improved the competitor-complete arena by about +0.002979. Empty-link
precision rose from 0.8495 to 0.9385 and empty-link recall from 0.2380 to 0.4349.

Arena-vs-test US/India shape is close enough for model comparison:

- mean list lengths differ by less than 0.005 after one-owner and less than
  0.002 after name-only
- empty prediction rates differ by less than 0.001
- arena US/India contest rate is 0.00348 versus test US/India 0.00259
- one-owner removed claims are 80.1% wrong in arena versus about 85.2% inferred
  from the public uploads
- France remains a major shift: 36,447 of 48,927 test contests are French

Use arena as the primary yardstick, but preserve test/France shift as a limitation.

## Retrieval status

P2 full union is complete:

- `/workspace/az-ml-new/v2/train_union/full/ready.json`
- 89,995,555 retained pairs over 2,072,821 eligible S1s
- link recall 0.987347
- retrieval oracle F0.5 0.995965
- no truth-edge injection

On medium TUNE, P2 full-key union has 1,437,538 pairs, link recall 0.992743 and
oracle F0.5 0.997707.

### India native retrieval — stop

The Astra/Cursor experiment under `ext-experiments/A/native_recall/` is correct
against dense K50: native phonetic top-5 recovers 90 Indian dense misses and
projects to 1.79 CPU-hours on full Indian test. However, the controlled follow-up
under `ext-experiments/D/` compared it with P2's stronger full-key union.

The strongest variant adds only 43 true links and 58,911 pairs beyond P2,
oracle delta +0.000495, 95% interval [0.000248, 0.000807]. Incremental total
link recall is +0.104 percentage points, below the C7 +0.2-point gate. Do not
generate the full-test branch. Exact tracked result:
`learnings/c7_union_incremental.json`.

## France pseudo-labels and forensic features

Integrated modules:

- `ber.forensic_features`
- `ber.derive_forensic_vocab`
- `ber.france_pseudo_labels`
- `ber.label_free_diagnostics`

Full label-free French pseudo-label generation completed locally in 141 seconds:

- French S1s: 259,452
- candidate pairs scanned: 12,972,600
- P+: 411,984 pairs across 189,936 S1s
- P-: 2,838,600
- rival P-: 2,476,098
- template/different-street P-: 491,721
- pseudo-holdout: 41,026 P+ and 503,136 P-

Artifacts:

- `output/france_pseudo/pseudo_labels.parquet`
  SHA256 `4453647736b77e76655e735c205fdb2d551e1c47def0c58289ac4916afa6f358`
- `output/france_pseudo/counts.json`
  SHA256 `f90e781506447297f7a60bc53e5d674e67ed5b9f894f74b675aeba65b526bb6e`
- `output/france_pseudo/audit_pos.txt`
  SHA256 `9329d34271a5d23f60218f26e620ba2e06b8cd7fe6bfd5ce9d21735d5fc8a1f2`
- `output/france_pseudo/audit_neg.txt`
  SHA256 `67078090122078fbb0163f5c087d117617c699b3184c7326ae8b3fb8f3afe862`

The P+ audit looks visually strong, but transfer precision is not yet measured.
Do not mix French pseudo-labels into C2 until the independent transfer audit
requested under `ext-experiments/E/` is complete or equivalent US/India
simulation proves the precision. The full generator peaked around 5.16 GB RSS,
violating its intended 2.5 GB local-memory bound; do not rerun it unchanged on a
constrained machine.

Generator forensics under `ext-experiments/A/` found no profitable new accept or
drop rule beyond name-only. Protect already-accepted high-confidence canonical,
case/state/reorder/dba/junk/close-typo links from aggressive resolver removal,
but do not force-add their rejected tail.

`ext-experiments/A/opencode-sol/` is incomplete. It mined 59,488,454 study pairs
and reached 53.5M/59.5M alignment rows before timeout, but did not checkpoint
counters, run TUNE evaluation or produce a report. It largely duplicates the
completed Cursor forensics; do not spend critical-path time restarting it unless
the GPU path is blocked.

## Fast BGE C2 implementation

Integrated in `8be54ef` plus pair-order fix `a208857`:

- canonical field-preserving text view
- exact 1:3 positive:negative sampling
- typed false accepts, contest losers, unowned look-alikes, digit substitutions
  and empty-address siblings
- 50% typed / 50% random gated negatives
- entity-value stake weights, class-normalized and capped at 5x
- fused AdamW, optional frozen input embeddings, BF16, gradient clipping
- length bucketing, pad-to-8 and background tokenization prefetch
- atomic hash-verified checkpoints and deterministic resume
- `train`, `score`, `evaluate`, and `adapt-g0b` commands
- immutable pair-order and sealed-ID validation

Tests: 234 passed, 6 GPU-dependent tests skipped. Focused suite: 13 passed.

### G0b status

The corrected historical adapter is complete:

- `/tmp/az-ml-v3/g0b_medium_v2/complete.json`
- SHA256 `1c9a15347fc892fe01fd26559f854d841e3855efdd9f2d10cfd85fb9747dde3b`
- positive pool 122,822
- negative pool 368,466
- TUNE references 12,000

The first parity training launch failed before loading the model because the
historical pair order is grouped by S1 but not sorted by right index. Commit
`a208857` fixes this with explicit historical pair-order semantics. No GPU time
was consumed by that failure. **Parity training has not been rerun.**

## Qwen3-Reranker-0.6B

Integrated commit `58f8be6` adds `ber.qwen_reranker` and 14 CPU tests. A real
A40 smoke passed:

- cached snapshot:
  `/workspace/amlc/cache/hf/hub/models--Qwen--Qwen3-Reranker-0.6B/snapshots/e61197ed45024b0ed8a2d74b80b4d909f1255473`
- license: Apache-2.0
- counted parameters: 595,776,512
- weight SHA256 `27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b`
- yes token ID 9693; no token ID 2152
- real 16-pair BF16 forward completed and wrote aligned scores
- smoke output: `/tmp/az-ml-v3/qwen_smoke/`

The adapter is viable but unscored. It expects a prepared pair parquet with
`pair_row`, four text fields, and for training `label`, `true_count`,
`negative_type`, and partition A. Build that input from the same C2 sampling
world. Run Qwen only after C2 or in otherwise idle GPU time. Kill if resolver
gain on SLICE is below +0.0005.

## ClaimNet and resolver status

The implementations exist but are not integrated:

- ClaimNet worktree:
  `.claude/worktrees/agent-claimnet/`
  - untracked `src/ber/claimnet.py`
  - untracked `tests/test_claimnet.py`
  - 10 focused tests passed; full CPU suite reported 231 passed, 6 skipped
- Resolver worktree:
  `.claude/worktrees/agent-resolver/`
  - untracked `src/ber/resolver.py`
  - untracked `tests/test_resolver.py`
  - 7 focused tests passed

Do not merge blindly. ClaimNet has no real Transformer forward/checkpoint test.
The resolver currently loads the full pair parquet into pandas and may need a
bounded streaming/aggregation adapter at 103.6M rows. Run independent review and
bounded real artifact smokes before integration.

## Claude Cloud sessions

Three sessions were created:

- stack/release: `session_01Bghgxgyk2GRDS1HYyZfeAV`
- ClaimNet audit: `session_01FLKekQXPMxSbeiViFuzucg`
- resolver audit: `session_01L3Dxj3HWZkagNqubzDxZfT`

Cloud checkout had no git remote, so no cloud commit was pushed.

### Stack/release cloud result

Cloud-only commit `16e313e` on branch `codex/stack-predict` added
`stack_predict.py` and tests, but the commit is not present locally or remotely.
It may be unrecoverable if the cloud container was reclaimed. Reimplement it
locally rather than assuming it exists. Reported tests: 36 passed, 3 skipped.
Known limitations: synthetic fixture only, no 86.6M-row run, no crash resume,
and expected-F0.5 ignored off-gate pairs in count estimation.

### Resolver cloud result

The cloud checkout did not contain `resolver.py`, so the resolver itself was not
audited. It found the normalized-empty name bug fixed in `9aac672`. Cloud-only
commit `d32ec7a` is not recoverable locally. The corrected 12k TUNE name-only
result is unchanged at +0.003961714; native/empty normalized names now fire zero
times. Rebuild any test file from the corrected code.

### ClaimNet cloud result

The status retrieval timed out and no branch, commit, PR or trustworthy findings
were returned. Treat the local ClaimNet implementation as unaudited.

## Immediate next commands

### 1. Wait for corrected arena finalization

```bash
ssh ... 'tail -n 50 /tmp/az-ml-v3/logs/arena_finalize_v2.log'
ssh ... 'test -f /workspace/az-ml-new/arena/finalize_v2/manifest.json && echo complete'
```

Read and record corrected `fidelity.json`. Use
`/workspace/az-ml-new/arena/finalize_v2/s1_partition.npy` for all new training.

### 2. Rerun G0b parity

Run from pod with the standard environment:

```bash
/opt/amlc-venv/bin/python -u -m ber.fast_trainer \
  --command train --mode g0b \
  --arena /tmp/az-ml-v3/g0b_medium_v2/arena \
  --labels /tmp/az-ml-v3/g0b_medium_v2/labels \
  --model /workspace/amlc/cache/hf/hub/models--BAAI--bge-reranker-v2-m3/snapshots/953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e \
  --allowlist /tmp/az-ml-v3/code/problem-statement/models.yaml \
  --output /workspace/az-ml-new/experiments/g0b_fast_parity_v2 \
  --checkpoint-root /workspace/az-ml-new/models/g0b_fast_parity_v2 \
  --device cuda --threads 2 --save-every 1800 --log-every 100
```

Then score TUNE partition B:

```bash
/opt/amlc-venv/bin/python -u -m ber.fast_trainer \
  --command score \
  --arena /tmp/az-ml-v3/g0b_medium_v2/arena \
  --partition /tmp/az-ml-v3/g0b_medium_v2/partition.npy \
  --checkpoint /workspace/az-ml-new/models/g0b_fast_parity_v2/step_001800 \
  --score-partition B \
  --output /workspace/az-ml-new/experiments/g0b_fast_parity_v2_scores \
  --device cuda --amp bf16 --inference-batch 128
```

Evaluate the historical fixed choice:

```bash
/opt/amlc-venv/bin/python -u -m ber.fast_trainer \
  --command evaluate \
  --arena /tmp/az-ml-v3/g0b_medium_v2/arena \
  --labels /tmp/az-ml-v3/g0b_medium_v2/labels \
  --partition /tmp/az-ml-v3/g0b_medium_v2/partition.npy \
  --scores /workspace/az-ml-new/experiments/g0b_fast_parity_v2_scores \
  --eval-partition B \
  --choice parity:0.75:0.775 \
  --output /workspace/az-ml-new/experiments/g0b_fast_parity_v2_eval
```

Gate: F0.5 must be within ±0.001 of historical 0.968992. Diagnose rather than
continue if it fails.

### 3. Run G0 throughput matrix

Use `python -m ber.fast_trainer --print-g0-matrix`. Run all six variants on the
real arena A with 300 steps each, checkpointing to separate durable paths.
Measure pairs/s and peak VRAM. Adopt embedding freeze only if G0b parity remains
within ±0.001. Do not compare variants on different S1s.

### 4. Run the C2 screen

Use:

- arena `/tmp/az-ml-v3/arena_data`
- labels `/workspace/az-ml-new/arena/labels`
- partition `/workspace/az-ml-new/arena/finalize_v2/s1_partition.npy`
- sealed reservation `/tmp/az-ml-v2/artifacts/v2_holdout2`
- initial model `/workspace/az-ml-new/experiments/r0_pipe14400_20260926b/checkpoint`
- allowlist `/tmp/az-ml-v3/code/problem-statement/models.yaml`
- mode C2, 15,000 steps, batch selected by G0, 1:3 negatives, constant LR
- durable checkpoint root under `/workspace/az-ml-new/models/`

Score SLICE with `--command score`, then evaluate fixed candidate blends with
`--command evaluate`. The current evaluator accepts fixed choices only. Select
thresholds with component-disjoint cross-fitting rather than repeatedly scanning
SLICE for a lucky threshold.

Promotion gate:

- delta at least +0.0005 versus v1/name-only on the same SLICE
- component-bootstrap 95% interval above zero
- no-match false merges no worse by more than 0.2 percentage points
- near-ties: fewer false links wins

### 5. Secondary models

Only after a usable C2 score exists:

- audit/integrate ClaimNet and score it on SLICE
- run Qwen 0.6B as a second-family resolver feature
- fit the bounded resolver on B-rest using component OOF
- do not run SetNet, noisy-student, Qwen-4B, or K100 until core path evidence
  exists and the GPU would otherwise idle

## Final release path

The stack-predict cloud code is absent. Reimplement the minimal final stack
locally, using `predict_streaming.py`, `arena_decision.py`, `name_only.py`, and
`validate_release.py` as references. Required behavior:

1. Verify all score arrays against the exact pair-order hash.
2. Apply frozen tree gate/model blend or resolver probabilities.
3. Enforce one owner per record.
4. Apply corrected nonempty-name-only logic last; exact-name additions evict
   other owners, but normalized-empty names never fire.
5. Cap lists at 11 and reject contested IDs.
6. Preserve every test S1 in original order.
7. Write LF `matching_results.tsv` and `candidate_pairs.tsv` atomically.
8. Run official validator with `--check-ids` and `ber.validate_release`.
9. Write hashes, provenance, and `ready.json` last.

Do not open HOLDOUT-2 until the complete C2/decision/model stack and thresholds
are frozen. Then score HOLDOUT-2 once against the full target pool. Start final
test inference early enough to finish strict validation, ZIP, README and
methodology before the deadline. Pseudo-label use must be disclosed.

## Documentation and ledger debt

Immediately after the next result:

- update `CONTEXT.md`: it still says arena scoring is running
- update `learnings/CURRENT_STATE.md` with completed arena/finalize_v2 metrics,
  Qwen smoke, France pseudo-labels and G0b status
- update `docs/PLAN_v2.md` checklist: forensic/France tooling, fast trainer,
  arena score/partition and Qwen adapter code are no longer pending
- update `learnings/EXPERIMENT_DECISIONS.md` with corrected arena findings
- regenerate the scoreboard if possible; the last attempt was blocked because
  the local ignored artifact `artifacts/v2_tree_screen/dense_compat_final/complete.json`
  was missing
- keep every failed compatibility launch and its reason

## What not to do next

- Do not upload U5 now.
- Do not generate India native test candidates.
- Do not restart the incomplete OpenCode Sol forensics on the critical path.
- Do not fit a giant tree.
- Do not tune on medium eval or v1 holdout.
- Do not open HOLDOUT-2.
- Do not assume Claude Cloud code exists locally.
- Do not compare sampled-pool metrics with full-pool metrics.

The immediate controller task is therefore: **read `finalize_v2`, run G0b
parity, run G0, then launch the arena-scale C2 screen.**
