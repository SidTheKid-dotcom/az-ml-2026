# Handoff — resume here

## Resume here — 2026-09-26 21:00 IST (best public 0.978287)

**Best public score: 0.978287** (user-reported, 26 Sep). The file is
`/home/sidvish/Downloads/s1-pair-tree/matching_results.tsv`, SHA256
`559f37b9b66f4e18fc28c0b25ae60d08543c386e0bdb32c619990b4c9cbd1a78`, the same bytes as pod
`/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv`.
- **US/India rows:** `ber.pair_tree_c2` "model" (CatBoost on v1 features plus C2
  rival/rank features, fit on B_REST halves), threshold 0.725, one owner, cap 11.
  Arena B_REST cross-fit is **0.985511** (US 0.987438, India 0.982626),
  precision 0.99789, recall 0.96096.
- **France rows:** C2 step-40k blend 0.9·C2 + 0.1·tree ≥ 0.10 (tree gate 0.01),
  plus the empty-address name-only rule.
- **C2:** bge-reranker-v2-m3 trained on arena A for 40k steps, plain
  `name | address`, checkpoint `/workspace/az-ml-new/c2_plain/ckpt/step_040000`.

**Where the loss is.** Details are in `learnings/c2_h100_run.md`, "S1 public = 0.978287".
- Train labels show the generator is country-independent: 3.46 true links per
  S1 and 5.58% singletons in both US and India.
- Test US/India borderline score densities match B_REST, and a withheld-S1
  distractor simulation is null. So test US/India ≈ 0.9848, which implies
  **France ≈ 0.941**. France is about 40% of all lost points.
- The French C2 blend is smeared. France has 0.545 pairs per S1 where the tree
  is ≥ 0.9 and C2 is < 0.05, against 0.016–0.020 in US/India (28×). At
  threshold 0.10 their acceptance turns on the tree's third decimal.
- Rescoring confirms stored C2 = fresh C2, so this is not a scoring bug. C2
  rejects different-street templated-name collisions (correct) and
  same-address records with an added or changed legal form (truth unknown).

**GPU (H100 pod, `ssh -p 16295 root@216.81.245.34`).**
- The union k=3 C2 scoring is done: `/workspace/az-ml-new/union/c2_{b_rest,test}_k3_step040000`.
- Qwen3-Reranker-0.6B training is running: `/workspace/az-ml-new/qwen/train_run1.log`,
  step ~1,850/3,250 at 20:45 IST, about 220 pairs/s. The chain
  `qwen_after_france.sh` then scores B_REST/test and builds the ensemble.
- France fine-tune (S2) is done, file SHA `b9a70d14…`
  (`/opt/az-ml-new/runs/s2_pair_tree_40k_france/model_us_india_c2france_ft/`). Not uploaded.

**Do not do** (measured, see decisions table):
- Drop French street-token-disjoint links. They are one-character street typos of true links.
- Apply a C2 floor on US/India. Accepted pairs with C2 < 0.005 are 73–84% true;
  B_REST 0.979583 → 0.979547.
- Re-tune for test distractors by withholding S1s. The effect is null (0.97958 → 0.97910).
- Loosen the French blend threshold. The [0.05, 0.1) band holds 1.23 pairs per S1,
  mostly templated look-alikes.
- Expect S2 to help. Strict pseudo-labels hold only easy positives and French
  lists get shorter (3.19 per S1 vs S1's 3.26; the prior is about 3.46).
- Still discarded: street-disjoint drops, key adds/moves, unique-name street adds,
  digitless name-only, count fill, cluster voting, expected-F0.5 decode, mutual
  top-1, resolver (all in the decisions table).

**Next (France first):** measure the generator's legal-form rule on the arena
(`/workspace/az-ml-new/distractor_sim/legal_rule.py`, running). Then build
France-only variants that change only French rows, so a public upload measures
the French effect exactly. Then stack the union and Qwen US/India gains if their
B_REST gates pass.

## Resume here — 2026-09-26 15:45 IST

EC2 is running one new CPU job and nothing else. GPU C2 is separate; do not start a second trainer.

`ber.rival_text_arena` (pid file `/opt/az-ml-new/rival-text.pid`, log `/var/log/az-ml-new/rival_text_arena.log`) fits the pair-tree candidates again with graded digit-edit and street-token features for the pair and for the rival claimant. Same 3M rows, seed 0, 80/20 split of A. Threshold chosen on A-tune. B_REST is read once against the saved pair tree at 0.70. SLICE is not read. It aborts if the reloaded pair tree's A-tune is not 0.976581. Fuzzy admin matching is off.

## Resume here — 2026-09-26 15:30 IST

The slow copy through this machine was stopped. The arena went EC2 → pod
directly, and `bge-reranker-v2-m3` came from Hugging Face onto the pod.
Hashes matched: reference `f6096c72…`, pool `a0308d52…`, pairs `16b301bf…`,
reserved IDs `356931fc…`. Model file is 2,271,071,852 bytes.

C2 is running on the H100, pid in `/opt/az-ml-new/c2_plain/train.pid`, log
`/opt/az-ml-new/c2_plain/train.log`. At 15:30 IST it was still loading the
arena (about 7 GB RSS, GPU not in the step loop yet). Command: plain text,
40,000 steps, batch 128, bf16, compile, prefetch 8, 8 tokenizer workers.
Checkpoints: `/workspace/az-ml-new/c2_plain/ckpt`. Do not start a second copy
or a second trainer. `matching_results.tsv` waits on a B_REST score that
beats v1. Nothing is uploaded.

Integer `s1_partition.npy` codes are 0=A, 1=B_REST, 2=SLICE. Training uses A.

## Resume here — 2026-09-26 14:15 IST

Full 36-hour factual handoff (scores, decisions, hashes, what was not done):
[`docs/briefs/HANDOFF_20260926_1430.md`](../docs/briefs/HANDOFF_20260926_1430.md).

Nothing is uploaded after 0.964563. No GPU exists. Nothing is running on the
GPU side and no new job was launched for this handoff.

**Preferred unuploaded file:** `output/pair_tree_us_india/matching_results.tsv`,
SHA256 `b7bc5a3d71d04a69002e4cf5f602fe400dcbf59fa657c2fbcbfc1c4dac3f83f7`.
- The partition-A pair CatBoost is used for US/India rows, and name-only rows
  are kept for France. The model passed the gate on arena B_REST against
  empty-address name-only: 0.976720 vs 0.975376, delta +0.001345, CI
  [+0.001240, +0.001469]. It was fit on 3M pairs from 80% of A components, with
  threshold 0.70 and "model alone" chosen on the other 20% of A. One owner, list
  cap 11. B_REST was read once; SLICE and HOLDOUT-2 are unread. No-match false
  merges went 3.38% → 2.53% and singleton accuracy 0.966 → 0.975. The rival-claim
  score carries most of the importance, and exact-key features are about 1% each.
- The priced public score is about 0.970 if the US/India gain carries. That is
  an estimate, not a public score, and it is still far from 0.985. 0.976720 is an
  arena score; there is no arena TSV.
- Validation: 1,732,544 rows, 0 contested, lists ≤ 11, LF, subset of the v1
  candidates, official validator `--check-ids` PASS.

**Do not prefer** `output/pair_tree/matching_results.tsv` (SHA256
`650bc4052e7006f2cc0576f94e13d98c12a895a8379dad34687ab0bb69cb99f4`). It
passes the same validation, but it applies the model in France, where it
removes 11,480 links and adds 8,815. That is untested: France has no labels.

**Fallbacks:**
1. `output/name_only_brest/matching_results.tsv`, SHA256
   `bab8ed5e02d32c63db0b012603e7c16b0c90d94dd34bc693eb78dbc283c27e26`. It is
   built on the 0.964563 file and validated. Its priced public delta is about
   +0.00420, which is not measured. Its arena rule is B_REST 0.975376
   (`/opt/az-ml-new/runs/name_only_arena/report.json`, no TSV).
2. Best public score, 0.964563: `/home/sidvish/Downloads/assignment-only-lf/matching_results.tsv`,
   SHA256 `3e0c6206c03ba06f8cb6d3d8e5a1e916230dff5c025061af3eab38120750bfbd`.
   Do not re-upload it. Do not upload the older U5 name-only file.

**Compute:** the only machine is the AWS CPU box (`docs/RUNBOOK.md`, "AWS CPU
machine"). RunPod pod `spl4sjzfwibvz3` was terminated with no network volume.
C2 and its step-5000 checkpoint are unrecoverable. Do not wait on RunPod
support. GPU quotas are 0 with requests `CASE_OPENED`; do not file more.

**If a GPU appears — C2 launch recipe (staged and CPU-smoked 2026-09-26 14:40 IST):**
public Apache `BAAI/bge-reranker-v2-m3` init, plain `name | address` (the
recipe behind 0.974288 on the 12k medium TUNE S1s at 14,400 steps; those
weights are gone), bf16, no gradient checkpointing, one GPU. Batch 128 on an
80 GB card, else 64. Checkpoint every 5,000 steps to a network volume without
hashing during the run. Screen at step 15,000; continue to 40,000 only if the
15k checkpoint beats v1 on arena B_REST (not SLICE). Run 40,000 steps with the
constant schedule and score the step-15,000 checkpoint while training continues;
changing `--steps` changes the config digest and blocks `--resume`.

```bash
cd code/business_entity_resolution && PYTHONPATH=src python -u -m ber.fast_trainer \
  --command train --mode c2 --text-view plain --steps 40000 \
  --batch-size 128 --microbatch 128 --fused --no-freeze-embeddings --amp bf16 \
  --length-buckets 64,128,256 --defer-checkpoint-hash --save-every 5000 [--compile] \
  --arena ARENA/c2_input --labels ARENA/labels \
  --partition ARENA/finalize_v1/s1_partition.npy \
  --sealed-reservation RES/v2_holdout2 --model MODELS/bge-reranker-v2-m3 \
  --allowlist ../../problem-statement/models.yaml \
  --output VOL/c2_plain/out --checkpoint-root VOL/c2_plain/ckpt --device cuda --threads 4
```

- New `fast_trainer` flags (additive, defaults unchanged, 11 unit tests pass):
  `--text-view plain` (C2 only; stored in the checkpoint profile, so scoring
  reuses it), `--length-buckets` (static pad widths), `--defer-checkpoint-hash`
  (large files recorded by size at intermediate saves; the final step is fully
  hashed; scoring hashes deferred files at verify time), `--attn-implementation`,
  and `--compile` (wraps only the forward pass; saves use the raw model).
  Attention is already SDPA by default under transformers 5.17.0; FlashAttention-2
  needs the `flash-attn` package and `--attn-implementation flash_attention_2`.
- Staged on the AWS box: `/opt/az-ml-new/artifacts/arena/c2_input` (symlinks
  to `arena/v1` plus a manifest recomputed from file bytes; reference/pool/pairs
  SHA256 `f6096c72…` / `a0308d52…` / `16b301bf…` match the original arena build),
  the ID-only reservation `/opt/az-ml-new/artifacts/v2_holdout2`
  (`reserved_ids.txt` SHA256 `356931fc…` matches `reservation.json`), and the public model
  `/opt/az-ml-new/models/bge-reranker-v2-m3` (`model.safetensors` 2,271,071,852
  bytes). The venv now has CPU torch 2.8.0 and transformers 5.17.0; a GPU box
  needs the CUDA wheel instead.
- CPU smoke (4 steps, batch 8, fp32, `/opt/az-ml-new/runs/c2_cpu_smoke/`): full
  arena load, sealed-ID check, typed negatives, and training took 3 min 10 s wall
  and 23 GB RSS, so GPU startup is cheap. Resume from the deferred step-2
  checkpoint reproduced the step-3/4 losses exactly, and the final weights and
  optimizer state were bitwise identical to the uninterrupted run. A 2-step
  `--compile` CPU run also finished (`compile.log`, exit 0) and saved a normal,
  verifiable `XLMRobertaForSequenceClassification` checkpoint. Its step-1 loss
  differs (1.994 vs 1.792 on the same batch), most likely because compiled
  dropout draws a different RNG stream (not verified), so compiled and eager
  runs are not bitwise comparable. Choose one mode
  for the whole run. This is a code and data check only, not a score.
- Throughput reference: the A40 did about 190 steps/min at batch 64 with
  pad-to-8 and hashed saves (~4 min stall per save). A second GPU will not help
  until the trainer is multi-GPU.

**Evaluation guards:** HOLDOUT-2 stays sealed: 50,000 S1s, seed 20260929, ID
SHA256 `7c98dcaef35d5a392a051e0525f854a2a6ceb55c366616ff2184abd98aa4ae63`. Do
not retune on the opened medium holdout (F0.5 0.968292). SLICE is not a tuning
set. France has no labels.

**Promotion gate:** delta ≥ +0.0005, component-bootstrap 95% CI above 0, and
no-match false merges no worse by more than 0.2 points. The last upload counts.
Do not upload an unmeasured file.

## Do not repeat — measured failures (26 Sep)

Reports are under `/opt/az-ml-new/runs/` on the AWS box. Decision rows are in
`EXPERIMENT_DECISIONS.md`.

- **Exact key as a hard rule.** Moves were 125/8,813 = 0.014 (US 0/3,042).
  Unowned exact-key adds were 3,576/12,935 = 0.276. Key-by-blend on partition A:
  no cutoff reached 0.80 precision with n ≥ 500. The best add band was 127/217 =
  0.585 at blend 0.3–0.5, and moves were ~0 in every band. Do not upload key
  move/add files. The key may be a model feature, not a rule.
- **Name-only beyond empty addresses.** The digitless extension scored B_REST
  delta −0.006952. Unique-name street adds scored −0.019117; identical streets
  are only 0.037 true. Name-only stays empty-address only.
- **Decision-layer tweaks on v1 scores.** The best CPU-wave threshold (0.70) gave
  B_REST +0.000436, below +0.0005. Resolver threshold 0.55 gave +0.000208.
  Responsibility decode gave −0.019, mutual top-1 −0.242, and small-gap drop
  −0.802.
- **France rules.** The strict France pseudo rule has precision 0.999895 on
  US/India but only fires at blend ≥ 0.9, so it adds no links. A different
  street + house + city key had unowned-add precision 0.056.

## Live — 2026-09-26 13:10 IST

- RunPod pod `spl4sjzfwibvz3` is terminated. The C2 step-5,000 checkpoint was on pod disk and is gone. GPU quotas in us-east-1, us-west-2, and ap-south-1 are still `CASE_OPENED`. Do not wait on that checkpoint.
- One EC2 `r7i.8xlarge` is up: `i-06fc1a94349dcd6de`, Virginia, 32 vCPU, 247 GiB. SSH: `ssh -i /home/sidvish/.ssh/az-ml-new-ec2 ubuntu@18.233.153.184`. Arena pairs are on the box. A second 256 GiB machine does not fit the 32-vCPU quota. Full procedure (SSH rule, S3, paths, job launch): `docs/RUNBOOK.md`, section "AWS CPU machine".
- CPU wave finished 2026-09-26 13:13 IST. B_REST, choice on A, SLICE unread. Baseline **0.972465**. Best screen is threshold 0.70: **0.972902**, delta **+0.000436**, below +0.0005. Responsibility decode **0.953493**, delta **−0.018972**. Claim propagation delta 0. Log: `/var/log/az-ml-new/cpu_wave.log`.
- Resolver finished. Chosen on B_REST: threshold **0.55** then one-owner. B_REST **0.972673**, delta **+0.000208**, CI [+0.000096, +0.000319]. SLICE was reported, not used to choose: **0.973301**, delta **+0.000353**. Below +0.0005. Do not promote. HOLDOUT-2 unread. Metadata: `/opt/az-ml-new/runs/resolver_blend_base/metadata.json`.
- Corrected name-only rule on B_REST: **0.975376**, delta **+0.002910**, CI [+0.002815, +0.003018]. Adds 62,649/67,388 = 0.929676. Drops include 7,874 true links of 16,287 (0.483453 still true). No-match false merges 1,082 → 976. Clears +0.0005. SLICE unread. Report: `/opt/az-ml-new/runs/name_only_arena/report.json`.
- Partition-A pair CatBoost **passes the gate** against empty-address name-only. Features: v1 blend, tree score, blend rank and gap in the S1 list, S1 and record counts of blend ≥ 0.5, best rival blend for the record, exact normalized name, name multiplicity, address kind, exact-key self/other. Fit on 3,000,000 pairs from 80% of A components; threshold 0.70 and variant "model alone" chosen on the other 20% of A (0.976581 vs name-only 0.975056). One B_REST read: **0.976720** vs **0.975376**, delta **+0.001345**, CI [+0.001240, +0.001469]. No-match false merges 976 → 731 (3.38% → 2.53%). Longest list 11, 0 contested. Rival blend carries 31% of importance; key features ~1% each. SLICE unread. Model `/opt/az-ml-new/runs/pair_tree_arena/model.cbm` SHA256 `35dcf133a0694e75be8caaf6cb186b901b52451af165438385119cb174538e26`; report beside it. Refactored features reproduce the saved arena probabilities exactly (`parity.json`, max diff 0.0).
- Two validated test files, not uploaded. Both: 1,732,544 rows, 0 contested, longest list 11, LF, official validator `--check-ids` PASS, subset of the v1 candidate file.
  - `output/pair_tree/matching_results.tsv`, SHA256 `650bc4052e7006f2cc0576f94e13d98c12a895a8379dad34687ab0bb69cb99f4`: the model on every country. Against name-only, US +16,775/−3,933 links, India +16,085/−6,224, France **+8,815/−11,480**. France moves opposite to the labeled countries and is unmeasured.
  - `output/pair_tree_us_india/matching_results.tsv`, SHA256 `b7bc5a3d71d04a69002e4cf5f602fe400dcbf59fa657c2fbcbfc1c4dac3f83f7`: model rows for US/India, name-only rows for France. This is the lower-risk candidate. If the arena gain carries to test US/India (85% of test S1s), about +0.0011 over name-only, i.e. roughly 0.970 public. That is an estimate, not a score.
  - Test inputs are in `s3://az-ml-new-20260926-533267274451/inputs/artifacts/v1_test_scores/` and `/opt/az-ml-new/artifacts/v1_test_scores/`. Rebuild: `python -m ber.pair_tree_test test --scores /opt/az-ml-new/artifacts/v1_test_scores --out DIR`.
- Exact-key edits split by v1 blend, cutoff chosen on partition A: no band qualifies. Unowned key adds peak at 127/217 = 0.585 (blend 0.3–0.5) and are 183/350 = 0.523 even at blend ≥ 0.775. Key moves are 0–20% in every band. No B_REST read was made because nothing passed A. The key stays a feature only. Report: `/opt/az-ml-new/runs/key_by_blend/report.json`.
- Unique-name street adds on top of the empty-address rule lost on B_REST: delta **−0.019117**, CI entirely below 0. Even identical street tokens are only 1,252/33,805 = 0.037036 true among records the name-only rule left unowned. Do not add them. Report: `/opt/az-ml-new/runs/street_name_arena/report.json`.
- Digitless-address extension of the name-only rule lost on B_REST: delta **−0.006952**, CI entirely below 0. Adds 172/2,889 = 0.059536. Drops removed 118,084 true links of 118,119. Do not extend the rule past empty addresses. Report: `/opt/az-ml-new/runs/name_nodigits_arena/report.json`.
- Test file built on the 0.964563 submission and not uploaded: `output/name_only_brest/matching_results.tsv`, SHA256 `bab8ed5e02d32c63db0b012603e7c16b0c90d94dd34bc693eb78dbc283c27e26`. 1,732,544 rows, 0 contested, longest list 11 after trimming one 12-ID row, official validator `--check-ids` PASS. Priced public delta about +0.00420 (France +0.000888, US/India +0.003309) before that one trim. Do not upload the older U5 file.
- Exact-key residual finished. Overall key precision **0.990890** (US 0.995216, India 0.983544), matching the manual's 0.9912. The disagreements do not: moves **125/8,813 = 0.014184** (US 0/3,042, India 125/5,771); unowned adds **3,576/12,935 = 0.276459** (US 168/3,063 = 0.054848, India 3,408/9,872 = 0.345219). Both are below the break-even for a public edit. Do not build or upload the move/add files. Report: `/opt/az-ml-new/runs/key_residual/report.json`.
- France-rule transfer audit on arena US/India, labels opened only after selection. P+ 1,710,947 pairs, precision **0.999895** (179 false). Rival negatives 13,607,551, precision **1.0**, zero true links rejected. The rule requires blend ≥ 0.9, so it cannot add links the current threshold misses. Safe as high-precision training pairs; not a recall gain and not French accuracy. Report: `/opt/az-ml-new/runs/france_transfer_audit/`.

## Active C2 arena run — checked 2026-09-26 11:51 IST, now dead

- RunPod A40 PID 90833 is training `ber.fast_trainer --mode c2` for 40,000
  steps, batch/microbatch 64, fused optimizer, trainable embeddings, initialized
  from the plain-14,400 checkpoint. It was at step **7,500** with 100% GPU use;
  no second GPU process is competing.
- Complete checkpoint:
  `/workspace/az-ml-new/models/c2_arena_v1/step_005000`; log:
  `/tmp/az-ml-v3/logs/c2_arena_v1.log`. Effective rate is about 190–192
  steps/minute. Each 6.8 GB checkpoint pauses training about four minutes for
  object-store write and hash readback; projected finish is around 15:10 IST.
- `/tmp` has only 3.4 GB free. Do not touch frozen `/tmp/az-ml-new` or write
  large new outputs under `/tmp`; use `/workspace/az-ml-new`.
- Current takeover brief:
  [`HANDOFF_20260926_1151.md`](../docs/briefs/HANDOFF_20260926_1151.md).

## Arena no-match filter — keep recipe, not current probabilities

- Unweighted CatBoost fit on A and applied after one-owner at p ≥ 0.5. On the
  confirm half of B_REST: **+0.000477**, CI [+0.000368, +0.000586]. On the one
  SLICE read: **0.972948 → 0.973402**, delta **+0.000454**, component-bootstrap
  CI **[+0.000292, +0.000632]**.
- SLICE false merges fell 3.33% → 2.25% (188 → 127 of 5,649 no-match S1s), but
  17 true lists were destroyed. This is below the +0.0005 promotion gate and
  specific to v1 scores. Keep its features for the resolver; rebuild and refit
  on C2 rather than applying saved probabilities.
- Report:
  `/home/sidvish/Documents/Work/Personal/ext-experiments/G1/g3/report.md`.

## Arena record-cluster voting — discarded

- Selected on B_REST and read once on SLICE: delta **+0.000032**, component
  bootstrap CI **[-0.000004, +0.000064]**. Same-name/same-digit true copies are
  mostly already accepted; precise additions are too rare and the digit veto
  removes more true than false links.
- Leave the rule off. Report:
  `/home/sidvish/Documents/Work/Personal/ext-experiments/G2/report.md`.

## Arena count/source fill — discarded 2026-09-26

- On the full arena, v1 blend ≥ 0.775 plus one owner is **0.972382** (6,784,096 links), matching the published baseline.
- The selected fill `source_ridge_bias1.0_drop0.90_add0.70` was fit on partition A and chosen on B-rest. One SLICE read (100,000 S1s): **0.973387** versus one-owner **0.972948**, delta **+0.000440**, component-bootstrap 95% interval **[+0.000225, +0.000655]**.
- It added 2,819 links (2,195 true) and deleted 0. False links rose from 2,798 to 3,422. Singleton accuracy fell from 0.966720 to 0.960347. Prune-only variants stayed within 0.00005 of the baseline.
- Below the +0.0005 promotion bar and past the 0.2-point singleton allowance. Leave the live decision as blend ≥ 0.775 plus one owner. Report: `/home/sidvish/Documents/Work/Personal/ext-experiments/G1/other_agent_deliverable/report.md`.

## France ownership key — discarded 2026-09-26

- Street + house number + city + distinctive name word. On arena partition A, unique-key hits the matcher already kept are true **460,307 / 464,179 = 0.992**.
- Edits fail the bars: add an unowned unique key **222 / 3,982 = 0.056** (bar 0.90); drop a street-disjoint accepted link **488 / 79,941 = 0.006** actually wrong (bar 0.40); reassign **0 / 5**; shared-key contest flip unmeasured.
- SLICE stays **0.972948**, delta **0**, bootstrap CI **[0, 0]**. Full arena baseline still **0.972382**.
- Priced onto test France with those US/India rates: drop 62,885, add 462, reassign 80, about **−7,135 entity-points (−0.004)**. That is a transfer estimate, not French accuracy. All four switches stay off. No France-only file. Report: `/home/sidvish/Documents/Work/Personal/ext-experiments/G1/report.md`.

## V2 R0 reranker follow-ups — completed 2026-09-26

- Field-tagged BGE at 7,200 steps scored **0.973686** on the same 12,000 medium
  TUNE S1s: precision 0.989378, recall 0.947705, 422 false links.
- Plain `name | address` BGE at 14,400 steps scored **0.974288**: precision
  0.990133, recall 0.948380, 392 false links, singleton accuracy 0.962697,
  India 0.970812 and US 0.976595. This is **+0.005296** over v1 and +0.000602
  over field-7,200, making it the current reranker checkpoint leader. The
  comparison changes both format and training steps, so it does not establish
  that plain formatting is intrinsically better. Exact artifacts are under
  `/workspace/az-ml-new/experiments/r0_{fields7200,pipe14400}_20260926b/`.

## V2 P2–P5 live work — checked 2026-09-25

- **P2 dense complete:** `/workspace/az-ml-new/v2/dense/dense_complete.json`
  seals **106,141,050** dense K50 pairs for **2,122,821** S1s (2,072,821
  eligible TRAIN/TUNE + sealed 50,000 HOLDOUT-2), against all 10,320,219
  targets; no labels read. Six mismatched durable query-vector chunks were
  repaired from hash-verified scratch without re-encoding; final embedding
  SHA256 `a08417f6c467c102d758932376cb73eb45d8b8c5cd6134fda2a63e8776aa83e2`.
- **P2 retrieval screens:** on the *same 12,000 medium TUNE S1s*, full key union
  has **1,437,538** pairs, link recall **0.992743**, oracle F0.5 **0.997707**
  versus saved union 1,243,184 / 0.991802 / 0.997254: **39** true links and
  **194,354** pairs added. Paired oracle gain **0.000453**, 10k entity-bootstrap
  CI **[0.000198, 0.000773]**; 301 links still missed. This is a retrieval
  ceiling, not matcher F0.5. Report: `/workspace/az-ml-new/v2/lexical/medium_tune_union/`.
  Separately, **10,000 eligible TRAIN S1s** against the full target pool yielded
  713,524 raw → 434,328 retained pairs (max 40 negatives/S1); union recall
  **0.988226** / oracle **0.996393**, versus dense **0.984368 / 0.995029**.
  `/workspace/az-ml-new/v2/train_union/screen/ready.json` is complete. The
  **full union is complete** at
  `/workspace/az-ml-new/v2/train_union/full/ready.json`: **89,995,555** retained
  pairs for 2,072,821 eligible S1s, **0.987347** link recall and **0.995965**
  oracle F0.5, with no truth-edge injection. Recompute candidate-dependent
  features before fitting.
- **P3:** a *dense-only compatibility* CatBoost fit on **1,200 medium TRAIN S1s**
  scored **0.937623 / 872 false links** on reused 12,000 medium TUNE; its
  600,000 candidates came from the full target pool, but it is neither a full
  TRAIN fit nor the P2 union matcher. Saved as
  [`dense_compat_final/results.json`](../artifacts/v2_tree_screen/dense_compat_final/results.json)
  and separately ledgered. The bounded 750k-pair local memory probe passed;
  1.5M pairs triggered swap and was stopped. No full-scale P3 tree/CE score yet.
- **P4/P5:** cluster-mate retrieval adds only **3 true links / 21,371 pairs**
  on dense TUNE; oracle gain **0.0000119**, below the **+0.003** gate, so cluster
  scoring stops (`/workspace/az-ml-new/v2/cluster/v2_cluster_screen/screen_complete.json`).
  Apache-licensed ByT5-small at 1,000 steps improves **full normalized-text
  exact hits on 1,024 sampled TUNE true links from 15 to 163**, with zero
  full-exact hits among 1,024 sampled nonlinks; keep for a bounded *feature*
  test, not as a matcher score (`/workspace/az-ml-new/v2/denoiser/screen1000/report.json`).
  P5 self-training has no completed result; a 128-row French text probe used
  no labels and measures no French accuracy. HOLDOUT-2 remains unopened.
- **C7 stopped after the incremental gate:** Cursor's best CPU India
  phonetic/address branch looked like +0.0023 oracle F0.5 versus dense K50, but
  against the already-complete P2 full-key union it adds only **43 true links / 58,911
  pairs**, +0.104 percentage points link recall and **+0.000495** oracle F0.5
  (entity-bootstrap 95% CI [0.000250, 0.000810]). This misses both predeclared
  promotion gates, so do not generate the full-test branch. Exact comparison:
  [`c7_union_incremental.json`](c7_union_incremental.json).

## V2 P0/P1 completed — 2026-09-25

- **P0 runtime:** scratch had **15.09 GiB free** after cleanup. On the saved
  12,000 medium TUNE S1s / 600,000 dense candidates from the full 10,320,219-target
  pool, selective BF16 inference (FP32 rescore within **±0.04** of threshold 0.775)
  exactly reproduced the **fresh FP32 accept set** on the 2,000-pair probe and
  all 12,000 references: **43,511** CE pairs, **1,065** rescored, **34.11s** vs
  **106.53s** full FP32 (3.12×). That fresh FP32 run accepted **one more link**
  than the saved v1 TUNE baseline (0.969008 vs 0.968992); do not count this as
  a new matcher gain. Batch256 reached 1,414–1,474 pairs/s on the fixed 4,000
  pairs, below the 1,700 gate; **retain batch128**. Complete report and arrays:
  `/workspace/az-ml-new/v2/amp_selective/` (scratch `/tmp/az-ml-v2/amp_selective/`).
- **P0 compatibility:** the bounded GPU-kind supervisor smoke completed with
  exit code 0; the local CatBoost/feature-parquet smoke passed on **120 quick
  TRAIN references / 4,474 sampled-pool pairs**. This was a compatibility check,
  not a full-pool matcher score. Reports: `../artifacts/v2_p0_runner/`.
- **P1 fixed-score decision:** on those same 12,000 medium TUNE S1s, the saved
  v1 matcher scores **0.968992 / 494 false links**. Entity-disjoint TUNE
  cross-fitted country calibration + TRAIN count-prior expected-F0.5 prefix,
  **no source bonus**, then pure one-to-one assignment (margin 0) scores
  **0.970099**, precision **0.988538**, recall **0.935722**, **450 false links**,
  singleton accuracy **0.959828**. Paired gain **+0.001107**, 95% TUNE interval
  **[0.000378, 0.001860]**. India **0.964359**, US **0.973909**; fixed dense
  retrieval link recall **0.984208**. Review/config/probability provenance:
  [`v2_p1_decision/decision_review.json`](../artifacts/v2_p1_decision/decision_review.json).
  Source bonus gives a near-tie **0.970222 / 462 false links**, but its isolated
  interval crosses zero; keep the higher-precision no-source candidate. The
  composite-key override adds **zero** links on existing candidates; a new
  key/digit retrieval branch has since passed a TUNE-only ceiling screen above.
- **Boundary / next step:** P0/P1 are complete; later retrieval/text screens
  above are not full-union matcher scores. HOLDOUT-2 is **reserved and
  unopened**: **50,000** S1s, seed **20260929**, sorted-ID SHA256
  `7c98dcaef35d5a392a051e0525f854a2a6ceb55c366616ff2184abd98aa4ae63`.
  Use [`reservation.json`](../artifacts/v2_holdout2/reservation.json) and its
  exclusion file: quick, medium, the older sealed 12,000, and HOLDOUT-2 must
  stay outside new TRAIN/TUNE. No HOLDOUT-2 or medium-eval labels were used
  for P0/P1 selection; France remains unmeasured. Freeze a complete v2
  candidate before its single full-pool holdout check.

## Verified first-submission holdout

The frozen model scored **F0.5 0.968292**, macro **F1 0.953967**, precision
**0.987270**, recall **0.939456** on all **12,001 untouched eval businesses**,
searching all **10,320,219 training targets**. F0.5 95% bootstrap interval:
**[0.966268, 0.970217]**. False links: **506**; singleton accuracy: **0.932650**.
India: 0.962826; US: 0.971919. This is close to TUNE, supporting generalization
on those countries; France remains unmeasured. Exact report:
[first_submission_holdout.json](first_submission_holdout.json).

## Frozen first-submission procedure

The selected **dense** model is `medium_dense_bge_ce1800`: E5-small fine-tuned200,
country-blocked dense K50, numeric CatBoost, and **75% BGE reranker1800**.
Accept blend ≥**0.775** after the original tree's **0.01** reranking gate.
On **12,000 TUNE S1s against all 10,320,219 training targets**, it scores
**macro F0.5 0.968992**, precision **0.987471**, recall **0.938760**, with
**494 false links**. India: 0.963339; US: 0.972744; singleton accuracy: 0.952654.

The controlled 600-step model scored 0.966067 with 531 false links on the same
entities, candidates and tree. The selected 1800 gains 206 true links, removes
37 false links and nine singleton mistakes. TRAIN has 35,999 businesses; no
medium eval labels were used to select either model or its threshold.

Exact preserved paths/digests are in [checkpoints.json](../docs/checkpoints.json).
[frozen_dense_pipeline.json](../docs/frozen_dense_pipeline.json) is the inference
configuration. Models and runtime are frozen at SHA256
`9ac4050a72c64b05c032a17ec26f2a13b339c1b8c135791bf07b737534371f5e`, preserved in
`/workspace/az-ml-new/models/submission_v1_pipeline.json`. CE uses length-sorted
batches of 128, padding to multiples of 8, reference batches of 1,000, maximum length 256. Exact upper-bound
rejection saves 26% of CE calls; all 50 candidates still receive tree scores.
Retrieval keeps query/pool tiles of 512/50,000. All 42 local checks passed. Do not
change models, gate, blend, threshold or source after inspecting the holdout.

## Results whose scopes differ

- Prior **quick-reference union** leader: F0.5 **0.969812**, precision 0.987683,
  97 false links, from lexical/phonetic/E5 candidates and 50% BGE600 at 0.55.
  This used **2,400 different TUNE S1s** against the full training pool. Its score
  is not the dense first-submission score or an apples-to-apples ranking.
- On those quick-reference lexical candidates, BGE600 beat MiniLM300 by 0.006166
  at equal 19,200 pair presentations and 124 false links; paired tune interval
  [+0.002883, +0.009499]. All such intervals condition on development selection.
- Small-pool screening reached 0.986490. The early frozen model fell from 0.981500
  to 0.832719 when its competing pool expanded on the same 1,000 references.
- Repeated tuning can overfit. France has no training labels; US/India results
  do not measure French accuracy. The user-reported public leaderboard F0.5 for
  the first scored submission is **0.95639**; the exact uploaded file is unconfirmed.

## First submission validated and available

The frozen v1 run is complete. The unchanged official validator with `--check-ids`
and our strict stream check passed with no warnings: exactly **1,732,544 test S1
rows**, **5,753,111 predicted links**, and **86,627,200 candidate links**.
`matching_results.tsv` has exactly the original test S1 IDs in the same order,
with no missing, extra or duplicate reference. Both local output hashes match the
verified `ready.json`; no local hidden-test metric has been computed.

Use the exact local files under `output/submission_v1/`; hard links to the two
TSVs also exist directly under `output/`. The **435,776,769-byte**
`Techquilla Shots_submission.zip` contains both output TSVs, source, pinned
requirements, README and filled methodology in the requested tree. Upload the
same `matching_results.tsv` separately to the leaderboard.
[submission_v1_release.json](../docs/submission_v1_release.json) records their
SHA256 hashes, sizes and validation. Model weights and raw data are external to
the ZIP so it fits the 500 MB code-upload field; exact replay needs those weights.

The user reports an earlier upload of the original v1 matching TSV as **FAILED**;
its error reason and whether the attempt consumed a daily slot are unknown. A
later first scored submission returned **public F0.5 0.95639 (user-reported)**.
The exact uploaded bytes, file variant and platform submission ID are not yet
confirmed; do not attach this score to the frozen v1 TSV hash without evidence.
The frozen release files still match the recorded hashes. Train ground truth and
test Source 1 use LF endings, while the original v1 outputs use CRLF; portal
incompatibility is unproven. An independently generated LF-only matching TSV is available at
`output/submission_v1_lf_diagnostic/matching_results.tsv` and passed the official
validator with the original candidate file; it is a diagnostic variant, not a
confirmed scored release. See `submissions.json` for both reported attempts. Keep subsequent
experiments separate from this frozen release; v1 holdout labels are no longer
untouched.

The v1 test predictor saved float32 blended scores for all **86,627,200**
candidate pairs at GPU scratch `/tmp/az-ml-new/output/submission_v1/probabilities.npy`
and the aligned tree scores at `base_probabilities.npy`, ordered by
`scored_pairs.parquet`. All three files were confirmed present on the pod. The
blend array stores zero for off-gate or provably rejected pairs (not a computed
CE score); only 7,068,690 pairs received CE calls. The optional
`cross_probabilities.npy` was not saved. On 26 Sep the three arrays were copied
and hash-verified to `/workspace/az-ml-new/releases/submission_v1_scores/` and
locally to `artifacts/v1_test_scores/`.

## Existing artifacts — inspect before relaunching

Paths are relative to GPU `/tmp/az-ml-new` unless absolute. Live state can change.

| Artifact | Last verified status |
| --- | --- |
| `artifacts/full_train_e5_ft200` | Complete; all 10,320,219 rows. Training vectors preserved durably; scratch symlink. |
| `artifacts/full_dense_medium` | Complete; TRAIN+TUNE only, 2,399,950 pairs; TUNE link recall 0.984208. |
| `artifacts/medium_dense_frozen_for_ce` | Exact original medium tree, SHA256 `b5c7ca0876adbd15ee9cb193098e437d5970025f1afb7ba5eec50bec534063cb`. |
| `artifacts/medium_dense_bge_ce600`, `...ce1800` | Both complete; supervisor 41896 finished; reports local and checkpoints durable. |
| `logs/test_candidates.log`, `artifacts/full_test_e5_ft200` | Complete: all test inputs searched, 86,627,200 candidate pairs. |
| `data/medium_hard` | Complete: 2,987,950 edges; extra union queue 39793 stopped. Do not resume training before first submission. |
| `artifacts/final_holdout_dense_v1`, `output/holdout_v1` | Complete: fixed F0.5 0.968292; labels now evaluated, no further tuning here. |

Medium seed **20260926** excludes all 12,000 quick references (seed 20260925).
Verified quick/medium ID overlap is zero. Split counts and exact ID fingerprints
are in [RUNBOOK.md](../docs/RUNBOOK.md); preserve these saved splits.
Original ZIP, statement files and raw dataset are unchanged, with all ten resource
files verified against original ZIP byte counts/CRC32 in
[source_integrity.json](../docs/source_integrity.json). Keep `/workspace/amlc` unchanged.

The ledger retains **107** historical model configurations, adds **one**
dense-only P3 compatibility model and **4** P1 TUNE-only decision variants:
**112 rows**, not 112 independent models. Retrieval/denoiser screens are not
model rows. See
[EXPERIMENT_DECISIONS.md](EXPERIMENT_DECISIONS.md) for what each established.

## Repository handoff audit

Portable replay, portal ZIP and archival packaging commands are in the code
README. `src/package_portal_submission.py` accepts a new validated release,
checks its ready marker and hashes, and enforces the required tree and 500 MB
cap. The current portal ZIP has the required tree and every archived output byte
was checked against the validated release. A full-size rebuild using the portal
command also passed under the cap. Team name is `Techquilla Shots`;
member names are unspecified. The validated submission ZIP and matching TSV now
exist locally; adding trained weights would exceed the portal code-upload limit.
The pre-portal expanded suite passed 47 tests, including rejection of incomplete
coverage, modified predictions/weights and archive overwrite; it also passed in
a fresh Git checkout without data/weights/credentials. Focused portal packaging
tests pass. Portable replay CLI dry run passed; full clean-machine GPU replay/retraining
has not been performed.

Git excludes binaries/data by design. A new agent needs the supplied data and
authorized GPU/key access; its working credential path, runtime and durable model
registry are in `docs/`. The failed attempt and a separate user-reported **0.95639**
scored attempt are recorded in `submissions.json`; uploaded-file identity and
the earlier portal failure reason remain unconfirmed.
For future experiments, consult the SageMaker migration in `docs/RUNBOOK.md` and
the executable final plan `docs/PLAN_v2.md`; neither changes the frozen v1 score or release.
