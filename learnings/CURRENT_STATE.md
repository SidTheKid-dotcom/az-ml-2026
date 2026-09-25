# Handoff — resume here

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
  **full union is running** (`ber.v2_train_union full --follow`); at inspection
  `full/raw_progress.json` was merging at 490,000/2,122,821 S1s, with no truth
  read or final union marker. Recompute candidate-dependent features before fitting.
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
