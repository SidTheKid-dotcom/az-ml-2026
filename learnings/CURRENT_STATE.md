# Handoff — resume here

## First validated submission is ready

`/workspace/az-ml-new/releases/submission_v1/ready.json` now confirms the frozen
v1 files are complete and durably copied. The unchanged official validator
(`--check-ids`) passed without warnings; the strict gate passed all **1,732,544**
test S1s and **86,627,200** actual candidate links. `matching_results.tsv`
SHA256: `5df9ddbf42137d4e403f0072fe85169aaabc0167a0b431f675f926754371e1ed`;
`candidate_pairs.tsv` SHA256:
`fc0ec7a43888208b25fef4deecea352a98dcce75797ea9288c0f7fb9e6ebca34`.
Pipeline SHA256 remains `9ac4050a72c64b05c032a17ec26f2a13b339c1b8c135791bf07b737534371f5e`.
[Ready](../artifacts/submission_v1_readiness_20260925/ready.json) and
[validation](../artifacts/submission_v1_readiness_20260925/validation.json)
manifests are copied locally; the release is preserved unchanged. The prior supervisor
**46635** exited and the A40 was free at the last check. **No platform upload,
public score or final ZIP is recorded yet**; format validation is not an accuracy
measurement. Upload only the matching TSV for a leaderboard measurement and
record the actual result in [submissions.json](submissions.json).

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
  do not measure French accuracy. **No official score is available.**

## Separate TRAIN/TUNE experiments

In isolated worktree `/home/sidvish/Documents/Work/Personal/az-ml-cpu-decision`
on `codex/cpu-tune-decision-20260925`, completed TRAIN/TUNE inputs were copied
selectively and used locally with CUDA hidden and two CPU threads. The frozen v1
source, weights and threshold were not changed. After release readiness, the
shortlisted-pair reranker ran in separate GPU scratch/durable directories.

- [Saved-score decision report](../artifacts/cpu_decision_v1_tune_20260925/results.json):
  independently reproduced the **0.968992** medium TUNE baseline, 494 false links,
  all 12,000 S1s/600,000 TUNE pairs aligned. Pair-level tree floors 0.25/0.50
  reduced F0.5 to **0.967733/0.964447**. A CE-conditional veto reached **0.968184**,
  precision **0.992738** and 280 false links, but lost 659 true links; no measured
  F0.5 gain. Keep v1's decision rule. [Configuration](../artifacts/cpu_decision_v1_tune_20260925/experiment_config.json).
- [Same-reference retrieval screen](../artifacts/cpu_medium_union_screen_20260925/results.json):
  dense top50 plus completed lexical/phonetic candidates recovers **315** of the
  dense branch's 655 missed true links on the same 12,000 TUNE S1s against all
  **10,320,219** targets. Recall 0.984208 → **0.991802**, **oracle** F0.5
  0.994090 → **0.997254** at the cost of **643,184** added pairs. Top-20 lexical
  rank plus phonetic hits recovers 152 with 177,515 added pairs (oracle 0.995513).
  No union-only pair has a valid saved tree/CE score; the oracle gain is **not**
  a matcher improvement. [Configuration](../artifacts/cpu_medium_union_screen_20260925/experiment_config.json).
- [CPU lexical-only matcher screen](../artifacts/cpu_lexical_rescue_20260925/results.json):
  scored all **177,515** extra top-20 TUNE candidates with a new TRAIN-fitted
  CatBoost, using a disjoint TRAIN calibration split (28,763 FIT / 7,236 CAL S1s).
  Just 91 of 107,189 CAL edges were true; even at score 0.97 it found 6 true and
  2 false links. Its fixed CAL `TP−5×FP` rule abstains, giving **no TUNE F0.5 gain**
  over v1's **0.968992**, 494 false links. Pair features took 84s; total CPU run
  100s. [Configuration](../artifacts/cpu_lexical_rescue_20260925/experiment_config.json).
- A 0.75 lexical-only CatBoost-probability gate chosen from TRAIN-CAL has frozen
  **118 CAL + 203 TUNE** candidate/text pairs for a small BGE experiment after
  v1 release. Individual shortlist pairs were selected without TUNE labels and
  their CE scores were computed separately as described below. Text/pairs stay
  outside Git; the
  [identity/count manifest](../artifacts/cpu_lexical_rescue_20260925/ce_shortlist_manifest.json)
  preserves the exact queue and checkpoint identities.
- [BGE shortlist gate 0.75](../artifacts/ce_shortlist_bge_v1_20260925/results.json):
  118 CAL + 203 TUNE extra pairs received valid frozen-BGE scores. TRAIN-CAL
  selected CE ≥**0.975** (14 true / 1 false there). Adding 27 true and 3 false
  links on all 12,000 medium TUNE S1s gives **F0.5 0.969245**, precision 0.987405,
  recall 0.939411, 497 false links; singleton accuracy remains 0.952654.
  Paired difference from v1 **+0.000253** [0.000093, 0.000479], conditional on
  reused TUNE choices. GPU scoring took 49s, including checkpoint verification.
- [BGE shortlist gate 0.50](../artifacts/ce_shortlist_bge_gate050_20260925/results.json):
  same records/checkpoint/runtime, only a wider lexical gate (352 CAL + 607 TUNE
  pairs). The **same frozen CE threshold 0.975** adds 37 true and 5 false links:
  **F0.5 0.969317**, precision 0.987358, recall 0.939652, 499 false links;
  singleton accuracy 0.952654. Versus the 0.75 gate, +0.000072 F0.5 has a paired
  interval **[-0.000003, 0.000151]**; the narrower gate is a higher-precision
  near-tie. GPU scoring took 46s. Both probability vectors and complete shortlist
  pairs are preserved durably under `/workspace/az-ml-new/artifacts/lexical_ce_shortlist_gate{075,050}_20260925`.
- A new **sealed confirmation holdout** was reserved by ID only from unused
  training S1s: 12,000 references, seed **20260928**, 4,787 India / 7,213 US,
  sorted-ID SHA256 `ec057370b8dc8b3dbef24e277a6920f905a08481ca2b35a5245dcdf9d22f5187`.
  All quick and medium S1s, including consumed eval, are excluded. Neither its
  truth nor the v1 eval errors were opened. [Reservation recipe/provenance](../artifacts/holdout_reserve_20260928/reservation.json);
  the ID list stays outside Git in that worktree's ignored `artifacts/` directory.

Next bounded hypothesis: the complementary lexical links beyond rank 20 may
recover more of the **315** dense-missed truths. Extend one rank cap while keeping
the full 10.32M competing target pool, regenerate all candidate-dependent
features, score all newly eligible pairs (no surrogate CE scores), and compare
complete medium TUNE F0.5, false links, country/script breakdown and cost against
the same v1 baseline. Preserve both 0.50 and higher-precision 0.75 near-ties;
freeze finalists before opening the reserved confirmation holdout. France remains
unmeasured; these small TUNE gains are not an official-test improvement.

## Remaining release work

1. Both official test TSVs and durable hashes are ready; the first platform upload
   and public score remain to be recorded. The leaderboard accepts only the
   matching TSV; its score covers a subset of hidden test labels.
2. Build the verified final ZIP from the ready release and exact promoted weights
   using the package command in the code README; no ZIP has been built yet.
3. Keep v1's source, weights, gate and threshold frozen. Its holdout has already
   been consumed; the new 12,000-S1 confirmation holdout remains sealed for a
   later frozen procedure. Do not rerun v1 inference or change its validated TSVs.

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
| `output/submission_v1`, `/workspace/az-ml-new/releases/submission_v1` | Complete and `ready.json` verified; official/strict validators PASS, no platform score. |
| `/tmp/az-ml-new/experiments/lexical_ce_shortlist*`, `/workspace/az-ml-new/artifacts/lexical_ce_shortlist_gate{075,050}_20260925` | Both label-free shortlist GPU scores complete, source/weights/probability hashes preserved; full TUNE reports local. |

Medium seed **20260926** excludes all 12,000 quick references (seed 20260925).
Verified quick/medium ID overlap is zero. Split counts and exact ID fingerprints
are in [RUNBOOK.md](../docs/RUNBOOK.md); preserve these saved splits.
Original ZIP, statement files and raw dataset are unchanged, with all ten resource
files verified against original ZIP byte counts/CRC32 in
[source_integrity.json](../docs/source_integrity.json). Keep `/workspace/amlc` unchanged.

The complete ledger records **109** scored configurations (including weights and
hyperparameters, not 109 independent approaches). See
[EXPERIMENT_DECISIONS.md](EXPERIMENT_DECISIONS.md) for what each established.

## Repository handoff audit

Portable replay and final ZIP packaging commands are in the code README. The ZIP
builder verifies complete release reports, exact checkpoint/source/output hashes,
required structure and every archived byte. Team name is `techquilla-shots`;
member names are unspecified. A production ZIP has not yet been built from the
now-ready release. The expanded suite passes 50 tests in this development worktree;
the earlier 47 release tests also passed in a fresh Git checkout using the
existing CPU environment. Portable replay CLI dry run passed; full clean-machine
GPU replay/retraining has not been performed.

Git excludes binaries/data by design. A new agent needs the supplied data and
authorized GPU/key access; its working credential path, runtime and durable model
registry are in `docs/`. No official submission is recorded yet.
