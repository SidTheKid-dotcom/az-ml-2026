# Handoff — resume here

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

## First submission validated and available

The frozen v1 run is complete. The unchanged official validator with `--check-ids`
and our strict stream check passed with no warnings: exactly **1,732,544 test S1
rows**, **5,753,111 predicted links**, and **86,627,200 candidate links**.
`matching_results.tsv` has exactly the original test S1 IDs in the same order,
with no missing, extra or duplicate reference. Both local output hashes match the
verified `ready.json`; no hidden-test metric has been computed.

Use the exact local files under `output/submission_v1/`; hard links to the two
TSVs also exist directly under `output/`. The **435,776,769-byte**
`Techquilla Shots_submission.zip` contains both output TSVs, source, pinned
requirements, README and filled methodology in the requested tree. Upload the
same `matching_results.tsv` separately to the leaderboard.
[submission_v1_release.json](../docs/submission_v1_release.json) records their
SHA256 hashes, sizes and validation. Model weights and raw data are external to
the ZIP so it fits the 500 MB code-upload field; exact replay needs those weights.

The user reports uploading the original v1 matching TSV and seeing **FAILED**
on the platform; its error reason and whether the attempt consumed a daily slot
are unknown. No platform F0.5 exists. The frozen release files still match the
recorded hashes. Train ground truth and test Source 1 use LF endings, while the
original v1 outputs use CRLF; portal incompatibility is unproven. An independently
generated LF-only matching TSV is available at
`output/submission_v1_lf_diagnostic/matching_results.tsv` and passed the official
validator with the original candidate file; it is a diagnostic variant, not a
scored release. See `submissions.json` for the reported attempt. Keep subsequent
experiments separate from this frozen release; v1 holdout labels are no longer
untouched.

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

The local ledger records **107** scored configurations (including weights and
hyperparameters, not 107 independent approaches). See
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
registry are in `docs/`. One user-reported platform attempt is recorded as FAILED
in `submissions.json`, with no reported public score or confirmed portal reason.
For future experiments, consult the SageMaker migration in `docs/RUNBOOK.md` and
the proposed `docs/PLAN_v2.md`; neither changes the frozen v1 score or release.
