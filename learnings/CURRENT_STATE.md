# Handoff — resume here

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

## Remaining release work

1. Frozen runtime is complete. Track `logs/frozen_holdout_v1.log`; the one fixed
   holdout job is running. Do not launch a duplicate or tune from its result.
2. Retrieve candidates for the **12,001 sealed medium eval S1s**, apply the frozen
   model once, then score committed predictions with complete truth, including
   missed links and empty singletons. Commands and guard limits are in the runbook.
3. Finish label-free official test encoding/retrieval and predict **all 1,732,544
   test S1s** against **9,969,589 test targets**. Verify ID/coverage/subset integrity
   and run the supplied validator with **`--check-ids`** before the user uploads.

Use `run_frozen_holdout.sh` followed by `run_first_submission.sh`, each with the
explicit finalized pipeline SHA256. No further model search is scheduled.
No official-test prediction files have been validated yet. Development TSVs are
not submission files. Notify the user when the first complete validated pair is ready.

## Existing artifacts — inspect before relaunching

Paths are relative to GPU `/tmp/az-ml-new` unless absolute. Live state can change.

| Artifact | Last verified status |
| --- | --- |
| `artifacts/full_train_e5_ft200` | Complete; all 10,320,219 rows. Training vectors preserved durably; scratch symlink. |
| `artifacts/full_dense_medium` | Complete; TRAIN+TUNE only, 2,399,950 pairs; TUNE link recall 0.984208. |
| `artifacts/medium_dense_frozen_for_ce` | Exact original medium tree, SHA256 `b5c7ca0876adbd15ee9cb193098e437d5970025f1afb7ba5eec50bec534063cb`. |
| `artifacts/medium_dense_bge_ce600`, `...ce1800` | Both complete; supervisor 41896 finished; reports local and checkpoints durable. |
| `logs/test_candidates.log`, `artifacts/full_test_e5_ft200` | All 9,969,589 target embeddings complete; test S1 encoding/retrieval running. |
| `data/medium_hard` | Complete: 2,987,950 edges; extra union queue 39793 stopped. Do not resume training before first submission. |
| `artifacts/final_holdout_dense_v1`, `output/holdout_v1` | Fixed holdout pipeline started; check log and completion markers. |

Medium seed **20260926** excludes all 12,000 quick references (seed 20260925).
Verified quick/medium ID overlap is zero. Split counts and exact ID fingerprints
are in [RUNBOOK.md](../docs/RUNBOOK.md); preserve these saved splits.
Original ZIP, statement files and raw dataset are unchanged, with all ten resource
files verified against original ZIP byte counts/CRC32 in
[source_integrity.json](../docs/source_integrity.json). Keep `/workspace/amlc` unchanged.

The local ledger records **107** scored configurations (including weights and
hyperparameters, not 107 independent approaches). See
[EXPERIMENT_DECISIONS.md](EXPERIMENT_DECISIONS.md) for what each established.
