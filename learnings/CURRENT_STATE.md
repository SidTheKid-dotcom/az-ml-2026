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

## Local CPU experiments while v1 inference runs

In isolated worktree `/home/sidvish/Documents/Work/Personal/az-ml-cpu-decision`
on `codex/cpu-tune-decision-20260925`, completed TRAIN/TUNE inputs were copied
selectively and used locally with CUDA hidden and two CPU threads. The frozen v1
source, weights, inference job and threshold were not changed.

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
  v1 release. Individual shortlist pairs were selected without TUNE labels; all
  their CE scores are still missing. The text/pairs remain outside Git; the
  [identity/count manifest](../artifacts/cpu_lexical_rescue_20260925/ce_shortlist_manifest.json)
  preserves the exact queue and checkpoint identities.
- A new **sealed confirmation holdout** was reserved by ID only from unused
  training S1s: 12,000 references, seed **20260928**, 4,787 India / 7,213 US,
  sorted-ID SHA256 `ec057370b8dc8b3dbef24e277a6920f905a08481ca2b35a5245dcdf9d22f5187`.
  All quick and medium S1s, including consumed eval, are excluded. Neither its
  truth nor the v1 eval errors were opened. [Reservation recipe/provenance](../artifacts/holdout_reserve_20260928/reservation.json);
  the ID list stays outside Git in that worktree's ignored `artifacts/` directory.

Strongest next bounded comparison after v1 release: score all 321 shortlisted
CAL/TUNE pairs with the **frozen BGE checkpoint**, choose an add-link decision
on TRAIN-CAL only, then measure the fixed rule's real macro F0.5/false links on
all 12,000 medium TUNE S1s. If promising, recompute candidate-dependent features
and train/score the fuller medium union matcher; only then freeze a finalist for
the reserved confirmation holdout. France remains unmeasured.

## Remaining release work

1. Holdout is complete and must not be rerun for model selection. Test candidates
   are also complete: **86,627,200 edges**, every S1 has its actual 50 candidates.
2. Full test scoring is running: supervisor **46635**,
   `logs/first_submission_v1.log`, `output/submission_v1`. Do not duplicate it.
   Last read-only check: supervisor still active; durable `ready.json` absent.
3. Its wrapper automatically runs both file validators and preserves verified
   release copies. Download final TSVs only after `ready.json` confirms success.

Reproduction uses `run_frozen_holdout.sh` then `run_first_submission.sh`, each
with the explicit finalized pipeline SHA256; both have already been launched. No further model search is scheduled before v1 release. After recording the first
platform result, resume bounded TRAIN/TUNE comparisons in separate outputs; the
evaluated v1 holdout must not steer new thresholds or training choices.
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

Portable replay and final ZIP packaging commands are in the code README. The ZIP
builder verifies complete release reports, exact checkpoint/source/output hashes,
required structure and every archived byte. Team name is `techquilla-shots`;
member names are unspecified. It requires a validated release; no production ZIP exists while inference is incomplete.
The expanded suite passes 47 tests, including rejection of incomplete coverage,
modified predictions/weights and archive overwrite. All 47 also pass in a fresh
Git checkout using the existing CPU environment, with no data/weights/credentials
copied. Portable replay CLI dry run passed; full clean-machine GPU replay/retraining
has not been performed.

Git excludes binaries/data by design. A new agent needs the supplied data and
authorized GPU/key access; its working credential path, runtime and durable model
registry are in `docs/`. No official submission is recorded yet.
