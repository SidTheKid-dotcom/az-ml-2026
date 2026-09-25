# Experiment evidence

[CURRENT_STATE.md](CURRENT_STATE.md) is the single live handoff. Read
[EXPERIMENT_DECISIONS.md](EXPERIMENT_DECISIONS.md) for controlled comparisons,
failures and keep/discard decisions. [scoreboard.md](scoreboard.md) and
`experiments.csv/json` are generated from completed reports; their rows include
blend/threshold choices, not independent model families.

The frozen v1 holdout score is **F0.5 0.968292** on **12,001 unseen businesses**
against **all 10,320,219 training targets**; exact results are in
[first_submission_holdout.json](first_submission_holdout.json). The earlier
0.969812 quick-union result uses a different tuning population and architecture.
Neither is an official test score; France remains unmeasured.

Fit on TRAIN, choose on TUNE, freeze before evaluating a new holdout. Both quick
eval and the now-evaluated medium holdout are unavailable for future unbiased
model selection. Count every reference, singleton and unretrieved true link.
Preserve full-pool candidate edges; remine only as an explicit new experiment.

The older focused reports retain useful failure/near-tie evidence:
`002_full_pool_failure.md`, `003_hard_negative_training.md`,
`full_pool_stress_diagnostics.md`, `hard_pool_cross_encoder.md` and
`hard_pool_saved_ce_comparison.md`. Their dated scores are historical, not current
winners. Reused tuning scores cannot establish absence of overfitting or bias.

After an experiment, save its config/report under `artifacts/<unique-name>/`,
add one decision-table row, then run `python -m ber.scoreboard`. Fetch remote
reports first; never regenerate a smaller ledger from an incomplete checkout.
Record actual platform submissions in `submissions.json` (currently none).
