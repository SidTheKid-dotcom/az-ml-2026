# Experiment learnings

Start with [EXPERIMENT_DECISIONS.md](EXPERIMENT_DECISIONS.md) for what each completed approach established and whether to keep it. [scoreboard.md](scoreboard.md), [experiments.csv](experiments.csv), and [experiments.json](experiments.json) are generated from completed local reports by `python -m ber.scoreboard`.

The highest completed full-pool development score is **0.950115 macro F0.5**, from the residual-name tree blended with 50% multilingual cross-encoder scores. A 25% blend is a higher-precision near-tie: **0.950051**, with **128 false links versus 159**. These results use 2,400 reused tuning references and candidates mined from all 10,320,219 training targets. They do not establish hidden-test performance or absence of overfitting.

The quick experiments use a 201,536-target sampled pool. Their roughly 0.986 scores must not be compared directly with the full-pool scores. On the same 1,000 tuning references, the frozen initial matcher dropped from **0.981500** in the sampled pool to **0.832719** against 9,290,818 targets. That result invalidated sampled-pool precision as a submission-readiness check.

## Validation discipline

- Fit on entity-disjoint TRAIN references; tune operating thresholds on TUNE only.
- Score every selected S1 and every true link, including singletons and unretrieved truth.
- Preserve original full-pool candidate edges in compact datasets; regenerating on retained records changes the experiment.
- Treat the initial quick eval split as development data because it has already been inspected.
- Keep the fresh medium eval labels untouched until the final pipeline is fixed.
- Reused tuning scores and paired tune-bootstrap intervals do not establish generalization.
- France has no training labels; US/India checks cannot estimate French accuracy.

## Detailed evidence

- [Full-pool failure](002_full_pool_failure.md) and [error diagnosis](full_pool_stress_diagnostics.md).
- [Hard-pool cross-encoder](hard_pool_cross_encoder.md).
- [Residual tree plus saved CE scores and near-tie uncertainty](hard_pool_saved_ce_comparison.md).

Full-pool hard-negative mining, tree/numeric/name-feature sweeps, country transfer, and CE comparisons have completed. Full-pool dense retrieval, fresh medium comparisons, and the final untouched holdout have no completed result in this local snapshot. Follow the root handoff/runbook for current remote job state and artifact locations; this folder records measured outcomes only.

No official test score or submission has been produced by these experiments. Failed launches and repeated baseline evaluations do not count as successful model configurations.
