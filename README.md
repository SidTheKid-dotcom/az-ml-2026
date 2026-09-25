# Amazon ML Challenge 2026 — entity resolution

Objective: maximize the official mean per-business **F0.5**, emphasizing correct
matches and correct abstention. Compete for first place; no local score establishes
a rank or guarantees the hidden-test result.

## Start here

1. [Current results and running work](learnings/CURRENT_STATE.md).
2. [Experiment loop](AGENTS.md): how to make the next useful experiment.
3. [GPU, data and scoring runbook](docs/RUNBOOK.md).
4. [What each experiment established](learnings/EXPERIMENT_DECISIONS.md).
5. [Full score ledger](learnings/scoreboard.md), with machine-readable CSV/JSON.
6. [Proposed v2 experiments](docs/PLAN_v2.md) and the [SageMaker migration
   workflow](docs/RUNBOOK.md#v2-compute-migration-sagemaker-ai) for work after v1.

The official [statement](problem-statement/statement.md),
[evaluation PDF](problem-statement/ps.pdf), and
[guidelines](problem-statement/guidelines.md) are retained verbatim.
[Data observations](problem-statement/task.md) and
[cached model inventory](problem-statement/models.yaml) came from a previous
controller project: their data facts are useful, but their controller paths and
commands are **not this project's active runtime**. Use the runbook below.

## Repository boundaries

- `code/business_entity_resolution/`: experiment/training modules, frozen inference,
  strict validation, portable replay, final ZIP builder and tests. Its
  [README](code/business_entity_resolution/README.md) gives executable commands.
- `learnings/`: compact conclusions, score ledger, failures, and handoff state.
- `docs/RUNBOOK.md`: connection, environment, file map, evaluation and release steps.
- `data/`, `artifacts/`, `output/`, `models/`, `releases/`, `.venv/`, and the supplied ZIP are intentionally
  excluded from Git. Their exact local/GPU locations are documented.
- Credentials and private-key contents must stay outside the repository.

[Original-file integrity](docs/source_integrity.json) records the verified ZIP
contents and all problem-statement files. Originals remain separate from experiments.

Small-pool scores around 0.986 were optimistic. Full-pool hard-negative training
is now the decision basis. The first full-test matching and candidate TSVs are complete and passed the
unchanged official validator plus strict coverage/subset checks. Exact release
hashes and local paths are in
[submission_v1_release.json](docs/submission_v1_release.json). A user-reported
first scored submission received public F0.5 **0.95639**; the exact uploaded
file has not been identified. An earlier upload was reported **FAILED**. See
[submission attempts](learnings/submissions.json); the frozen release hashes
remain the reference for investigating the upload identity and failure.
Development TSVs must not be uploaded as test predictions.

The frozen first matcher uses **35,999 labeled training businesses**, with 12,000
separate tuning businesses and a 12,001-business holdout. Its verified holdout
F0.5 is **0.968292**; full official-test format and integrity validation passed. E5 was fine-tuned on
the earlier quick TRAIN split. Searching all 10,320,219 training targets does
not mean fitting on every labeled training business. Official inference covers
all 1,732,544 test businesses and all 9,969,589 test targets; labels remain hidden.

Regenerate the compact ledger after fetching completed reports:

```bash
PYTHONPATH=code/business_entity_resolution/src .venv/bin/python -m ber.scoreboard
```

A fresh agent with this checkout, supplied data and authorized GPU access can
follow the loop through experiments, fixed-model replay, validation and final
packaging. Git alone cannot supply private SSH credentials or model/data binaries.
The 500 MB submission ZIP includes both output TSVs, source, pinned dependencies
and methodology. Model weights are external; the private Git repository stays
small. Use `learnings/submissions.json` for actual platform uploads/scores.

The current architecture is replaceable. Inventory entries and prior results are
evidence, not an organizer model mandate or proof of universal superiority.
Model selection bias and unseen-France performance remain explicit limitations.
