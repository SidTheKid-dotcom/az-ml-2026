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

The official [statement](problem-statement/statement.md),
[evaluation PDF](problem-statement/ps.pdf), and
[guidelines](problem-statement/guidelines.md) are retained verbatim.
[Data observations](problem-statement/task.md) and
[cached model inventory](problem-statement/models.yaml) came from a previous
controller project: their data facts are useful, but their controller paths and
commands are **not this project's active runtime**. Use the runbook below.

## Repository boundaries

- `code/business_entity_resolution/`: experiment, training, retrieval, scoring,
  frozen prediction, tests, and queued-run scripts.
- `learnings/`: compact conclusions, score ledger, failures, and handoff state.
- `docs/RUNBOOK.md`: connection, environment, file map, evaluation and release steps.
- `data/`, `artifacts/`, `output/`, `.venv/`, and the supplied ZIP are intentionally
  excluded from Git. Their exact local/GPU locations are documented.
- Credentials and private-key contents must stay outside the repository.

[Original-file integrity](docs/source_integrity.json) records the verified ZIP
contents and all problem-statement files. Originals remain separate from experiments.

Small-pool scores around 0.986 were optimistic. Full-pool hard-negative training
is now the decision basis. No official test score or submission-ready test TSV
exists yet; development TSVs must not be uploaded as test predictions.

Regenerate the compact ledger after fetching completed reports:

```bash
PYTHONPATH=code/business_entity_resolution/src .venv/bin/python -m ber.scoreboard
```
