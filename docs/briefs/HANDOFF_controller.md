# Handoff prompt — take over as controller (written 26 Sep ~03:30 IST)

You are taking over as controller of team Techquilla Shots' Amazon ML Challenge
2026 entity-resolution effort. Deadline **27 Sep 23:59 IST; the last upload
counts** (public board = subset of test). Uploads left: 4 on 26 Sep (U5 is
about to use one), 5 on 27 Sep. Goal: beat the leaders (0.988; ranks 4–14 at
0.984–0.985) without overfitting or using anything but the supplied data.

## Read first (in order)
`/home/sidvish/Documents/Work/Personal/az-ml-new/`: `AGENTS.md`, `CONTEXT.md`,
`docs/PLAN_v2.md` (the executable final plan; live checklist §17),
`learnings/negative_side_analysis_manual.md`, `docs/briefs/`.

## Repo and access
- Repo: `/home/sidvish/Documents/Work/Personal/az-ml-new`, branch
  `codex/experiment-loop` (single integration branch), latest `148763a`.
  No git identity is configured: commit with
  `git -c user.name="Claude" -c user.email="noreply@anthropic.com" commit`,
  message ending `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  `.claude/` (agent worktrees) is ignored.
- Local Python: `.venv/bin/python` (no torch). Local data:
  `data/raw/dataset/{train,test}/`; v1 test scores `artifacts/v1_test_scores/`.
- Pod: `ssh -F /dev/null -o BatchMode=yes -o IdentitiesOnly=yes -i /home/sidvish/amlc-private/ssh/id_ed25519 -p 22141 root@69.30.85.107`
  (scp uses `-P 22141`). A40 46 GB; cgroup 7.65 cores / 50 GB; `/tmp` ~6 GB free.
  Runtime: `/opt/amlc-venv/bin/python`, env
  `PYTHONPATH=/opt/az-ml-new-extra:<code>/src HF_HOME=/workspace/amlc/cache/hf HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2`.
  Launch every job detached: `nohup setsid bash -c '…' > log 2>&1 < /dev/null &`
  (an SSH exit killed an earlier run).
- Pod layout: `/tmp/az-ml-new` = **frozen v1 code/data, never edit**;
  `/tmp/az-ml-v3/code` → exported integration commit (`git archive` → scp →
  extract to `/tmp/az-ml-v3/code-<commit>`, symlink `code`);
  `/tmp/az-ml-v3/{arena_data,arena_v1,logs}`; durable `/workspace/az-ml-new/`
  (`arena/`, `experiments/`, `models/`, `releases/submission_v1_scores/`).

## Scores and evidence
- Best public: **0.964563** (#3 = v1 + one owner per record).
- **U5** = #3 + name-only rule: `~/Downloads/u5-name-only/matching_results.tsv`
  (sha256 `2b61949c…`), validated, predicted +0.0035 to +0.0042. The user
  uploads; record the five-decimal score in `learnings/submissions.json`.
- Measured facts: empty-address records are ~98% owned; v1 accepts only 25% of
  their true links; name-only rule +0.003962 on the 12k tuning set. France holds
  74.5% of test contests (templated names; street name decides). Decision rules
  chosen on 12k samples mislead: contests only appear when all businesses
  compete (the arena). The expected-F0.5 decoder loses on raw pair
  probabilities (−0.0015): it needs list-aware calibration (resolver) first.
  Field-format BGE, 7,200 steps, same data: **0.973686** vs v1 0.968992 on
  medium TUNE (422 vs 494 false links).

## Running right now
1. **Arena scoring** (frozen v1 code, all 2,072,821 eligible train S1s ×
   top-50 = 103,641,050 pairs; HOLDOUT-2 excluded): log
   `/tmp/az-ml-v3/logs/arena_v1_score.log`, output `/tmp/az-ml-v3/arena_v1/`
   (`probabilities.npy` blend, `base_probabilities.npy` tree, aligned to
   `scored_pairs.parquet` = `arena_data/pairs.parquet`). ETA ~06:00 IST. Then
   copy to `/workspace/az-ml-new/arena/` and compact arrays to local
   `artifacts/arena/`.
2. **Arena labels** (no HOLDOUT-2 rows): log `/tmp/az-ml-v3/logs/arena_labels.log`
   → `/workspace/az-ml-new/arena/labels/{label.npy,true_count.npy,rec_owner.npy,labels_summary.json}`.
3. **R0 plain 14,400-step BGE** (legacy trainer, medium data): log
   `/tmp/az-ml-v2/logs_r0/`, output
   `/workspace/az-ml-new/experiments/r0_pipe14400_20260926b/results.json`
   (`selected_by_tune.tune.macro_f05`). Compare with field-format 0.973686 to
   choose the text format for C2.
4. **Subagents of the previous session** (may stop when it ends; their work is
   committed on branches `worktree-agent-*`, worktrees in `.claude/worktrees/`):
   agent-D → `src/ber/resolver.py` (C4 list-aware resolver; merged earlier:
   `arena_decision.py`, `name_only.py`); agent-F → `forensic_features.py`,
   `france_pseudo_labels.py`, `label_free_diagnostics.py`, outputs
   `output/france_pseudo/`; agent-T → `fast_trainer.py` (C2) and `claimnet.py`
   (C3). Check each branch: merge what is finished and tested; finish the rest
   from the specs in `docs/PLAN_v2.md` §7 and §13.2.
5. **External CPU agents run by the user** (Cursor/Grok, opencode GPT-6
   sol/astra) on briefs `docs/briefs/{A,B,C}_*.md`; reports land in
   `/home/sidvish/Documents/Work/Personal/ext-experiments/{A,B,C}/`.

## Next steps (in order)
1. Record U5's score.
2. When the arena lands: partitions by claim-graph component
   (`arena_decision.claim_graph_components`, edges where both blends ≥ 0.05;
   A ≈ 70% train, B ≈ 30%, SLICE = 100k-S1 subset of B) → `s1_partition.npy`;
   fidelity check against test (contest rate, list length, empty rate).
3. C4 resolver on B out-of-fold → apply to v1 test scores → upload if it beats
   the best on the arena and prices positive.
4. Deploy `fast_trainer.py`: speed bench, then parity (reproduce medium TUNE
   within ±0.001 of 0.968992), then the C2 screen on arena A (field format,
   canonical view, stake weights, arena negatives); evaluate on SLICE and
   medium TUNE.
5. ClaimNet screen as a resolver input; idle-GPU bets per §8.
6. Final trainings; **final test inference starts by 27 Sep 10:00 IST**; freeze
   13:00; HOLDOUT-2 once; final file uploaded last (keep a slot); ZIP, README,
   methodology by 20:00.

## Rules
Never read HOLDOUT-2 labels (`az-ml-v2/artifacts/v2_holdout2/reserved_ids.txt`);
no external data; never edit frozen v1 code or releases; LF outputs; official
validator with `--check-ids` plus strict checks (row order, subset of
candidates, zero contested IDs); decisions on the arena, not on 12k samples;
promote only when the component-bootstrap interval is above zero; pseudo-labels
are disclosed in the methodology; price every upload before it goes out.
