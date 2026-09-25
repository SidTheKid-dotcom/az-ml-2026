# Handoff — resume here

## Verified result and release state

- Best completed full-pool development matcher: **F0.5 0.969812**, fine-tuned E5
  dense + lexical/phonetic candidate union, numeric-conflict CatBoost blended with
  **50% BGE reranker600**, threshold 0.55; **precision 0.987683**, recall 0.934743,
  **97 false links**. Configuration: `artifacts/hard_union_bge_ce600/results.json`.
  Scope: **2,400 tuning S1 businesses against 10,320,219 original training targets**.
- Its union tree alone scored **0.956008**, 85 false links; the BGE blend recovers
  306 more true links at a cost of 12 additional false links.
- On the larger disjoint sample, dense tree training on 35,999 S1 scores
  **0.957134** on 12,000 TUNE S1 (precision 0.988471, 437 false links).
  This is another development population, not an apples-to-apples model ranking.
- Dense-only tree reaches **0.954765**, precision 0.984243, recall 0.908304,
  121 false links. Union's incremental value needs independent confirmation.
- On fixed lexical/phonetic candidates, BGE reranker600 + tree reaches **0.954969**
  versus MiniLM300 + tree **0.948804** at the same 19,200 training pair presentations.
  Both make 124 false links. Paired tune difference is +0.006166, 95% interval
  [+0.002883, +0.009499], conditional on development selection.
- Earlier residual-tree/MiniLM blend scored 0.950115; its 25% blend was a precision
  near-tie at 0.950051. Preserve its original 0.01 gate if reproducing that pipeline.
- Small-pool best 0.986490 is an optimistic screening score, not the best credible
  whole-pool estimate. Repeated tuning can overfit; final holdout has not been scored.
- **No official test prediction files or public score exist yet.** Never upload
  development outputs. Test labels are hidden, so no entire-test F0.5 is available.

## Running pipeline — check live files before relaunching

All paths below are relative to GPU `/tmp/az-ml-new`. PIDs can become stale.

| Stage | Output / status | State at handoff |
| --- | --- | --- |
| Fine-tuned E5 full training embeddings | `artifacts/full_train_e5_ft200/complete.json` | Complete, all 10,320,219 rows verified |
| Full dense quick retrieval | `artifacts/full_dense_quick/complete.json` | Complete: tune link recall 0.982334, oracle F0.5 0.993129 |
| Dense quick matcher sweep | `artifacts/hard_dense_sweep`, `logs/hard_dense_sweep.log` | Complete, 12 variants; best 0.954765 |
| Quick candidate union + tree sweep | `data/hard_union`, `artifacts/hard_union_sweep` | Complete, 12 variants; best 0.956008 |
| Fresh medium dense mining | `artifacts/full_dense_medium/complete.json` | Complete, 2,399,950 pairs; tune recall 0.984208 |
| Fresh medium lexical/phonetic mining | `data/medium_hard`, `logs/medium_hard.log` | Running, 47,999 TRAIN+TUNE refs; no eval |
| Larger dense / union tree + residual sweeps | `artifacts/medium_*sweep`, `logs/medium_*sweep.log` | Dense tree complete (0.957134), residual running; union waits for lexical mining |
| BGE-M3 reranker comparisons | `artifacts/hard_bge_ce600`, `artifacts/hard_union_bge_ce600` | Complete, best blends 0.954969 / 0.969812 |
| Larger dense BGE600 / BGE1800 | `artifacts/medium_dense_bge_ce{600,1800}`, supervisor PID 41896 | Two serial training budgets; TRAIN/TUNE only, frozen tree in `medium_dense_frozen_for_ce` |
| Official test E5 encoding and dense candidates | `logs/test_candidates.log`, `artifacts/full_test_e5_ft200`, `artifacts/full_dense_test` | Running label-free preparation via `run_test_candidates.sh` |

The medium sample excludes every quick reference: 35,999 train, 12,000 tune,
12,001 **sealed eval**. Never fit or choose a threshold with its eval labels.
Dense-only validation cannot validate a lexical/dense union: candidate generation
and feature semantics must match training. Full-pool lexical indexing is the CPU
bottleneck; reuse a constructed index across all needed development queries.
The first release can use the dense pipeline after its own validation; do not
claim the union's score for dense-only inference. Full-test lexical search would
be substantially slower. Dense-only BGE is being measured before making that tradeoff.

## Next actions

1. Collect full-pool dense/union scores and larger-data results; append measured
   keep/discard decisions and refresh `ber.scoreboard`. Favor useful diversity over
   more nearly identical settings. Preserve every promoted checkpoint durably.
2. Freeze one leader plus a justified precision alternative. Prepare full-pool
   candidates for the 12,001 final eval S1; score once with fixed thresholds and
   report per-country results, false links, singleton accuracy and paired uncertainty.
3. Generate all official test candidates/predictions with the chosen pipeline;
   validate coverage/IDs/subsets and create the first versioned submission. An early
   valid public score is more useful than endless local tuning; keep a separate
   holdout discipline and do not optimize repeatedly against public feedback.

## Bounded model ideas, with reasons

- **BGE reranker v2 M3** improved the controlled hard-pool comparison; test its
  interaction with dense/union candidates and larger TRAIN next. Multilingual,
  Apache-2.0, already cached.
  [Official model card](https://huggingface.co/BAAI/bge-reranker-v2-m3).
- If India remains the main weakness, test a **MuRIL pair-classifier expert** on
  India only; it was pretrained on Indian languages and transliteration. Do not
  use it as a universal French matcher. [Official model card](https://huggingface.co/google/muril-base-cased).
- Sequence-pair fine-tuning and field-focused inputs have precedent in
  [Ditto](https://arxiv.org/abs/2004.00584); those benchmark results do not predict
  this competition's score. Keep the experiment small and use only supplied labels.

See `docs/RUNBOOK.md` for SSH, environments, data, storage and exact scoring;
`AGENTS.md` for the experiment loop; `EXPERIMENT_DECISIONS.md` for completed evidence.
