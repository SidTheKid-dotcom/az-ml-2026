# C2 on the H100 (26 Sep 2026) — results and where to improve

**Run:** BGE reranker v2-m3 from the public pretrained checkpoint, trained on
arena partition A (1.45M training S1s). Plain `name | address` text, batch 128,
bf16, `torch.compile`, 40,000 steps, constant LR 2e-5. The run is live on the
H100 pod (`/opt/az-ml-new/c2_plain/train.log`).

## Measured on arena B_REST

B_REST is 521,846 S1s, all competing; SLICE and HOLDOUT-2 are unread. The
decision is blend p = w·C2 + (1−w)·tree on tree-gated pairs, then one owner per
record and a cap of 11 per list. (w, t) was cross-fitted on component halves.

| Model | Macro F0.5 | Precision | Recall | False links | No-match merges |
| --- | ---: | ---: | ---: | ---: | ---: |
| v1 (0.775) | 0.972465 | 0.99146 | 0.93776 | 14,599 | 3.74% |
| C2 5k (w 0.9, t 0.10) | 0.977867 | 0.99619 | 0.94135 | 6,510 | 1.50% |
| C2 10k (w 0.9, t 0.10) | 0.978722 | 0.99538 | 0.94603 | 7,936 | 1.95% |
| v1 + name-only | 0.975373 | 0.99203 | 0.94532 | 13,719 | 3.38% |
| C2 5k + name-only | **0.981331** | 0.99596 | 0.95275 | 6,976 | 1.57% |

- C2 5k + name-only vs v1 + name-only: **+0.005958**, CI [+0.005765, +0.006154].
- C2 probabilities are calibrated low: the best threshold is at or below 0.10.
  The grid now reaches 0.02.
- Gains per step shrink: 5k → 10k added +0.0009.

## Where we lost time (fix next run)

1. **Scoring on the GPU while training** halved training speed (5.6 →
   2.0–2.7 steps/s). Score only key checkpoints, or after training ends.
2. **Scorer tokenization was CPU-bound** at ~2,000 pairs/s: per-microbatch
   Python padding. Fixed in `score_external` (parallel Rust encoding, NumPy
   padding, prefetch pool); parity mean |diff| 6e-6.
3. **Coarse length buckets** (64/128/256) waste GPU on padding. Use finer
   buckets (48/64/80/96/128/160/256) and batch 256.
4. **Launch preprocessing** takes ~5 min per trainer start (typed-negative
   Python loop, hashing). Cache the flags per arena and thresholds.
5. **The test inputs and scorer were not staged on the new pod.** S3
   pre-signed URLs moved them in under a minute.

## Improvements still open

- **Decision layer with C2 features:** pair CatBoost with rival C2 scores,
  fitted on B_REST halves (agent P, running).
- **French fine-tune:** audited strict pseudo-labels (P+ precision 0.9999 on
  US/India) with rival negatives and US/India replay. Queued after the 40k test
  scoring. It adds no French recall; it teaches the model to reject rival
  claimants.
- **Checkpoint weight averaging** of 30k/35k/40k: no extra inference cost,
  needs one B_REST scoring.
- **Retrieval:** the P2 union lifts the arena recall ceiling 0.9835 → 0.9873.
  It would need test union mining (the pod now has 128 cores) plus tree features
  for the new pairs.
- **Recall is the remaining loss:** 0.946 vs a 0.9835 retrieval ceiling.

## Next-submission stage (set up 26 Sep ~17:10 IST, in parallel with the live run)

Expected public score of the step-40k C2 + name-only file: about 0.975–0.979.
This assumes B_REST ≈ 0.982–0.984 and a public score ~0.008 below it, as
v1's was. That is an estimate. 0.99 is not realistic by the deadline: the
leaders are at 0.988 and the dense retrieval ceiling is ≈ 0.994. A realistic
stretch for the next file is ≈ 0.980–0.984, stacking:

| Lever | Owner | Expected | Gate |
| --- | --- | --- | --- |
| Decision layer with C2 rival scores (pair CatBoost, B_REST halves) | agent P (pod CPU) | +0.001–0.002 | B_REST cross-fit CI > 0 |
| French fine-tune (strict pseudo-labels, rival negatives, US/India replay) | chained after the 40k test scoring | France-dependent | Monitor US/India no regression; France-only upload |
| Union retrieval (lexical/phonetic/key, top-k new pairs scored by C2) | agent U (pod CPU) + GPU scoring | +0.001 | B_REST oracle and matcher gain CI > 0 |
| Checkpoint averaging 30k/35k/40k | controller | 0 to +0.001 | One B_REST scoring |
| Qwen3-Reranker-0.6B second family, ensembled with C2 | agent Q prepares; GPU overnight | +0.001–0.003 | B_REST ensemble cross-fit CI > 0 |

GPU order after 40k: final file → French fine-tune → union scoring →
Qwen train and score → ensemble assembly on 27 Sep morning.
