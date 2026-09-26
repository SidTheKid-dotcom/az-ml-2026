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

## S1 public = 0.978287 — where the loss is (26 Sep ~20:15 IST)

- **The generator is country-independent (train labels).** US and India have
  3.459 and 3.465 true links per S1, singleton rates 5.58% and 5.59%, and
  0.740 true links per pool record in both.
- **Test US/India look like the arena.** The test pool has 5.8 records per S1
  (train 4.67). But the borderline C2-blend band densities per S1 match
  B_REST: US [0.05,0.1) 0.161 vs 0.186, India 0.141 vs 0.115. Withholding
  20–30% of B_REST S1s, which turns their records into orphans, moves the plain
  C2 score only 0.97958 → 0.97910 (`/workspace/az-ml-new/distractor_sim/`).
  Test-shift loss for US/India is small.
- **So France ≈ 0.941.** Take US/India test ≈ arena cross-fit (US 0.98744,
  India 0.98263, weighted 0.98480). Test weights are France 0.1498 and
  US/India 0.8502, giving France = (0.978287 − 0.8502·0.98480)/0.1498 ≈ 0.941.
  France is about 40% of all lost points.
- **French scores are smeared.** Per S1, France has 1.23 pairs in blend
  [0.05,0.1), against 0.14–0.19 for US/India. It has 0.41 in [0.1,0.3)
  (US/India 0.09–0.12), 0.059 in [0.3,0.7) (0.004–0.008), and only 3.11
  confident (≥0.7) links (US/India 3.18–3.22). France predicts 3.26 links
  per S1, against a generator prior of about 3.46 true.
- **Street-disjoint French drops are refuted by inspection.** The "different
  street" records are one-character typos of the same street (Beaumarchais →
  Beauuarchais, Mollien → Molilen). Dropping them would lose true links, as
  G1's −0.004 pricing said.
- S2 (strict French pseudo-label fine-tune) shortens French lists further
  (3.19 per S1), away from the prior. Expect it not to help. Its upload would
  still measure the French delta exactly.
- **Tree-yes / C2-no pairs (tree ≥ 0.9, C2 < 0.05).** France has 0.545 per S1
  (about 141k), US 0.019, India 0.026 (B_REST 0.016/0.020). 39% of the French
  ones are accepted at 0.10.
  - B_REST truth: 0% when another S1 claims the record at ≥ 0.7. With no better
    claimant, 12–25% for C2 < 0.005 and 88–97% for C2 0.005–0.05.
  - Stored C2 equals a fresh rescore, so this is not a bug.
  - French examples: different-street templated collisions ("Bordeaux Parents
    SAS", 54 Rue de Talence vs 54 Rue du Quai Bourgeois) and same-address
    legal-form additions ("Club du Cours" vs "Club du Cours SARL").
- **C2 floor on accepted pairs.** B_REST 0.979583 → 0.979547 at 0.005. Accepted
  C2 < 0.005 is 73–84% true in US/India, so the floor stays off there. France
  has 19,991 such links; the upside is ≤ +0.002.
- Diagnostic scripts: `code/business_entity_resolution/diagnostics/20260926_france/`
  (pod copies in `/workspace/az-ml-new/distractor_sim/`).
