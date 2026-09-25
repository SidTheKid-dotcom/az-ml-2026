# V2 master plan — full-scale training, structural hacks, speed fixes

Status: proposed 2026-09-25; no v2 gain measured. The first scored public
submission is user-reported F0.5 **0.95639** (uploaded file unconfirmed).
Baseline: frozen v1, holdout F0.5 **0.968292**
(precision 0.987270, recall 0.939456, 506 false links, 2,529 missed links,
singleton accuracy 0.932650). Goal: a v2 that beats v1 with measured,
paired evidence at every step. Nothing here touches the frozen v1 release.
This is a hypothesis backlog, not a record of adopted changes. Follow `AGENTS.md`
and `docs/RUNBOOK.md` (especially the SageMaker migration) for current execution;
the RunPod-only scheduling and throughput estimates below need remeasurement on
the chosen hardware. Prioritize bounded experiments over executing every phase.

## Candidate bottlenecks and hypotheses

| # | Bottleneck | Evidence | Fix | Phase |
| --- | --- | --- | --- | --- |
| B1 | CE mixed precision needs parity/throughput checks on new hardware | CUDA bf16/AMP is **already implemented** in `rerank_inference.py` and `cross_encoder_experiment.py` | benchmark existing mode and any proposed change on fixed pairs and accepted sets | P0 |
| B2 | CE batch 128 | 1,174 pairs/s benchmark | batch 256 with existing length sorting + pad8 | P0 |
| B3 | Concurrent GPU work may improve throughput | Old A40 occupancy estimate; not benchmarked on SageMaker | test one bounded job first, then add compatible work only if the validation path stays fast | P0 |
| B4 | Old RunPod scratch was nearly full | Historical `du` snapshot | preserve originals and verified durable artifacts; reclaim only redundant scratch copies after hash checks | P0 |
| B5 | Old pod had 7.65 CPU cores | runbook | check actual new-job CPU/RAM limits before moving feature preparation or CatBoost | P0 |
| B6 | Trained on 36k of 2.2M labeled S1s | medium split; gains from additional data are a hypothesis | screen larger training subsets before mining ~2.08M S1s (fresh holdout excluded) | P2/P3 |
| B7 | Dense-only K50 candidates | quick-reference union recall 0.991227; this is **not** medium recall | test lexical+phonetic+dense union on the same medium TUNE references against the full pool | P2 |
| B8 | Independent per-pair scoring | one-to-one + duplicate structure unexploited | assignment pass, cluster-aware scoring, k-prior decision rule | P1/P4 |
| B9 | 86.6M-pair CE rerank is expensive | 1,174 pairs/s measured on the old setup | benchmark larger batches and a cascaded shortlist; any speedup or score parity remains unmeasured | P0/P6 |
| B10 | France zero labels, test distractor shift (5.8 vs 4.7 targets/S1) | task.md | denoiser canonicalization + self-training on high-confidence test pseudo-labels | P4/P5 |

## Parallel execution map (what runs when)

- **GPU track A (critical path)**: S1 encoding (P2.2) → dense mining (P2.3) → CE training (P3.2) → CE inference/eval.
- **GPU track B (co-resident, when VRAM allows)**: lexical/phonetic mining chunks (P2.4), second CE family (P3.3), byt5 denoiser (P4.2), later test-set pseudo-label inference (P5.1).
- **Local workstation (if capacity allows)**: decision-rule prototyping on saved TUNE scores (P1), feature engineering, cluster purity tooling (P4.1).
- **SageMaker job CPU**: streaming feature/parquet prep; size threads against measured cgroup limits.
- Historical A40 limits were 46 GB usable VRAM, 7.65 cores and 50 GB RAM; verify the new job's actual limits, budget and available disk before parallelism.

Guardrail: never modify `models/submission_v1_pipeline.json` or anything under
`releases/submission_v1`. All v2 work happens in a new output/code checkout.

---

## Measured data structure (train-only analysis, 2026-09-25; no eval/test labels touched)

Preliminary draft observations: sample of 120k S1s (415,706 true links) + streamed
full pool (10,320,219 rows). The detailed 120k analysis report is not included
in this checkout. Reproduce these percentages from TRAIN and save a report before
using them as quantitative evidence for any change; the one-to-one count and
match-count prior are separately recorded in `problem-statement/task.md`.

- **One-to-one verified**: all 7,638,365 links hit distinct targets. A surviving
  multi-claim is a guaranteed false link (basis of P1.1).
- **Composite key `(country, norm-name-no-suffix, address-digit-sequence)`**:
  99.3% of S1 keys are unique; **28.2% of true links uniquely identify their
  correct S1** by exact key; only 1.1% collide with a *different* S1, 0.5%
  ambiguous. A deterministic channel with ~96% precision before any ML.
- **Digits are the most stable address signal**: true-pair digit-set Jaccard
  mean 0.902 (75.1% exact set equality; 67.7% exact sequence); name-colliding
  distractors reach digit-set equality only ~50% (best-over-S1s, upward bias).
- **Distractors are engineered look-alikes**: 49.9% of unmatched pool rows share
  a normalized name with some S1; half of those share its full digit set; mean
  best addr token-Jaccard 0.429. Hard-negative mass concentrates in
  same-name/same-digits/slightly-different-detail pairs — unit/door numbers are
  the discriminative signal (matches the v1 FP examples).
- **Name noise inventory (true pairs)**: 43.0% exact after norm+suffix-strip+
  accent-fold; 15.8% native/accent script; 6.2% typo ≤2; 11.9% edit 3–5;
  20.8% heavy edit >5; 2.3% reorder-only. → denoiser (P4.2) targets the ~36%
  heavy+script mass; normalization channel (N1) covers the 43%.
- **Intra-S1 match structure**: within-source match–match token-Jaccard median
  0.667 vs cross-source 0.455; group sizes 2–5 typical; 6.3% of pool rows are
  exact duplicates (645k). Clustering at J≈0.5 groups same-S1 records with
  high purity potential (P4.1); pool canonicalization removes ~6.5% of
  retrieval compute with zero recall loss.
- **Cross-source co-occurrence**: 80.5% of non-singleton S1s match both S2 and
  S3 → source-balance prior for the decision rule (P1.3).
- **k is text- and country-independent**: corr(k, name/addr tokens) ≈ 0.003;
  India 3.462 vs US 3.461. One global k-prior; no per-country k model.
- **Singletons are not textually separable**: 23.1% of singleton S1s vs 22.9%
  of matched S1s have a look-alike in the same unmatched sample — identical. A
  heuristic "look-alike ⇒ not singleton" detector is dead on arrival; only
  calibrated scores + expected-F0.5 + one-to-one can decide singletons.
- **Name alone is ambiguous**: 5.6M unique names in a 10.3M pool (56.7% of
  rows share a name with another row); 39% of S1s share a name with another S1.
  Address digits/details are mandatory discriminators.

Treat these as TRAIN-only hypotheses, not guaranteed generator invariants.
Fit derived calibration/priors on TRAIN and choose decision thresholds on TUNE.

## P0 — Speed and infrastructure (unlocks everything)

1. **Check resources and preservation.** Verify account credits/quota and the new
   job's storage/CPU/GPU limits as in the runbook. Preserve the supplied archive
   and all frozen release files. Reclaim scratch only after durable-copy hashes
   match; retain original inputs and successful checkpoints.
2. **Mixed-precision parity.** Existing CE training/inference already uses CUDA
   autocast (bf16 when supported). On a new GPU, compare fixed saved TUNE pairs
   and complete S1 decisions to the v1 reference before adopting any new mode.
3. **Larger inference batches.** Screen batch 128 → 256 (length sorting and pad8
   unchanged); benchmark throughput, memory, accepted-set parity and macro F0.5.
4. **Job scheduling.** Start with one short training job, then overlap CPU prep
   and compatible GPU jobs only when measured total throughput improves. Avoid
   creating a second queue/dashboard for already completed v1 work.

Exit criteria: validated small-job parity and measured speed/resource report.

## P1 — Decision-layer wins on saved scores (no training, no new mining)

Work on the saved v1 TUNE probabilities (12,000 S1s, full pool) — zero GPU needed,
runs on the workstation in minutes.

1. **One-to-one assignment pass (Hack A).** Ground truth: every S2/S3 record
   matches at most one S1 (7,638,365 links, 0 shared — measured in `task.md`).
   Build the conflict set: targets predicted for ≥ 2 S1s. Resolve greedily by
   blended score with margin δ ∈ {0, 0.05, 0.1, 0.2}: drop the weaker claim when
   the stronger exceeds it by δ; an S1 left with no claims predicts empty.
    A conflict contains ≥ 1 false link under the TRAIN invariant, but dropping
    the wrong claim may hurt S1 macro F0.5. Choose δ on TUNE only;
   keep a δ=0 fallback (pure constraint) if the margin underperforms.
2. **Calibration.** Fit isotonic regression on disjoint or out-of-fold TRAIN
    scores (US, India; pooled fallback for France later). Select the rule and
    assess reliability on TUNE; do not fit and evaluate on the same S1s.
3. **Expected-F0.5 decision rule with k-prior.** Replace the flat 0.775
   threshold: for each S1 with sorted calibrated probs p₁ ≥ p₂ ≥ …, pick the
   prefix k* maximizing E[5·TP/(4k+T)] where each pᵢ gets a Beta posterior and T
   (true match count) uses the training prior (0→5.6%, 1→5.4%, 2→17.0%,
    3→24.1%, 4→21.9%, 5→14.6%, 6+→11.4%, mean 3.46; start with a global
    prior, then test calibration). Add the **source-balance prior**: a preliminary 80.5%
   of non-singleton S1s match both S2 and S3, so given confident matches in one
   source, lower the bar slightly for the other source (weight fitted on TUNE).
   Small DP, no GPU.
    Test whether the per-entity rule beats the global threshold; probabilities
    and the match-count model may be miscalibrated or dependent.
4. **Composite-key deterministic channel (new, from data analysis).** For
   candidates whose `(country, norm-name-no-suffix, digit-sequence)` exactly
   and uniquely equals an S1's key, auto-accept after an address-token sanity
   check. Measured: covers 28.2% of true links at ~96% precision pre-ML (1.1%
   wrong-S1 collisions). Implement as (a) a retrieval branch guaranteeing these
   candidates exist, and (b) a decision override evaluated on TUNE.
    Check whether this improves precision and retrieval recall against the full
    pool; the preliminary ~96% precision is below v1's measured precision.
5. **Evaluate.** Macro F0.5 + false links + singleton accuracy on TUNE for
   {baseline, A, rule, A+rule}; paired bootstrap (10k resamples) vs baseline.
   Keep only components whose 95% CI lower bound is ≥ 0, or precision-neutral
   F0.5 wins. These are decision-layer modules that apply to ANY matcher,
   including the v2 from P3.

Record: `learnings/` note + ledger rows; copy exact configs to `/workspace`.

## P2 — Full-scale candidate mining (long-running GPU work)

1. **Reserve the fresh holdout first.** From the 2,206,821 train S1s, exclude
   quick (12,000) + medium (60,000) IDs; sample **50,000** disjoint S1s with a
   recorded seed as HOLDOUT-2. They get mined (so they can be scored) but never
   train or tune anything. This is the one-shot final gate for v2.
2. **Encode** the remaining ~2.08M S1s with the frozen E5-ft200 encoder
    (screen bf16/batch 1,024 for memory and decision parity). Pool embeddings
    already exist — verify their hashes and relocation before use; do not
    re-encode the 10.32M targets without a concrete failure.
3. **Dense top-50** for those S1s against the full pool (tiles 512 × 50k;
    benchmark time on the chosen hardware before scaling).
4. **Lexical + phonetic mining** resumed from the `medium_hard` design, chunked;
    union with dense; screen two proposed branches: **digit-sequence token
   index** (75.1% of true links share the address digit set; 67.7% the exact
   sequence — a recall branch orthogonal to name) and the **composite-key
   lookup** (P1.4). Canonicalize the pool first: exact duplicates are 6.3% of
    rows (645k) — retrieve against unique representatives, expand back after
    (possible speedup only if ordered candidates and recall are preserved).
    Compact to retrieved positives
   + ≤ 40 hard negatives per S1 (~85–105M pairs, parquet, ~3 GB), where hard
   negatives are mined preferentially from **same-name/same-digit-set non-linked
   pairs** (the measured distractor design). Keep every true link in scoring
   even when retrieval missed it (misses must be measurable).
5. **Truth join** for TRAIN S1s only. HOLDOUT-2 gets candidates + truth frozen
   separately, untouched.

Gate for expansion: candidate recall on the **same TUNE S1s** improves with
manageable pairs/runtime; pair counts and digests recorded, durable S3 copies
and a verified manifest. Do not promote based on a quick-reference ceiling.

## P3 — Full-scale training (the biggest single lever)

1. **CatBoost numeric tree at larger scale** (on measured CPU/GPU capacity):
    compare with the frozen medium tree (depth 10, negative weight 2, numeric on)
    using compatible candidate features; train on
   the full pair set (subsample to ~60M pairs if RAM-bound). Eval on TUNE.
   Why: trees gate the CE (0.01) and their false-link behavior sets the cascade;
    test whether more hard negatives reduce near-twin confusions.
2. **BGE CE training, full data, bf16**: batch 256, lr 2e-5 cosine, hard
   negatives = tree-gated uncertain band ∪ top-dense-rank ≤ 50 non-positives,
   mined preferentially from same-name/same-digit-set non-linked pairs (the
    proposed distractor design); include retrieved TRAIN positives in fitting,
    while keeping every unretrieved truth link in evaluation. Steps:
   evaluate at 10k / 20k / 40k on TUNE; stop at plateau. Expected 6–15 h on
    the old A40 (not a SageMaker estimate). Checkpoint each eval point.
    Hypothesis: the 600→1800 comparison improved +0.0029 on fixed medium
    candidates; scaling to millions of S1s may saturate or regress. Never
    inject unretrieved truth edges into candidate evaluation.
3. **Noise-op synthetic augmentation (new, from data analysis).** The noise
   inventory is measured: 43% exact-after-norm, 6.2% typo ≤2, 11.9% edit
   3–5, 20.8% heavy edit, 15.8% native/accent script, 2.3% reorder-only.
   Re-synthesize additional positives from TRAIN S1 records by applying the
   same op distribution (including **accented-Latin ops for French-style
   text** — France's only source of matched-looking training signal), and hard
   negatives as same-name/different-digit variants. Mix with real pairs at a
   capped weight; gate on TUNE. Label leakage: none — ops are estimated from
   TRAIN only; augmentation uses S1 text, never eval labels.
4. **Optional diversity CE**: same recipe on mDeBERTa-v3-base (or
   bge-reranker-v2-m3 init) for a second failure mode; only if P3.2 shows a
   plateau before the step budget.
5. **Blend + decision layer**: tree gate 0.01 → CE blend {0.5, 0.75} →
   threshold tuned on TUNE → apply the P1 winners (assignment + expected-F0.5
   rule + composite-key channel). Compare to v1 on the same TUNE population
   with paired bootstrap.

## P4 — Structural modules (each independently gated)

1. **Target-side duplicate clustering (Hack B).** Cluster the 10.32M pool with
   FAISS/LSH on existing E5 embeddings. Measured feasibility: same-S1 matches
   sit at median token-Jaccard 0.667 within-source / 0.455 cross-source and
   6.3% of pool rows are exact duplicates — LSH at J≈0.5 should group them;
    composite-key ambiguity is only 0.5–1.1%; cluster→S1 purity needs actual
    measurement, not inference from these figures. Verify on TRAIN, then:
   (a) retrieval: add cluster-mates of top hits as candidates (recall);
   (b) feature: cluster-mean CE score (precision);
   (c) one-to-one applied at cluster level.
   Gate first on retrieval-recall ceiling delta (cheap); build scoring only if
   recall ceiling rises ≥ +0.003.
   Why: duplicates within S2/S3 are generated per-source; a missed member is
   usually findable via its near-twin, including native-script India rows.
2. **byt5-small denoiser.** Train on the 7.6M matched noisy→clean pairs
   (input: S2/S3 "name | address", target: the S1 record's text), bf16,
   ~100k steps. Canonicalize pool + queries; add an exact-normalized-match
   channel (high-precision feature + retrieval branch).
    Hypothesis: a learned inverse may help noisy names/addresses, including
    France, which has no training labels. A French benefit would remain unmeasured.
3. **LLM adjudication (only if P3 plateaus).** Qwen3-4B yes/no on pairs with
   final prob ∈ [0.15, 0.85]; deterministic decoding. Parameter budget:
   4.022 + 0.568 + 0.118 (+0.279 optional) ≈ 4.99B < 8B cap. Only worth it if that
   band holds most of the remaining TUNE error mass.

## P5 — Test-time adaptation

1. **Self-training.** Run the frozen v2 over all 1.73M test S1s. Keep pairs with
   calibrated p ≥ 0.98 (and one-to-one-consistent) as pseudo-positives and
   p ≤ 0.02 as hard negatives; mix into CE training at ~0.3 weight, retrain
   5–10k steps from the v2 checkpoint. Eval on TUNE as usual.
   Why: adapts to the test distractor mix (5.8 vs 4.7 targets/S1) and gives the
    model French pseudo-positives for the first time — using predictions, never
    labels. Their correctness must not be assumed.
2. **France label-free diagnostics**: CE score distribution shift vs US/India,
   cluster coverage, canonicalization round-trip quality. No tuning on these.

## P6 — Freeze, validate, release

1. Freeze every component, threshold, blend and decision rule **on TUNE only**.
2. Single evaluation of the complete frozen v2 on HOLDOUT-2 (all 50k S1s vs
    the full 10.32M pool). Do not iterate on an opened holdout; reserve another
    entity-disjoint holdout for a subsequent selection cycle.
3. Regenerate official test TSVs with the frozen pipeline (reuse the existing
    test candidates only if the candidate set and feature semantics are unchanged;
    otherwise regenerate candidates and all candidate-dependent features), run
   `ber.validate_release` + official validator `--check-ids`, package via
   `package_portal_submission.py`, record in `submissions.json`.

## Overfitting protection (non-negotiable)

- HOLDOUT-2 opened once after selection; TUNE carries all threshold choice.
- Every keep/drop decision gets a paired bootstrap CI on complete S1 entities.
- Pseudo-labels: ≥ 0.98 / ≤ 0.02 confidence only, capped weight, one-to-one
  consistency required.
- δ, blend, threshold and cluster settings: selected on TUNE. Compare the
  complete frozen candidates to v1 on HOLDOUT-2 once; do not select modules on
  that opened holdout.
- France: never claimed measured; report label-free diagnostics only.
- Every experiment gets a ledger row + one-paragraph keep/discard note, per AGENTS.md.

## Execution order and concurrency

| Order | GPU | Local workstation |
| --- | --- | --- |
| 1 | P0 SageMaker parity pilot, benchmarks | P1 decision rules on saved scores |
| 2 | P2 mining (long-running), then P3.2 CE training | P3.1 CatBoost full scale |
| 3 | P3.2 finishes + blend/eval | P4.1 clustering tooling |
| 4 | P4.2 denoiser, optional 2nd CE | P4.1 scoring, P4.3 if needed |
| 5 | P5.1 pseudo-label inference + retrain | P6 packaging prep |
| 6 | HOLDOUT-2 one-shot, official rerun + validation | release |

---

## Execution prompt (hand this to the working agent)

Read AGENTS.md, README.md, learnings/CURRENT_STATE.md, docs/RUNBOOK.md,
learnings/EXPERIMENT_DECISIONS.md, then this proposal. Run a bounded parity
pilot on authorized SageMaker resources before selecting a v2 hypothesis; phases
P0→P6 are a backlog, not a required serial job queue. Rules: (1) never touch frozen v1
artifacts or `/workspace/amlc`; develop in a separate checkout; (2) split by
S1, fit on TRAIN, choose on TUNE, HOLDOUT-2 is opened once per selection
cycle; (3) compare alternatives with a paired bootstrap CI on the same S1s and
write a ledger entry with
config, seed, split digests and paths; (4) stop weak variants early but record
them; (5) keep the full 10.32M target pool for all comparisons; (6) reuse
existing embeddings/candidates — never re-encode the pool; (7) bf16 or any
inference change must pass the parity gate before adoption; (8) report
official-comparable macro F0.5 plus precision, recall, false links, singleton
accuracy, country breakdown; (9) update CURRENT_STATE.md and the decision table
after each completed phase; (10) when in doubt on a merge, prefer the
higher-precision near-tie. Check live jobs and saved artifacts before starting.
