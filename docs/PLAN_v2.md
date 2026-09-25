# Final plan — attack portfolio (26–27 Sep 2026)

Read `AGENTS.md`, `CONTEXT.md` and `learnings/negative_side_analysis_manual.md`
first. This is a portfolio of structural bets, each with a kill rule, plus the
infrastructure that measures them. It is not a minute-by-minute schedule: the
GPU queue runs continuously and bets are promoted or killed as results land.
Only the hard deadlines in §8 are fixed. The TODO list in §9 is the live
checklist; tick items in place.

**Stance.** We attack: GPU time goes to new model structures, not to
hyperparameter sweeps of v1. Tuning is allowed only as far as a bet needs to
work (one learning-rate/step check). Every bet is still measured on the arena
before it spends an upload — #2 lost to #3 because bold, unpriced rules cost
528 entity-points. No plan guarantees first place (leaders 0.988, us 0.964563).

---

## 0. Facts

| Fact | Value |
| --- | --- |
| Deadline | 27 Sep 23:59 IST; the **last upload counts** |
| Uploads left | 4 on 26 Sep (user-confirmed), 5 on 27 Sep |
| Current best | #3 v1 + one-owner, public **0.964563** |
| Compute | 1× A40 46 GB; pod 7.65 cores / 50 GB; `/tmp` 5.6 GB free; `/workspace` durable |
| Local | 8 cores / 14 GB; raw data; v1 test scores being copied to `artifacts/v1_test_scores/` |
| Model rule | MIT/Apache, all models together ≤ 8B params |
| Pseudo-labels | **Adopted** (controller decision; user undecided): labels derived from unlabeled supplied test text, disclosed in the methodology. Kept separable so the final can drop them |
| v1 test scores | Hash-verified at `/workspace/az-ml-new/releases/submission_v1_scores/` |
| v2 code | Committed `1a55287` on `codex/v2-plan-20260925` (worktree `az-ml-v2`) |

---

## 1. Evidence (measured 26 Sep on v1 TUNE scores + label-free test)

- **Loss on US/India TUNE (3.10 points/100 S1s):** partial miss 1.43, dilution
  0.67, abstention 0.53, singleton false merge 0.28, mixed 0.19. 655 true links
  never retrieved; 1,885 retrieved but rejected (252 at the tree gate).
- **Name-only records:** empty addresses are 4.4% of owned vs 0.3% of unowned
  records (~98% owned). v1 accepts 25% of their true links (1,177 of the 1,885
  rejections). Exact name unique among the country's S1s: precision 0.978;
  shared by 2: 0.469. Accept-unique + drop-shared: **+0.003962** on TUNE
  (421 better / 63 worse). Fuzzy additions: −0.00093. Address-side mirror
  (unrelated name, matching unique address): 51% precise — not a rule.
- **False links (494):** 246 belong to another S1 (120 empty-address siblings,
  117 co-located trade names); 248 unowned decoys, mostly same-length digit
  substitutions (57% of decoys vs 22% of true links with changed digits).
- **Retrieval misses (655):** 392 with digits, 213 empty, 194 native-script
  names, 69% India. Union retrieval recovered ~315 on TUNE; reverse exact-name
  lookup would recover 7.
- **France (test, label-free):** 36,447 of 48,927 v1 contests. Templated names
  ("Nantes Club/Conseil/Centre EURL"), repeated house numbers, 10.9% shared S1
  addresses; the street name is the discriminator. Estimated France ≈ 0.934
  and US/India ≈ 0.970 after one-owner (inferred from uploads).
- **Why TUNE misled:** one-owner touched 2 contests on TUNE vs 48,927 on test;
  the P1 count rule was +0.0011 on TUNE and −0.0003 on the board (US/India).
  Decision rules need a competitor-complete world: the arena (§4).

---

## 2. Training approaches (what gets trained, how, from what)

Nothing is trained from scratch except small heads and gradient-boosted
models. Pretrained multilingual models already know Indic scripts and French;
from-scratch transformers would need orders of magnitude more compute than we
have. All neural models are **fine-tuned from pretrained checkpoints**.

| Component | Model (licence, params) | Init | Technique |
| --- | --- | --- | --- |
| Retriever | multilingual-e5-small FT200 (MIT, 0.118B) | existing v1 fine-tune | Frozen; bi-encoder dense K50. Only A7 may change retrieval |
| Gate / feature tree | CatBoost | from scratch | Gradient-boosted trees on string, numeric and forensic features; GPU training, bounded rows |
| **Workhorse** pair reranker | bge-reranker-v2-m3 (Apache, 0.568B) | v1 BGE1800 checkpoint | Pointwise cross-encoder, BCE, **arena-mined hard negatives**, **cost-sensitive (stake) weights**, optional **diff-augmented input** |
| **ClaimNet** (record-centric listwise) | bge-reranker-v2-m3 backbone + new marker head | BGE1800 backbone; head from scratch | **Listwise softmax over claimant set + NULL class** (one record, its K claimant S1s in one sequence) |
| **SetNet** (entity-centric set) | bge-reranker-v2-m3 backbone + new marker and count heads | BGE1800 backbone; heads from scratch | **Multi-label set prediction** (one S1, its top-M records in one sequence) + **match-count head**; masked loss for partial labels |
| Second family | Qwen3-Reranker-0.6B (Apache, 0.596B) | pretrained | Decoder-LLM reranker, score = logit(yes) − logit(no); full fine-tune |
| LLM adjudicator (bet) | Qwen3-Reranker-4B (Apache, 4.022B) | pretrained + LoRA (from scratch) | **LoRA** fine-tune on the uncertain band only; inference on that band only |
| Resolver / stacker | CatBoost | from scratch | **Stacked generalization** over all model scores + structural features; OOF by component; isotonic calibration |
| Decoder | — | — | **Bayes-optimal expected-F0.5 decoding** (best prefix of sorted calibrated p) + **iterated best-response assignment** (one owner per record) |

Parameter budget if everything ships: 0.118 + 3×0.568 + 0.596 + 4.022 = 6.44B
(< 8B). A shared backbone would lower it; not required.

Shared training techniques: bf16 autocast, fused AdamW, frozen word embeddings
(after the G0b parity check), length-bucketed batches, background tokenization,
warmup then constant LR for screens and cosine for finals, checkpoint every
2,000 steps, checkpoint averaging (last 3), order-swap test-time augmentation
(pair models only, if it gains on the slice).

Semi-supervised techniques: **multi-view pseudo-labeling** for France and
**noisy-student self-training** on test (A4).

---

## 3. The attack portfolio

Each bet: hypothesis → build → data → cost *(est.)* → kill rule. Screens use
the arena SLICE (§4) with a fixed decision rule unless stated.

### A0 — Name-only ownership rule (structural, CPU, ready now)
Accept empty-address candidates whose normalized name is unique among the
country's S1s; drop empty-address links whose name is shared by ≥ 2 S1s. On
top of #3; the exact-name owner wins a contest. Upload as U5.
Kill: the arena disagrees in sign, or the upload loses.

### A1 — ClaimNet: listwise ownership over claimants (flagship)
- **Hypothesis:** pointwise models score each look-alike claimant ~0.8+ in
  isolation (France: up to 274 claimants of one record). Seeing all claimants
  at once turns "is this similar?" into "which one is it, if any?".
- **Input:** `record: <name | address> [SEP] <c1> <S1 text> <c2> <S1 text> … <cK> …`,
  K ≤ 8 claimants (v1 tree ≥ 0.01, top by v1 blend). Claimant order shuffled
  in training to prevent position bias.
- **Head:** linear on each marker's hidden state → K logits; linear on CLS →
  NULL logit; softmax over K+1; cross-entropy on the owner index or NULL. An
  auxiliary per-claimant sigmoid gives pair-level scores.
- **Data:** arena A records: ~40% multi-claimant owned, ~30% single-claimant,
  ~30% NULL. The owner counts as a claimant only if the record is in its
  top-50 (same as inference). France: records with a P+ owner (§5) among claimants.
- **Cost:** ~1–1.5 h training; test inference over ~5M records ~40 min.
- **Kill:** as a resolver input, SLICE F0.5 +0.001 over the workhorse-only
  resolver, or contest win accuracy +5 points. Else stop.

### A2 — SetNet: entity-level set prediction + count head
- **Hypothesis:** an S1's true records agree with each other while decoys
  differ systematically; seeing its top-M records jointly recovers noisy true
  records (partial misses are the largest local loss). A count head predicting
  how many matches the S1 has feeds the decoder and singleton decisions.
- **Input:** `S1 text [SEP] <r1> record … <rM> record`, M ≤ 12 by v1 score.
- **Heads:** per-marker sigmoid (BCE, stake-weighted); count head softmax over
  0…11 (target = true count, including unretrieved links). Masked loss for
  French pseudo-labels (unknown records carry no loss).
- **Cost:** ~1–1.5 h training; test inference over 1.73M S1s ~50 min.
- **Kill:** +0.001 SLICE F0.5 as a resolver input, or recall +0.5 points at
  equal precision.

### A3 — Workhorse at arena scale (needed by everything)
- Pointwise BGE on 40× more businesses than v1 (arena A), negatives from the
  arena's own v1 mistakes (false accepts, contest losers, negatives with
  blend ≥ 0.3, same-length digit substitutions, same-address different names,
  empty-address siblings). **Stake weights:** positive 1/(5T−4); negative
  4/(5T+4) for T ≥ 1, 1.0 for a singleton; class-normalized, capped at 5×.
- Variants (each only if the previous gains): **diff-augmented input** (A6),
  French pseudo-labels at 15–30%.
- Cost: ~45 min per screen, 3–4 h final.
- Kill a variant: −0.002 on SLICE at the 5k-step checkpoint.

### A4 — Noisy-student self-training on test (domain adaptation)
- Teacher: best full stack on test. Pseudo-positives: assignments with
  resolver p ≥ 0.97 that are one-owner consistent (France additionally:
  multi-view P+). Pseudo-negatives: their rival claimants, and records whose
  every claimant has p ≤ 0.02.
- Student: continue the best model on arena A + test pseudo-labels (weight
  0.3–0.5) with input noise (case, accents, token drops).
- Why: test has 5.75 records per S1 vs 4.68 in train, and France has no labels.
- Kill: SLICE regresses > 0.0005, or France diagnostics worsen.

### A5 — Second family and LLM adjudicator
- A5a Qwen3-Reranker-0.6B fine-tune (zero-shot scorer exists:
  `score_qwen_reranker_gpu.py` in the follow-up code copy). Keep if the blend
  with the workhorse gains ≥ +0.0005 on SLICE.
- A5b Qwen3-Reranker-4B + LoRA (r=16, q/k/v/o), trained on the arena uncertain
  band (~200k pairs), applied only to test pairs with stack p ∈ [0.1, 0.95].
  First measure the band size on test; inference must fit in ≤ 2 h. Kill after
  1 h of training if band AUC does not beat the stack's.

### A6 — Diff-augmented input (forensic hints in text)
Prefix the pair with compact alignment tags, e.g.
`[digits sub 118→119] [street mismatch] [names shared 3] [address empty]`.
The cross-encoder then sees the generator's tell-tale differences
explicitly. Cheap variant of A3. Kill: < +0.0005 on SLICE.

### A7 — Retrieval attack (only if the probe says so)
Probe recall@K (50/100/200) on SLICE. If ≥ half the misses sit within top-200,
raise K for the gate. Otherwise, measure lexical/phonetic union mining speed on
50k test S1s and add it only if it fits. A retrained bi-encoder is out of time
budget. Any retrieval change rewrites `candidate_pairs.tsv` and recomputes
every candidate-dependent feature.

### A8 — Decision layer (CPU, continuous)
One-owner → name-only rule → learned resolver (stacker) → exact expected-F0.5
decoding with a Poisson term for unretrieved links (λ≈0.055) → iterated
best-response exclusivity (keep a record's best claimant only if p ≥ its list
break-even b(k) = [4/(5k+4)] / [1/(5k+1) + 4/(5k+4)], i.e. 0.727, 0.759, 0.771
for k = 1, 2, 3; re-decode the losers; repeat until stable). Blend weights and
gate (0.003 vs 0.01) are re-fit here, on the arena, not on TUNE.

**Resolver features:** all model scores (tree, workhorse, ClaimNet share and
NULL probability, SetNet probability and predicted count, Qwen); rank in the
S1 list; S1 list stats (count ≥ 0.5, top-1, top-2, gap, sum p); number of
claimants, rank among claimants, margin to the best rival, softmax share;
address kind; source; exact normalized-name equality; S1 name and address
multiplicity; record-name multiplicity; exact key; digit relation and edit-type
counts; street-token Jaccard (French-normalized); native-script flag.
**No country feature** (France is unseen); France uses pooled calibration.

### Not doing
Hyperparameter sweeps of v1; blanket removals/additions; country thresholds;
fuzzy name-only additions; address-side trade-name rules; cluster retrieval;
unbounded tree fits; ByT5 denoiser; Qwen3-4B zero-shot adjudication without
LoRA; anything selected on the opened medium holdout or on 12k-sample decision
tuning.

---

## 4. Infrastructure that makes attacking safe

### 4.1 Arena
- All training S1s × top-50 of the full 10,320,219 pool. P2 already mined
  2,122,821 S1s (`/workspace/az-ml-new/v2/dense/`); mine the missing ~84k.
- **HOLDOUT-2 (50,000) excluded together with its ground-truth rows**; its records look
  unowned, which is correct for a world without those businesses.
- Score with frozen v1 (tree gate 0.01, BGE1800, blend 0.75), selective BF16.
- Save `pairs.parquet`, `v1_tree.npy`, `v1_blend.npy`, `v1_ce_scored.npy`,
  `label.npy`, `s1_meta.parquet`, `rec_meta.parquet`, `manifest.json` under
  `/workspace/az-ml-new/arena/`; copy the compact arrays locally.
- **Partitions by claim-graph component** (S1s linked by a shared record with
  both blends ≥ 0.05; raise the cut if the largest component > 5%).
  **A ≈ 70%** trains models (quick/medium TRAIN components forced here);
  **B ≈ 30%** is never trained on (medium TUNE / v1-holdout components forced
  here). **SLICE** = fixed 100k-S1 subset of B for model screens; B-rest for
  resolver OOF and reporting. Bootstrap unit: component.
- **Fidelity check:** v1 + one-owner in the arena must reproduce test's US/India
  contest rate, contested share by address kind, list length, empty rate, and
  ≈85% wrong-rate of removed claims. If contest density is too low, add a
  random share of test US/India records as absent-owner records and redo B's
  top-50. If still unfaithful by 26 Sep 09:00, use the arena only as a relative
  yardstick for model screens and set decision thresholds by pricing against
  the board.

### 4.2 GPU orchestration
- One supervisor started with `setsid nohup` (SSH exit cannot kill it); two
  slots (`train`, `infer`), VRAM cap 38 GB, per-job threads, `/tmp` floor 4 GiB,
  append-only queue re-read every 30 s, completion markers, **no automatic
  retries**, `status.json` per job (step, pairs/s, VRAM, ETA).
- The queue always holds the next job, so the GPU never idles between them.
  While new code is being written, a cheap legacy job fills the gap.
- Co-run train + infer only if G0 shows combined throughput ≥ 1.15× serial.
- Thread budget: GPU jobs 2 each; pod CPU jobs ≤ 3; total ≤ 7.

### 4.3 Throughput
- Current trainer: 147 pairs/s; synchronous tokenization; non-fused AdamW over
  all 568M params. Fixes: batch 64/128, fused AdamW, frozen word embeddings,
  prefetch tokenization thread, length bucketing. Target ≥ 2× (est.).
- **G0b parity:** the fast trainer reruns the BGE1800 recipe on medium data and
  must land within ±0.001 of 0.968992; else drop the embedding freeze.
- Inference: length sorting, pad-8, upper-bound skipping, selective BF16
  (3.12×, identical accepted set on TUNE).
- Feature caches only for the gated band plus a stratified sample; never 110M
  × 100 features.

### 4.4 Measurement and kill protocol
- Screens: score SLICE gated pairs → blend with the v1 tree at w ∈ {0.5, 0.75,
  1.0} → fixed decision (threshold + one-owner + A0) with the threshold
  cross-fitted over two component halves. Structured models (A1/A2) are judged
  as resolver inputs.
- **Promote:** Δ ≥ +0.0005 and component-bootstrap 95% CI > 0. **Kill:** −0.002
  at the 5k-step checkpoint, or two levels of a factor without gain.
- **France:** diagnostics must improve (contested claims ↓, street-mismatch
  rate ↓, exact-key agreement ↑, list length 3.2–3.5, empty rate 5–6.5%) with
  no US/India regression; then a France-only factorial upload.
- **Before upload:** per-country diff and arena-priced prediction (manual §4).
  **After:** five decimals in `learnings/submissions.json`; attribute; if the
  board contradicts the arena by > 0.0005 in sign, stop that line and recheck
  fidelity.

---

## 5. France pseudo-labels (multi-view)

- **French normalization:** fold accents; expand `r.`, `av.`, `bd`, `pl.`,
  `chem.`, `all.`, `imp.`, `fbg`; drop region and département names; street
  tokens = non-digit tokens minus city and street-type words.
- **P+:** unique French exact key ∧ street-token Jaccard ≥ 0.5 ∧ v1 blend ≥ 0.9
  ∧ no rival claimant ≥ 0.5 ∧ record not empty-address.
- **P−:** every other claimant of a P+ record (one owner per record), plus
  same-template names in the same city with the same house number and a
  different street.
- **Audit:** print 200 P+ and 200 P−, judge by eye; require ≥ 98% apparent
  precision or tighten.
- **Pseudo-holdout:** 10% of French components held out: P+ recall, P−
  rejection, key-owner contest win rate (upper bounds, not French accuracy).
- **Diagnostics** on every file: contested claims, list length, empty rate,
  exact-key agreement, S2+S3 co-occurrence, street-mismatch rate.

---

## 6. Final stack (assembled from whatever survives)

Gate tree → workhorse (+ Qwen 0.6B if kept) on gated pairs → ClaimNet on
records → SetNet on S1s → LLM adjudicator on the uncertain band (if kept) →
resolver/stacker → expected-F0.5 decoding + iterated best-response → name-only
rule check → validation. Estimated full test inference: 3–4 h *(est.)*.

## 7. Orchestration and agents

- **Controller (main session):** owns the pod, the queue, GPU code (fast
  trainer, arena scorer, ClaimNet, SetNet, adapters), every decision and all
  docs; prepares every upload file with a predicted score. Background watchers
  wake the controller when jobs finish.
- **agent-D (decision):** arena evaluator (component bootstrap), exact
  expected-F0.5 decoder, iterated best-response, resolver harness; must
  reproduce v1 TUNE 0.968992 and the P1 numbers; builds the A0/U5 file.
- **agent-F (features/France):** vectorized forensic features, French
  normalization, multi-view pseudo-labels and audit printouts, France
  diagnostics.
- Workers run locally in their own git worktrees, never touch the pod, and
  deliver code + tests for controller review.
- The user clicks uploads (no script uploads). If the terminal session closes,
  the pod queue keeps running but no new decisions are made; run `/loop` for
  unattended operation.

## 8. Hard deadlines (IST)

| When | What |
| --- | --- |
| 26 Sep, as soon as validated | U5 = A0 |
| 27 Sep 10:00 | Final-stack test inference starts (3–4 h) |
| 27 Sep 13:00 | Freeze; HOLDOUT-2 one-shot |
| 27 Sep 16:30 | Intended final uploaded |
| 27 Sep 20:00 | ZIP, README, reproduce, methodology done |
| 27 Sep 20:30 | Re-upload the chosen final if needed; no experiments after |

## 9. TODO (live checklist)

Owners: `ctl` controller, `D` agent-D, `F` agent-F, `user`.

### Now
- [x] ctl — Relaunch legacy follow-ups detached (`r0_fields7200_20260926b`, `r0_pipe14400_20260926b`)
- [x] ctl — Preserve v1 test score arrays on `/workspace` (hash-verified)
- [x] ctl — Commit v2 worktree code (`1a55287`)
- [ ] ctl — Copy v1 test scores + test reference/pool locally (running)
- [ ] D — Build A0/U5 file on top of #3; validate (official `--check-ids` + strict); per-country diff and price
- [ ] user — Upload U5; report five decimals
- [ ] ctl — Read R0 results against 0.968992

### Infrastructure
- [ ] ctl — Supervisor v2 (two slots, detached, markers, no retries, status.json, queue re-read)
- [ ] ctl — Fast trainer (batch, fused AdamW, frozen embeddings, prefetch, bucketing, checkpoints/resume)
- [ ] ctl — G0 speed benchmark; G0b parity vs 0.968992
- [ ] ctl — Move test pool vectors from `/tmp` to `/workspace` (hash, symlink)

### Arena
- [ ] ctl — Mine the missing ~84k S1s
- [ ] ctl — Score the arena with v1 (selective BF16); save arrays + manifest
- [ ] D — Components, partitions A/B/SLICE, component bootstrap
- [ ] D — Fidelity check vs test; density-matched fallback decision
- [ ] D — Arena D-grid: one-owner, A0, break-even contest rule, expected-F0.5 decoder, blend/gate re-fit

### Bets
- [ ] F — Forensic features + French normalization (vectorized, tested)
- [ ] F — France P+/P−, audit printouts, pseudo-holdout, diagnostics
- [ ] ctl — A3 workhorse screen on arena A (stake weights, arena negatives)
- [ ] ctl — A1 ClaimNet build + screen
- [ ] ctl — A2 SetNet build + screen
- [ ] ctl — A6 diff-augmented input screen
- [ ] ctl — A5a Qwen3-Reranker-0.6B screen; A5b band-size check → 4B LoRA screen
- [ ] ctl — A7 recall@K probe
- [ ] D — Resolver/stacker on B-rest OOF with every surviving score
- [ ] ctl — U6: resolver on v1 test scores (if arena-positive)
- [ ] ctl — First full stack on test → U7; France factorial → U8
- [ ] ctl — A4 noisy-student round from the best stack

### Final
- [ ] ctl — Final training runs of surviving bets (cosine, checkpoint averaging)
- [ ] ctl — B-rest re-score; resolver refit; test inference (starts by 27 Sep 10:00)
- [ ] ctl — HOLDOUT-2 one-shot after freeze
- [ ] ctl — Final file, validation, U-final; reserve the last slot
- [ ] ctl — ZIP, README, reproduce path, methodology (pseudo-labels disclosed)
- [ ] ctl — Update `learnings/`, CONTEXT.md, submissions.json after each result

## 10. Expected outcome (estimates, not measurements)

| Source | Public Δ |
| --- | --- |
| A0 name-only rule | +0.002 to +0.004 (measured on TUNE) |
| A8 resolver + decoder | +0.001 to +0.004 |
| A3 workhorse at scale | +0.002 to +0.006 |
| A1/A2 structured models | 0 to +0.005 (untested bets) |
| France (A1, pseudo-labels, A4) | +0.002 to +0.008 |
| **Total** | **≈ 0.972 to 0.990+**, if every bet lands |
