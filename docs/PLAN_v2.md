# Final plan — 26–27 Sep 2026

The single executable plan. Read `AGENTS.md`, `CONTEXT.md` and
`learnings/negative_side_analysis_manual.md` first. The live checklist is §17;
tick items in place and commit. Estimates are marked *(est.)* and must be
replaced by measurements.

No plan guarantees first place. Leaders are at 0.988, ranks 4–14 at
0.984–0.985, and we are at 0.964563. This plan maximizes expected score per
GPU-hour and never risks the best file.

---

## 1. First principles

The data generator, as far as the train labels show:

1. Source 1 is a clean, deduplicated list of businesses.
2. Each Source 2/3 record is either a **noisy copy of exactly one business**
   (74%) or **not a copy of any listed business** (26%: engineered look-alikes
   and records of absent businesses).
3. Noise on copies: typos, suffix changes, reordering, native-script
   transliteration, uppercase, junk tokens (`##`, `<NULL>`, `>>`), digits
   dropped or inserted, and whole addresses removed.
4. Look-alikes: same name with a different unit/house number (digits
   substituted at the same length), a different business at the same address,
   and same-name businesses elsewhere.

So the only question is: **for each record, who owns it, if anyone?** The
score then follows from how each business's list is built. The whole system
is three steps:

- **Estimate ownership well** (models).
- **Decide optimally** under one owner per record and the F0.5 metric.
- **Measure truthfully** in a world where every business competes.

---

## 2. Stance

- **Attack.** GPU time goes to new structure and scale, not to sweeps of v1's
  hyperparameters. Each approach starts from our best guess of its full recipe
  ("best guess first"); an ingredient is ablated only if the recipe fails or
  the GPU is otherwise idle.
- **Measure before spending uploads.** #2 was bold and unmeasured, and lost 528
  entity-points to #3.
- **Keep it simple.** One core path (§5); everything else waits for idle GPU.

---

## 3. Facts and rules

| Fact | Value |
| --- | --- |
| Deadline | 27 Sep 23:59 IST. **The last upload counts**; the public board is a subset of test |
| Uploads left | 4 on 26 Sep (user-confirmed), 5 on 27 Sep |
| Current best | #3 = v1 + one-owner, public **0.964563** (`~/Downloads/assignment-only-lf/matching_results.tsv`) |
| GPU pod | 1× A40 (46 GB); cgroup 7.65 cores, 50 GB RAM; `/tmp` 30 GB overlay (≈5.6 GB free); `/workspace` durable, slower |
| Local machine | 8 cores, 14 GB RAM, 772 GB free disk |
| Data | All train (S1–S3 + ground truth) and test (S1–S3) files locally at `data/raw/dataset/` and on the pod at `/tmp/az-ml-new/data/raw/dataset/`, identical byte sizes |
| v1 test scores | 86,627,200 aligned pairs, blend + tree scores: pod `/workspace/az-ml-new/releases/submission_v1_scores/`, local `artifacts/v1_test_scores/` (hash-verified both) |
| Models allowed | MIT/Apache; all models used together ≤ 8B parameters |
| External data | None. Vocabularies (e.g. French region names) are derived from the supplied data |
| Pseudo-labels | Adopted: labels derived from unlabeled supplied test text; disclosed in the methodology; kept separable so the final can drop them |

---

## 4. Targets

From leaderboard arithmetic (a common 0.984–0.985 profile implies about 97%
recall at 99–99.5% precision; with France 15% of test, France must be ≈ 0.97+
for a 0.987 average). These are inferred, not measured; the arena measures all
but France.

| Target | Now | Goal |
| --- | --- | --- |
| US/India micro recall | 0.939 | ≥ 0.97 |
| US/India micro precision | 0.987 | ≥ 0.99 |
| No-match businesses wrongly merged | 4.7% (TUNE) – 6.7% (holdout) | ≤ 2% |
| France F0.5 | ≈ 0.934 (inferred) | ≥ 0.97 |
| France empty-list rate | 4.7% (v1) | 5.6–6.0% |
| Contested record IDs in any file | 0 | 0 |

Point values: each 1% of no-match businesses wrongly merged ≈ 0.056 points;
France +0.01 ≈ +0.0015 overall; 0.001 overall ≈ 1,733 entity-points.

---

## 5. Evidence behind the choices (measured 26 Sep)

On the 12,000 medium TUNE S1s with v1 scores against the full pool, joined to
raw text and full-train ownership; plus label-free analysis of the test file.

- **Loss (3.10 points per 100 S1s):** partial misses 1.43, dilution 0.67,
  wrongly empty 0.53, no-match businesses merged 0.28, mixed 0.19.
- **Recall:** 655 true links never retrieved (69% India, 194 native-script
  names); 1,885 retrieved but rejected (252 by the tree gate; 706 just under
  the 0.775 threshold).
- **Name-only records:** empty addresses are 4.4% of owned vs 0.3% of unowned
  records (~98% owned). v1 accepts 25% of their true links (1,177 of the
  1,885 rejections). Exact name unique in the country: 97.8% correct; shared
  by two: 47%. Accept-unique + drop-shared: **+0.003962** on TUNE.
- **False links (494):** half belong to another business (sibling or
  co-located), half are look-alikes (mostly same-length digit substitutions:
  57% of look-alikes vs 22% of true links with changed digits).
- **Not a rule:** an unrelated name at a matching unique address is only 51%
  correct (the generator plants same-address look-alikes).
- **France:** 74.5% of all test contests; templated names ("Nantes Club /
  Conseil / Centre EURL"), repeated house numbers; the street name decides.
- **Why TUNE misled:** one-owner touched 2 contests on TUNE vs 48,927 on test;
  the P1 count rule gained on TUNE and lost on the board.
- **R0 field-format result (26 Sep 02:32):** BGE fine-tuned 7,200 steps with
  field-tagged text on the same 36k medium businesses scores **0.973686** on
  the same TUNE vs BGE1800's 0.968992 (+0.0047): precision 0.9894, recall
  0.9477, 422 false links (494), no-match merges 3.6% (4.7%), India 0.9701,
  US 0.9761. Confounded: 4× steps and the field format; the plain 14,400-step
  run separates them. Evidence that more training alone still pays.

---

## 6. The system being built

```
test S1 ──E5 dense top-50 (existing)──► candidates (86.6M pairs)
                                        │  (+ C7 native-script branch, if it passes)
                                        ▼
                         v1 tree (gate, tree ≥ 0.01; 0.003 tested)
                                        ▼
               C2 pair reranker (arena-scale BGE, canonical text)
                                        ▼
               C3 ClaimNet: each record vs all its claimants + "none"
                                        ▼
    C4 resolver: stacks all scores + structural features → calibrated p
                                        ▼
    C4 decoder: expected-F0.5 list per business, one owner per record,
                name-only rule, no-match probability
                                        ▼
                  matching_results.tsv (+ candidate_pairs.tsv)
```

Idle-GPU bets (§8) plug in as extra scores into the C4 resolver.

---

## 7. Core approaches (in build order)

### C0 — Structural rules (CPU, now)

- **What:** one-owner (exists) + name-only rule: accept empty-address
  candidates whose normalized name is unique among the country's S1s; drop
  empty-address links whose name is shared by ≥ 2 S1s; the exact-name owner
  wins any contest.
- **Why:** generator facts (§5), measured +0.003962 on TUNE.
- **How:** agent-D builds `name_only_rule.py`, reproduces the TUNE number, writes
  the U5 file on top of #3, validates, and prices it.
- **Leads to:** U5; expected public +0.002 to +0.004.

### C1 — The arena (competitor-complete yardstick)

- **What:** every training business scored by v1 against its top-50 records of
  the full 10,320,219 pool, all at once, so contests appear as on test.
- **Why:** 12k-sample tuning cannot see contests; every decision rule chosen
  there has misled us.
- **How:**
  1. Reuse P2's dense top-50 for 2,122,821 S1s
     (`/workspace/az-ml-new/v2/dense/candidate_pairs.parquet`); mine the
     missing ~84k S1s with the same E5 encoder and pool vectors.
  2. **Exclude HOLDOUT-2** (50,000 S1s) together with its ground-truth rows.
     Its records then look unowned, which is correct in a world without them.
  3. Score with frozen v1 (tree gate 0.01, BGE1800, blend 0.75), selective
     BF16, streaming predictor pointed at train data.
  4. Store the arrays listed in §13.
  5. **Partitions by claim-graph component:** connect S1s that share a record
     where both pairs have v1 blend ≥ 0.05 (raise the cut if the largest
     component exceeds 5% of S1s). **A ≈ 70%** trains models (components with
     quick/medium TRAIN S1s forced here); **B ≈ 30%** is never trained on
     (components with medium TUNE / v1-holdout S1s forced here). **SLICE** =
     fixed 100k-S1 subset of B for model screens; **B-rest** for resolver
     out-of-fold fitting and reporting.
  6. **Fidelity check:** v1 + one-owner in the arena must reproduce test's
     US/India contest rate, contested share by address kind, list length,
     empty rate, and the ≈85% wrong-rate of removed claims (inverted from
     uploads). If contests are too rare (train has 4.68 records per S1, test
     5.75), add a random share of test US/India records as absent-owner
     records and redo B's top-50. If still unfaithful by 26 Sep 09:00, use the
     arena only as a *relative* yardstick for models and set decision
     thresholds by pricing against the board.
- **Cost:** ~10 min mining + ~75 min scoring (GPU CE ~30 min, pod CPU
  features ~45 min, overlapped) *(est.)*.
- **Leads to:** a local number that behaves like the board; training data for
  C2/C3; out-of-fold data for C4.

### C2 — Arena-scale pair reranker (the workhorse)

- **What:** bge-reranker-v2-m3, fine-tuned from our v1 BGE1800 checkpoint, on
  ~40× more businesses than v1, with the arena's own mistakes as negatives,
  weighted by what each mistake costs.
- **Why:** beating engineered look-alikes (unit numbers, one street word) is
  learned from volume of hard examples; v1 saw 57,600 pair presentations
  from 36k businesses.
- **How (best-guess recipe, screened first):**
  - *Input:* the **field-tagged** format of the R0 run (unless the plain
    14,400-step run beats it), in the **canonical view**: Latin accents folded (combining marks removed only
    from Latin letters; Indic scripts untouched), lowercase, junk tokens
    (`<NULL>`, `##`, `>>`, `<<`) removed, whitespace collapsed. Why: France's
    injected accents ("Àmicale", "Çlub") and vendor uppercase then look like
    the training text.
  - *Positives:* all retrieved true pairs of sampled arena-A businesses.
  - *Negatives (1 positive : 3 negatives):* half typed hard negatives — v1
    false accepts; contest losers (another business's record claimed with
    blend ≥ 0.3); unowned look-alikes with blend ≥ 0.3; same-length digit
    substitutions; empty-address siblings — and half random gated negatives
    (tree ≥ 0.01) to keep calibration.
  - *Loss:* BCE with stake weights: positive 1/(5T−4); negative 4/(5T+4) for
    T ≥ 1 and 1.0 for a no-match business (T = the business's true count);
    class-normalized, capped at 5×.
  - *France:* 20% of each batch from French pseudo-labels (§9) when available.
  - *Optimizer:* fused AdamW, LR 2e-5, 5% warmup then constant (screens) or
    cosine (final), weight decay 0.01, batch 64, max length 256, bf16,
    word embeddings frozen if the G0b parity check passes.
  - *Screen:* 15k steps (~1M pairs, ~45–60 min). *Final:* 60–80k steps
    (~4–5M pairs, ~3.5–4 h) *(est.)*, checkpoint every 2k steps, average the
    last three.
- **Ablations (only if the screen fails, or idle GPU):** canonical vs raw
  text; stake weights on/off; typed vs untyped negatives.
- **Leads to:** better pair scores for everything downstream; expected
  +0.002 to +0.006.

### C3 — ClaimNet (listwise ownership; the flagship bet)

- **What:** one model input = one record plus all businesses that claim it;
  output = which of them owns it, or "none".
- **Why:** it asks the generator's question directly. Pointwise models score
  each look-alike claimant ~0.8 in isolation (France: up to 274 claimants of
  one record); side by side, the street name or unit number decides. The
  "none" option learns look-alikes whose owner is absent, which is exactly
  the no-match false merge (§4).
- **How:**
  - *Claimants:* S1s whose top-50 contains the record and whose v1 tree ≥ 0.01,
    top K = 8 by v1 score. The true owner is a claimant only if it retrieved
    the record (same as at inference); otherwise the label is "none".
  - *Input:* `record text [SEP] S1_1 text [SEP] S1_2 text … [SEP]`, canonical
    view, ≤ 400 tokens. Candidate spans tracked by token offsets; no new
    vocabulary.
  - *Head:* mean-pooled hidden states over each candidate span, concatenated
    with the CLS state → MLP → one logit per candidate; CLS → "none" logit;
    softmax over K+1. Auxiliary per-candidate sigmoid (weight 0.5) yields
    pair-level probabilities.
  - *Loss:* cross-entropy on owner index or "none", plus the auxiliary BCE.
    Claimant order shuffled every example (no position shortcut).
  - *Data from arena A:* ~40% records with ≥ 2 claimants, ~30% single-claimant
    owned, ~30% "none" (unowned, owner not retrieved, or claimed by no-match
    businesses). France: records whose owner is a pseudo-positive (§9).
  - *Init:* the BGE1800 encoder weights; new heads from scratch.
  - *Training:* batch 16 sequences, LR 2e-5 (heads 1e-4), bf16. Screen 8k steps
    (~60–90 min); final 25–30k steps (~2.5–3 h) *(est.)*.
  - *Inference:* every test record with ≥ 1 gated claimant (≈ 4–5M records,
    ~40 min *(est.)*); two claimant orders averaged if it helps on SLICE.
- **Kill rule:** added to the C4 resolver, SLICE F0.5 must rise ≥ +0.001, or
  contest win accuracy ≥ +5 points, or no-match false merges ≥ −1 point.
- **Leads to:** fewer wrong owners and fewer merged no-match businesses,
  mostly in France; expected 0 to +0.005 (untested).

### C4 — Resolver, decoder and rules (CPU)

- **What:** turn all scores into F0.5-optimal lists with one owner per record.
- **Why:** the metric rewards whole-list decisions; per-pair thresholds cannot
  see list length, rivals or the chance that a business has no match.
- **How:**
  - *Rows:* the uncertain band — pairs with tree ≥ 0.01 and blend in
    [0.02, 0.998], or contested, or empty-address.
  - *Resolver:* CatBoost, 5-fold out-of-fold by component on B-rest; isotonic
    calibration by address kind (France uses the pooled calibration).
  - *Features:* every model score (tree, v1 blend, C2, ClaimNet share and
    "none", idle-bet scores); rank in the business's list; list stats (count
    ≥ 0.5, top-1, top-2, gap, sum); number of claimants, rank among claimants,
    margin to the best rival, softmax share; address kind; source (S2/S3);
    exact normalized-name equality; business name and address multiplicity;
    record-name multiplicity; exact key; digit relation and edit counts
    (same-length substitutions, insertions, deletions); street-token Jaccard
    (French-normalized); native-script flag. **No country feature** (France is
    unseen in training).
  - *Decoder:* per business, the prefix of candidates sorted by calibrated p
    that maximizes E[5·TP/(4k+T)], where T = true links in the list + in the
    rest of the candidates + Poisson(λ≈0.055) never-retrieved links; k = 0 is
    allowed and wins when the no-match probability is high.
  - *Exclusivity (iterated best response):* a record claimed by several
    businesses stays only with its highest-p claimant, and only if p ≥ that
    list's break-even b(k) = [4/(5k+4)] / [1/(5k+1) + 4/(5k+4)] (0.727, 0.759,
    0.771 for k = 1, 2, 3); losers are re-decoded; repeat until stable (≤ 5
    rounds).
  - *Rules:* name-only rule and zero contested IDs enforced last; lists capped
    at 11 (train maximum).
  - *Grid (minutes each):* blend weights, gate 0.01 vs 0.003, λ, calibration
    grouping.
- **Leads to:** an upload on v1 scores; the final decision layer; expected +0.001 to
  +0.004 beyond C0.

### C5 — France track

- **Canonical view** in C2/C3 (§7 C2).
- **Pseudo-labels** from agent-F (§9) mixed into C2 and C3 training.
- **Street-token features** in C4.
- **France gate** on every candidate final: contested claims ↓, street-mismatch
  rate ↓, exact-key agreement ↑, mean list length 3.2–3.5, empty rate
  5.6–6.0%, and no US/India regression. Then a France-only upload to measure
  it (§11).
- Leads to: France from ≈ 0.934 toward 0.97 (hypothesis); each +0.01 ≈ +0.0015.

### C6 — No-match (singleton) track

- Report the no-match false-merge rate per country in every arena evaluation,
  next to F0.5.
- Sources of improvement: ClaimNet's "none", the decoder's k = 0 option, the
  name-only drop rule, and list-level features (top score, gap, rival claim on
  the top candidate, look-alike digit fingerprint of the top candidate,
  single-source support).
- If the rate is still > 3% after C3 + C4, add a dedicated business-level
  no-match classifier (CatBoost) whose probability feeds the decoder.
- Leads to: 4.7–6.7% → ≤ 2%, worth ≈ +0.0015 to +0.0025.

### C7 — India native-script recall probe

- **What:** transliterate native-script record names and addresses
  (Unidecode), embed with the existing E5 encoder, search Indian S1s against
  this small index (top-10), add any new candidates.
- **Why:** 69% of retrieval misses are Indian; 194 of 655 are native-script.
- **How:** first on the arena/TUNE (encode ~1.5M train native-script records,
  ~10 min *(est.)*); keep only if recall of true links rises ≥ +0.2 points.
  Then build for test (~850k records) and add new pairs to candidates,
  features, CE scores and `candidate_pairs.tsv`.
- Leads to: up to +0.001 to +0.002 (upper bound from recall).

---

## 8. Idle-GPU bets (queued only when the core is not using the GPU)

| Order | Bet | What / why | Kill rule |
| --- | --- | --- | --- |
| B1 | Qwen3-Reranker-0.6B fine-tune | Second model family (decoder LLM, score = logit(yes) − logit(no)); diversity for the resolver. Zero-shot scorer already exists (`score_qwen_reranker_gpu.py`) | Resolver gain < +0.0005 on SLICE |
| B2 | Noisy-student on test | Teacher = best stack on test; pseudo-positives p ≥ 0.97 and one-owner consistent (France: plus multi-view agreement); rivals as negatives; student trains on arena A + test labels (weight 0.3–0.5) with input noise. Adapts to test's denser pool and France | SLICE regresses > 0.0005 or France checks worsen |
| B3 | SetNet | One business + its top-12 records in one input; per-record match + count head (0–11). Targets partial misses and gives the decoder a per-business count | Resolver gain < +0.001 |
| B4 | Diff tags in text | Prefix pairs with `[digits sub 118→119] [street mismatch] [names shared 3] [address empty]` | < +0.0005 |
| B5 | Qwen3-Reranker-4B + LoRA | Only on the uncertain band; first measure band size (inference must fit in ≤ 2 h) | Band AUC not above the stack after 1 h of training |
| B6 | K = 100 / union retrieval | Only if the C7 probe shows many misses sit in ranks 51–200 | Recall gain < +0.2 points |

Parameter budget if everything ships: E5 0.118 + C2 0.568 + C3 0.568 +
B3 0.568 + B1 0.596 + B5 4.022 = 6.44B (< 8B).

---

## 9. France pseudo-labels (agent-F)

- **Normalization:** accent folding; street abbreviations expanded (`r.`,
  `av.`, `bd`, `pl.`, `chem.`, `all.`, `imp.`, `fbg`), derived from the data
  where possible; region/département names as admin tokens, **derived from
  the data** (components without digits occurring in ≥ 500 French addresses).
- **Positives (P+):** unique French exact key ∧ street-token Jaccard ≥ 0.5 ∧
  v1 blend ≥ 0.9 ∧ no rival claim with blend ≥ 0.5 ∧ record address not empty.
- **Negatives (P−):** every other claimant of a P+ record (one owner per
  record); same-template names in the same city with the same house number
  and a different street.
- **Audit:** 200 of each printed; ≥ 98% must look right by eye, else tighten.
- **Pseudo-holdout:** 10% of French businesses (deterministic hash) for
  P+ recall, P− rejection and contest-win checks (upper bounds only).
- **Diagnostics script** for any matching file: empty rate, mean and longest
  list, contested IDs, S2+S3 co-occurrence, exact-key agreement,
  street-mismatch rate.

---

## 10. GPU scheduling

### 10.1 Lanes and budgets

| Lane | Holds | Threads |
| --- | --- | --- |
| GPU `train` slot | One training job | 2 |
| GPU `infer` slot | Arena/SLICE/test scoring | 2 |
| Pod CPU | Feature computation, parquet builds | ≤ 3 (≤ 5 while arena features run alone) |
| Local CPU | Agents, resolver, decoder, upload files, diagnostics | ≤ 6, ≤ 10 GB RAM |

The `train` and `infer` slots co-run only if the G0 benchmark shows combined
throughput ≥ 1.15× serial; otherwise the critical-path job goes first.

### 10.2 Supervisor

- One process launched with `setsid nohup … < /dev/null` (an SSH exit killed
  the earlier 14,400-step run at step 11,240).
- Append-only queue file (`/workspace/az-ml-new/queue/queue.json`) re-read
  every 30 s; jobs are added without restarts.
- Per-job: slot, VRAM reservation (cap 38 GB total), threads, timeout,
  `/tmp` floor 4 GiB, completion marker, `status.json` (step, pairs/s, VRAM,
  ETA). **Failed jobs are never retried automatically.**
- Training jobs checkpoint every 2,000 steps to `/workspace` (keep the last two
  plus the best on SLICE) and resume from the latest checkpoint.
- The queue always holds the next job; while new code is written, a cheap
  legacy job fills the gap.
- A local background watcher wakes the controller when a job ends.

### 10.3 Throughput work (before the first screen)

- Current trainer: 147 pairs/s (4.6 steps/s × 32); tokenization synchronous;
  AdamW over all 568M parameters (256M are word embeddings).
- Fixes: batch 64, fused AdamW, frozen word embeddings, background
  tokenization thread (prefetch 4) with pinned memory, length-bucketed
  batches. Target ≥ 2× *(est.)*.
- **G0 benchmark:** 300 steps each of batch 32/64/128 × {fused, frozen}; pick
  the fastest stable setting.
- **G0b parity:** the fast trainer reruns the BGE1800 recipe on medium data
  (1,800 steps × 32 from pretrained) and must land within ±0.001 of TUNE
  0.968992; otherwise drop the embedding freeze.
- Inference keeps length sorting, pad-to-8, batch 128, upper-bound skipping
  (−26% calls) and selective BF16 (3.12×, identical accepted set on TUNE).

### 10.4 Queue order and estimates

| # | Job | Slot | Needs | Est. |
| --- | --- | --- | --- | --- |
| Q0 | R0 legacy: field-format 7,200 → plain 14,400 steps (running) | train | — | until ~03:35 |
| Q1 | G0 benchmark + G0b parity | train | fast trainer | 40 min |
| Q2 | Arena: mine missing 84k S1s → v1 scoring | infer | arena code | 1.5 h |
| Q3 | C2 screen (best-guess recipe) | train | arena, fast trainer | 1 h |
| Q4 | C3 ClaimNet screen | train | arena, ClaimNet code | 1.5 h |
| Q5 | SLICE scoring of each screen checkpoint | infer | screens | 5 min each |
| Q6 | C7 native-script probe | infer | encoder | 30 min |
| Q7 | B1 Qwen3-Reranker-0.6B screen | train | adapter | 1 h |
| Q8 | C2 final (60–80k steps) | train | C2 screen passes | 3.5–4 h |
| Q9 | C3 final (25–30k steps) | train | C3 screen passes | 2.5–3 h |
| Q10 | B-rest re-score with finals | infer | finals | 1 h |
| Q11 | **Final test inference** (tree features, C2, C3, extras) | infer | finals | 2.5–3.5 h, **starts by 27 Sep 10:00** |
| Q12 | B2 noisy-student / B3–B5 | train | idle slots | as time allows |

About 21 GPU-hours of core work within ~31 hours until 27 Sep 10:00; the
slack goes to idle-GPU bets and reruns.

### 10.5 Storage on the pod

- `/tmp` is scratch only. Move the ~15 GB test pool vectors
  (`artifacts/full_test_e5_ft200`) to `/workspace` after a hash check and
  leave a symlink.
- Everything that must survive lives on `/workspace` (§13).
- Feature caches only for the gated band plus a stratified sample; never
  110M × 100 features (≈ 44 GB).

---

## 11. Uploads (9 left: 4 on 26 Sep, 5 on 27 Sep)

Uploads are trigger-based, not scheduled: **upload whenever a candidate beats
the current best** on the arena (promotion rule §12) and its priced prediction
is positive. France-only variants are uploaded when a France change needs
measuring. The only fixed rule: the last upload must be the chosen final, so
one slot on 27 Sep stays in reserve until the end.

Every upload: per-country diff against the current best, arena-priced
prediction, official validator with `--check-ids`, strict checks (row order,
subset of candidates, zero contested IDs, LF endings), five-decimal result in
`learnings/submissions.json`, attribution (manual §5).

---

## 12. Decision protocol

1. **Yardstick:** arena macro F0.5 over every business in the evaluated
   population (including never-retrieved links and no-match businesses), with
   paired bootstrap over components (2,000 resamples for screens, 10,000 for
   finals). Also report precision, recall, false links, no-match false-merge
   rate, country, address kind, contest win accuracy.
2. **Model screens:** SLICE, with a fixed decision rule (threshold + one-owner
   + name-only) whose threshold is cross-fitted over two component halves;
   structured models (C3, B3) are judged as resolver inputs.
3. **Promote:** Δ ≥ +0.0005 with the 95% interval above zero, and no-match
   false merges not worse by > 0.2 points. Near-ties: fewer false links wins.
4. **Kill:** −0.002 at one third of a run's budget; a factor after two levels
   without gain.
5. **France:** the C5 gate, then a France-only upload.
6. **Board:** if an upload contradicts the arena's sign by > 0.0005, stop that
   line and recheck arena fidelity.
7. **Final:** best on B-rest, consistent with HOLDOUT-2 and the board;
   ignore public gains < 0.0003.
8. **HOLDOUT-2:** opened once after the 27 Sep 13:00 freeze: add it back to the
   arena, score its pairs with the frozen models, run the frozen decision
   layer, report once.

---

## 13. How the repository and storage evolve

### 13.1 Branches

- `codex/experiment-loop` (main checkout `az-ml-new`) becomes the **single
  integration branch**. First step on "begin": merge
  `codex/v2-plan-20260925` (worktree `az-ml-v2`, v2 modules at `1a55287`) into
  it, resolving the `learnings/` conflicts by keeping both records.
- Worker agents develop on their own worktree branches; the controller reviews
  and merges into the integration branch.
- The pod runs an exported copy of a committed revision
  (`git archive` → `/tmp/az-ml-v3/code-<commit>/`); every job records the
  commit it ran.

### 13.2 New code (under `code/business_entity_resolution/`)

| File | Contents | Owner |
| --- | --- | --- |
| `src/ber/name_only_rule.py` | C0 rule, U5 builder | agent-D |
| `src/ber/arena_decision.py` | Evaluator, component bootstrap, components, one-owner, decoder, iterated best response | agent-D |
| `src/ber/forensic_features.py` | Digit relations, street tokens, keys, multiplicities, French normalization | agent-F |
| `src/ber/france_pseudo_labels.py` | P+/P−, audit, pseudo-holdout | agent-F |
| `src/ber/label_free_diagnostics.py` | Per-country file diagnostics | agent-F |
| `src/ber/text_views.py` | Canonical view | controller |
| `src/ber/arena_build.py` | Missing-S1 mining, arena assembly, partitions, fidelity check | controller |
| `src/ber/arena_score.py` | v1 scoring of the arena (streaming, resumable) | controller |
| `src/ber/fast_trainer.py` | C2 trainer (typed negatives, stake weights, prefetch, bucketing, checkpoints/resume) | controller |
| `src/ber/claimnet.py` | C3 dataset, model, trainer, inference | controller |
| `src/ber/resolver.py` | C4 stacker + calibration | controller |
| `src/ber/stack_predict.py` | Full-stack test inference (resumable) → TSVs | controller |
| `run_queue_v3.py` | Two-slot supervisor | controller |
| `tests/test_*.py` | One test file per module | each owner |

Frozen v1 code and release files are never edited.

### 13.3 Durable storage on the pod (`/workspace/az-ml-new/`)

```
arena/                 pairs.parquet, v1_tree.npy, v1_blend.npy, v1_ce_scored.npy,
                       label.npy, s1_meta.parquet, rec_meta.parquet, manifest.json
models/<run>/          checkpoint-best/, checkpoint-last/, run.json
experiments/<run>/     config.json, log, slice_scores.npy, metrics.json, run.json
test_scores/<model>/   float16 scores aligned to v1 scored_pairs (+ new-pair tables)
queue/                 queue.json, status/, complete/
releases/<release>/    matching_results.tsv, candidate_pairs.tsv, validation.json, ready.json
releases/submission_v1_scores/   preserved v1 test arrays (hash-verified)
```

Every `run.json` records: git commit, full config, seed, arena manifest hash,
data identities, checkpoint path + SHA-256, elapsed time, metrics.

### 13.4 Local storage (not in Git)

```
artifacts/arena/           compact arena arrays for the decision layer
artifacts/v1_test_scores/  v1 test arrays (+ dense_test reference/pool)
artifacts/test_scores/     per-model test scores copied from the pod
artifacts/resolver/        resolver models, out-of-fold predictions
output/<release>/          upload files, validation, diff and price reports
output/france_pseudo/      pseudo-labels, audits, diagnostics
```

### 13.5 In Git

Code and tests; small configs; `docs/PLAN_v2.md` (this plan, checklist
ticked); `CONTEXT.md` (running state); `learnings/` — one short keep/kill note
per experiment, the regenerated ledger (`experiments.csv/json`,
`scoreboard.md`), `submissions.json`. Commits after every promoted or killed
experiment and every upload, with the attribution line.

---

## 14. Orchestration

- **Controller (main session):** pod, supervisor and queue; GPU code (§13.2);
  every decision; documents; builds and prices every upload file. Background
  watchers wake it when jobs end.
- **agent-D and agent-F** (already running on the groundwork): local CPU only,
  own worktrees, deliver code + tests + reports for review.
- **agent-R** (27 Sep morning): release path — portable reproduction of the
  final pipeline, ZIP builder update, methodology draft.
- **User:** clicks uploads and reports five-decimal scores. If the terminal
  session closes, the pod finishes its queue but no new decisions are made;
  `/loop` keeps the controller running unattended.

---

## 15. Biases, risks and guards

| Risk | Guard |
| --- | --- |
| Picking a lucky winner among many screens | Promotion interval gate; B-rest reporting; HOLDOUT-2 once |
| Country: only US/India measurable | France pseudo-labels, gate, France-only uploads |
| Arena easier than test (4.68 vs 5.75 records per business) | Fidelity check; density-matched fallback |
| Anchoring on v1 (retrieval, negative mining) | C7 probe; typed negatives beyond v1's mistakes |
| Pseudo-label echo | Multi-view agreement, 20% cap, audit |
| Public board is a subset | Arena decides; ignore < 0.0003 public gains |
| Engineering time (the real bottleneck) | Two workers in parallel; best-guess recipes; no speculative infrastructure |
| Job death / memory | Detached supervisor, checkpoints, bounded tree fits |
| Last upload counts | One slot reserved for the chosen final |
| Final inference overruns | Starts by 27 Sep 10:00; the previous uploaded file is the fallback |

**Not doing:** hyperparameter sweeps of v1; blanket removals/additions; country
thresholds; fuzzy name-only additions; address-side trade-name rules; cluster
retrieval; tree retraining (forensic features go to the resolver instead);
ByT5 denoiser; selection on the opened medium holdout or 12k-sample decision
tuning.

---

## 16. What this leads to (estimates, not measurements)

| Scenario | What lands | Public score |
| --- | --- | --- |
| Floor | C0 + C4 on v1 scores | ≈ 0.968–0.971 |
| Base | + C2 at arena scale, partial France | ≈ 0.975–0.980 |
| Strong | + C3 ClaimNet and France pseudo-labels land | ≈ 0.982–0.986 |
| Stretch | + noisy-student, native-script recall, second family | ≈ 0.987–0.990 |

First place needs the stretch scenario. Each upload tells us which scenario we
are in; spend the remaining GPU on whatever is paying.

---

## 17. Checklist

Owners: `ctl` controller, `D` agent-D, `F` agent-F, `R` agent-R, `user`.

### Done
- [x] ctl — Relaunch legacy follow-ups detached (`r0_fields7200_20260926b`, `r0_pipe14400_20260926b`)
- [x] ctl — Preserve v1 test arrays on `/workspace` (hash-verified)
- [x] ctl — Commit v2 worktree code (`1a55287`)
- [x] ctl — Copy v1 test arrays and test reference/pool locally (hash-verified)

### Groundwork (running)
- [ ] D — C0 rule reproduced on TUNE; U5 file built, validated, priced
- [ ] D — `arena_decision.py` + tests (reproduces 0.968992, 0.969029, +0.003962)
- [ ] F — `forensic_features.py` + tests
- [ ] F — France P+/P−, audits, pseudo-holdout; diagnostics on v1 and #3
- [x] ctl — R0 field-format 7,200: 0.973686 vs 0.968992 (+0.0047)
- [ ] ctl — R0 plain 14,400 (running): separates steps from format

### On "begin"
- [ ] ctl — Merge `codex/v2-plan-20260925` into the integration branch
- [ ] ctl — `run_queue_v3.py` supervisor; move test pool vectors off `/tmp`
- [ ] ctl — `fast_trainer.py`; Q1 G0 + G0b
- [ ] ctl — `arena_build.py`, `arena_score.py`; Q2 arena
- [ ] D — Components, partitions, fidelity check, arena D-grid
- [ ] user — Upload U5

### Core
- [ ] ctl — `text_views.py`; Q3 C2 screen
- [ ] ctl — `claimnet.py`; Q4 C3 screen
- [ ] ctl — `resolver.py`; C4 on B-rest; apply to v1 test scores; upload if it beats the best
- [ ] ctl — Q6 C7 native-script probe
- [ ] ctl — First full stack on test; France-only variant when France changes
- [ ] ctl — Q8/Q9 finals; Q10 B-rest re-score; resolver refit

### Idle-GPU bets
- [ ] ctl — B1 Qwen3-Reranker-0.6B
- [ ] ctl — B2 noisy-student
- [ ] ctl — B3 SetNet
- [ ] ctl — B4 diff tags
- [ ] ctl — B5 Qwen3-Reranker-4B LoRA (after the band-size check)
- [ ] ctl — B6 K=100 / union (only if the C7 probe says so)

### Final (27 Sep)
- [ ] ctl — Q11 final test inference (starts by 10:00)
- [ ] ctl — 13:00 freeze; HOLDOUT-2 once
- [ ] ctl — Final file validated and uploaded last; one slot kept in reserve
- [ ] R — Reproduction path, ZIP, methodology (pseudo-labels disclosed)
- [ ] ctl — `learnings/`, CONTEXT.md, `submissions.json` updated after every result
