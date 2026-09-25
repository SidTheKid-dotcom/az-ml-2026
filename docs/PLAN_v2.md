# Final plan — ownership-aware matching (26–27 Sep 2026)

Replaces the earlier v2 backlog. Read `AGENTS.md`, `CONTEXT.md` and
`learnings/negative_side_analysis_manual.md` first. This plan is executable:
every step has an owner lane, a cost estimate, a gate and a fallback.
Estimates marked *(est.)* must be replaced by measurements as they arrive.

**No plan guarantees first place.** Leaders are at 0.988; our best is
0.964563; the private board is the unseen part of test. The plan maximizes
expected score per GPU-hour and keeps an uploadable file at every phase.

---

## 0. Fixed facts and rules of the game

| Fact | Value | Consequence |
| --- | --- | --- |
| Deadline | **27 Sep 23:59 IST** | Freeze models 27 Sep 13:00; final upload by 20:30 |
| Uploads | 5/day; count used today is uncertain (IST vs UTC day) | Check the portal; plan uses U5–U12 (+1 emergency) |
| Which upload counts | **The last one** | Reserve the final slot; never end on an experiment |
| Scored file | `matching_results.tsv` only | Candidates file must still contain every predicted ID |
| Compute | 1× A40 46 GB; pod cgroup 7.65 cores / 50 GB; `/tmp` 30 GB overlay (5.6 GB free); `/workspace` durable, slow | One training + one inference job at most; CPU is the co-bottleneck |
| Local | 8 cores / 14 GB, all raw data, v1 test TSVs | Decision-layer sweeps and test-file construction run here |
| Models | MIT/Apache, total ≤ 8B params | E5-small 0.118 + BGE-m3 reranker 0.568 + one second family ≤ 0.6 → ≤ 1.3B |
| No external data | Only supplied files | Pseudo-labels from unlabeled test text are transductive; disclose in methodology |

Current best upload: #3 (v1 + one-owner), public **0.964563**. Frozen v1
scores for all 86,627,200 test pairs are preserved and hash-verified at
`/workspace/az-ml-new/releases/submission_v1_scores/`.

---

## 1. Evidence this plan rests on

Measured 26 Sep 01:30–02:00 IST on the saved v1 TUNE scores (12,000 medium TUNE
S1s, 600,000 dense pairs, full 10.32M pool) joined to raw text and full-train
ownership, plus label-free analysis of the v1 test file. Scripts are in the
session scratchpad (`err_anatomy.py`, `sibling.py`, `empty_rule.py`,
`test_contests.py`, `dba.py`, `misses.py`); port them to `code/` when reused.

### 1.1 Where v1 loses points (US/India TUNE, 3.10 points lost per 100 S1s)

| Bucket | Points | Entities |
| --- | ---: | ---: |
| Partial miss only (true IDs missing, no wrong IDs) | 1.43 | 2,041 |
| Dilution only (wrong IDs on a matched S1) | 0.67 | 352 |
| Abstention (matched S1 predicted empty) | 0.53 | 63 |
| Singleton false merge | 0.28 | 33 |
| Mixed wrong + missing | 0.19 | 75 |

Recall is the largest local loss. Of 41,476 true links: 655 never retrieved,
1,885 retrieved but rejected (252 by the tree gate, 1,633 by the blend; 706 sit
just under the threshold at 0.70–0.775).

### 1.2 The name-only mechanism (largest single fixable hole)

- Empty-address records (blank, `<NULL>`, `##`) are **4.4% of owned records but
  0.3% of unowned records** → an empty-address record is owned ~98% of the
  time. The generator does not make empty-address decoys.
- v1 accepts only **25%** of empty-address true links; they are 1,177 of the
  1,885 rejected true links (62%).
- Exact normalized name, unique among the country's S1s: precision **0.978**,
  v1 accepts 41%. Name shared by 2 S1s: precision 0.469 (coin flip). ≥3: 0.028.
- Rule "accept exact-unique, drop exact-shared" on TUNE: **+0.003962**
  (421 entities better, 63 worse; false links 494 → 411; +282 true links).
  On top of the P1 count rule: +0.003265. Fuzzy-name additions: −0.00093 (don't).
- The mirror rule (unrelated name + matching address) is **only 51% precise**
  even at a unique address: the generator plants same-address decoys. Not a rule.

### 1.3 What the false links are (494 on TUNE)

- 246 belong to **another S1** (contest-resolvable only with competitors
  present); 120 of these are empty-address sibling records; 117 are
  same-address co-located businesses whose record name is a trade name.
- 248 are unowned decoys: 49% of unowned hard negatives are **same-length digit
  substitutions** (a different unit/house number). True links with changed
  digits mostly change *length* (drop/insert): same-length share 22% for true
  links vs 57% for decoys → a forensic "digit edit type" feature.

### 1.4 Retrieval misses (655)

392 have digits, 213 empty, 50 no digits; 194 native-script names; 69% India.
Exact-name reverse lookup would recover only 7. The lexical+phonetic+dense
union recovered ~315 of these on the same TUNE (recall 0.984 → 0.992) — the
only material recall branch, and expensive on test.

### 1.5 France (label-free, test)

- France holds **36,447 of 48,927** v1 contests (74.5%). 9.7% of French links
  with digits are contested vs 0.16% (US) and 0.63% (India).
- French S1 names are templated from a small vocabulary ("Nantes Club EURL",
  "Nantes Conseil EURL", "Nantes Centre EURL"), house numbers repeat, 10.9% of
  French S1 addresses are shared (4–5% US/India). The **street name** is the
  discriminator ("5 Rue Coustou" vs "5 Rue Bisson"); v1 never learned it.
- Estimated from our uploads: US/India ≈ 0.970 and France ≈ 0.934 after
  one-owner (inferred, not measured).

### 1.6 Why local TUNE failed us

The one-owner rule touched 2 contests on TUNE (+0.00004) and 48,927 on test
(+0.0082 public). The P1 count rule gained +0.0011 on TUNE and lost 0.0003 on
the board for US/India. A 12k sample of S1s has no competitors, so it cannot
see contests. **Every decision rule must be chosen in a competitor-complete
world (the arena, §4).**

---

## 2. Strategy

Stop thresholding pairs independently; **assign records to owners**. Every
Source 2/3 record has at most one owner. Three levers, one yardstick:

1. **Ownership decision layer** (cheap, CPU): name-only rule, learned
   contest/ownership resolver, exact expected-F0.5 list builder, singleton
   probability. Targets partial misses, siblings, contests.
2. **Matcher v2** (GPU): reranker trained on 40× more businesses, on the
   arena's own mistakes, weighted by entity-point stake, plus French
   pseudo-labels with contest-loser negatives. Targets the 0.969 → 0.994 gap
   between matcher and retrieval oracle, and France.
3. **France-specific signal**: street-token features, French normalization,
   multi-view pseudo-labels, factorial uploads (the only French meter).
4. **Yardstick: the arena** — all 2.2M training S1s scored together against
   the full pool, so contests appear as on test.

---

## 3. Compute plan and GPU scheduling

### 3.1 Lanes and budgets

| Lane | Runs | Thread budget |
| --- | --- | --- |
| GPU-train | One reranker/tree training job at a time | 2 (tokenizer prefetch + main) |
| GPU-infer | Arena/slice/test scoring, CatBoost-GPU | 2 |
| Pod CPU | Feature computation, parquet builds | 3 (never exceed 7 total) |
| Local CPU | Decision-layer grid, resolver, pseudo-labels, test-file builds | 6 of 8 cores, ≤ 10 GB RAM |

Co-run GPU-train and GPU-infer only if the benchmark (§3.4) shows combined
throughput ≥ 1.15× serial; otherwise serialize with inference first when it is
on the critical path. Never let GPU wait on CPU: build the next job's inputs
(parquets, pseudo-labels) while the current job trains.

### 3.2 Supervisor (replace `keep_gpu_busy.sh` and the ad-hoc queues)

- One supervisor, launched with `setsid nohup … < /dev/null` so SSH exit cannot
  kill it (the 14,400-step run died this way at step 11,240).
- Two slots (`train`, `infer`) with VRAM accounting (cap 38 GB total), per-job
  thread counts, and a refusal when `/tmp` has < 4 GiB free.
- Append-only queue file re-read every 30 s: add jobs without restarting.
- Completion marker per job; **failed jobs are never auto-retried** (no loops).
- Every training job checkpoints every 2,000 steps to `/workspace`, keeps the
  last 2 checkpoints plus any "best on slice", and resumes from the latest
  checkpoint when relaunched with the same name.
- Every job writes `status.json` (step, pairs/s, VRAM, ETA) for monitoring.

### 3.3 Throughput fixes (trainer is the real GPU bottleneck)

Current trainer: 4.6 steps/s × batch 32 = **147 pairs/s**; tokenization runs
synchronously in the step loop; AdamW (non-fused) updates all 568M params
including 256M word-embedding params every step.

| Fix | Why |
| --- | --- |
| Batch 64 (test 128), LR 2e-5 → 3e-5 | More tokens per kernel launch; amortize optimizer step |
| `AdamW(fused=True)` | Optimizer step is memory-bound on 568M params |
| Freeze word embeddings | Cuts optimizer traffic ~45%, VRAM ↓; standard for reranker fine-tuning |
| Background tokenization thread (prefetch 4), pinned memory, `non_blocking` | GPU never waits on the tokenizer |
| Length-bucketed batches (sort within windows of 50 batches) | Less padding (dynamic padding already on) |
| Warmup 5%, then constant LR for screens; cosine for final | Intermediate checkpoints are comparable → one run gives a learning curve |

Target ≥ 2× pairs/s *(est.)*, verified by G0 before any screen.
**G0b quality parity:** the fast trainer (frozen embeddings, fused AdamW,
prefetch) retrains the BGE1800 recipe on medium data from the pretrained model
(1,800 steps × 32) and must land within ±0.001 of medium TUNE 0.968992;
otherwise drop the embedding freeze and rerun. Speed without parity is not
adopted.

**Never idle:** while new code is being written, keep one cheap, informative
legacy job queued (e.g. the 1,800-step conflict-only negatives run from
`experiments/queue.json`, output to `/workspace`). Inference
already has length sorting, pad-to-8, batch 128, exact upper-bound skipping
(−26% calls) and selective BF16 with FP32 rescoring within ±0.04 of threshold
(3.12× faster, identical accept set on TUNE). v1 test inference took 1.9 h in
FP32 (CE 5,913 s); with BF16 expect ~1 h *(est.)*.

### 3.4 Storage

- `/tmp` (5.6 GB free): move `artifacts/full_test_e5_ft200` (test pool vectors,
  ~15 GB) to `/workspace` after hash check, keep a symlink.
- All checkpoints, arena arrays and score arrays live on `/workspace`; active
  memory maps are copied to `/tmp` only while in use.
- Feature caches: only the gated band (v1 tree ≥ 0.01, ~9M arena pairs) plus a
  2M stratified sample of the rest. Never cache 110M × 100 features (44 GB).

---

## 4. The arena (competitor-complete training world)

### 4.1 Build (GPU ~1 h + pod CPU ~45 min, est.)

1. Candidates: P2 dense K50 already mined for 2,122,821 S1s (106,141,050 pairs,
   `/workspace/az-ml-new/v2/dense/`). Mine the remaining ~84k S1s (quick,
   medium, sealed v1 holdout) with the same E5-ft200 encoder and pool vectors
   (minutes). Result: every training S1 × top-50 of the full 10,320,219 pool.
2. **Exclude HOLDOUT-2 (50,000 S1s) from the development arena entirely**,
   including its ground-truth rows. Records it owns then look unowned — which
   is correct for a world without those businesses. HOLDOUT-2 returns once at
   the end (§8.3).
3. Score all pairs with frozen v1 (tree gate 0.01, BGE1800 CE, blend 0.75),
   reusing the streaming test predictor pointed at train data, selective BF16.
4. Save to `/workspace/az-ml-new/arena/`: `pairs.parquet` (s1_idx, rec_idx),
   `v1_tree.npy`, `v1_blend.npy`, `v1_ce_scored.npy`, `label.npy`,
   `s1_meta.parquet` (country, component, partition, name/address
   multiplicity), `rec_meta.parquet` (source, address kind, owner idx),
   `manifest.json` with hashes. Download the compact arrays locally (~2 GB).

### 4.2 Partitions (by claim-graph component)

- Build a graph over S1s; connect two S1s when they share a record where both
  pairs have v1 blend ≥ 0.05. Take connected components. If the largest
  component exceeds 5% of S1s, raise the edge threshold (0.2, then 0.5) and
  record the number of cut cross-partition claims.
- **A (≈70%)**: matcher training. All components containing quick or medium
  TRAIN S1s go here (v1/E5 saw them).
- **B (≈30%)**: never trained on by any matcher. Components containing medium
  TUNE or v1-holdout S1s go here, so medium TUNE stays a legacy comparison.
  - **SLICE**: fixed 100k-S1 subset of B components for matcher screens.
  - B-rest: resolver development (5-fold by component) and final reporting.
- Bootstrap unit everywhere: **component**, not S1 (contests couple S1s).

### 4.3 Metric

Official macro F0.5 over every S1 in the evaluated population, including
retrieval misses (count as FN), singletons and empty predictions. Report
precision, recall, false links, singleton accuracy, country, address kind,
contest count and contest win accuracy.

### 4.4 Fidelity check (gate before trusting the arena)

v1 + one-owner in the arena must reproduce the test's US/India pattern: contests
per 1,000 links, share of contested links by address kind, mean list length,
empty rate, and removed-claim wrong-rate ≈ 85% (inverted from uploads). Train
has 4.68 records per S1, test 5.75. If arena contest density is materially
lower, build a density-matched arena: add a random share of test US/India
records to the pool as absent-owner records (vectors already exist) and redo
dense top-50 for B only. Decide within 30 minutes of the arena landing.
If fidelity is still not reached by 26 Sep 09:00, use the arena only as a
*relative* yardstick for matcher screens, and set decision-layer thresholds by
pricing against the board (manual §5) rather than by arena optimum.

---

## 5. Experiments

Each factor is tested one at a time from the current best, then winners are
combined. Full factorials are not affordable on one GPU.

### 5.1 Decision layer — cached arena scores, minutes each (local CPU)

| ID | Experiment | Details |
| --- | --- | --- |
| D0 | v1 baseline | Threshold 0.775 |
| D1 | + one-owner | Keep highest blend; ties → lower S1 index |
| D2 | + name-only rule | §1.2 rule using S1 name multiplicity by country |
| D3 | Contest rule variants | Keep best only if p ≥ list break-even b(k) = [4/(5k+4)] / [1/(5k+1) + 4/(5k+4)] (k=1: 0.727, 2: 0.759, 3: 0.771); else drop from all. Margin δ ∈ {0, 0.05, 0.1} |
| D4 | Exact expected-F0.5 list builder | Calibrated p (isotonic by address kind × country group), best prefix of sorted p maximizing E[5·TP/(4k+T)], T = in-list TP + other-candidate TP + Poisson(λ≈0.055) retrieval misses; T=0 handled (singleton). Reuse `v2_decision.py` |
| D5 | Learned ownership resolver | CatBoost on (record, claimant) rows in the uncertain band (tree ≥ 0.01 and blend ∈ [0.02, 0.998], or contested, or empty address). Features below. 5-fold OOF by component on B-rest. Then D4 + exclusivity (argmax p per record if ≥ b(k), else drop; re-run D4 for losers) |
| D6 | Blend/gate re-fit | CE weight {0.5, 0.65, 0.75, 0.85}, gate {0.003, 0.01} (gate recall costs 252 true links on TUNE) |

**Resolver features:** blend, tree, CE logit; rank in the S1's list; S1 list
stats (count ≥ 0.5, top-1, top-2, gap, sum p); record-side: number of
claimants (tree ≥ 0.01; blend ≥ 0.5), this S1's rank among claimants, margin to
best rival, softmax share of blend across claimants; address kind; source
(S2/S3); exact normalized-name equality; S1 name multiplicity and record-name
multiplicity (country); S1 address multiplicity; exact-key hit (manual §8);
digit relation class and edit-type counts (same-length substitutions,
deletions, insertions, transpositions); street-token Jaccard (address tokens
minus digits, city, region/département and street-type words; French
normalization §5.4); native-script flag; E5 cosine between the record and the
claimant's most confident other record ("cluster vote", optional; pool
vectors exist). **No country feature**: France is unseen in training, and an
unseen category is undefined behavior. Country enters only through
calibration groups (US, India; France uses the pooled calibration).

### 5.2 Tree — CatBoost on GPU, 15–30 min each (GPU-infer lane)

| ID | Experiment |
| --- | --- |
| T1 | v1 feature set, trained on 5M stratified rows from A (all positives of 150k S1s + gated negatives + 10% random), depth 10, 2,000 iterations |
| T2 | T1 + forensic features (digit edit type, street tokens, multiplicities, address kind, key) |
| T3 | T2 with gate 0.003 and gate-recall report (how many true links enter CE) |

Evaluate tree-only and tree + v1 CE on SLICE. A new tree changes which pairs
reach the CE, so final scoring re-runs CE on the new gated set.

### 5.3 Reranker — GPU-train lane, ~45 min each (after G0 speedup)

Fixed unless varied: init from BGE1800 checkpoint; fast trainer; batch 64;
15k steps; checkpoints at 5k/10k/15k scored on SLICE; max length 256.

| ID | Factor | Levels |
| --- | --- | --- |
| R0 | Legacy follow-ups on medium data (running now) | field-format 7,200 steps; plain 14,400 steps. Read against medium TUNE 0.968992 |
| R1 | Data scale (base) | 150k A-S1s vs the 36k medium TRAIN on the same step budget |
| R2 | Diversity | 500k A-S1s, same steps |
| R3 | Negatives | N0 current sampler / N1 arena mistakes (v1 false accepts, contest losers, negatives with blend ≥ 0.3) / N2 typed decoys (same-length digit substitutions, same-address different name, empty-address siblings) / N3 N1+N2 |
| R4 | Stake weights | Positive weight 1/(5T−4); negative weight 4/(5T+4) for T ≥ 1, 1.0 if the S1 is a singleton; class-normalized, capped at 5× |
| R5 | Text | plain `name | address` / field-tagged (from R0) / + context tokens (`[names:3] [addr:empty]`) |
| R6 | Loss | BCE / BCE + group softmax over claimants of the same record (owner vs rivals) |
| R7 | France mix | 0% / 15% / 30% French pseudo-labels (§5.4) |
| R8 | Family | Qwen3-Reranker-0.6B with yes/no adapter (score = logit(yes) − logit(no) at the final token; code exists in `score_qwen_reranker_gpu.py` for zero-shot) / mDeBERTa-v3-base single logit (allowlist role check must admit `encoder`) |
| R9 | Init | Pretrained bge-reranker-v2-m3 vs BGE1800 |

Priority order: R1 → R3 → R4 → R7 → R8 → R5 → R6 → R2 → R9.

**Screen protocol (isolates matcher quality):** score SLICE's gated pairs (v1
tree ≥ 0.01, ~0.5M pairs, ~3 min) → blend with v1 tree at w ∈ {0.5, 0.75, 1.0}
→ fixed decision (threshold + D1 + D2) with threshold cross-fitted across two
component halves of SLICE → report mean. Also report medium TUNE for legacy
comparability.

### 5.4 France — no labels

| ID | Step |
| --- | --- |
| F1 | French normalization: fold accents; `r.`→rue, `av.`→avenue, `bd`→boulevard, `pl.`→place, `chem.`→chemin, `all.`→allée, `imp.`→impasse, `fbg`→faubourg; drop region and département names (region vs département is noise); street tokens = remaining non-digit tokens minus city |
| F2 | Pseudo-positives P+: unique French exact key ∧ street-token Jaccard ≥ 0.5 ∧ v1 blend ≥ 0.9 ∧ no rival claimant with blend ≥ 0.5 ∧ record not empty-address. Multi-view agreement, not one signal |
| F3 | Pseudo-negatives P−: every other claimant of a P+ record (one-owner makes these near-certain) plus same-template names in the same city with the same house number and a different street. These are exactly the France confusions |
| F4 | Audit: print 200 P+ and 200 P−, review by eye; require ≥ 98% apparent precision, else tighten F2 |
| F5 | French pseudo-holdout: 10% of French components held out; measure P+ recall, P− rejection, key-owner contest win rate (an upper bound, not French accuracy) |
| F6 | Optional second round: re-derive P+ from the final model's confident, one-owner-consistent French predictions ∧ key agreement; retrain 5k steps |

**French label-free diagnostics** on every candidate file: contested claims
before resolution, mean list length (3.2–3.5), empty rate (5–6.5%),
exact-key agreement (share of P+ present; key links given to a different
owner), S2+S3 co-occurrence (~80%), and **street-mismatch rate** among
accepted links (template name matches but street tokens disagree).

### 5.5 Retrieval — gated, lower priority

| ID | Experiment | Gate |
| --- | --- | --- |
| X1 | K=100 dense on SLICE | Keep only if matcher-scored SLICE F0.5 rises ≥ +0.0005 |
| X2 | Transliterated / address-only E5 branch for native-script India records | Same |
| X3 | Lexical+phonetic union on test | Measure mining throughput on 50k test S1s first; only if it fits before 27 Sep 06:00 |

Any retrieval change on test requires recomputing features and CE for the new
pairs and rewriting `candidate_pairs.tsv`.

### 5.6 Inference-time extras (cheap)

E1 average the last 3 checkpoints; E2 order-swap TTA (score record-first and
S1-first, average logits; +1× CE cost); E3 two-family blend (weights fit on
B-rest OOF).

---

## 6. Decision protocol

1. **Yardstick:** arena macro F0.5 (US/India) on the relevant population, with
   paired bootstrap over components (2,000 resamples for screens, 10,000 for
   finals).
2. **Promote** a factor if Δ ≥ +0.0005 and the 95% CI lower bound > 0. For
   |Δ| < 0.0005, keep the variant with fewer false links, or the simpler one.
3. **Kill** a GPU run at the 5k-step checkpoint if it trails the base curve by
   ≥ 0.002. Stop a factor after two levels show no gain.
4. **France:** accept a France change only if diagnostics move the right way
   (contest rate ↓, street-mismatch ↓, key agreement ↑, list length and empty
   rate within range) and US/India does not regress; then confirm with a
   factorial upload (France-only change vs not).
5. **Before every upload:** diff against the current best per country (links
   added/removed, owner changes, newly empty/filled rows, contested IDs,
   longest list). Price each changed set with the arena-measured precision of
   the same rule and the per-link values (manual §4). Record the predicted Δ.
6. **After every upload:** record five decimals in
   `learnings/submissions.json`, attribute (manual §5), update CONTEXT.md. If
   the board contradicts the arena's sign by > 0.0005, stop that direction and
   recheck arena fidelity (§4.4).
7. **Final choice:** best arena B-rest OOF, consistent with HOLDOUT-2 and the
   board. Do not select on a public gain < 0.0003; the private board is the
   other part of test.

---

## 7. Final modeling recipe (filled in from screens, 26 Sep ~19:00)

1. Tree: best of T1–T3 (features + gate).
2. Reranker FINAL-A: winning R factors combined, trained on 1–1.5M A-S1s
   (~5M pair presentations, ~3.5–4 h *(est.)*), cosine LR, E1 checkpoint
   average.
3. FINAL-B: second family (R8 winner) ~2 h, only if the R8 blend gained on
   SLICE; otherwise a second BGE seed.
4. Score B-rest with FINAL-A/B (~1 h); fit blend weights and the D5 resolver
   (5-fold OOF); choose D3/D4 settings.
5. Re-score test: new tree features (pod CPU ~40 min) + CE on the new gated
   set (~1 h BF16); apply resolver + list builder + exclusivity; validate.

---

## 8. Timeline (IST), GPU queue and uploads

### 8.1 26 Sep

| Time | GPU-train | GPU-infer / pod CPU | Local / controller | Upload |
| --- | --- | --- | --- | --- |
| 02:00–03:45 | R0 field-format 7,200 → R0 plain 14,400 (running) | — | Commit `az-ml-v2` code; supervisor v2; fast trainer; arena scorer; U5 file build | |
| 03:45–04:15 | G0 trainer benchmark (batch 32/64/128 × fused/frozen) | G1 mine missing 84k S1s | U5 diagnostics + pricing | |
| 04:15–05:45 | — | **G2 arena v1 scoring** (critical path) | Features/decision agents coding | |
| 05:45–06:30 | R1 starts | Arena partition + fidelity check | Download arena arrays | **U5**: #3 + name-only rule (after 05:30 in case the day is UTC) |
| 06:30–19:00 | R1, R3, R4, R7, R8, R5, R6, R2, R9 (~12 runs) | SLICE scoring per checkpoint; T1–T3; X1 | D1–D6 grid; resolver; F1–F5 pseudo-labels | **U6** ~12:30: D5 resolver on v1 test scores |
| 19:00–20:30 | — | Best screen model on B-rest + test | Resolver refit | |
| 20:30–01:30 | FINAL-A training | — | France diagnostics | **U7** ~21:30: best screen model + resolver; **U8** ~23:00: France factorial |

### 8.2 27 Sep

| Time | GPU | Local / controller | Upload |
| --- | --- | --- | --- |
| 01:30–03:30 | FINAL-B (if justified) | — | |
| 03:30–06:30 | B-rest re-score, resolver refit, test re-score | Blend weights, D3/D4 | |
| ~08:00 | — | Validate | **U9**: final pipeline |
| 08:00–12:00 | F6 or E2 (whichever SLICE favors) | — | **U10** ~12:30 |
| **13:00** | **Freeze** | HOLDOUT-2 one-shot | |
| 13:30–16:30 | Final test file if changed | Official + strict validation | **U11** ~16:30: intended final |
| 16:00–20:00 | — | ZIP, README, reproduce, methodology | |
| 20:30 | — | — | **U12**: re-upload the chosen final if U11 is not it |
| 20:30–23:59 | No experiments | Buffer | Emergency slot only |

### 8.3 HOLDOUT-2 (one shot)

After freeze, add HOLDOUT-2 back to the arena, score its pairs with the frozen
models (~20 min), run the frozen decision layer on the full arena, and score
HOLDOUT-2 S1s only. Report once. No changes after reading it except aborting a
clearly broken component (then fall back to the previous uploaded file).

### 8.4 Fallbacks

Every phase leaves an uploadable file: U5 needs only CPU; U6 needs only the
arena and v1 scores; if FINAL-A fails, the best screen model is final. The
fallback final is always the best-scoring uploaded file, kept locally with its
hash.

---

## 9. Agents

- **Controller (main session):** pod supervisor and queue, all decisions,
  uploads guidance, docs, integration. Only the controller touches the pod.
- **Worker agents** (background, separate worktrees, local CPU only, each with
  a spec, fixtures and acceptance tests; controller reviews before deploy):
  1. *features*: forensic + French normalization features, vectorized
     (≥ 1M rows/s/core target), unit tests.
  2. *decision*: list builder (exact E[F0.5] DP), exclusivity, resolver
     harness, component bootstrap; must reproduce v1 TUNE 0.968992 exactly.
  3. *release* (27 Sep morning): `reproduce.py` and packaging for the new
     pipeline, methodology draft.

---

## 10. Release

Final ZIP: `output/{matching_results.tsv, candidate_pairs.tsv}` (the uploaded
final bytes), `code/business_entity_resolution/{src,README.md,requirements.txt}`
with every component that produced the file, filled `Documentation_template.md`
(decision layer, arena, resolver, pseudo-labels disclosed as transductive use
of unlabeled test text). LF line endings. Official validator `--check-ids` plus
strict coverage/subset. Weights stay external (500 MB cap).

---

## 11. Risks and what we stop doing

| Risk | Mitigation |
| --- | --- |
| Jobs die with SSH / OOM | Detached supervisor, checkpoints every 2k steps, bounded CatBoost |
| Arena unlike test (density 4.68 vs 5.75) | §4.4 fidelity check; density-matched arena if needed |
| Winner's curse over many screens | Promotion CI gate; B-rest reporting; HOLDOUT-2 one-shot |
| France unmeasurable | Multi-view pseudo-labels, audit, diagnostics, factorial uploads |
| Public-board overfitting | Arena decides; board confirms; ignore gains < 0.0003 |
| Last upload counts | Final slot reserved; no experiments after 20:30 on 27 Sep |
| Transductive pseudo-labels judged unfair | Uses only supplied data; disclosed. Team decision required before R7/F6 run; without it, France relies on F1 features and the resolver only |
| Pod loss | All arrays/checkpoints on `/workspace`; best file local with hash |

Stop: blanket removals/additions, country thresholds, fuzzy name-only
additions, address-side trade-name rules, cluster retrieval, unbounded tree
fits, any selection on the opened medium holdout or on 12k-sample decision
tuning, Qwen3-4B adjudication (cost), ByT5 denoiser (weak screen).

---

## 12. Expected outcome (estimates, not measurements)

| Step | Evidence | Public Δ |
| --- | --- | --- |
| Name-only rule | Measured on TUNE | +0.002 to +0.004 |
| Ownership resolver + list builder | Inferred from uploads (perfect contest choice ≈ +0.0023) plus TUNE list-builder gain | +0.001 to +0.004 |
| Matcher v2, US/India | 600→1800 steps gave +0.003; data scale untested | +0.002 to +0.007 |
| France (features + pseudo-labels) | Hypothesis; France ≈ 15% of score at ≈ 0.934 | +0.002 to +0.008 |
| **Total** | | **≈ 0.972 to 0.988** |

The top of the range needs every lever to land. France is the largest
unmeasured term; spend uploads to measure it.
