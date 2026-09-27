# Fact-check: colleague's critique of the pipeline (27 Sep 2026, checked 12:35–12:55 IST)

**Scope.** This check is read-only. Sources:
- the repo (code and learnings);
- today's S3 agent outputs under `s3://az-ml-new-20260926-533267274451/run27/{K,U,L,S,G,R,J,Y}`;
- two primary external sources: ComEM, arXiv 2405.16884v3, and the Foursquare 1st-place write-up on Kaggle (discussion 336055).

No training or GPU job was run.

**Tags.** [M] measured by us; [I] inference; [R] external source. Code references are relative to
`code/business_entity_resolution/src/ber/` unless another path is given.

## Bottom line

- The critique is mostly right about what we did:
  - the rerankers are trained on A only;
  - the level-2 GBDTs are unweighted;
  - B_REST is reused;
  - the test pool is denser;
  - the public board's country mix is skewed.
- Most of its prescriptions do not fit a **stacked** design or today's 19:00 freeze.
- **Reranker refit on A+B_REST+SLICE: don't.**
  - It would make the reranker scores in-sample on B_REST, where every level-2 model is trained.
  - It cannot be validated today.
  - Our measured learning curves say the gain would be tiny.
- **Price-table weights already exist** in the C2 and Qwen-0.6B trainers (`stake_weights`). Using the same table in the
  decoder measured negative.
- **Listwise and competition modelling already exist:** K2 claimant features, and Agent L's listwise cross-encoder,
  which is in the finalist.
- **Overshoot:** upload #1 cannot identify it. The next upload can measure it exactly if it changes US/India only.
- **Our estimate for "refit + cost weights + seed ensembles":** 0 to +0.0003, most of it already in flight. The critique
  claims +0.001–0.002.

## Truth table

| # | Claim | Verdict | Decisive evidence | Worth doing today? |
|---|---|---|---|---|
| 1a | No model is refit on 100% of labels; C2 and Qwen-0.6B are A-only | **TRUE** | `fast_trainer.py:980` (`in_train = partition == "A"`); `qwen_train.py:231`; Qwen 3,250 × 128 = 416k pairs | — |
| 1b | 30% of labels (B_REST + SLICE) are never trained on | **PARTLY TRUE** | B_REST (25%) is the training set of every level-2 model. Only SLICE (4.8%) is unused | — |
| 1c | Refit rerankers on A+B_REST+SLICE: +0.0005–0.001, needs no validation | **FALSE** for our stack | See §1: the refit breaks out-of-fold stacking; learning curves are nearly flat; the C2 40k and Qwen weights are not in S3 | **No** |
| 2a | CatBoost used flat class weights [2,1] | **PARTLY TRUE** | In the current stack only the v1 base tree uses them (some v1-era experiments did too). Today's level-2 CatBoost is plain Logloss (`pair_tree_c2.py:64`, `:221–230`) | — |
| 2b | Nobody trained with per-example price-table weights | **FALSE** for the rerankers, **TRUE** for level 2 | `fast_trainer.py:110–132` `stake_weights` implements manual §4.3 exactly. It is used by C2 (`:1069–1070`) and Qwen-0.6B (`qwen_train.py:299`) | Optional CPU test |
| 2c | Cost weighting would add on top of OOF-tuned thresholds | **UNVERIFIABLE** (expected ≈ 0) | Decoder-side use of the same table: P5 −0.000541 [−0.000613, −0.000465]; K2 expected-F decode −0.00005 to −0.00014 | Optional, idle CPU only |
| 3a | The pair model never sees the competition | **PARTLY TRUE** | True of C2, Qwen and the v1 CE. False of the system: rival features, K2 +0.001293, L listwise CE +0.000640 in C | Already done |
| 3b | ComEM +16% F1 | **PARTLY TRUE** (mis-attributed) | The +16.02% is "select" vs "match" for zero/few-shot prompted LLMs. ComEM itself adds "up to 4%" | — |
| 3c | Foursquare fused numeric features into the transformer (+0.03 CV) | **PARTLY TRUE** (right numbers, wrong comparison) | CatBoost 0.878 → mDeBERTa + ~90 features 0.907. Our GBDT already ingests text-model scores; their ensemble step was +0.004 | No |
| 3d | Seed-ensembling the rerankers is the cheap version | **FALSE** today | One C2 run is 40k steps, ≈ 2 h of H100, plus scoring. The stack-correct version (C2 v2 as a second OOF feature, K's run F) is already running | Only K's run F |
| 4a | Predictions overshoot; realized ≈ 60–70% of B_REST gains | **UNVERIFIABLE** now | Upload #1 changed both country blocks. The level check shows no overshoot (public US/India 0.98559 vs mix-adjusted arena 0.98503) | **Measure it** (next upload) |
| 4b | Upload #1 fits the 60–70% realization | **FALSE** as evidence | It fits r = 1.0 (France +0.0071) as well as r = 0.65 (France +0.0112) | — |
| 4c | B_REST reuse creates a winner's curse | **PARTLY TRUE**, small | Kept-family CIs are ±0.00006–0.00008 around +0.0002–0.0013, so selection accounts for about 10% or less. Shift is the real risk | — |
| 5 | New-candidate arms decay on test (5.8 vs 4.67 records per S1) | **PARTLY TRUE** | Route rows over-accept 1.4–1.7× (guarded at 0.955). Reverse rows accept 0.82× in India: decay is plausible there | Measured by the next upload |
| 6 | Public ≈ 43/43/15; private is India-heavier; India arms are worth more | **TRUE** (inferred; modest) | harshgitty58's drops give US weight ≥ 0.414 > 0.383. README: private = the remainder. India private weight ≈ 0.48–0.51 | Tie-breaker only |
| 7 | Remaining US/India loss is information-poor; only new evidence helps | **PARTLY TRUE** | Shared-name EA ≈ 0.003–0.0045 is coin-flip. But L run2 gained +0.00064 on existing candidates | No new work |
| 8 | Refit + cost weights + seed ensembles are worth +0.001–0.002 | **FALSE** (unsupported) | Parts: refit ≤ +0.00005; weights ≈ 0 ± 0.0002; seed ensembles either already present or infeasible | — |

## Evidence by claim

### 1. Refit on 100% of labels, given our stacked design

**What is true [M]:**
- C2 (bge-reranker-v2-m3, 40k steps × 128) samples training pairs only from partition A (`fast_trainer.py:980`). Its
  training config is at `run27/stage/c2_plain/ckpt/step_040000/training_config.json`.
- Qwen-0.6B is A-only (`qwen_train.py:231`; 416k pairs).
- Agent L's listwise CE trains only on records whose claimants are all in A (`listwise_ce.py:167`).
- J's 4B uses A only (443,264 pairs).
- SLICE is used by nothing today. HOLDOUT-2 is sealed by design.

**What is not true [M]:**
- "30% never trained on" is wrong for B_REST.
- B_REST (521,846 S1s) is the training set of every decision model:
  - the pair tree;
  - U4/u8r (CatBoost, 5-fold component-grouped);
  - K final C (LightGBM, 5-fold).
- These models take the A-trained reranker scores as out-of-sample features.

**Why the prescription fails for a stack [M + I]:**
1. **Refit the rerankers, then refit level 2 on B_REST.** B_REST reranker scores become in-sample (over-confident).
   Level 2 learns to over-trust them and is wrong on test, where the scores are out-of-sample. This is exactly plan
   mistake #5 (`docs/briefs/PLAN_TO_099_20260927.md:177`). Its B_REST CV would also stop being a valid estimate.
2. **Refit the rerankers, keep level 2.** Test features then come from a different model than the one level 2 learned
   on, with shifted calibration.
   - This matters: C2 is "calibrated low" (`learnings/c2_h100_run.md:23`).
   - 18,563 test rows already flip across the 0.70 threshold between fold models (`run27/U/u8r_v2/output_report.json`).
   - No labelled set is left to check it: B_REST would be in-sample, SLICE was never scored, and HOLDOUT-2 has one read.

**The gain would be small anyway [M]:**
- **Level-2 learning curve**, going from 50% to 80% of B_REST:
  - pair tree +0.000018 [−0.000021, +0.000059] (`run27/U/reports/audits.md`, "U4 status");
  - K-robust 0.986804 → 0.986899, +0.000095 (`run27/K/k2b_robust` vs `k5_robust`).
- So a refit on 100% of B_REST is worth about +0.00001–0.00005.
- **The rerankers are not label-limited.**
  - C2 saw 40,000 × 128 = 5.12M sampled pairs out of ≈ 72.5M A pairs (50 per S1). That is at most one pass over ~26%
    of A's ≈ 5M positives.
  - Its gains per step flatten: 5k→10k +0.00085, 10k→40k +0.00086.
  - Qwen-0.6B matched C2's plain blend with 416k pairs (0.98011 vs 0.97958).
  - More labelled S1s without more steps adds diversity, not signal volume.

**Feasibility today [M]:**
- The C2 step-40k and Qwen step-3250 `.safetensors` are not in S3:
  - a bucket-wide listing finds only R's e5, C2-FR 14k/18k, the base BGE and the v1 CE;
  - `run27/stage/c2_plain/ckpt/step_040000` holds metadata only;
  - U reports them lost.
- C2 40k scored only B_REST (2,576,129 gated pairs; population `b_rest_s1.npy`,
  `podbackup/.../c2_step040000/b_rest_scores/complete.json`). **SLICE was never scored**, so it cannot be added to
  level 2 with C2/Qwen features.
- **HOLDOUT-2:** U already reports that the finalist cannot be re-run on HOLDOUT-2. A refit would add another component
  that is neither reproducible nor validated in front of our only untouched check.

**What a correct version looks like:**
- **(a) K-fold cross-fitted rerankers over A ∪ B_REST (∪ SLICE).**
  - Each fold model excludes one component-grouped fold, and every row's level-2 feature comes from a model that did
    not see it.
  - Level 2 then trains on about 4× more rows.
  - Test gets the fold average; check its calibration on HOLDOUT-2.
  - Cost: K × (≈ 2 h training + scoring) of H100 per reranker family, i.e. multi-day. Not today.
- **(b) Level 2 on B_REST + SLICE.** Valid in principle, but impossible now (SLICE is unscored and the weights are
  missing), and worth ≤ +0.00005 by the curve above.
- **(c) What is valid today: keep the rerankers on A and add a second A-trained model as an extra OOF feature.**
  - Agent S's C2 v2 is retrained from base on A, which is correct.
  - K's run F (C + C2 v2 score, rank and gap) started at 12:45 and is exactly this.

For contrast: Foursquare 1st place did retrain every stage on "all data" for submission (repo notebooks `*_all`). Its
stage-3 models, however, were combined by a fixed linear blend, not stacked under a learned second level, so there the
refit is safe [R].

### 2. Objective vs metric

**The [2,1] class weights [M]:**
- `tabular_sweep.py:79` sets `class_weights=[negative_weight, 1.0]`.
- The selected v1 base tree is `d10_neg2_numeric1`, i.e. [2,1]. That tree is `b5c7ca08…`
  (`artifacts/medium_dense_sweep/results.json`).
- Its TUNE score was 0.957134 against 0.957044 for the same tree at negative weight 1.
- Across the 6 paired sweep configs, a negative weight of 2 ranged from −0.0008 to +0.0001.
- That tree is a level-1 input ("base"/"blend"). The current level-2 CatBoost uses Logloss with no class or sample
  weights (`pair_tree_c2.py:64`, `:221–230`; `pair_tree_arena.py:207`; U's `u4_tree.py` reuses them).

**Price-table weights already exist [M]:**
- `stake_weights` (`fast_trainer.py:110–132`) is manual §4.3 exactly:
  - positives get 1/(5T−4), the cost of missing one of T links;
  - matched negatives get 4/(5T+4), the cost of a wrong addition;
  - no-match negatives get 1.0;
  - weights are class-normalized and capped at 5.
- It is used for C2 (`:1069–1070`) and Qwen-0.6B (`qwen_train.py:299`).
- Unweighted models:
  - K LightGBM, binary (`k_integrate.py:39`, `:115`) and grouped softmax (`record_assign.py:168–185`);
  - L listwise CE (`listwise_ce.py:450`);
  - J 4B (`llm_judge.py:24`, `:245`, "Weights are uniform");
  - G string GBDT.

**Does it add on top of OOF thresholds? Probably not [M + I]:**
- The price table's leverage is in the decision rule, and that has been tried:
  - P5 per-S1 expected-F0.5 with calibration and a count posterior: −0.000541 [−0.000613, −0.000465]
    (`run27/U/reports/audits.md`);
  - K2 expected-F decode vs threshold, within each half: −0.00005 to −0.00014 (`run27/K/k2/report.json`);
  - per-country thresholds: +0.000005 / +0.000031.
- Level 2 already sees list context (S1 row counts, ranks, rival scores).
- Level 2 is near-calibrated: Σp 1,768,994 vs 1,769,048 positives (`docs/briefs/PLAN_TO_099_20260927.md:1104`).
- Weighting therefore mostly re-allocates capacity. Expected 0 ± 0.0002.

**Cheap test** (CPU only, about 30 min on EC2, only if cores are idle):
- **Variants:**
  - W1: in `k_integrate.fit`, pass `weight = stake_weights(y, T_s1)`, where T is the S1's full true count from the
    training folds' B_REST labels.
  - W2: W1^0.5.
- **Held fixed:** C's families and sources, 500 rounds, parameters, threshold grid, cross-fitting, one owner, cap 11, and
  the 0.955 route guard.
- **Readout:** paired component bootstrap against C's per-S1 vector (0.989254) on the U salt-1 and salt-2 fold files;
  false links, singleton accuracy and per-country Δ.
- **Keep** only if Δ ≥ +0.0002 with CI > 0 on both salts, and test acceptance by row type stays within the B_REST range.
- **Chance of passing:** about 20–30% [I].

### 3. Competition-aware models; ComEM; Foursquare; seed ensembles

**What we have [M]:**
- **Level 2 sees the competition through features:** `rival_c2`, `rival_q`, `q_minus_rival` (K2's top gain feature),
  `rec_best_q_grp`, `rec_n_claim_*` and `rival_k3_all`.
- **K2 is a genuine record-centric grouped softmax** with a "none" option: +0.001293 [+0.001213, +0.001383] vs the
  pair tree.
  - The objective itself adds nothing: binary with the same features scores 0.986852 vs softmax 0.986805
    (`k2b/report.json` vs `k2/report.json`).
  - The gain is in the claimant features.
- **Agent L's listwise CE** (record + up to 8 claimant S1s + none; multilingual-e5-small):
  - standalone −0.000626 vs the pair tree;
  - as features +0.000302 in K-lite and **+0.000640 [+0.000576, +0.000710] in the finalist C**
    (`run27/L/run1/kcheck_report.json`, `run27/K/kf_C/report.json`).

**ComEM [R]:**
- The +16.02% F1 (17.58% in v2) is the "select" strategy vs the "match" strategy for **zero/few-shot prompted LLMs**.
- ComEM adds "up to 4%" on top of that.
- The benchmarks sit at F1 ≈ 46–86: Mistral-7B matching averages 45.91, and supervised Ditto 80.69.
- The paper's limitations section confines the study to basic zero-/few-shot prompting.
- It says nothing about a fine-tuned cross-encoder + GBDT stack at 0.989.

**Foursquare 1st place [R]:**
- Stage-3 CatBoost ("same features as 2nd") scored CV 0.878.
- mDeBERTa-v3-base with text plus ~90 features (FGM, EMA) scored CV 0.907, i.e. +0.029. XLM-R-large used ~70 features.
- The ensemble scored 0.911, i.e. +0.004.
- The +0.03 is a text model vs a features-only GBDT. Our GBDT already consumes text-model scores (C2, Qwen, L, the v1
  CE). The analogous remaining step for us is their ensemble step (+0.004 at an IoU of ~0.91), not +0.03.
- The metric (IoU), the task (lat/lon POI dedup) and the accuracy regime also differ.

**Seed ensembles [M]:**
- **Rerankers:** a C2 run is 40k steps (≈ 2 h of H100 at 5.6 steps/s) plus hours of scoring. The GPUs are committed:
  J's 4B until about 14:20, Y's Laya run started 12:32, S's C2 v2 chain, and the HOLDOUT-2 GPU steps.
- **Level 2:** test predictions are already the mean of 5 fold models.
  - The fold spread is small: mean std 0.00101, and each fold model's link count is within 769 (≤ 0.03%) of the
    averaged model's count in each country (`u8r_v2/output_report.json`).
  - More seeds would give ≤ +0.0001 [I].
- **The stack-correct "second seed" is K's run F** (C2 v2 as an extra A-trained feature). It is already running.

### 4. Overshoot, winner's curse, 60–70%

**Not identifiable from upload #1 [M]:**

public Δ = 0.850248 · r · 0.001965 + 0.149752 · ΔFrance = 0.002735 (from 0.978287 to 0.981022)

| r (realization) | Implied ΔFrance | France |
|---|---|---|
| 1.00 | +0.0071 | ≈ 0.944 (the ledger's reading assumes r = 1) |
| 0.65 | +0.0112 | ≈ 0.948 |
| 1.64 | 0 | ≈ 0.937 |

All three are compatible with the label-free French evidence, so the upload "fits" any r from 0 to about 1.6.

**What we do know [M]:**
- **Level transfer (S1 file):** arena B_REST 0.985511 vs decoded public US/India 0.98559 (France-blank probe
  0.846352).
  - At the public mix (about 50/50 US/India) the arena predicts 0.98503.
  - Public is therefore **+0.0006 above** the prediction, ±0.0009 for the French no-match-rate assumption.
  - This is no sign of overshoot, but it is a level, not a delta.
- **Clean deltas:** no US/India-only delta has been measured on public since the v1 era (#4 vs #3: −0.000305, with no
  arena prediction).
- **Selection noise is small:**
  - Kept families have paired CIs of ±0.00006–0.00008 around +0.0002 to +0.0013.
  - Null families were dropped (`run27/K/status.md` 12:35):
    - stacked U probability, +0.000009;
    - G v2, +0.000021;
    - collective, +0.000081, dropped for adversarial AUC 0.78–0.83;
    - U's `u1`, which lost −0.000121 at test-like density.
  - A bias of about 1 SE per kept family is ≤ ~0.0003 of the +0.0037 stack gain, i.e. **≥ ~90% realization from
    selection alone** [I].
- **The real risk is covariate shift, and it differs by arm (§5).**
- **External [R]:**
  - Epic021 and rajesh08114: "held-out gains have matched leaderboard gains except France".
  - Density and cluster-count features failed on the leaderboard for GozonLegendent, harshgitty58 and mayankgoplani431.
- **A blanket 0.6–0.7 discount is not supported.** A component discount is defensible: ~0.9–1.0 for dense-row features,
  and less certain for the route and reverse arms.

**How to measure it:** make the next upload **US/India-only** [M arithmetic]:
- **File:** C's rows (`run27/K/kf_C/us_india_rows_route0.955.tsv`, SHA `ec44d42f…`) plus upload #1's French rows,
  byte-identical.
- **Why it is exact:** effects add exactly across countries (manual §5.1; countries share no records).
- **Predicted Δ:**
  - +0.001512 at the B_REST mix (0.850248 × 0.001778);
  - +0.001756 at public weights US 0.43 / India 0.43 (US +0.000718, India +0.003365).
- **r = Δpublic / prediction.**
  - r ≥ 0.9: B_REST gates transfer.
  - r of 0.6–0.8: discount the new-candidate arms.
  - r < 0.5: use upload #1's US/India rows in the final.

### 5. New-candidate arms on test

**Density fact [M]:**
- True: 4.67 records per S1 in train vs 5.8 in test.
- But the arena is S1-denser. US claimants per record are 18.65 in the arena vs 8.68 in test
  (`run27/K/adversarial.json`).

**Route rows [M]:**
- Accepted per 1k rows, B_REST → test: US 3.3 → 4.5 (1.39×), India 8.0 → 13.3 (1.66×)
  (`run27/U/u8r_v2/rowtype_diag.json`). K's model B showed 1.5×.
- Cause: test routes reach records outside the dense pool (185 accepted per 1k such rows). In the arena, the compact pool
  sends those links through reverse rows instead.
- Combined orphan acceptance on test is 15.9 per 1k S1s, against ~20 per 1k S1s of true orphan links reachable on
  B_REST.
- K's 0.955 guard brings C's test route acceptance to India 8.5 / US 3.4 per 1k vs OOF 8.9 / 3.1.
- This is over-acceptance, and it is handled. It is not decay.

**Reverse rows [M]:**
- C's test acceptance: India 13.7 vs OOF 16.7 per 1k (0.82×); US 6.4 vs 5.8.
- This is the one arm where decay is plausible. The alternative explanation is that test has fewer true orphan links:
  India test has 0.135 EA records per S1 vs 0.145 in train.

**Adversarial AUCs [M]:**
- route rows: 0.615 (India) / 0.631 (US);
- `rev3` block: 0.52–0.54;
- L run2: 0.54 / 0.58;
- edit-op block on reverse rows: 0.61–0.65.

The US/India-only upload in §4 measures these arms jointly.

### 6. Public/private country mix

**Public mix [R + M arithmetic]:**
- harshgitty58 blanked each country from a 0.969 base: France 0.836, India 0.580, US 0.578. The drops are 0.133 /
  0.389 / 0.391.
- Since a drop equals w_c·(F_c − s_c) and F ≤ 1 with s ≈ 0.056, **w_US ≥ 0.391 / 0.944 = 0.414 > 0.383** (US share of
  test S1s).
- With F ≈ 0.975 / 0.975 / 0.93, the weights come out at ≈ 0.43 / 0.42 / 0.15 (US / India / France).
- Our France-blank probe (0.846352) is consistent with w_F ≈ 0.15, but cannot split US and India on its own.

**Private mix:**
- The README says the private board is "the remaining portion of the test set" and decides the final ranking
  (`data/raw/README.md:213–215`).
- Private India weight = (0.467 − 0.423·f)/(1 − f), which is **0.48–0.51** for public fractions f = 0.25–0.5.
  Private US weight is 0.34–0.37.
- An India gain is therefore worth ~15–20% more on private than on public, and a US gain ~15–20% less.
- **Use:** a tie-breaker only.
  - C's gain is India-heavy (+0.0034 India vs +0.0007 US), so on private it is worth more than its public delta.
  - U's per-country model (+0.00009–0.00012 overall; India +0.00014–0.00020) is an India-leaning near-tie.

### 7. Is the remaining US/India loss information-poor?

**Where C's remaining loss sits [M]:**
- C's B_REST loss is 1 − 0.989254 = 0.0107.
- 0.0032 of it lies beyond the current candidates (reverse-augmented oracle 0.99683). Reverse rank ≤ 20 lifts the oracle
  to 0.99788, but at 71 pairs per S1 and 0.16–0.8% true.

**The information-poor part [M + R]:**
- Shared-name empty-address copies:
  - 0.003209 of the pair tree's 0.014483 loss (U10);
  - a ceiling of 0.9950 (US) / 0.9960 (India), i.e. ≈ 0.0045, if only they were lost.
- The owner is the best name match for only 23% of missed EA links. The best unique-margin rule reaches precision
  0.71, below the 0.73–0.78 addition break-even (U2).
- Other teams report the same coin flip: GozonLegendent 19.6% vs 21.9% random; Epic021 0.407 accuracy when 2–3 S1s
  share a name.

**But not "only new evidence helps":**
- Since upload #1's US/India rows, C gained +0.001778. Of that:
  - +0.000904 came from L1 v3 plus reverse rows (new candidates);
  - +0.000640 came from L run2 on existing candidates.
- Within-candidate headroom is ≈ 0.0076. About 0.003–0.0045 of it is the ambiguous pool, so ~0.003 is still learnable
  in principle.

### 8. "+0.001–0.002 over the plan"

Unsupported. The parts, from our own measurements:

| Part | Estimate | Status |
|---|---|---|
| Level-2 refit on 100% | ≤ +0.00005 | Low value |
| Reranker refit | Unsafe and infeasible today | Don't |
| Level-2 cost weights | 0 ± 0.0002 | Rerankers already weighted; decoder-side use negative |
| Level-2 seed ensembles | ≤ +0.0001 | Already 5-model averages |
| Reranker seed ensembles | — | Infeasible today |
| C2 v2 as an extra feature (K's run F) | +0 to +0.0003 [I] | Already running |

**Total:** 0 to +0.0003.

## What to actually do today (ranked by expected public gain per hour)

Constraints:
- freeze at 19:00, final upload about 18:30;
- 3–4 uploads left (probe C′ is logged 27 Sep; confirm whether it counted);
- HOLDOUT-2 may be read once;
- option A: no test-derived training data. Nothing below uses test-derived training data.

1. **Next upload (by ~13:30): US/India = C, France = upload #1, byte-identical.**
   - **Expected:** +0.0015 to +0.0018 public, if r = 1. Realized is plausibly +0.0010 to +0.0018, given India reverse
     rows at 0.82×.
   - **Effort:** ~30 min (the rows exist; run the validator, the byte checks and the candidate-pairs merge).
   - **Also measures r**, which settles claims 4 and 5.
   - **Prefer it** over the combined `full_kfC_route955_fr_u1rp.tsv`, which again confounds r with ΔFrance.
   - **Risk:** under-realization of the India reverse and route arms. The fallback is upload #1's US/India rows.
2. **Gate the in-flight, stack-correct families** under the existing rules (CI > 0 on both salts, adversarial AUC
   ≲ 0.6):
   - K's run F (C2 v2 features, now);
   - J's 4B `llm_p` (due ~14:20; the 4B keeps its +0.0005 bar).

   Expected 0 to +0.0005 each; ~0.5–1 h of CPU each.
3. **Change one country block per upload**, so every delta stays identified:
   - **#3 (~15:30–16:00):** France-only change on top of #2, e.g. u1_refine_plus or the later Y/S/fr_swap candidates.
   - **#4 (~18:00–18:30):** the final, i.e. the best measured blocks plus any US/India family that passed since #2.
     That is again a single-block change.
   - **Reserve:** re-upload the best measured file if #4 loses. The last upload counts.
4. **Optional, only on idle EC2 cores, never on the critical path:**
   - the stake-weighted LightGBM test (§2);
   - 3-seed bagging of C;
   - a per-country version of C (India-leaning).

   Expected ≤ +0.0001–0.0002 each; keep only if Δ ≥ +0.0002 with CI > 0 on both salts.
5. **Decide HOLDOUT-2 now**, choosing among U's options (a), (b) and (c).
   - The finalist uses C2 40k and Qwen 3250 features whose weights are not in S3, so it cannot be re-run exactly.
   - If K's run F brings C2 v2 into the finalist, option (b) becomes closer to the real predictor. Qwen stays missing.
   - Read HOLDOUT-2 once, after #4's composition is frozen, and state its scope.
   - Adding any refit component would widen this gap.
6. **Not today:**
   - refitting C2 or Qwen on A+B_REST+SLICE;
   - refitting level 2 on 100% of B_REST;
   - reranker seed ensembles;
   - a feature-fused transformer;
   - a blanket 0.6–0.7 discount on B_REST deltas (measure r instead, with item 1).

## Hygiene notes found during the check

- `run27/L/status.md` on S3 contains Agent G's status text, not L's. `run27/K/status.md` starts with G's text too.
  L's numbers above come from its report JSONs.
- U's status headers say "updated 14:15" and "14:55 IST", but the files were written at 12:24 and 12:41 IST. Use the S3
  timestamps.
