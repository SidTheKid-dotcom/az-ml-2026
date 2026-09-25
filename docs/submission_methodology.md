# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** techquilla-shots

**Draft status:** Frozen holdout evaluation, complete official-test matching and both supplied/strict validators are verified. Durable `ready.json` confirms the frozen v1 files. No leaderboard score or platform upload is claimed.

## 1. Executive Summary

We retrieve the 50 nearest same-country target records using a task-fine-tuned multilingual E5 encoder, then combine a numeric-aware CatBoost matcher with a multilingual BGE cross-encoder. Full-pool hard negatives, explicit number/name conflicts and conservative per-business F0.5 selection address false merges that sampled candidate pools concealed.

## 2. Methodology

### 2.1 Problem Analysis

Business names and addresses contain spelling differences, abbreviations, reordered text, transliteration and missing fields. Shared addresses and near-identical names can represent different businesses; unit/house-number differences are informative but are not unconditional rejection rules. A reference may match multiple S2/S3 records or none.

Only supplied business records and training labels were used; no external business lookup, geocoding or additional task labels were added. Pretrained components are `intfloat/multilingual-e5-small` (MIT, 0.118B parameters) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 0.568B), approximately 0.686B combined.

### 2.2 Solution Strategy

**Approach Type:** Country blocking, dense retrieval, supervised pair classification and cross-encoder blending.

**Core Contribution:** Train and select against the complete competing target pool, preserve each S1's full candidate group for relative features, and combine multilingual semantic evidence with explicit identity-bearing string and numeric differences.

The initial quick sample used seed 20260925. A separate 60,000-S1 medium sample used seed 20260926 and excluded all 12,000 quick references. Country-stratified, entity-disjoint medium partitions contain 35,999 TRAIN, 12,000 TUNE and 12,001 eval S1s (sealed until v1 was frozen). E5 used quick TRAIN only; the selected tree and BGE used medium TRAIN, with model/threshold selection on medium TUNE. Split fingerprints and country counts are recorded in the repository runbook; replay instructions are in the code README.

## 3. Candidate Generation (Blocking)

- **Blocking key:** Normalized country equality; search S2 and S3 together within that country.
- **Representation:** Original-script `business_name | business_address`, prefixed `query: ` for S1 and `passage: ` for targets; maximum length 128, 384 dimensions, L2 normalization.
- **Retrieval rule:** Exhaustive top 50 dot-product search, without an approximate-neighbor index. Target vectors are stored as float16; dot products use float32 with float32 queries. This is exact search over the stored representations, not a claim of unquantized encoder equivalence.
- **Training target scope:** All 10,320,219 original S2/S3 training records. Medium TRAIN+TUNE has 2,399,950 candidate pairs; TUNE has 600,000.
- **Official-test scope:** All 1,732,544 S1s against 9,969,589 S2/S3 targets. Candidate generation produced **86,627,200 unique pairs**, exactly 50 for every S1. Both final TSVs passed the unchanged official `--check-ids` validator without warnings and the stricter complete-coverage/subset gate. The durable release is `/workspace/az-ml-new/releases/submission_v1`; these checks establish format/integrity, not hidden-test accuracy.
- **Miss accounting:** No positive candidate injection. Ground-truth targets missing from retrieval may be retained as compact records solely for scoring; their missing edges still count as missed links. TUNE link recall is 0.984208; perfect classification of these candidates would yield macro F0.5 of 0.994090.

This selected release uses dense candidates. Earlier lexical/phonetic/dense union experiments are separate procedures and are not silently substituted during inference.

## 4. Matching Model

**Features used:** 105 ordered features, including normalized/core-name and address fuzzy similarities, token overlap, simplified phonetic similarity, exact/missing/native-script indicators, text lengths, country agreement, S2/S3 source indicator, dense score/rank and within-S1 gaps. Numeric features cover short-number differences, digit similarity/transposition, postcode-like numbers, alphanumeric prefixes/suffixes, unit identifiers and numbers in names. Numerical entity-ID suffixes and row positions are not model inputs. Dense-only candidate semantics set unavailable lexical retrieval scores to zero.

**Training budgets and model types:**

- **E5:** 200 steps, batch 32, symmetric in-batch contrastive loss, temperature 0.05, AdamW learning rate 2e-5, weight decay 0.01, maximum length 128. Each batch selects distinct positive TRAIN businesses. Seed 20260925.
- **CatBoost:** 650 saved trees, depth 10, learning rate 0.08, L2 leaf regularization 5, Logloss, negative/positive class weights [2,1]. Fit on 1,799,950 medium TRAIN candidate pairs; the TUNE Logloss evaluation set controls best-iteration selection. The selected bundle has best iteration 649 (zero-based). Seed 20260925.
- **BGE:** Start from the original pretrained reranker; 1,800 steps ×32 pairs = 57,600 presentations, BCE-with-logits, AdamW learning rate 2e-5, weight decay 0.01, maximum paired length 256. Batches contain 16 positives and 16 negatives. TRAIN negative selection prioritizes high tree scores and numeric/name conflicts; the retained negative pool is 3× the positive pool, with up to 50% conflict-focused negatives. The run saw 28,800 unique positive and 28,800 unique negative pairs. Seed 20260925.

**Inference and threshold selection:** Rerank only pairs with raw tree probability ≥ 0.01. For those pairs, `score = 0.25 × tree_probability + 0.75 × sigmoid(BGE_logit)`; accept score ≥ 0.775. Pairs whose maximum possible blend (CE=1) cannot reach 0.775 skip the CE call; this preserves accepted sets. Unreranked pairs receive zero final score. Choose blend/threshold solely on TUNE macro F0.5; preserve the exact tree, E5 and BGE checkpoint hashes before eval. Reference batches are 1,000; CE uses stable length-sorted batches of 128, padding to a multiple of 8 and BF16 autocast. Runtime and source hashes are frozen in the release pipeline manifest.

## 5. Results & Error Analysis

For each S1 with true matches, `F0.5 = 5×TP / (4×predicted_count + true_count)`. A true singleton scores 1 only when prediction is empty, otherwise 0. Average over every reference, including missing-candidate cases.

| Measure | Selected model: 12,000 medium TUNE S1s / full training target pool |
| --- | ---: |
| Macro F0.5 | 0.968992 |
| Link precision / recall | 0.987471 / 0.938760 |
| True / false predicted links | 38,936 / 494 |
| Singleton accuracy / erroneous singleton merges | 0.952654 / 33 |
| India / US macro F0.5 | 0.963339 / 0.972744 |
| Unretrieved true links / retrieved true links rejected | 655 / 1,885 |

Development stress identified wrong merges between near-twin names at shared addresses and distinct units/house numbers. It also showed that strict numeric rejection can discard legitimate address variants. These earlier diagnostic examples do not establish the frequency of error types in the selected model's 494 false links.

The matched 600-step comparison scored 0.966067 with 531 false links; the selected 1,800-step model adds 206 true links and removes 37 false links on the same TUNE population. Repeated tuning can overfit. The untouched **12,001-S1 holdout** scores **macro F0.5 0.968292** (95% business-bootstrap interval [0.966268,0.970217]) and **macro F1 0.953967**. Precision/recall: 0.987270/0.939456; false links: 506; missed links: 2,529; singleton accuracy: 0.932650. India/US F0.5: 0.962826/0.971919. Models, thresholds and source hashes were frozen before access; no adjustment followed this result. **France has no supplied training labels or measured validation accuracy; official-test labels remain hidden.**

## 6. Conclusion

Full-pool retrieval and difficult negatives substantially changed model selection compared with the easier sampled pool. The frozen dense/tree/BGE procedure retained its development performance on the untouched holdout. Complete official-test prediction and validation have succeeded; no official-test accuracy is measurable without a platform result. Later TRAIN/TUNE experiments are separate procedures and do not change this release.

## Appendix

### A. Code Artefacts

Source, tests and pinned CPU/GPU environments ship under `code/business_entity_resolution/`. Its `README.md` gives the exact data preparation, training, inference and validation commands. The finalized package includes the selected weights in `models/` and configuration, source/model digests, measured runtime, holdout evidence, the unchanged organizer validator and completed release reports in `resources/`.

- `src/reproduce.py --dataset <supplied-dataset> --work-dir <empty-fast-directory>` verifies the included frozen weights/source, prepares the complete test candidate set, predicts every S1 and validates both files. Only filesystem locations change in a recorded relocated configuration; model, gate, blend and threshold stay fixed.
- Preparation/training modules: `ber.prepare`, `ber.hf_experiment`, `ber.full_embeddings`, `ber.full_dense_candidates`, `ber.tabular_sweep`, `ber.cross_encoder_experiment`. TRAIN/TUNE comparisons select a model; retraining is not claimed to produce bit-identical GPU weights.
- `ber.validate_release` invokes the unchanged supplied validator with `--check-ids`, then enforces exact coverage, valid IDs, duplicate rejection and match-subset consistency. It checks format/integrity, not test accuracy.
- `src/package_submission.py` builds the final ZIP only from a completed, validated release and the exact recorded weights/source. Every archived file is verified against its SHA256. The original TSVs are unchanged in `output/`; no script uploads to the platform.

### B. Additional Results

Full configurations and rejected approaches are in `learnings/scoreboard.md` and `EXPERIMENT_DECISIONS.md`. The earlier union score 0.969812 used 2,400 different TUNE S1s and a different candidate rule; it is not the selected dense model's score. The complete fixed eval report is `learnings/first_submission_holdout.json`; readiness, official-validator report and final file hashes are in `learnings/CURRENT_STATE.md` and the immutable durable release manifests.
