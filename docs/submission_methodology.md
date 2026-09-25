# ML Challenge 2026: Business Entity Resolution Solution

**Draft status:** Selected method documented; final holdout score, complete official-test candidate count and supplied-validator results are pending verification. This document does not claim submission readiness or a leaderboard score.

## 1. Executive Summary

We retrieve the 50 nearest same-country target records using a task-fine-tuned multilingual E5 encoder, then combine a numeric-aware CatBoost matcher with a multilingual BGE cross-encoder. Full-pool hard negatives, explicit number/name conflicts and conservative per-business F0.5 selection address false merges that sampled candidate pools concealed.

## 2. Methodology

### 2.1 Problem Analysis

Business names and addresses contain spelling differences, abbreviations, reordered text, transliteration and missing fields. Shared addresses and near-identical names can represent different businesses; unit/house-number differences are informative but are not unconditional rejection rules. A reference may match multiple S2/S3 records or none.

Only supplied business records and training labels were used; no external business lookup, geocoding or additional task labels were added. Pretrained components are `intfloat/multilingual-e5-small` (MIT, 0.118B parameters) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 0.568B), approximately 0.686B combined.

### 2.2 Solution Strategy

**Approach Type:** Country blocking, dense retrieval, supervised pair classification and cross-encoder blending.

**Core Contribution:** Train and select against the complete competing target pool, preserve each S1's full candidate group for relative features, and combine multilingual semantic evidence with explicit identity-bearing string and numeric differences.

The initial quick sample used seed 20260925. A separate 60,000-S1 medium sample used seed 20260926 and excluded all 12,000 quick references. Country-stratified, entity-disjoint medium partitions contain 35,999 TRAIN, 12,000 TUNE and 12,001 sealed eval S1s. E5 used quick TRAIN only; the selected tree and BGE used medium TRAIN, with model/threshold selection on medium TUNE. Split fingerprints and country counts are recorded in `docs/RUNBOOK.md`.

## 3. Candidate Generation (Blocking)

- **Blocking key:** Normalized country equality; search S2 and S3 together within that country.
- **Representation:** Original-script `business_name | business_address`, prefixed `query: ` for S1 and `passage: ` for targets; maximum length 128, 384 dimensions, L2 normalization.
- **Retrieval rule:** Exhaustive top 50 dot-product search, without an approximate-neighbor index. Target vectors are stored as float16; dot products use float32 with float32 queries. This is exact search over the stored representations, not a claim of unquantized encoder equivalence.
- **Training target scope:** All 10,320,219 original S2/S3 training records. Medium TRAIN+TUNE has 2,399,950 candidate pairs; TUNE has 600,000.
- **Official-test scope:** All 1,732,544 S1s against 9,969,589 S2/S3 targets. Final produced pair count and coverage await completed artifacts and validation.
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

The matched 600-step comparison scored 0.966067 with 531 false links; the selected 1,800-step model adds 206 true links and removes 37 false links on the same TUNE population. Repeated tuning can overfit. **Final holdout results are pending; France has no supplied training labels and no measured validation accuracy; official-test labels remain hidden.**

## 6. Conclusion

Full-pool retrieval and difficult negatives substantially changed model selection compared with the easier sampled pool. The frozen dense/tree/BGE procedure is ready for its fixed holdout and complete test-generation pipeline; performance and release claims will be updated only from completed, verified artifacts.

## Appendix

### A. Code Artefacts

Source is under `code/business_entity_resolution/src/ber/`, with `README.md`, `requirements.txt` and unit tests in the same code package. Python 3.12 and the measured CUDA environment are documented in `docs/RUNBOOK.md` and `docs/gpu_runtime.json`.

- Preparation and training: `ber.prepare`, `ber.hf_experiment`, `ber.full_embeddings`, `ber.full_dense_candidates`, `ber.tabular_sweep`, `ber.cross_encoder_experiment`. Saved configs record exact seeds, candidate/data digests and training choices.
- Preserved weights and SHA256 identities: `docs/checkpoints.json`; selected modeling settings: `docs/frozen_dense_pipeline.json`. The finalized executable release configuration is `/workspace/az-ml-new/models/submission_v1_pipeline.json`.
- From the GPU project root, `bash code/business_entity_resolution/run_frozen_holdout.sh "$selected_pipeline_sha"` freezes the finalized configuration and produces the one fixed eval report. `bash code/business_entity_resolution/run_test_candidates.sh` prepares label-free test candidates; inspect existing completion markers before launching it.
- `bash code/business_entity_resolution/run_first_submission.sh "$selected_pipeline_sha"` requires that same pipeline's complete holdout, writes `output/submission_v1/matching_results.tsv` and `candidate_pairs.tsv`, invokes `ber.validate_release` with the unchanged supplied validator's `--check-ids`, and checks exact coverage/ID/subset integrity. Set the variable to the explicitly recorded finalized pipeline digest.
- Only successful validation and hash-verified durable copies create `ready.json` under `/workspace/az-ml-new/releases/submission_v1`. The script does not upload to the competition platform. Original files remain unchanged; `docs/source_integrity.json` records ZIP/statement integrity.

### B. Additional Results

Full configurations and rejected approaches are in `learnings/scoreboard.md` and `EXPERIMENT_DECISIONS.md`. The earlier union score 0.969812 used 2,400 different TUNE S1s and a different candidate rule; it is not the selected dense model's score. The frozen model's complete eval report, official validator report and final file hashes will be appended after verification.
