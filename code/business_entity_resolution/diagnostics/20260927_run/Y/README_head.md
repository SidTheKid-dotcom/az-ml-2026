# Agent Y: Laya-multilingual C2-FR (27 Sep 2026)

All numbers here are measured on gpu3 (A10G 24 GB) unless marked *inferred*. No labels were read for French, test, SLICE
or HOLDOUT-2. B_REST monitor labels are real US/India TRAIN labels, and the Frenchified TUNE labels come from how
that world was built.

## Model (verified on the HF card and Hub API)
- `convaiinnovations/laya-multilingual`, HF revision `e4e9ddf21a7b1903b7acffd8814ad4307bf63a67`.
  - Licence: Apache-2.0.
  - Parameters: 321,908,998, from the safetensors metadata and matching my own count.
- Structure: an mmBERT-base encoder plus a Laya decision head.
  - The encoder is ModernBERT: 22 layers, hidden size 768, 256k vocabulary, 8k positions. Its `encoder.*` tensors hold
    306,939,648 params.
  - Its base, `jhu-clsp/mmBERT-base`, is MIT-licensed. Laya's encoder weights differ from the base by a median relative
    L2 of 7.6% (maximum 19%), so Laya fine-tuned the encoder rather than freezing it.
- The Laya checkpoint loads as a standard encoder, so no fallback to mmBERT-base was needed.
  - `ber.laya_convert` maps `encoder.*` one-to-one onto `ModernBertForSequenceClassification.model.*`. No
    remote code is involved.
  - It drops the decision head: 2 transformer layers, an option scorer, an act head, a type embedding and a temperature,
    14,969,350 params in total.
  - It adds a fresh 1-logit BCE head (dense + norm + linear, 591,361 params, seed 20260927, mean pooling).
  - Weights are stored in FP32. The trained model has 307,531,009 params.
- Checks:
  - The pair template is `<bos> A <eos> B <eos>`.
  - Logits do not change with padding width (maximum difference 1e-6).
  - The tokenizer saved with the checkpoint produces the same ids as the original tokenizer.
- Allowlist: `code/models_laya.yaml` is a copy of `problem-statement/models.yaml` with the entry
  `convaiinnovations/laya-multilingual`, apache-2.0, 0.322B, role reranker. The repository file is unchanged.

## Training
- Recipe: Agent S's gpu3 C2-FR(base) command (`S/g3_train.sh`), with the Laya backbone in place of
  bge-reranker-v2-m3.
  - It runs `ber.fast_trainer --mode c2` unchanged, called through `ber.laya_c2`. The only change is that the
    checkpoint verifier accepts the Laya model id.
  - Data: arena_fr_v3 partition A only, i.e. partition-A US/India replay plus train-derived Frenchified pairs
    (option A, no test text). HOLDOUT-2 stays sealed.
  - Settings: plain `name | address` text, batch 128 (microbatch 64), lr 2e-5 with cosine decay and 5% warmup,
    seed 20260925, bf16, length buckets, torch.compile, 2 tokenizer workers.
- Planned for 12,000 steps and **stopped at the step-10,000 checkpoint**, so that the French scores could reach S3 by
  14:20 IST as the lead asked.
  - At step 10,000 the cosine learning rate is 1.48e-6, 7% of peak, so the schedule was nearly finished.
- Throughput: 0.271 s/step at steady state, i.e. **about 472 pairs/s**. That is 2.1 times the 222 pairs/s of
  bge-reranker-v2-m3 on the same GPU with Agent S's recipe.
  - Step 1, including compilation, arrived after 108.6 s.
  - Wall time for 10,000 steps was 2,925.6 s, which includes an 86 s pause for an inference-throughput probe.
  - Peak GPU memory was about 18 GB.
- Inference runs at about 1,720–1,780 pairs/s in bf16 with batch 512.

## Key findings (measured unless marked inferred)
- **French, label-free:**
  - The French pieces went to `french/` at 13:58 IST, before the 14:20 deadline. `french/README_french.md` has a table
    comparing Laya with C2-FR.
  - At the #5 rule (0.10), Laya adds 51,232 links and removes 17,453 relative to the S1 file. At 0.70 it adds 33,227
    and removes 28,041.
  - **Red flag:** 21.9% of the links added at 0.10 carry a decoy word (16.3% at 0.70); for C2-FR the figure is 2.9%.
    They are same-address templated siblings, e.g. "X Ecole SA" → "X SA France".
    - At 0.10 there are 11,409 such additions: groupe 5,402, développement 2,979, france 2,915.
    - The other families almost never accept them: C2-FR 821, the K 3-family vote 387, G 37.
    - *Inferred:* the training arena has decoy words only together with a house-number shift, so the unshifted case
      was never trained.
  - Removals agree with the other families: C2-FR also removes 75% of them, G 82% and the K vote 61%.
  - Probe A (T2r) acceptance is 0.840 at 0.10 and 0.691 at 0.70, against 0.788 for C2-FR. Probe C′ acceptance is
    0.961 and 0.948, against 0.962 for C2-FR.
- **B_REST monitor (US/India, real labels, #5 rule):** 0.973345 against 0.976718 for stored C2 40k, Δ −0.00337 ± 0.00053.
  C2-FR H100 was −0.00100. Laya has lower precision at the 0.10 threshold (0.98493 against 0.99132).
- **Frenchified TUNE (synthetic labels):** 0.97218 at 0.10. The best threshold is 0.70, which gives 0.97772; C2-FR H100's
  best was 0.97705 at 0.60.
- **Agent U's uncertain US/India band** (u8r_v2 p in [0.02, 0.98]; B_REST TRAIN labels; Laya never saw B_REST):
  - On 264,747 scored B_REST pairs, AUC is 0.8980 for u8r_v2 p, 0.8089 for Laya alone, and **0.9045 for a 2-feature
    logistic model** (5-fold, S1-grouped).
  - On reverse rows (row_type 2, 8,720 pairs with text), **Laya's AUC is 0.994 against 0.906 for u8r**. These are the
    rows where U's model has no C2 or Qwen score.
  - These are screening AUCs, not F0.5. U would need to add the feature and gate it in its own cross-validation.
  - Test uncertain pairs: 782,728 rows, 697,272 scored. Rows whose record lies outside the dense pool have no text here
    and are left as NaN.

## Recommendation (inferred)
- **France:** do not use the Laya rows as a French file. As a 4th vote member, use the 0.70 rows or Platt-calibrated
  `laya` scores, and block additions whose record adds a decoy word (K's `decoy_added`) together with the existing twin
  guard. Its most useful role is to confirm removals. Do not let a Laya + C2-FR pair (trained on the same arena) add
  links on its own.
- **US/India:** Laya is weaker than C2 40k as a direct scorer. Try it as a feature in U's model, especially for reverse
  and route rows, and gate it on B_REST cross-validation as usual.

## Files
- `model/`: the step-10,000 checkpoint (FP32 safetensors, config, tokenizer, training config, markers) and
  `initial_model_provenance.json` from the conversion.
- `train/train.log`: the training log.
- `results/`:
  - `eval_laya_step_010000.json`: B_REST monitor and TUNE.
  - `laya_french_pair_scores.parquet`
  - `laya_t0{10,70}_*`: French rows and diagnostics.
  - `agreement.json` and `laya_t0{10,70}_decoy_check.json`
  - `laya_us_india_uncertain_{brest,test}.parquet`, with columns `s1_id, rec_id, row_type, rec_outside_pool,
    u8r_v2_p, has_text, laya`.
  - `laya_us_india_uncertain_brest_auc.json`
  - `parity_check_c2_40k_diagnostics.json`: the stored C2 40k run through the same rebuild reproduces the S1 file
    exactly (SHA `559f37b9…`).
- `fr_full_index/`: `scores.npy` and `scored_mask.npy` over the full 86,627,200-pair test index, in the layout
  `s_rebuild.py` expects.
- `code/`: `laya_convert.py`, `laya_c2.py`, the gpu3 scripts and `models_laya.yaml`.
