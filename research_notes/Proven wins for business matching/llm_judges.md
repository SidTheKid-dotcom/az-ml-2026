# LLM judges (1B–8B, open licences) compared with fine-tuned cross-encoders and GBDT stacks for entity and business matching

Scope: whether a Qwen3-Reranker-4B LoRA "LLM judge", applied only to uncertain candidate pairs, is likely to add measurable accuracy to a strong stack. The stack is a record-centric GBDT with a 568M cross-encoder, Qwen3-Reranker-0.6B and listwise e5 features. It serves US and India data (including native-script names) and an unlabelled French segment. Research date: 2026-09-27. Numbers are copied from the sources. Anything marked ESTIMATE is my own calculation.

## Q1. F1 of fine-tuned or prompted 1–8B LLMs against fine-tuned PLM cross-encoders on EM benchmarks, in and out of distribution

### Takeaway
The best available evidence comes from the Mannheim group (Peeters, Steiner and Bizer). On six standard benchmarks, a LoRA-fine-tuned Llama-3.1-8B stays **below** fine-tuned RoBERTa-base and Ditto on in-distribution F1. For example, WDC Products scores 69.2 against 77.5 and 84.9, and Amazon-Google 50.0 against 79.3. The clear LLM advantage is robustness to unseen entities and datasets, and it is shown mainly for very large hosted models (GPT-4, GPT-4o). Fine-tuning a small LLM improves transfer within a domain but hurts transfer across domains. No EM benchmark reports Qwen3-Reranker results, and I found no study of cross-lingual business name and address matching with small fine-tuned LLMs.

### Cited Findings
**PLM baselines (in-distribution)**
- Fine-tuned PLM F1 (RoBERTa-base / Ditto), Table 4:
  - WDC Products 77.53 / 84.90
  - Abt-Buy 91.21 / 91.31
  - Walmart-Amazon 87.02 / 86.39
  - Amazon-Google 79.27 / 80.07
  - DBLP-Scholar 93.88 / 94.31
  - DBLP-ACM 99.14 / 99.00
  — [Peeters, Steiner, Bizer, "Entity Matching using LLMs" (arXiv 2310.11244v4; EDBT 2025)](https://arxiv.org/html/2310.11244v4)

**Zero-shot LLMs**
- Zero-shot F1 with the best prompt:
  - GPT-4: WDC 89.61, Abt-Buy 95.78, Walmart-Amazon 89.67, Amazon-Google 76.38, DBLP-Scholar 89.82, DBLP-ACM 98.41.
  - "Llama 3.1": WDC 83.67, Abt-Buy 89.84, Walmart-Amazon 84.85, Amazon-Google 73.99, DBLP-Scholar 86.32, DBLP-ACM 98.81.
  - Mixtral: WDC 53.37, Amazon-Google 40.98.
  - The extraction did not confirm the Llama 3.1 parameter size. See Gaps.
  — [Peeters et al.](https://arxiv.org/html/2310.11244v4)
- Without task-specific training, GPT-4 "outperforms fine-tuned PLMs on 3 out of 4 e-commerce datasets and achieves a comparable performance for bibliographic data." — [Peeters et al. PDF](https://arxiv.org/pdf/2310.11244)

**Fine-tuned small LLMs (Steiner et al.)**
- LoRA fine-tuning of Llama-3.1-8B-Instruct used r=64, alpha=16, dropout 0.1, lr 2e-4 and 10 epochs. The 8B model trained on one RTX6000 and the 70B on four. — [Steiner, Peeters, Bizer, "Fine-tuning LLMs for Entity Matching" (arXiv 2409.08185v2)](https://arxiv.org/html/2409.08185v2)
- Llama-3.1-8B F1, zero-shot → fine-tuned:
  - Abt-Buy 56.57 → 87.34
  - Amazon-Google 49.16 → **50.00**
  - Walmart-Amazon 42.04 → 65.65
  - WDC Products 53.36 → 69.19
  - DBLP-ACM 85.52 → 97.42
  - DBLP-Scholar 67.69 → 92.95
  — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- Fine-tuned GPT-4o-mini F1: Abt-Buy 94.09, Amazon-Google 80.25, Walmart-Amazon 78.85, DBLP-ACM 99.10, DBLP-Scholar 96.22. — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- WDC-small results for the other models:
  - GPT-4o-mini: 81.61 zero-shot, 83.41 standard fine-tune, 84.38 with structured explanations. One extracted table shows 84.38 as its standard result; Tables III and VI give 83.41.
  - GPT-4o: 81.6 → 87.1.
  - Llama-3.1-70B: 75.19 → 72.66, so fine-tuning **hurt**.
  — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- Training set sizes (positive + negative pairs):
  - WDC-small: 500 + 2,000
  - WDC-medium: 1,500 + 4,500
  - WDC-large: 8,471 + 11,364
  - DBLP-Scholar: 4,277 + 18,688
  — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- Changes to how training data is represented or selected (WDC):
  - Structured explanations: Llama-8B +4.94 (69.19 → 74.13), GPT-4o-mini +0.97, GPT-4o −3.90.
  - Error-based filtering, relevance filtering and synthetic examples: Llama-8B +3.18 to +5.18.
  - The same filtering and synthetic examples gave GPT-4o-mini −6.25 and −6.42.
  — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- On transfer, fine-tuning "improves the generalization to in-domain datasets while hurting cross-domain transfer."
  - Product → scholarly transfer: Llama-8B averages −11.05 F1 below its own zero-shot, and GPT-4o-mini −2.07.
  - "All Llama 8B and half of GPT-4o-mini models underperform their zero-shot baselines" across domains.
  — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- Models fine-tuned on WDC Products and tested on other datasets (Table 7):
  - "Llama 3.1": WDC 72.05, Abt-Buy 83.47, Walmart-Amazon 76.92, Amazon-Google 63.97, DBLP-Scholar 65.25, DBLP-ACM 80.91.
  - GPT-4o-mini: WDC 88.89, Abt-Buy 92.49, Walmart-Amazon 88.61, Amazon-Google 77.78, DBLP-Scholar 86.82, DBLP-ACM 97.28.
  — [Peeters et al.](https://arxiv.org/html/2310.11244v4)

**Unseen entities and out of distribution**
- PLMs trained on Abt-Buy and tested on unseen WDC Products entities: RoBERTa 55.52 (−22.01) and Ditto 48.74 (−36.16). "PLM-based matchers perform poorly on entities that are not part of any pair in the training set." — [Peeters et al.](https://arxiv.org/html/2310.11244v4)
- Jellyfish-13B (Llama-2-13B instruction-tuned for data preprocessing), compared with Ditto and GPT-4:
  - Unseen Abt-Buy: 89.58 (Ditto 89.33, GPT-4 92.77)
  - Unseen Walmart-Amazon: 89.42 (Ditto 86.76, GPT-4 90.27)
  - Seen Amazon-Google: 81.34 (Ditto 75.58)
  - Seen DBLP-Scholar: 98.51 (Ditto 95.60)
  — [Jellyfish (arXiv 2312.01678v2)](https://arxiv.org/pdf/2312.01678v2)
- AnyMatch fine-tunes GPT-2 (124M) in a leave-one-dataset-out transfer setup.
  - Its mean zero-shot F1 over nine datasets is 81.96, against 86.36 for MatchGPT[GPT-4]. That is second best of 14 methods, ahead of MatchGPT with Mixtral, SOLAR, Beluga2 and GPT-3.5.
  - Per dataset: Abt-Buy 86.05, Amazon-Google 55.08, Walmart-Amazon 61.51, WDC 63.31.
  - Swapping GPT-2 for BERT lowers mean F1 by 9.04%; T5 lowers it by 2.38%.
  - Data selection was central: an AutoML filter for difficult pairs, attribute-level examples and control of label imbalance.
  — [AnyMatch (arXiv 2409.04073v2)](https://arxiv.org/html/2409.04073v2)

**Pairwise against listwise prompting**
- Plain pairwise ("matching") prompting of open 7–8B instruction LLMs is weak. Mean F1 over 8 datasets is 47.75 for Llama-3.1-8B, 45.91 for Mistral-7B and 62.11 for Qwen2-7B.
  - For Llama-3.1-8B, choosing among candidates ("selecting") raises this to 76.33, and ComEM to 80.17.
  - Ditto (supervised) averaged 80.69 and HierGAT 83.34 in the same table. Ditto's Walmart-Amazon is only 57.75 there, well below other papers, which suggests a different training regime.
  - "Probabilities on short-form labels are misaligned for common open-source chat-tuned LLMs."
  — [Wang et al., ComEM (COLING 2025)](https://aclanthology.org/2025.coling-main.8.pdf)

**Multi-script and multilingual evidence**
- OpenSanctions Pairs has 755,540 expert-labelled pairs from 293 sources across 45 jurisdictions. It is multi-script and 76.9% positive; a constant-positive predictor gets F1 0.870.
  - Rule baseline: 91.3% F1. GPT-4o: 98.95%.
  - DeepSeek-R1-Distill-Qwen-14B: 98.23% with prompt optimization, 97.76% without.
  - Llama-3.1-8B: 94.0% zero-shot unoptimized, 95.94% at best.
  - LLM errors "concentrate in two narrow modes: **cross-script transliteration differences (Latin / Cyrillic / Arabic)** … and over-weighting of minor data-entry noise … into spurious negatives."
  — [OpenSanctions Pairs (arXiv 2603.11051v2, Aug 2026)](https://arxiv.org/pdf/2603.11051)
- Qwen3-Reranker covers "100+ languages" under Apache-2.0. Reranking scores on the top-100 candidates from Qwen3-Embedding-0.6B (MTEB-R / CMTEB-R / MMTEB-R / MLDR):
  - Qwen3-Reranker-0.6B: 65.80 / 71.31 / 66.36 / 67.28
  - **Qwen3-Reranker-4B: 69.76 / 75.94 / 72.74 / 69.97**
  - Qwen3-Reranker-8B: 69.02 / 77.45 / 72.94 / 70.19
  - BGE-reranker-v2-m3 (listed as 0.6B): 57.03 / 72.16 / 58.36 / 59.51
  - gte-multilingual-reranker-base (0.3B): 59.51 / 74.08 / 59.44 / 66.33
  — [Qwen3-Reranker-4B model card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)

### Inferences
- **The in-distribution prior is negative for "an LLM judge beats a tuned CE."** A generic 8B instruction LLM fine-tuned with thousands of pairs fell below RoBERTa-base on all six benchmarks. The cross-paper comparison is not fully controlled: training set sizes differ, although the authors and test sets are the same. Only much larger or hosted fine-tuned models (GPT-4o-mini, GPT-4o) reach PLM level.
- **Qwen3-Reranker-4B is a better starting point than Llama-3.1-8B-Instruct.** It was trained for yes/no relevance, which avoids ComEM's "misaligned short-label probabilities" problem. Its zero-shot multilingual reranking is much stronger than BGE-reranker-v2-m3, by about +14.4 MMTEB-R points. If the stack's 568M CE is of that XLM-R-large class, the 4B plausibly adds knowledge in the zero-shot or low-label regime. After in-domain fine-tuning the gap usually narrows (the EM literature pattern). No EM benchmark confirms this.
- **4B is the right size in the Qwen3-Reranker family.** The 8B scores slightly below the 4B on MTEB-R (69.02 against 69.76) and only +0.2 on MMTEB-R.
- **France:** the documented LLM advantage is robustness out of distribution, which is exactly the French segment's situation. Fine-tuning a small LLM on US and India could erode it, as in Steiner's cross-domain loss of −11 F1 for Llama-8B. France is the same domain in a different language, a shift none of these papers measured. Treat French performance as unknown.
- **India native script:** transliteration is the main LLM failure mode even for frontier models on OpenSanctions. Do not expect a 4B judge to fix cross-script pairs unless transliterated or romanized fields are also given to it.

### Gaps
- There are no EM benchmark numbers for Qwen3-Reranker (0.6B, 4B or 8B), fine-tuned or zero-shot. This pass also did not retrieve Unicorn, Qwen2.5-7B-LoRA or Mistral-7B-LoRA EM numbers.
- Peeters et al. do not confirm the size of "Llama 3.1" in their Tables 2, 7 and 9 in the extracted text. It is probably 70B for zero-shot, but this is unverified.
- I found no study of cross-lingual transfer (for example US/India → France) for fine-tuned LLM matchers, and no LLM benchmark on business or POI name and address matching. OpenSanctions (sanctions entities) is the closest multi-script evidence.
- Numbers for the later Jellyfish-7B and 8B variants were not retrieved.

## Q2. Stacking or cascading after a strong cheap model: incremental gains, calibration, and LLM scores as GBDT features

### Takeaway
Published "LLM only on the hard or uncertain subset" designs mostly show **cost** savings at small quality loss, not large accuracy gains. Where a larger reranking stage is added on top of an already strong one, the gains are small. In Eedi, a 32B stage added +0.010 and a 72B stage +0.013 MAP on about 0.62. ComEM gains up to 4% F1 over a single LLM strategy. I found no EM paper that adds an LLM-judge probability as a GBDT feature. LLM judges are documented as overconfident and need post-hoc calibration.

### Cited Findings
- **ComEM cascade:** a medium LLM ranks or filters the blocked candidates, and a stronger LLM makes the final choice among the top-k. This "improve[s] average F1 of the single selecting strategy by up to 4% while reducing the cost." Open 3B–11B LLMs handle the ranking stage well. — [ComEM](https://aclanthology.org/2025.coling-main.8.pdf)
- **Eedi 1st place (retrieval then rerank):**
  - A Qwen2.5-14B pointwise LoRA reranker scores all 32–64 retrieved candidates and keeps the top 8.
  - A Qwen2.5-32B reranker re-scores those 8 and keeps 5: private LB 0.615 → 0.625 (+0.010).
  - A Qwen2.5-72B listwise reranker on the top 5: 0.625 → 0.638 (+0.013).
  - Pseudo-labelling and distillation for the 14B model: 0.531 → 0.575 (+0.044). Chain of thought: +0.019.
  — [Eedi 1st place write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution)
- **BoostER:** existing ER tools produce match probabilities. Pairs where the tools disagree (high entropy) are sent to an LLM for verification to limit cost. No F1 or cost numbers were extracted. — [BoostER (arXiv 2403.06434)](https://arxiv.org/abs/2403.06434)
- **General LLM cascades:** "the error patterns of small and large models are correlated." Early abstention reduced overall test loss by 2.2% on average across six benchmarks. — [Zellinger et al. 2025](https://arxiv.org/abs/2502.09054)
- **Calibrated 4B → 12B cascade (production NER on H100, not EM):**
  - 4B micro-F1 0.847 at cost 1.00; 12B 0.932 at cost 3.02.
  - Calibrated routing cut cost by 31% (95% CI 27–35%) at micro-F1 0.91 and reduced ECE from 0.12 to 0.03.
  — [UCCI (arXiv 2605.18796)](https://arxiv.org/pdf/2605.18796v1)
- **Complementary errors:** the rule-based matcher and LLMs "fail in complementary ways: rules over-match, while LLMs struggle with cross-script transliteration." The authors expect future gains from "blocking, clustering, and uncertainty-aware review" more than from pairwise prompt tuning. — [OpenSanctions Pairs](https://arxiv.org/pdf/2603.11051)
- **Practitioner lessons ("ER in Practice", 2026):**
  - "Which matcher wins depends on the dataset."
  - An LLM teacher costs about $450 per 1M pairs at about 2 s per pair. The distilled matcher costs $12 per 1M pairs, 37.5× cheaper. LightGBM matchers run at 222–263K pairs/s on CPU.
  - Transitive closure is safe when baseline precision is above 0.9. Below 0.5 it creates mega-clusters.
  - Verified merge gives the best pair-F1 on 4 of 6 datasets.
  — [Entity Resolution in Practice (arXiv 2607.26298)](https://arxiv.org/html/2607.26298v1)
- **Overconfidence after post-training:** post-training "often comes at the cost of degraded calibration—introducing overconfidence", while pre-trained LMs "typically exhibit more accurate confidence estimates." — [NeurIPS 2025, "Your Pre-trained LLM is Secretly an Unsupervised Confidence Calibrator"](https://papers.neurips.cc/paper_files/paper/2025/file/ef6b667990416111e3ebbeeba4a8f283-Paper-Conference.pdf)
- **Overconfident LLM judges:** "LLM judges are known to be overconfident." Calibrated judges let systems send easy cases to cheaper models and reserve expensive judges for uncertain ones. — [Calibrating LLM Judges (ACL 2026 Industry)](https://aclanthology.org/anthology-files/pdf/acl/2026.acl-industry.14.pdf)
- **EM calibration (RoBERTa only):** the study examines temperature scaling and other methods for a RoBERTa-based EM model, not generative LLMs. It found "no correlation between the ECE, MCE, or RMSCE metric values and the datasets' F1 scores, sizes, or mean entry pair sizes." — [Kamsteeg et al., UncertaiNLP 2025](https://aclanthology.org/2025.uncertainlp-main.12/)
- **Distillation from large teachers (LMSYS 1st place):** logits from Llama3-70B and Qwen2-72B teachers were distilled into a Gemma-2-9B student. The student's 5-fold CV log loss was 0.858–0.876, against 0.869–0.881 for the Qwen2-72B teacher. — [LMSYS 1st place write-up](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)

### Inferences
- **The value comes from error diversity, not raw accuracy.** The 4B shares its family and training recipe with the stack's Qwen3-Reranker-0.6B, so its errors are likely correlated with a component the GBDT already uses (not measured). The cascade literature says small and large models' errors are correlated in general.
- **Expected size of the gain (ESTIMATE).** The analogous gains for an added stage are about 1.5–2% relative (Eedi +0.010 and +0.013 on about 0.62 MAP). At a stack operating point near F0.5 0.97–0.98, a realistic prior for a 4B uncertain-band judge is a few tenths of a point at most, possibly within tune-set noise. The gain is more likely on out-of-distribution and French-like pairs than on US in-distribution pairs.
- **Preferred integration:** feed the raw logit (yes−no) as an **out-of-fold** GBDT feature trained by cross-fitting on TRAIN businesses. Add a "not scored" indicator for pairs outside the band, which is safer than a hard override. If it is used as a direct veto or boost rule, calibrate on TUNE with Platt scaling or isotonic regression. Raw P(yes) from a fine-tuned LLM should be assumed overconfident.
- **Band definition:** route by disagreement or entropy between existing components (the BoostER idea) or by a GBDT margin band. Keep the band large enough (several thousand TUNE pairs) for paired differences to be measurable.
- **Replay cost:** if the judge helps, the LMSYS and Eedi distillation pattern (teacher → smaller student) is a documented way to keep the gain at lower cost. Check the project's final parameter cap: a 4B judge adds about 4B parameters to the release.

### Gaps
- I found no EM paper that (a) adds LLM-judge scores as GBDT or stacking features, or (b) measures the incremental F1 of an LLM applied only to the uncertain band of a strong supervised EM stack.
- F1 and cost numbers for BoostER and the related Li et al. 2024 ("cost-efficient approach", arXiv 2401.03426) were not extracted.
- There are no ECE numbers for generative LLM entity matchers, fine-tuned or prompted. The only EM calibration paper found studies RoBERTa.

## Q3. Throughput, fine-tuning cost, training-pair needs and failure modes (single H100)

### Takeaway
Qwen3-Reranker-4B is an Apache-2.0 causal LM that scores with yes/no logits. vLLM can serve it as a sequence-classification scorer through the score API with automatic prefix caching (APC). I found **no published H100 pairs/s benchmark** for it. A first-principles ESTIMATE is a few hundred short pairs per second for scoring alone, and roughly 10–20 minutes of LoRA training per 50K-pair epoch. Documented failure modes are prompt sensitivity, position and order bias, overconfidence, cross-script transliteration errors, and fine-tuning that fails on some datasets.

### Cited Findings
**Serving**
- Qwen3-Reranker usage:
  - The system prompt reads "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
  - The score comes from the logits of the "yes" and "no" tokens.
  - The card's vLLM example sets `enable_prefix_caching=True` and `max_model_len=10000`.
  - Leaving out a task instruct drops retrieval performance by about 1–5%.
  — [Qwen3-Reranker-4B card](https://huggingface.co/Qwen/Qwen3-Reranker-4B)
- vLLM: the original yes/no formulation "requires computing logits for all 151,669 tokens in the vocabulary, making it inefficient and incompatible with vLLM's score() API." The fix is to serve it as `Qwen3ForSequenceClassification` with `--hf_overrides '{"classifier_from_token": ["no","yes"], "is_original_qwen3_reranker": true}'` and `--runner pooling`. The converted model "is significantly more efficient" and "fully supports vLLM's score() API." — [vLLM qwen3_reranker example](https://github.com/vllm-project/vllm/blob/main/examples/pooling/score/qwen3_reranker_online.py); [vLLM docs](https://docs.vllm.ai/en/v0.10.2/examples/offline_inference/qwen3_reranker.html)
- vLLM APC: KV blocks are hashed by their tokens plus the prefix before them and reused across requests that share a prefix. It is "almost a free lunch and won't change model outputs." Only full matching blocks hit the cache. LoRA IDs are part of the block hash, so caches are per adapter. — [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching)
- The Eedi winner trained with a classification head and later switched to yes/no tokens "mainly to take advantage of vllm inference + in-built prefix caching. Both Yes/No approach and classifier head gave similar scores." "For 14b ranker, prefix caching was key." — [Eedi 1st place write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution)
- Docker's VRAM estimates for Qwen3-Reranker in F16: about 1.2 GiB (0.6B), 8 GiB (4B) and 16 GiB (8B). — [Docker Hub ai/qwen3-reranker](https://hub.docker.com/r/ai/qwen3-reranker)

**Throughput and cost anchors**
- AnyMatch throughput table (tokens/s, batch size):
  - Llama2-13B (Jellyfish): 26,721 at batch 128
  - Mixtral-8x7B: 2,108
  - SOLAR-70B: 752
  - GPT-2 124M: 693,999 at batch 8,192
  - Cost per 1K tokens: AnyMatch $0.0000038, Jellyfish $0.000025, MatchGPT[Mixtral] $0.00063.
  - The paper says the models fit on one A100-40GB, but its cost table lists "8x on p4d.24xlarge". See Inferences.
  — [AnyMatch](https://arxiv.org/html/2409.04073v2)
- Sequential latency per prompt, not batched throughput:
  - "Llama 3.1": 0.54 s zero-shot, 0.30 s fine-tuned and quantized.
  - GPT-4o-mini fine-tuned: 0.46 s, at 0.280¢ per prompt including fine-tuning.
  — [Peeters et al.](https://arxiv.org/html/2310.11244v4)
- On OpenSanctions, DeepSeek-14B takes about 70 s and Llama-3.1-8B about 25 s per example. These are generation or reasoning runs, specific to that setup, so Llama-8B is "the better accuracy-throughput point." — [OpenSanctions Pairs](https://arxiv.org/pdf/2603.11051)

**Training cost and data needs**
- Steiner et al. fine-tuned Llama-3.1-8B on one RTX6000 for 10 epochs. 2,500 WDC-small pairs gave +15.83 F1, and about 23K DBLP-Scholar pairs gave +25.26. — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- The Eedi winner's reranker recipe:
  - LoRA r=64, alpha=128, on all linear layers, with `lm_head` frozen.
  - Each batch had 1 positive and batch_size−1 negatives; a commenter summarizes this as a 1:24 ratio. The loss was cross-entropy on the yes−no logit difference.
  - A commenter reports that LoRA-training a 14B yes/no reranker "took > 5 hours" on rented H100s, and that six hard-negative ablations took "more than 2 days."
  — [Eedi write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution)
- The LMSYS 1st place trained on 8× A100-80G with ZeRO-2 and data parallelism, a sequence-classification head, batch 8, max length 1024 and QLoRA. The LoRA adapters were averaged over 5 folds and quantized to 8-bit with GPTQ. — [LMSYS 1st place](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)

**Failure modes**
- **Prompt sensitivity:** the mean standard deviation of F1 across prompts is 2.26 for GPT-4, 14.86 for GPT-4o, 18.45 for GPT-4o-mini, 18.54 for Llama 3.1, 6.18 for Llama 2 and 13.68 for Mixtral. "There is no single best prompt." — [Peeters et al.](https://arxiv.org/html/2310.11244v4)
- **Few-shot does not reliably help small models:** GPT-4o-mini's mean F1 is 85.51 at best zero-shot but 73.76 and 76.56 with 6 and 10 related demonstrations. GPT-4o rises from 88.10 to 91.21. — [Peeters et al.](https://arxiv.org/html/2310.11244v4)
- **Position bias:**
  - ComEM's Figure 3 shows F1 varying with the position of the matching record among the candidates. — [ComEM](https://aclanthology.org/2025.coling-main.8.pdf)
  - The Eedi winner notes that "LLM based listwise rankers typically suffer from candidate position bias", with shuffled test-time augmentation as the fix (not run for lack of time). — [Eedi write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution)
  - The LMSYS 4th-place gold solution became a library with "position-bias correction" for pairwise A/B judges. — [pairjudge](https://github.com/DaoyuanLi2816/pairjudge)
- **Fine-tuning that fails or backfires:** Llama-8B gained only +0.84 on Amazon-Google, Llama-70B lost −2.53 on WDC, and filtered or synthetic data cost GPT-4o-mini −6.25 and −6.42. — [Steiner et al.](https://arxiv.org/html/2409.08185v2)
- **Transliteration and over-weighted noise:** cross-script transliteration errors, and minor data-entry noise read as a mismatch, produce false negatives. — [OpenSanctions Pairs](https://arxiv.org/pdf/2603.11051)

### Inferences
- **Scoring ESTIMATE (assumptions: about 8 GFLOP per token forward for 4B; H100 SXM dense BF16 peak about 989 TFLOPS, an NVIDIA spec not fetched here).**
  - A 150–250-token pair prompt (instruction plus two name and address records) costs about 1.2–2.0 TFLOP.
  - At 30–50% MFU, one H100 scores about **150–400 pairs/s (about 0.5–1.4M pairs per hour)**, prefill only with no decode.
  - A shared system and instruct prefix of about 60–80 tokens can be cached by APC, saving roughly that fraction of compute.
  - Qwen3-Reranker-0.6B is about 6–7× cheaper per pair.
  - Scoring both record orders (A,B) and (B,A) to cancel order bias doubles cost and is still cheap.
- **Training ESTIMATE.** LoRA training costs about 2–3× a forward pass (3× with gradient checkpointing), or roughly 50–130 pairs/s. A 50K-pair epoch then takes about 7–17 minutes, and a 2–3 epoch screen about 0.5–1 hour. The Eedi report of more than 5 hours for a 14B model with long CoT and few-shot prompts is consistent with this once size and length are accounted for.
- **The AnyMatch 13B figure is likely aggregated.** 26.7K tokens/s for a 13B forward pass needs about 0.7 PFLOP/s, above one A100's roughly 312 TFLOPS BF16 peak. It is probably summed over the 8 GPUs in the p4d cost table, so do not treat it as a per-GPU figure.
- **Training data:** the published recipes use 2.5K–23K labelled pairs (Steiner), mined hard negatives at about 1:24 (Eedi), and filtering for difficult pairs (AnyMatch, Steiner). An uncertain-band judge should be trained on the band's distribution plus hard negatives, not on random pairs.

### Gaps
- No measured Qwen3-Reranker-4B pairs/s on H100 (vLLM or HF Transformers) was found.
- A search summary claimed "vLLM is 3.2–3.9× faster than Hugging Face" for reranking, and gave per-query latencies for Qwen3-Reranker-0.6B (831 ms) and 8B (1,876 ms) on the top-20. The underlying source could not be identified, so these are unverified and not used.
- Unsloth's H100 LoRA timing numbers were not retrieved. A search summary claims about 2× speed for Llama-3.1-8B QLoRA on the [Unsloth benchmarks page](https://unsloth.ai/docs/basics/unsloth-benchmarks), which I did not verify.
- I found no measurement of record-order (A,B) against (B,A) asymmetry for pairwise EM with LLMs.

## Q4. Kaggle and LLM-era competition evidence (2024–2026)

### Takeaway
The 2024 Kaggle ranking and preference winners used LoRA-fine-tuned 9B–72B Qwen, Gemma and Llama models as pointwise or listwise rerankers and judges, often distilled from larger teachers. They used yes/no-logit or classifier heads and served with vLLM and prefix caching. In Eedi the LLM rerankers replaced encoder cross-encoders entirely. The closest *matching* precedent, multilingual POI matching in Foursquare Location Matching (2022, pre-LLM), was won by LightGBM stacks over transformer-similarity features. I found no 2024–2026 Kaggle entity or POI matching competition reporting an LLM-judge result.

### Cited Findings
- **Eedi (2024) 1st place:** the backbones listed in the MIT-licensed code are all 7B–72B decoder LLMs: Qwen2.5-Math-7B, Qwen2.5-14B, 32B and 72B, e5-mistral-7b-instruct and bge-en-icl. No encoder cross-encoder is listed. The retrieval → 14B → 32B → 72B cascade and its stage gains are listed under Q2. — [rbiswasfc/eedi-mining-misconceptions](https://github.com/rbiswasfc/eedi-mining-misconceptions); [write-up](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/writeups/mth-101-1st-place-detailed-solution)
- **Eedi 11th place** also used Qwen2.5-32B-Instruct as the retrieval embedding model, with a separate LLM reranker. — [MilchstraB/Eedi](https://github.com/MilchstraB/Eedi)
- **LMSYS Chatbot Arena (2024) 1st place:**
  - Teachers: Qwen2-72B (5-fold CV log loss 0.875, 0.881, 0.869, 0.880, 0.875) and Llama3-70B (0.874, 0.877, 0.877, 0.873, 0.873).
  - Distilled Gemma-2-9B student: CV 0.862, 0.876, 0.858, 0.872, 0.868.
  - LoRA adapters merged and quantized to 8-bit: LB 0.882 (0.876 with TTA); final private score 0.96898.
  — [LMSYS 1st place](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion/527629)
- **LMSYS 4th place** (gold, out of 1,849 teams) became a pairwise-judge library covering gemma-2-9b-it on 4× A100, position-bias correction and pseudo-label distillation. — [pairjudge](https://github.com/DaoyuanLi2816/pairjudge)
- **Foursquare Location Matching (2022):**
  - A silver-medal pipeline trains XLM-RoBERTa with ArcFace. It uses cosine similarity plus coordinate distance to generate candidates, then feeds cosine similarity, distance, LCS and TF-IDF features to LightGBM. — [dongkyunk/Kaggle-Foursquare](https://github.com/dongkyunk/Kaggle-Foursquare)
  - Another top solution builds second-stage LightGBM features from multilingual-BERT embeddings with SVD. — [TakoiHirokazu repo](https://github.com/TakoiHirokazu/Kaggle-Foursquare-Location-Matching)

### Inferences
- The Kaggle wins where LLM rerankers took over from encoders involved reasoning or world-knowledge-heavy text: math misconceptions and chat preference. For short, structured name and address pairs, the only documented winning pattern is a GBDT over many similarity features, as in Foursquare, which is the current stack's pattern. The evidence therefore favours an LLM judge as an **added stage or feature**, not a replacement.
- A recipe consistent with these winners:
  - a yes/no-logit or seq-cls head;
  - LoRA r=64 on all linear layers, with `lm_head` frozen;
  - mined hard negatives;
  - 5-fold or out-of-fold training (which doubles as GBDT features);
  - vLLM serving with prefix caching;
  - optional distillation back into the 0.6B or CE if the 4B helps.
- **Suggested bounded test (inference):**
  - Fix the TRAIN and TUNE businesses and the full target pool.
  - Train a 4B LoRA on TRAIN pairs from the uncertain band plus hard negatives, one epoch first.
  - Add its out-of-fold logit as a GBDT feature, then compare paired TUNE F0.5, precision and false links, broken down by country and script.
  - Stop if the paired gain does not beat seed noise or precision drops.

### Gaps
- WSDM Cup 2025 and 2025–2026 Kaggle competitions (for example MAP Math Misunderstandings, Jigsaw rules) were not retrieved in this pass.
- The Foursquare 1st-place details were not retrieved; the write-up page returned no content.
- I found no published DeBERTa-against-LLM numbers from LMSYS or Eedi write-ups.
