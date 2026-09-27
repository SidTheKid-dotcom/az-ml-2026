# Proven methods for multilingual business/POI record matching (India native-script and France emphasis)

Scope note: this research pass was done on 2026-09-27 using papers, competition write-ups and library documentation. Each number is as reported by its source, on that source's own data and metric (IoU, F1, accuracy or Recall@1). None of these numbers can be compared directly with this project's macro F0.5. Licence constraint: only MIT, Apache or BSD tools and models may be used. Tools that fail this constraint, or whose licence I did not verify, are flagged.

## Q1. What works for Indic-script ↔ Latin name matching (transliteration, phonetic keys, multilingual embeddings, char models), and how much does it help?

### Takeaway
Transliteration alone is not accurate enough to use as an exact key. The strongest open Indic transliterator is IndicXlit (MIT, about 11M parameters). On the named-entity test set, its top-1 accuracy is only about 13–61%, depending on language and direction, and vowel errors are the largest error class. Transliteration output should therefore feed fuzzy, phonetic or learned similarity (ideally using top-k outputs), not exact matching. On Indian names, the largest measured gains come from supervised pairwise models. On a 50K-pair held-out set of Indian KYC person names, F1 was 0.726 for Levenshtein, 0.806 for fine-tuned mBERT, and 0.946–0.994 for a fine-tuned 8B LLM with augmentation and a curriculum. I found no published comparison of LaBSE, mE5 and bge-m3 on Indic ↔ Latin business names, so that comparison must be measured locally.

### Cited Findings
**Transliteration tools and licences**
- IndicXlit is a transformer-based multilingual transliteration model of about 11M parameters. It covers 21 Indic languages in both directions (Roman → native and native → Roman) and is MIT-licensed. — [IndicXlit GitHub](https://github.com/AI4Bharat/IndicXlit)
- It is trained on Aksharantar: 26 million transliteration pairs across 21 Indic languages and 12 scripts. Sources include Wikidata entity names, the Samanantar parallel corpus and IndicCorp. — [Aksharantar abstract (alphaXiv mirror of arXiv 2205.03018)](https://www.alphaxiv.org/abs/2205.03018v2)
- IndicXlit improves accuracy by 15% on the Dakshina test set compared with prior results. — [ACL Anthology, Findings EMNLP 2023](https://aclanthology.org/2023.findings-emnlp.4/)
- Top-1 accuracy on the Aksharantar **named-entity** test subset is modest:
  - English → Indic ranges from about 20.2% to 58.9% across languages.
  - Indic → English ranges from about 12.9% to 60.8%.
  - Source: [IndicXlit README, evaluation tables](https://github.com/AI4Bharat/IndicXlit)
- Re-ranking beam candidates with a language model (interpolation weight α=0.9) raised accuracy by about 12% on average. The gain was largest for native-language words and high-resource languages. — [alphaXiv overview (secondary summary)](https://www.alphaxiv.org/overview/2205.03018v2); per-language "+rerank" table (top-4 re-ranking) in [arXiv PDF](https://arxiv.org/pdf/2205.03018v2)
- Vowel errors are the most common error type across languages, at 45% of errors. — [Aksharantar arXiv PDF](https://arxiv.org/pdf/2205.03018v2)
- Aksharamukha (script conversion across Indic scripts and romanization formats such as IAST, ITRANS and ISO) is released under **GNU AGPL 3.0**. It is therefore excluded by the MIT/Apache/BSD constraint. — [Aksharamukha GitHub](https://github.com/virtualvinodh/aksharamukha)
- The Python `indic_transliteration` package is MIT-licensed. — [indic_transliteration_py GitHub](https://github.com/indic-transliteration/indic_transliteration_py)

**Phonetic keys**
- `indic-soundex` 0.1.0 is a Soundex variant for Indian names:
  - It handles aspirated consonants (bh, dh, gh), compound characters (ksh, gy), the Tamil zh↔l alternation, and v/w, q/k and f/ph.
  - Examples: Sharma/Sarma → S650, Krishna/Krushna → K625, Bharat/Bharath → B630.
  - The package reports no evaluation.
  - Source: [PyPI indic-soundex](https://pypi.org/project/indic-soundex)
- An ACL 2026 paper states that Soundex and Metaphone are tuned for English pronunciation and are ill-suited to Indian languages, where morphological and cross-script variation dominates. It cites Mhaske et al., 2022. — [SGER, arXiv 2605.23597](https://arxiv.org/html/2605.23597)

**Supervised pairwise models on Indian names (Dream11 KYC, 50,000 held-out pairs)**
- Results by method:

  | Method | F1 | Precision | Recall |
  |---|---|---|---|
  | Levenshtein (threshold 0.8) | 0.726 | 75.1% | 70.2% |
  | Fine-tuned `bert-base-multilingual-cased` | 0.806 | | |
  | GPT-4o few-shot | 0.911 | | |
  | Llama-3-8B SFT | 0.946 | | |
  | Llama-3-8B SFT + augmentation | 0.973 | | |
  | SGER curriculum (parse name structure first, then match) | 0.994 | 99.95% | 98.9% |

  Source: [SGER, arXiv 2605.23597](https://arxiv.org/html/2605.23597)
- Growing the training set from 20K curated pairs to 50K+ pairs with structural-variant augmentation raised F1 from 0.946 to 0.973. — [SGER](https://arxiv.org/html/2605.23597)

**Multilingual and cross-script encoders**
- LaBSE reaches 83.7% bitext retrieval accuracy over 112 languages on Tatoeba, compared with 65.5% for LASER. — [LaBSE paper page](https://huggingface.co/papers/2007.01852)
- LaBSE is trained only on translation pairs. English MTEB averages are: LaBSE 45.2, mE5-small 57.9, mE5-base 59.5, mE5-large 61.5, mE5-large-instruct 64.4. — [Multilingual E5 technical report](https://arxiv.org/html/2402.05672)
- On cross-lingual semantic discrimination (DE↔FR Recall@1, mean over four sets), LaBSE scores 94.43. The others score: paraphrase-multilingual-mpnet 91.63, mE5-large 91.21, gte-multilingual-base 90.46, mE5-base 87.07. Retrieval-tuned models such as mE5 benefit from pivoting through English, while bitext models such as LaBSE work best directly across languages. — [arXiv 2502.08638](https://arxiv.org/html/2502.08638v1)
- Symphonym (2026) is a character-level student model distilled from IPA articulatory features:
  - It maps names from 20 writing systems into a 128-dimensional phonetic space.
  - It reaches Recall@1 of 85.2% and MRR of 90.8% on the MEHDIE Hebrew/Arabic cross-script benchmark.
  - An ablation using raw articulatory features alone reaches only 45.0% MRR.
  - Source: [arXiv 2601.06932](https://arxiv.org/abs/2601.06932)
- ParaNames provides 118 million names across 400 languages. It is a possible source of cross-script name pairs for training. — [ParaNames, arXiv 2202.14035](https://arxiv.org/html/2202.14035v3)
- Overture's production name similarity for divisions uses a cross-lingual XLM-RoBERTa sentence transformer. It embeds every name variant and average-pools them, so "München" and "Munich" come out similar without a translation table. — [Overture workshop: matching concepts](https://github.com/OvertureMaps/workshop/blob/main/8-matching-concepts.md)
- On OpenSanctions Pairs (31 countries, cross-script names), LLM matchers fail mainly on cross-script transliteration (Arabic, Cyrillic, Latin). Blocking uses an inverted index over character n-grams with transliteration handling. — [OpenSanctions Pairs, arXiv 2603.11051](https://arxiv.org/html/2603.11051v1)
- Transliterating into the knowledge-base language and then matching monolingually is "a reasonable approach." However, name matching only needs a score for each pair, not a generated transliteration. — [McNamee et al., Cross-Language Entity Linking, IJCNLP 2011](https://aclanthology.org/I11-1029.pdf)

### Inferences
- **Ranked for India native-script ↔ Latin (my synthesis):**
  1. Keep a supervised pairwise scorer (cross-encoder or fine-tuned LLM) that sees the raw native-script string and its romanization. Supervision gave the largest measured jumps (0.726 → 0.806 → 0.946+ F1, from SGER).
  2. Romanize native-script names with IndicXlit (Indic → English, beam top-k, MIT). Compute string similarities on both the romanization and a vowel-collapsed "consonant skeleton." Because about 45% of transliteration errors are vowel errors, a skeleton key probably absorbs many of them. This is an estimate and has not been measured.
  3. Add a script-aware retrieval channel: character 3-gram TF-IDF over the romanized skeleton, plus dense retrieval. Measure recall@k separately for each script.
  4. Use Indic phonetic keys as features, not as hard blocking keys, because the library is unevaluated.
- Sentence-level evidence favours translation-aligned encoders (LaBSE) for direct cross-lingual similarity. Business names, however, are mostly *transliterated*, not translated, and are short. The choice between LaBSE, mE5 and bge-m3 therefore has to be measured on native ↔ Latin pairs from this project's own training split.
- A cheap augmentation hypothesis: generate native-script variants of Latin training names with IndicXlit English → Indic, to create cross-script positives and hard negatives. This mirrors SGER's augmentation gain of +2.7 F1 but has not been tested on business names.

### Gaps
- I found no published head-to-head of LaBSE, mE5 and bge-m3 (or char-CNN models) on Indic native-script ↔ Latin **business** names.
- I found no study that measures the record-linkage recall or F1 gain from adding IndicXlit. The only numbers are transliteration accuracy.
- I did not retrieve Mhaske et al. (2022), the study cited for phonetic algorithms on Indian names.
- `indic-soundex`: licence not shown in the excerpt and no evaluation.
- Symphonym: Indic-script coverage and licence not verified.
- SGER data are person names from a proprietary KYC set, not business names. The licences of the Llama-family models it used were not checked against the MIT/Apache constraint.

## Q2. What is known about company-name matching (legal forms, token-frequency weighting, acronyms, DBA aliases), and which steps give the largest gains?

### Takeaway
Evidence supports four steps:
1. Extract the legal form into its own attribute rather than deleting it, because some companies differ only in legal form.
2. Down-weight frequent or generic tokens (TF/IDF weighting, "core words," local name distinctiveness).
3. Add explicit number-mismatch and asymmetric extra-token penalties.
4. Use a learned or LLM judge framed as contradiction detection.

Measured end-to-end gains are scarce. The exception is LLM versus rule-based matching on OpenSanctions Pairs (91.33 → 98.23–98.95 F1). In France, most legal names carry no legal form at all, so a missing legal form must be treated as neutral.

### Cited Findings
- **Legal form inside the name field.** Kruse et al. (2021) showed that classic measures (normalized Levenshtein, Jaro-Winkler, Jaccard, Soft TF-IDF) cannot separate companies that differ only in legal form. After splitting "cleaned name" from "legal form," cleaned-name similarity reached 100% and the legal-form attribute separated the true matches. Prior work that simply *deleted* forms such as "inc."/"ltd." failed because some companies differ only in their legal form. — [Kruse et al. 2021, BIS](https://www.tib-op.org/ojs/index.php/bis/article/download/44/114/1115)
- **Legal-form extraction accuracy** on German company names:
  - A Bi-LSTM sequence labeller reached about 99.2% F1.
  - Hybrid classifiers reached weighted F1 of 99.70% (SVC) and 99.57% (random forest).
  - An adapted rule-based `cleanco` reached exact-match ratios of only 0.696 (classification) and 0.705 (extraction), more than 20% behind the DL and hybrid approaches.
  - `cleanco` was best only on the "no legal form" class (0.969).
  - Source: [Kruse et al. 2021](https://www.tib-op.org/ojs/index.php/bis/article/download/44/114/1115)
- Loster et al. split company names into core name, legal form, person names and location names, and recognized the legal-form tag with 98% F1 (as reported in Arimond et al.). — [Arimond et al. 2023, arXiv 2310.12766](https://arxiv.org/pdf/2310.12766)
- **France specifically:**
  - France has the most legal-form (ELF) codes of any jurisdiction (165).
  - Most French and Belgian entities carry no legal form in their legal names, and French entities are not legally obliged to include it.
  - France and Belgium score markedly lower on legal-form classification.
  - Their LENU library is CC0.
  - Source: [Arimond et al. 2023](https://arxiv.org/pdf/2310.12766)
- `cleanco` is MIT-licensed. It strips legal-type terms, infers the organization type, and suggests possible countries (e.g., "Oy" suggests Finland). It covers 16 entity types mapped to 66 countries. — [cleanco GitHub](https://github.com/psolin/cleanco); coverage from [Arimond et al.](https://arxiv.org/pdf/2310.12766)
- libpostal's French `company_types` dictionary (MIT) maps spelled-out forms and dotted or spaced variants. Entries seen in the excerpt: sarl/s.a.r.l., sas/s.a.s., eurl/e.u.r.l., snc, sc, scs, sca, sep, ei, asbl, sprl, sprlu, ss, fcp. — [libpostal fr/company_types.txt](https://raw.githubusercontent.com/openvenues/libpostal/master/resources/dictionaries/fr/company_types.txt)
- De Nederlandsche Bank's `name_matching` (MIT/X):
  - Matches company names with options `legal_suffixes`, `common_words` and `top_n=50` (TF-IDF preselection).
  - Combines distance metrics such as `bag`, `typo` and `refined_soundex`.
  - Its algorithms are based on the Abydos package.
  - Source: [name_matching GitHub](https://github.com/DeNederlandscheBank/name_matching)
- **Term-frequency weighting (Splink).** A match on a common value is worth less than a match on a rare one. On FEBRL demo data, a first-name match is worth +6, increased for a rare name such as "Portia" and reduced for a common one such as "Jack." In one example, each rare field contributed about +5 extra match weight. TF adjustments are added as independent terms per comparison. — [Splink docs, term-frequency adjustments](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)
- **OpenSanctions logic-v2 weights** (practitioner rules, not measured gains):
  - name_match 1.00; registry-identifier matches 0.95; weak_alias_match 0.80.
  - `nm_number_mismatch` penalty of 0.3 for mismatching numbers in company names.
  - Unmatched name parts weigh 0.8 on the query side and 0.2 on the result side.
  - It uses "cultural reference data" for cross-language and cross-script matching.
  - Source: [OpenSanctions matcher](https://www.opensanctions.org/matcher)
- OpenSanctions' `rigour` library keeps an organisation-type file (`org_types.yml`, about 3.5k lines). It has a `compare` form that standardizes variants without abstracting as far as a `generic` form. — [rigour org_types.yml](https://github.com/opensanctions/rigour/blob/main/resources/names/org_types.yml)
- OpenSanctions defines `weakAlias` as "a relatively broad or generic alias that should not be used for matching in screening systems." This is a direct model for handling DBA and trade names. — [OpenSanctions data dictionary](https://www.opensanctions.org/reference/)
- **OpenSanctions Pairs benchmark:**
  - Size: 755,540 labelled pairs from 293 sources across 31 countries, with multilingual and cross-script names. It includes about 48K companies and 47K organizations.
  - Rule-based nomenklatura RegressionV1 (18 features: token overlap, Levenshtein, phonetic, dates, IDs, country): F1 91.33, precision 84.46, recall 99.42. It over-matches, for example scoring 0.98 on a common-name non-match.
  - Llama-3.1-8B, optimized zero-shot prompt: F1 95.94 (P 97.98).
  - DeepSeek-R1-Distill-Qwen-14B, optimized prompt: F1 98.23 (P 98.60, R 97.86).
  - GPT-4o: up to 98.95 F1.
  - Framing the task as *contradiction detection* rather than similarity was the key prompt insight.
  - Few-shot examples did not improve on optimized zero-shot prompts.
  - Source: [arXiv 2603.11051](https://arxiv.org/html/2603.11051v1)
- LLM precision risk: ChatGPT zero-shot recall was ≥98% with lower precision, meaning it tended to over-match. In-context demonstrations raised F1 by up to 7.85% with similarity-based selection, and by 4.92% with a fixed set of 10 hand-picked demonstrations. — [Peeters & Bizer, Using ChatGPT for EM](https://arxiv.org/html/2305.03423)
- GPT-4 given natural-language matching rules *lost* 1–3% F1 compared with its best zero-shot prompt. — [Peeters et al., Entity Matching using LLMs](https://arxiv.org/html/2310.11244v4)
- Overture's POI name similarity treats common "core words" (e.g., shop, restaurant) separately. It also uses "spatial context (how distinctive a name is within its local area)" as a feature. — [Overture Places guide](https://docs.overturemaps.org/guides/places)

### Inferences
- **Ranked for company names (my synthesis):**
  1. Split the legal form into a separate field using the libpostal/cleanco lists plus a curated French list (SASU, SA, SCI, SELARL and others). Encode it as a three-way feature: agree, disagree or missing. Missing must be neutral, especially for France.
  2. Weight tokens by frequency, computed per country and ideally per locality ("local distinctiveness"). Handle generic business-type words (pharmacie, boulangerie, restaurant, kirana, store) separately from the distinctive core.
  3. Add a number-mismatch feature for branch or unit numbers, e.g. "Store 12" vs "Store 21".
  4. Add asymmetric token-coverage features: the share of the shorter name's tokens found in the longer name, with different penalties for extra tokens on each side.
  5. Take the max over alias pairs (DBA/brand names), treating generic aliases as weak.
  6. Add an acronym feature: the initials of one name equal a token of the other. This is unmeasured.
  7. Use an LLM or cross-encoder judge framed as contradiction detection for hard pairs, calibrated for precision because the metric is F0.5 and zero-shot LLMs can over-match.
- Deleting legal forms outright, which cleanco does by default, can create false links between sibling entities that differ only in legal form. That directly hurts precision-weighted metrics.

### Gaps
- I found no published measurement of precision/recall gains from acronym handling or DBA/alias expansion in company matching.
- I found no GLEIF or OpenCorporates matching study with reported accuracy. The GLEIF API offers "fuzzy" name/address matching but publishes no numbers.
- The Splink docs illustrate the TF-adjustment mechanism but report no benchmark gain.
- Senzing was not assessed. The only related source found was libpostal's note that Senzing supplied some parser training data.
- Licences of Splink, dedupe, rigour/fingerprints and Abydos (the basis of DNB `name_matching`) were not re-verified in this pass.
- The libpostal excerpt was truncated, so I could not confirm entries for SASU or SA.

## Q3. For POI/address matching (Foursquare 2022, Overture conflation, address parsing), which features and models gave the biggest measured gains, especially cross-country/multilingual, and what about Indian and French address conventions?

### Takeaway
The best-documented multilingual POI result is the Foursquare 2022 competition (1.5M+ noisy places worldwide). The winners used a staged pipeline:
1. A union of spatial-kNN and name-embedding/TF-IDF kNN candidates, giving a recall ceiling of 0.979–0.994 IoU.
2. A GBDT with about 120 string and distance features plus per-record aggregate statistics (CV 0.875–0.878).
3. Multilingual cross-encoders (XLM-R-large, mDeBERTa-v3-base) that read the concatenated fields plus tabular features (CV 0.907; ensemble 0.911).
4. Set-merging post-processing (CV 0.9166).

Academic work adds that explicit distance embeddings and neighbourhood context give +2.7 to +7.2 F1 over an LM-only matcher and transfer better to unseen cities. In India, PIN codes are informative but noisy, and addresses are landmark-heavy with vernacular street terms. In France, libpostal's MIT dictionaries cover street types and legal forms but contain ambiguities and apparent gaps.

### Cited Findings
**Foursquare Location Matching (Kaggle, 2022)**
- The dataset has more than 1.5M place entries deliberately altered with noise, duplicates and incorrect information. Scoring is sample-wise mean IoU of predicted match sets. — [Kaggle competition page](https://www.kaggle.com/c/foursquare-location-matching)
- 1,079 teams took part. — [Kaggle overview](https://www.kaggle.com/competitions/foursquare-location-matching/overview)
- **1st place, stage 1 (candidates):**
  - 100 nearest neighbours by lat/lon plus 100 by cosine similarity of `bert-base-multilingual-uncased` name embeddings (cuML kNN).
  - LightGBM kept the top 20 from each channel, about 40 candidates per record.
  - Max IoU after this stage: 0.979.
  - Source: [1st place write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
- **1st place, stage 2 (GBDT):**
  - About 120 features: Levenshtein, Jaro-Winkler, per-id statistics (max, min, mean) of each similarity and ratios to them, lat/lon distance, and SVD-reduced name embeddings.
  - LightGBM CV 0.875.
  - A 0.01 threshold removed about 90% of candidates while keeping max IoU at 0.974.
  - Source: [1st place write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
- **1st place, stage 3 (models and ensemble):**
  - CatBoost: CV 0.878.
  - xlm-roberta-large and mdeberta-v3-base were fed text of the form "name + categories + address + city + state," plus about 70–90 tabular features concatenated into the head. mDeBERTa, trained with FGM and EMA, reached CV 0.907.
  - Ensemble (weights LGB 0.01, CatBoost 0.32, XLM-R 0.29, mDeBERTa 0.38): CV 0.911.
  - Source: [1st place write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
- **1st place, stage 4 (post-processing):** merge two ids' match sets when shared ids exceed 50% from either side, then re-score the new pairs with XLM-R. This raised CV to 0.9166. — [1st place write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)
- The 1st-place leaderboard jump from 0.900 to 0.971 came from exploiting train/test overlap (a leak). It is not a transferable technique. The code is MIT-licensed. — [write-up](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution); [code](https://github.com/TakoiHirokazu/Kaggle-Foursquare-Location-Matching)
- **2nd place (T0m's part):**
  - 128 candidates per id from lat/lon neighbours plus name-TF-IDF cosine neighbours, mixed with **country-specific optimized ratios**. Max IoU 0.9895.
  - A custom Transformer "candidate blocking" model. Absolute features: lat/lon, country embedding, category embedding, XLM-R-base name embedding, TF-IDF-SVD. Relative features: edit distances on name, address, zip, phone, URL and categories. Single model LB 0.918.
  - A 0.005 threshold plus union-find kept 4.1 candidates per id at max IoU 0.986.
  - Source: [2nd place discussion](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336072)
- **9th place:**
  - 400 candidates per id (350 from lat/lon kNN, 50 from lat/lon+name-TF-IDF kNN) gave max IoU 0.994 over 420M pairs. A CatBoost prune at 0.005 cut this to 3M pairs at max IoU 0.987.
  - Text encoding inside CatBoost added +0.010. The model also used 12 string-similarity methods and target encoding of name pairs and category pairs.
  - Leaderboard steps: CatBoost 0.925 → + graph post-processing 0.930 → + LightGBM 0.933 → + training on all data 0.943 → + 5-model ensemble 0.945.
  - Source: [9th place discussion](https://www.kaggle.com/c/foursquare-location-matching/discussion/336415)
- A simple silver-medal pipeline trained XLM-RoBERTa with ArcFace loss for name embeddings, used cosine plus coordinate distance for candidates, and a GBDT on cosine, distance, LCS and TF-IDF features. — [dongkyunk/Kaggle-Foursquare](https://github.com/dongkyunk/Kaggle-Foursquare/)

**Academic and production POI matching**
- **GeoER (WWW 2022)** combines a language model, a distance embedding and neighbourhood attention over nearby entities:
  - It beat Ditto (RoBERTa) on all 8 datasets: F1 89.8–97.6 against Ditto's 82.6–94.4, a gain of +2.7 to +7.2 per dataset.
  - In ablation, the distance embedding contributed more than the other components.
  - Cross-city test (train on other cities, test on Singapore): GeoER lost 3.0 F1 on average, while Ditto lost 1.9–7.4. The authors attribute Ditto's loss to unseen POI and street words.
  - Source: [Geospatial Entity Resolution, ACM WWW 2022](https://dl.acm.org/doi/fullHtml/10.1145/3485447.3512026)
- **Overture Places conflation:**
  - ML-based pairwise matching followed by clustering, which resolves chains of matches.
  - Features: name similarity (string comparison with common "core words" handled separately, plus text-embedding similarity), address similarity, website domain and path, phone, category-embedding similarity, spatial distance, and local name distinctiveness.
  - Labels come from human annotators plus LLM labels produced by an optimized prompt built on an evidence hierarchy: functional identity, then phone, then name uniqueness and distance, then website, category and brand.
  - Source: [Overture Places guide](https://docs.overturemaps.org/guides/places)
- In Overture's divisions workshop, blocking uses H3 cells whose resolution varies by admin subtype. A near-perfect geometric match with a very different name "scores poorly" in the composite. — [Overture workshop](https://github.com/OvertureMaps/workshop/blob/main/8-matching-concepts.md)
- A multi-feature POI matcher (spherical distance, edit distance and Jaro-Winkler over name, address and distance, weighted by AHP) raised recall by 17.43% and 5.17% over name-only and distance-only baselines respectively. Accuracy rose by 4.37% and 1.22%. — [Zhao et al., ISPRS Annals 2022](https://isprs-annals.copernicus.org/articles/X-3-W2-2022/87/2022/isprs-annals-X-3-W2-2022-87-2022.html)
- A PRISMA systematic review groups POI similarity into name, type and spatial footprint measures, with Jaro and Jaro-Winkler among the common name measures. — [Sun et al. 2023, CEUS](https://www.acsu.buffalo.edu/~yhu42/papers/2023_CEUS_POIConflation.pdf)

**Indian address conventions**
- Flipkart pre-trained RoBERTa on a large address corpus and reached about 90% sub-region classification accuracy for North and South Indian cities with minimal preprocessing. It beat Word2Vec+TF-IDF and Bi-LSTM. Preprocessing combined edit distance with phonetic algorithms. — [Mangalgi et al. 2020, arXiv 2007.03020](https://arxiv.org/abs/2007.03020)
- A PIN code may cover about 50 sq km, and customers often enter wrong PINs. Place names romanized phonetically from local languages vary in spelling. Cities designate streets with labels such as "blocks," "mains," "sectors" or "phases." — [HBS Digital Initiative student post on Flipkart (secondary)](https://d3.harvard.edu/platform-rctom/submission/flipkart-using-machine-learning-to-solve-unique-problems-in-indian-e-commerce)
- Addresses can run to about 200 words with directions (e.g., from the nearest bus stop), while 9–10 words usually suffice. — [Livemint 2018 interview with Flipkart (news)](https://www.livemint.com/Companies/G12niFPALtvPjjrxVexopL/Heres-how-Flipkart-is-using-AI-to-resolve-Indias-address-p.html)
- Indian addresses vary in language, script and format, with regional terms such as street, main/cross and colony. Meesho's GeoIndia geocodes them by predicting hierarchical H3 cells. — [GeoIndia, EMNLP 2024 Industry](https://aclanthology.org/2024.emnlp-industry.29.pdf)
- **bharataddress** open-source parser (self-reported, 200-address gold set):
  - F1 by component: pincode 0.995, state 0.971, district 0.965, city 0.959, landmark 0.918, locality 0.796, building name 0.679.
  - It lists 409 vernacular terms (Marg, Salai, Rasta, Veedhi) and supports 6 Indic scripts.
  - Source: [bharataddress site](https://bharataddress.nativerse-ventures.com/)
- Transformer address parsers score near-perfectly on synthetic data but degrade on real Indian addresses. Retrieval from an OSM-derived knowledge base raised SUBURB F1 from 0.162 to 0.519 and ROAD from 0.011 to 0.151 (500 real addresses). This depends on external OSM data. — [ETASR 2026](https://etasr.com/index.php/ETASR/article/view/18510)
- A typical Indian address NER tag set: area_locality_name, city_town, flat_apartment_number, landmark, society_name, street, sub_locality, pincode. — [Indian-Address-Parser (GitHub)](https://github.com/malharnd/Indian-Address-Parser-and-Entity-Matching)
- Deepparse is **LGPL-3.0**, which falls outside the MIT/Apache/BSD list. Its BPEmb variant generalizes better than FastText to Indian addresses not seen in training. — [Deepparse, arXiv 2311.11846](https://arxiv.org/pdf/2311.11846)

**French address conventions**
- libpostal (MIT) expands abbreviations in 60+ languages. For example, "Quatre vingt douze R. de l'Église" becomes "92 rue de l eglise", folding accents, splitting the elision and converting spelled-out numbers. Its CRF parser was trained on about 1.2B records from OpenAddresses, OSM and Senzing-supplied data. — [libpostal GitHub](https://github.com/openvenues/libpostal); [ecosyste.ms mirror of README](https://awesome.ecosyste.ms/projects/github.com/openvenues/libpostal)
- libpostal's French street-type dictionary includes:
  - avenue|av|ave|aven|avn…
  - boulevard|bd|bde|blv|blvd|boul|bvd|bld…
  - rue|r; chemin|ch|che; impasse|imp; place|pl; route|rt|rte; allée|all
  - route nationale|rn; rond point|rdpt; résidence|res; square|sq; cours|crs; hameau|ham
  - grand rue|gr and grand boulevard|gbd variants
  - **quai|au**: "au" is also a very common French word
  - Source: [libpostal fr/street_types.txt](https://raw.githubusercontent.com/openvenues/libpostal/master/resources/dictionaries/fr/street_types.txt)
- NF Z10-011 is the French postal-address line standard. The French banking committee CFONB maps it to ISO 20022 structured fields, including department. — [CFONB guidance PDF](https://www.cfonb.org/sites/www.cfonb.org/files/documents/CFONB_Rules_Guide_for_transposition_into_structured_postal_address_according_to_ISO20022_standard_0.pdf)

### Inferences
- **Ranked for POI/business matching (my synthesis):**
  1. Take a union of retrieval channels: spatial/postcode plus name-embedding plus character TF-IDF. Tune mixing ratios per country, then add a light pruning model. This produced the recall ceilings of 0.979–0.994 IoU.
  2. Use per-record aggregate ("relative") features: each similarity compared with the max, min and mean over that record's candidate set.
  3. Use a multilingual cross-encoder with tabular features injected into the head (a +0.029 CV gain over CatBoost in the 1st-place pipeline).
  4. Add distance embedding and local name-distinctiveness features (GeoER, Overture). These also transfer across cities.
  5. Ensemble (+0.004 CV).
  6. Use transitivity or set-merge post-processing (+0.005 CV and LB) with care. It trades precision for recall, so under F0.5 and this project's singleton convention it must be tuned on TUNE.
- **India:**
  - Treat the PIN code as a strong but noisy signal: use it as a soft feature and as one retrieval channel, never as a hard filter.
  - Segment landmark clauses ("near," "opp," "behind") away from the core address, so shared landmarks do not create false similarity.
  - Normalize vernacular street and area terms (marg/salai/rasta/veedhi → road; nagar/colony/layout/sector/phase) before comparing tokens.
- **France:**
  - Use libpostal expansion, but guard ambiguous abbreviations such as "au" → quai.
  - Add types that were absent from the street-types file I retrieved: faubourg/fbg, ZI/ZA/ZAC, lieu-dit, lotissement, cité, domaine, quartier. They may exist in other libpostal files, so verify.
  - Derive the département from the postcode: the first two digits, with Corsica 2A/2B and overseas 97x as exceptions. This is standard French postal convention (domain knowledge, not sourced in this pass).
- The Foursquare metric (IoU over match sets) rewards recall-oriented post-processing more than a precision-weighted F0.5 does. Its thresholds and set-merging rules should not be copied directly.

### Gaps
- The Foursquare write-ups do not report per-country (India or France) IoU, so cross-country difficulty cannot be quantified from them.
- Overture's documentation describes its matching features and labelling but gives no precision/recall numbers in the retrieved text.
- I found no measured study of Indian *business-address matching* (as opposed to parsing or geocoding). Parser numbers are small or self-reported. GeoIndia's numbers were not captured.
- I found no quantified gain from French street-type or legal-form normalization in a linkage pipeline.

## Q4. Transferring a matcher trained in one country/language to another (e.g., US → France) without target labels: what minimized the loss?

### Takeaway
Published evidence comes mostly from product and bibliographic matching, not businesses, but it points the same way on five things:
1. **Multilingual backbones plus mixing source-language pairs with even small target sets recover most of the gap.** mBERT on German went from 67.1 to 88.0 F1 with 450 German pairs plus 7,200 English pairs.
2. **Fine-tuned PLMs transferred to unseen entities can collapse (−22 to −61 F1), while LLM judges are far more robust.** Every LLM beat the best transferred PLM by at least 8 F1, and GPT-4 by 40–68.
3. **Language-agnostic spatial features transfer across cities much better than LM-only matching.** GeoER lost 3.0 F1 against Ditto's 1.9–7.4.
4. **Domain adaptation (DADER) can add about 14% on some dataset pairs**, but the gain depends on the dataset.
5. **Overture uses LLM-generated labels in production** (evidence-hierarchy prompt) to get labels at scale across countries.

I found no study of transferring a business matcher from the US to France.

### Cited Findings
- **Cross-language product matching** (German test set, 1,200 pairs, 25% matches). F1 trained on 1,800 German pairs only, then with 7,200 English pairs added:

  | Model | German only | + English | Gain |
  |---|---|---|---|
  | SVM | 71.00 | 71.05 | +0.05 |
  | English BERT | 65.27 | 74.29 | +9.02 |
  | German BERT | 73.43 | 89.83 | +16.40 |
  | mBERT | 87.69 | 91.44 | +3.75 |
  | XLM-R | 73.40 | 86.98 | +13.58 |

  Source: [Peeters & Bizer 2022, arXiv 2110.03338](https://arxiv.org/pdf/2110.03338)
- **The value of added English data shrinks as target data grows** (mBERT, Peeters & Bizer):

  | German pairs | 0 English | 7,200 English | Gain |
  |---|---|---|---|
  | 450 | 67.11 | 87.97 | +20.86 |
  | 1,800 | 87.69 | 91.44 | +3.75 |
  | 3,600 | 93.63 | 94.46 | +0.83 |

  - With 450 German pairs, most of the gain came by 3,600 English pairs (86.82).
  - Any mix totalling fewer than about 2,000 pairs (with ≤900 German) stayed at about 75% F1 or less.
  - Source: [arXiv 2110.03338](https://arxiv.org/pdf/2110.03338)
- A Mannheim project used English as the high-resource language, German as few-shot, and French and Spanish as zero-shot targets. Transformers beat classical methods on the pairwise tasks, but no numbers were captured. — [Mannheim DWS student project](https://data.dws.informatik.uni-mannheim.de/Web-based-Systems-Group/StudentProjects/2020/Cross-lingual-Product-Matching-using-Transformers)
- Training on English product data transferred to Portuguese gave "promising results" with reduced target data. — [SAC 2024, Cross-Lingual Learning Strategies for Product Matching](https://dl.acm.org/doi/10.1145/3605098.3636001)
- **Unseen-entity transfer:**
  - Fine-tuned RoBERTa/Ditto models applied to WDC Products (a different product set) lost 36–56 F1 for Ditto and 22–61 F1 for RoBERTa. The in-domain Ditto reference is 84.90.
  - All LLMs scored at least 8 F1 above the best transferred PLM, and GPT-4 scored 40–68 above it.
  - GPT-4 zero-shot beat fine-tuned PLMs on 3 of 4 e-commerce datasets.
  - Fine-tuning LLMs added 1–26 F1 over zero-shot, and fine-tuned GPT-mini generalized to related domains.
  - Source: [Peeters, Steiner & Bizer, Entity Matching using LLMs](https://arxiv.org/html/2310.11244v4)
- **Cross-city transfer:** GeoER lost 3.0 F1 on average on unseen Singapore, while Ditto lost 1.9–7.4. The spatial components (distance embedding, neighbourhood attention) are robust to unseen words. — [GeoER](https://dl.acm.org/doi/fullHtml/10.1145/3485447.3512026)
- **Domain adaptation:**
  - DADER's domain adaptation improved Walmart–Amazon by 14.2% using Abt–Buy as the source.
  - Methods include gradient reversal (GRL), inverted-label GAN (InvGAN) and InvGAN plus knowledge distillation.
  - It can add active learning over a small verified target sample.
  - Source: [DADER demo, PVLDB 15](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)
- The full DADER paper groups feature-alignment methods into discrepancy-based, adversarial-based and reconstruction-based. — [Tu et al., SIGMOD 2022](https://dl.acm.org/doi/10.1145/3514221.3517870)
- DAME transfers from multiple source domains through a mixture of experts and studies zero-shot target performance. No numbers were captured. — [DAME, arXiv 2204.09244](https://arxiv.org/pdf/2204.09244)
- A zero-shot cross-language entity-linking system trained with multilingual representations "does reasonably well." Training changes improved recall and often matched in-language performance. — [MRL 2022](https://aclanthology.org/2022.mrl-1.4/)
- Supervised re-rankers trained on English relevance data "rarely" improve cross-lingual retrieval under language plus domain shift ("monolingual overfitting"). — [On cross-lingual retrieval with multilingual text encoders](https://www.researchgate.net/publication/359066092_On_cross-lingual_retrieval_with_multilingual_text_encoders)
- Retrieval-tuned mE5 benefits from machine-translating to an English pivot, while LaBSE works best directly across languages. — [arXiv 2502.08638](https://arxiv.org/html/2502.08638v1)
- Overture trains and evaluates its matcher on human labels plus **LLM-generated labels at scale**, sampled across providers, countries and categories, using an evidence-hierarchy prompt. — [Overture Places guide](https://docs.overturemaps.org/guides/places)
- Across 31 countries and cross-script names, zero-shot LLMs (e.g., an open 14B model at 98.23 F1) beat a tuned rule system (91.33) without task-specific training. — [OpenSanctions Pairs](https://arxiv.org/html/2603.11051v1)
- **Distribution shift in France:** most French legal names carry no legal-form token, unlike typical US names with suffixes such as LLC or Inc. A legal-form feature learned on US data therefore sees a different distribution in France. — [Arimond et al. 2023](https://arxiv.org/pdf/2310.12766)

### Inferences
- **Ranked levers for France with no labels (my synthesis):**
  1. Canonicalize surface forms before modelling (street types, legal forms, accents, elisions such as l'/d', saint/st, spelled-out numbers) so that French records contain fewer unseen tokens. GeoER attributes LM transfer loss to unseen words.
  2. Lean on language-agnostic features: distance, phone, website domain, postcode/département equality, number mismatch, token rarity.
  3. Train one multilingual backbone (mBERT, XLM-R or mDeBERTa families) jointly on all countries. In Peeters & Bizer, mBERT transferred best, and XLM-R needed more data.
  4. Create French pseudo-labels from agreement between diverse models, or from an open-licence LLM using an Overture-style evidence hierarchy. Self-train on them and audit precision on a small manually checked sample, if project rules allow LLM pseudo-labels on the supplied records.
  5. Try adversarial domain adaptation (DADER-style) only as a second-tier experiment, because its reported gains vary by dataset.
- Per-country thresholds for France cannot be tuned without French labels. A single global threshold with a precision margin is the safer default under F0.5. That is a judgement, not a measured result.

#### Cross-cutting ranked list (synthesis of Q1–Q4; strength = quality and relevance of evidence)
| Rank | Technique | Best reported effect (source metric) | Evidence strength |
|---|---|---|---|
| 1 | Union retrieval: spatial kNN + name-embedding/TF-IDF kNN, per-country mixing ratios, light pruning GBDT | Max IoU 0.979–0.994 before pruning; 4.1 candidates/id at 0.986 ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution), [2nd](https://www.kaggle.com/competitions/foursquare-location-matching/discussion/336072), [9th](https://www.kaggle.com/c/foursquare-location-matching/discussion/336415)) | Strong (large multilingual competition) |
| 2 | GBDT on rich string/distance features + per-record aggregate ("relative") stats | CV 0.875–0.878 ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)) | Strong |
| 3 | Multilingual cross-encoder (XLM-R-large / mDeBERTa-v3) with tabular features injected | CatBoost 0.878 → mDeBERTa 0.907; ensemble 0.911 ([1st](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)) | Strong |
| 4 | Fine-tuned or zero-shot LLM judge (contradiction framing) for hard pairs | Indian names: mBERT 0.806 → 8B SFT 0.946–0.994 F1 ([SGER](https://arxiv.org/html/2605.23597)); rules 91.33 → open 14B 98.23 F1 ([OS Pairs](https://arxiv.org/html/2603.11051v1)) | Strong on those tasks; cost and licence checks needed; zero-shot LLMs may over-match ([ChatGPT EM](https://arxiv.org/html/2305.03423)) |
| 5 | Distance embedding + neighbourhood/local-distinctiveness features | +2.7 to +7.2 F1 over Ditto; smaller cross-city drop ([GeoER](https://dl.acm.org/doi/fullHtml/10.1145/3485447.3512026)) | Moderate (English-speaking cities) |
| 6 | Mix source-language training pairs; multilingual backbone | +3.75 to +20.86 F1 on German target ([Peeters & Bizer](https://arxiv.org/pdf/2110.03338)) | Moderate (products, not businesses) |
| 7 | Legal form as a separate attribute (not deleted); missing = neutral in France | Extraction F1 about 99% (ML) vs 0.70 exact match (cleanco) ([Kruse](https://www.tib-op.org/ojs/index.php/bis/article/download/44/114/1115)); France mostly lacks forms ([Arimond](https://arxiv.org/pdf/2310.12766)) | Moderate for extraction; end-to-end gain unmeasured |
| 8 | Token-rarity weighting, generic "core words" handled separately, number-mismatch penalty | Mechanism only ([Splink](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html), [Overture](https://docs.overturemaps.org/guides/places), [OpenSanctions](https://www.opensanctions.org/matcher)) | Practitioner consensus; unmeasured |
| 9 | Transliteration to a common script (IndicXlit, MIT) + skeleton/phonetic features | Transliteration top-1 only about 13–61% on named entities; LM re-rank about +12% ([IndicXlit](https://github.com/AI4Bharat/IndicXlit), [overview](https://www.alphaxiv.org/overview/2205.03018v2)) | Moderate for transliteration; linkage gain unmeasured |
| 10 | Set-merge / transitivity post-processing | +0.0056 CV (1st), +0.005 LB (9th) | Moderate; may hurt precision under F0.5 |
| 11 | Address normalization (libpostal MIT; Indian vernacular terms; PIN as soft key) | Parser-level numbers only ([libpostal](https://github.com/openvenues/libpostal), [bharataddress](https://bharataddress.nativerse-ventures.com/)) | Weak to moderate |
| 12 | Adversarial domain adaptation (DADER) | +14.2% on one dataset pair ([DADER](https://www.vldb.org/pvldb/vol15/p3666-fan.pdf)) | Weak/variable |

### Gaps
- I found no published study transferring a business or POI matcher from the US (or any country) to France without labels. All transfer numbers above come from product, bibliographic, entity-linking or cross-city POI settings.
- DADER's full results table (average gains, and cases where adaptation hurts) was not retrieved; only the 14.2% example was captured.
- The zero-German-label result (English-only training evaluated on German) from Peeters & Bizer was not in the retrieved excerpt.
- Unicorn and AnyMatch (universal/zero-shot matchers) were not examined in this pass.
- Licences of the LLMs used in the cited studies (Llama 3, DeepSeek-R1-Distill-Qwen) and of the embedding models (LaBSE, mE5, bge-m3) were not re-verified here. The project's `models.yaml` remains the checked inventory.
