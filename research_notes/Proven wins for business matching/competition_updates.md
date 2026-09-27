# Competition updates: Amazon ML Challenge 2026 entity resolution, public activity on 27 Sep 2026

**Scope and method.** Sweep run 27 Sep 2026, 11:55–12:25 IST (06:25–06:55 UTC).

- **GitHub:** searched repos, code (`matching_results.tsv`, `candidate_pairs.tsv`, `source1_entity_id`) and commits. Then triaged the READMEs, root/doc `.md` files and commit messages of the **278 challenge repos pushed after 03:00 UTC (08:30 IST) on 27 Sep**.
- **Hugging Face:** API search plus cards of the known leads.
- **Web:** Exa (date-filtered from 26 Sep) and Parallel/Web search over LinkedIn, X, Reddit, Medium, YouTube, dev.to and Unstop.

Only public README/doc text, cards and commit messages were read. **No other team's data, embeddings, checkpoints or code was downloaded or run.** Times are UTC, with IST (UTC+5:30) in brackets.

This complements, and does not repeat, `research_notes/Entity resolution missing big wins/competition_public_intel.md` (written 08:40 IST). Every leaderboard (LB) number below is a **self-reported public-LB score (a subset of test)** from that team's repo. None is verified by me, and none is a full-test or private-LB score.

---

## 1. What did newly published teams at or near the top (LB ≥ 0.983) do?

### Takeaway
One new team at ≥0.985 published a complete, LB-measured pipeline and history: **GozonLegendent, public 0.987326**. Its main points:
- dense e5 retrieval;
- three listwise cross-encoders, including bge-reranker-v2-m3;
- per-record argmax;
- a label-free threshold of 0.75, not the validation-optimal 0.36;
- removal-only post-rules, including a **French same-address qualifier-swap rule worth +0.0011 on the LB**.

Two more new write-ups are just below 0.985: GarvRajput17 (0.983362) and Epic021 (0.983). Mveen3 (0.984819) is also new. **No team at ≥0.990 (e.g. Banana, 0.990816) has published anything I could find.**

### Cited Findings
- **GozonLegendent (public 0.987326, submission "v6pf")** — docs pushed 06:08–06:32 UTC (11:38–12:02 IST). The handoff says "~11:45 IST … The top team is at 0.992" and "one leaderboard attempt left". — [Documentation_template.md](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md), [TEAM_HANDOFF.md](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md), [commit f8fcd0edaf](https://github.com/GozonLegendent/amazon-ml/commit/f8fcd0edaf)
  - **Retrieval:**
    - Model: `intfloat/multilingual-e5-small` fine-tuned with symmetric InfoNCE (τ = 0.05), single-country batches and one mined same-name hard negative per pair, on 3M true pairs.
    - Search: exact fp16 GPU kNN inside each country label, record→S1 top-10 ∪ S1→record top-10.
    - Recall: 99.73%, or 99.36% within each record's top-10.
    - Ranking quality: the encoder ranks the correct entity first for 97.96% of held-out true pairs.
  - **Pruner:** LightGBM, ≤4 candidates per record with p ≥ 0.003, recall 99.12%. `candidate_pairs.tsv` has 8,702,689 pairs, down from 102,303,092 after blocking.
  - **Scorers:** three listwise cross-encoders, each with a learned "no match" slot:
    - e5-small (800k queries);
    - `intfloat/multilingual-e5-base` (MIT, 1.5M queries);
    - `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M, 800k queries).
  - **Final ranker:** LightGBM with 97 features. The top features by gain are the bge-reranker margin over the record's runner-up, the e5-base margin, the two logits and the pruner probability.
  - **Decision:** each record keeps only its argmax entity, and the link is kept if p ≥ **0.75**. The validation-optimal threshold was 0.36 (0.99305). Using 0.75 costs 0.0008 on validation (0.9922).
  - **Removal-only post-rules:**
    - `fr_swap` (France): 8,703 links removed;
    - `namesake`: no address and ≥2 S1 share the exact name; 1,366 links;
    - `cap`: >5 S2 or >6 S3 links per entity; 121 links.
  - **Validation protocol:** S1 folds by crc32 % 10.
    - Unmatched records take the fold of their top-1 retrieved S1, so decoys sit in the same fold as the entity they compete with. Otherwise "false merges look rarer than they are".
    - Folds 5–9 train the base models, folds 1–4 the rankers, and fold 0 is held out.
- **GozonLegendent threshold logic (label-free):**
  - Method: US/India validation true links per entity, divided by test links per entity, per probability band, give an "implied test precision". It is **40–54% below p = 0.75 and 73–94% above**.
  - Break-even: a false link costs an entity about 0.19 and an extra true link gains about 0.056, so a link pays only above about 77% precision.
  - Handoff rule of thumb: "adding a link pays only if it is right ≥ ~74% of the time; removing a set of links pays if ≥ ~26% of them are wrong."
  - — [Documentation_template.md §4–5](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md), [TEAM_HANDOFF.md §2](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)
- **GarvRajput17 (public 0.983362)**, commits 05:58 and 06:51 UTC (11:28 / 12:21 IST).
  - **Blocking:**
    - a fine-tuned multilingual-e5-small dense channel plus **S2↔S3 cross-links**, which raised "India's train recall ceiling from 96.2% to 99.65%";
    - a two-stage learned pruner that cut candidates from 11.9 to ~6 per entity "while raising recall";
    - a joint name+address channel.
  - **Matching:** house-number distance features (signed delta, ratio, edit distance). A fine-tuned xlm-roberta-base cross-encoder blended by 2-way CV took validation from 0.99006 to 0.99203; this blend had not been LB-scored at the time of the handoff.
  - — [commit 67ffd09e37](https://github.com/GarvRajput17/MLChallenge/commit/67ffd09e37), [commit 7678c5d690](https://github.com/GarvRajput17/MLChallenge/commit/7678c5d690)
- **Epic021 (public 0.983, "STACK-B5")**: a stage-2 stacker over its own model plus a teammate's model (holdout 0.98792). The teammate's model alone scored 0.975.
  - Retrieval: a `number|token` address-key retriever raised the oracle ceiling from 0.9867 to 0.9917 (India +0.011).
  - More training data was flat: 10% / 30% / 50% gave 0.97472 / 0.97551 / 0.97574.
  - Stacker v2 (holdout 0.98830) was awaiting LB.
  - — [STRATEGY.md](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md), [commit 21b82828cd](https://github.com/Epic021/amazon-ml-challenge-2026/commit/21b82828cd) (05:35 UTC / 11:05 IST)
- **Mveen3 (public 0.984819)**: baseline trained on 500k entities.
  - Out-of-fold: 0.98933 (US 0.98964, India 0.98888); ceiling 0.99671.
  - Precision 0.9985, recall 0.9705, singleton F0.5 0.9946.
  - About 70% of misses are no-address records whose name several S1 share.
  - — [docs/Documentation_template.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Documentation_template.md) (pushed 04:16 UTC / 09:46 IST)
- **Public LB snapshots mentioned today:**
  - "The top team is at 0.992" (~11:45 IST) — [GozonLegendent TEAM_HANDOFF.md](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)
  - "Leader: 0.9906" — [rajesh08114 HANDOFF.md](https://github.com/rajesh08114/business_entity_resolution/blob/master/HANDOFF.md)
  - "The leaderboard top (0.990) is not exploiting a trivial shortcut". The same file reports a leak check: IDs are uniform random 9-digit numbers, a linked record's ID correlates with its S1's ID at Spearman +0.0008 (S2) / −0.0016 (S3), and distractors are placed uniformly. — [TarunSitaraman EXPERIMENTS.md](https://github.com/TarunSitaraman/amazon-ml-challenge/blob/main/EXPERIMENTS.md)

### Inferences
- The ≥0.985 public recipe is now concrete and close to ours in family: E5 retrieval, trees, cross-encoders and one-owner decoding. The differences that were **measured on the LB** are all **decision-layer and France-specific**, not model capacity:
  1. a label-free threshold of 0.75;
  2. dropping density/cluster-count features;
  3. a French word-swap removal rule;
  4. per-S1 caps and namesake removal.
- GozonLegendent's cross-encoder ladder shows how much larger rerankers are worth: each added cross-encoder brought +0.0001 to +0.0009 on the LB. That is real but small.
- The gap from 0.9873 to 0.992 is not explained by any public write-up.

### Gaps
- No write-up from any team at ≥0.990 was found on GitHub, HF or social sites. GozonLegendent's "top team is at 0.992" is a second-hand snapshot.
- The LB outcomes of GozonLegendent's `fr_swap_ext` (v6pfx, "expected 0.9875–0.9877"), GarvRajput17's xlm-r blend and Epic021's stacker v2 were still pending at sweep time.

---

## 2. France (no labels): what was measured on the leaderboard?

### Takeaway
Two independent teams used LB probes and found **the same dominant French error**. French *sibling* businesses at the **same address and house number, with exactly one name word ("qualifier") swapped**, are being over-accepted.
- GozonLegendent: removing them gave **+0.0011 LB** (0.986201 → 0.987326; 8,703 links).
- Epic021: a stacker that dropped mostly such links raised **implied France from ~0.94 to ~0.950**.

Self-training or bootstrapped French dictionaries helped one team: France est. ~0.917 → ~0.945 (GarvRajput17). Hand-written French rules, synthetic French data and simply changing the France threshold did not help on the LB.

### Cited Findings
- **GozonLegendent `fr_swap` (removal only).** A link is removed when it meets all three conditions:
  - same first house number;
  - exactly one core word replaced by another;
  - both words are in a qualifier vocabulary **learned from the file's own French links**: words removed ≥200 times and added 0.2–5× as often, minus words that true pairs add or drop as noise.

  Example: "Regards Ecole" vs "Regards Amis". Each qualifier is swapped with ~25 different partners at random, unlike a synonym. **LB: 0.986201 without, 0.987326 with.** — [Documentation_template.md §4–5](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md)
- **Swaps that are ordinary true-pair noise and must not be removed:** swaps INTO "Services", "Fils", "Associés" or "France", and garbled typos. The same noise appears in US/India true pairs (Clinic → Services, Center).
- **`fr_swap_ext`:** also removes cases where the record's NEW word is a French qualifier and the entity's word is any other non-noise word ("Palestine Sportive" → "Palestine Club"). About 3,200 more links; LB result pending.
- — [TEAM_HANDOFF.md §4](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md), [commit f8fcd0edaf](https://github.com/GozonLegendent/amazon-ml/commit/f8fcd0edaf)
- **GozonLegendent's French address structure, from unlabelled test inputs:**
  - The S1 admin component is always a région.
  - On S2/S3 it is the same région about ⅓ of the time, the département about ⅓, and absent about ⅓.
  - So "whole-address similarity for French S2 looks like training S3". The team compares the street component and the finest locality instead, and drops the source indicator from the model.
  - France gets 3.31 links per entity vs 3.39 for US/India (training truth 3.46).
  - Ruled out: "Lowering the France threshold … Not safe." French normalisation was already adequate; only an apostrophe gap ("L'X" vs "L X") remains, worth ~0.0001.
  - — [Documentation_template.md §2.1, §5](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md), [TEAM_HANDOFF.md §5](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)
- **Epic021 (LB 0.983), same finding via the "word-swap class"** (same address and number, one word swapped, sole claimant):
  - Test pairs per S1 in this class: **France 4.3, India 2.1, US 0.95**.
  - Model probability on it: **France p ≈ 0.31 vs ≈ 0.70 in India/US**, where the training truth rate is ≈ 70%.
  - The stacker drops 0.135 French predictions per S1 (6× India/US), and **29k of 35k are same-number word swaps**. LB 0.983 then implies France +0.009, so "France errs by over-accepting those swaps".
  - — [Epic021 STRATEGY.md §4.4](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md)
- **Epic021's other France diagnostics:**
  - Implied France by version: B1 0.88 → B2 0.94 → BLEND-B4 0.94 → STACK-B5 **0.950**, from LB ≈ 0.468·India + 0.383·US + 0.15·France with India/US taken at holdout levels.
  - Expected F0.5 says France ≈ 0.967 while the LB implies ≈ 0.94: **"confident errors"**.
  - Borderline pairs (0.2 ≤ p ≤ 0.8): France 0.32 per S1 vs 0.11–0.14 elsewhere.
  - Its two models disagree 2× more in France.
  - Label-free "word roles" learned from unlabelled top-1 pairs give a French decoy vocabulary: *participations, développement, groupe*.
  - The classic decoy proxy (changed number + added word) fires on only 0.2–0.4% of French predictions, so classic decoys are already handled.
  - A pilot removing rank features (blamed by SHAP on crowded French streets) **failed**: −0.001 holdout, swap p unchanged.
  - Leave-one-country-out (LOCO): "no principled stricter threshold for unseen countries". US→India loses 0.04, and no feature group hurts transfer.
  - France-only re-threshold probes at 0.75/0.85 were awaiting LB.
  - — [Epic021 STRATEGY.md §4.4](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md)
- **GarvRajput17 (LB 0.983362):** self-training for unlabelled countries. The matcher's own high-confidence matches bootstrap France's cleaning dictionary, then France is re-cleaned and re-scored. Results:
  - "+0.0061 F0.5 on a simulated unseen country";
  - "on the real leaderboard, France's estimated score rose from ~0.917 to ~0.945";
  - house-number distance features addressed "the direct cause of France's excess false-match rate".

  The same repo also uses adaptive per-country stop words fitted from the records alone. — [commit 67ffd09e37](https://github.com/GarvRajput17/MLChallenge/commit/67ffd09e37)
- **Mveen3 (LB 0.984819):** "four diagnostic submissions blanked or invalidated one country's rows. Their exact scores are consistent with India/US scoring as out-of-fold and **France ≈ 0.960**, i.e. about 40% of the leaderboard loss from 15% of the entities." France handling:
  - a French parsing profile ("St" = Saint);
  - country-agnostic features;
  - optional monotone constraints;
  - no pseudo-labels (see §5).

  — [docs/Documentation_template.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Documentation_template.md), [docs/Pipeline_Architecture.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Pipeline_Architecture.md)
- **Negative LB results for France:**
  - **rajesh08114:** "v5 + French hand rules: 0.9834 val → 0.9738 LB (worse, reverted)" vs 0.9754 without; "French maps: −0.0016". — [HANDOFF.md](https://github.com/rajesh08114/business_entity_resolution/blob/master/HANDOFF.md)
  - **ayushmanjha52 (LB 0.9709):**
    - Approach: re-applies its reverse-engineered noise generator to unlabelled French S1 records to synthesise French true variants and sibling decoys.
    - Probe: France scored by the mean of the French and US/India models gave the "same leaderboard score as 0.9709".
    - A "test-like" synthetic gain of +0.020 locally "did not show up: real test siblings are easier than the synthetic ones".
    - — [docs/EXPERIMENT_LOG.md](https://github.com/ayushmanjha52/business-entity-resolution/blob/main/docs/EXPERIMENT_LOG.md), [README](https://github.com/ayushmanjha52/business-entity-resolution)
- **Unscored or proxy-only France methods:**
  - **Siva402-ai (LB 0.970441):** one-round self-training (France pseudo-labels at p ≥ 0.98 → match, ≤ 0.02 → non-match), checked on a US↔India transfer proxy. Not LB-scored. It reports "France has twice the uncertain band" and a 0.04–0.05 loss on a labelled unseen-country proxy. — [Documentation_template.md](https://github.com/Siva402-ai/amazon-ml-challenge-2026/blob/main/Documentation_template.md), [README](https://github.com/Siva402-ai/amazon-ml-challenge-2026)
  - **Yashika2211 (no LB given):** a "pseudo stage" that mines aliases and noise tokens from confident French test pairs and re-runs inference. Before it, France had about twice as many uncertain candidates. — [Documentation_template.md](https://github.com/Yashika2211/amazon-ml-challenge-2026/blob/main/Documentation_template.md)
  - **INFINITY-ISHANT (LB 0.977):** target encodings also learn from confident test pseudo-labels. Test average matches per S1: France 3.29, US 3.34, India 3.25. — [README](https://github.com/INFINITY-ISHANT/byte-builders-entity-resolution), [submissions_log.md](https://github.com/INFINITY-ISHANT/byte-builders-entity-resolution/blob/main/submissions_log.md)
- **Other blanking/implied-France numbers:**
  - rishang19dx (LB 0.933414): France blank gives 0.809967, so US/India ≈ 0.944 and France ≈ 0.874. — [README](https://github.com/rishang19dx/AMZN_MLC2026)
  - Abhigyanv23 (best LB 0.9683): implied France ~0.87 → ~0.91 → ~0.91 → ~0.93 across v001–v005, assuming a test mix of 47/38/15%. — [Documentation_template.md](https://github.com/Abhigyanv23/Amazon-ML-Challenge-2026/blob/main/Documentation_template.md)

### Inferences
- **The French word-swap sibling is the single best-evidenced France lever, and it is removal-only.**
  - GozonLegendent's +0.001125 LB, with France at ~15% of the public-LB weight, implies about **+0.0075 France F0.5 from one rule**.
  - Epic021 found the same class from the other direction: France ≈ 0.31 model probability vs a 70% truth rate in India/US, and 4.3 such pairs per French S1.
  - A qualifier vocabulary learned from our own French predictions is a cheap, test-internal and label-free step. Its gain can only be measured by an LB probe or by our France proxies.
  - Under GozonLegendent's asymmetry arithmetic, removal pays if ≥ ~26% of the removed links are wrong.
- Our last logged upload is 0.981022 (local git commit `fd59533` in this repo: "Log upload: 0.981022 (K US/India + routes, France 3-family vote)"), and our France probe was ≈0.937 (`research_notes/Entity resolution missing big wins/competition_public_intel.md`). That suggests French over-acceptance may remain in our 3-family vote. This is an inference and needs a probe.
- Teams that lowered or loosened thresholds for France, or added hand rules, lost LB. Teams that **removed** a specific French decoy pattern gained. This matches NFG's earlier "stricter is better" France result.
- Self-training that changes **normalisation dictionaries**, not labels, is the only positive self-training report (GarvRajput17, +0.028 est. France). The "estimated" France comes from their LB arithmetic, not a direct measurement.

### Gaps
- No team reports a direct France F0.5 measurement; all France numbers are implied from blanking arithmetic under an assumed country weight.
- `fr_swap_ext` and Epic021's France-only threshold probes had no LB result yet.
- GarvRajput17's France gain is bundled with other changes in one 0.978 → 0.983 step. The France share (~0.917 → ~0.945) is their estimate.

---

## 3. India/US and decision-rule findings that did (or did not) transfer to the LB

### Takeaway
New LB-validated lessons:
1. **Cluster/density-count features break on test** (GozonLegendent 0.969 → 0.986306 after removing them and raising the threshold).
2. Only **precision gains reach the leaderboard** (Tony-AJ; INFINITY-ISHANT's stricter threshold gave +0.005 LB despite −0.003 validation).
3. **Name-uniqueness and fuzzy house-number features** carried over (rikkicoder 0.9793 → 0.9811).
4. About 75% of the remaining validation loss is **address-less records with shared names**, which several teams now call irreducible.

On the India side specifically, one team found non-Devanagari Indic names being erased by normalisation.

### Cited Findings
- **Why cluster features failed on test (GozonLegendent):**
  - v2 kept **9.1% of test decoy-signature links vs 0.07% on validation**.
  - On test, sibling decoys come as several records sharing the moved house number (mean 0.90 vs 0.19 on validation).
  - Removing four cluster-count features (`cl_same_qnum, cl_same_snum, cl_n, pr_s_n`) cut test decoy acceptance to 0.98%.
  - — [Documentation_template.md §5](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md)
- **GozonLegendent sibling-decoy statistics (training labels):**

  | Signal | Decoys | True pairs |
  |---|---|---|
  | First house number equal | 0.9% (US) | 77% |
  | Number moved to a nearby value | ~90% | 0.5–1.7% |
  | "Group"/"Holdings" added to the name | ~18% | 0.00% |

  - 57.6% of unmatched US S2 records are sibling decoys (India 36.1%).
  - India injects a first number ("H.NO 53") in 14% of true pairs.
  - Learned "different business" words: holdings, group, pvt, public, east, west, north, south, metro, lakeside.
  - — [Documentation_template.md §2.1, §4](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md)
- **Loss budget:**
  - GozonLegendent (v5 at t = 0.75, total loss 0.0080): about 75% comes from the 4.4% of true pairs with no address. When the name is shared by ≥2 entities (22 on average), following another source's identical-name record "picks the right entity 19.6% of the time vs 21.9% at random" (119,814 records). Records with an address lose 0.0021 in total.
  - Epic021 agrees: 97.7% of empty-address records are matched; accuracy is 0.978 when the name is unique vs 0.407 when 2–3 S1 share it; "calibrated coin flips, irreducible".
  - — [GozonLegendent Documentation_template.md §5](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md), [Epic021 STRATEGY.md](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md)
- **rikkicoder v4, public 0.9811 (from 0.9793).**
  - Commit 567b34b477 (00:34 UTC / 06:04 IST) adds:
    - counts of visible S1/pool records sharing the core name;
    - an exact-name-unique-in-S1 flag;
    - digit edit distance, containment and relative difference between house numbers;
    - a stage-2 count of high-probability same-core-name siblings.
  - Reason given: "66% of rejected true matches are pool copies with an empty address, and house-number typos (6420 vs 642, 104 vs 04) break the exact-number features".
  - Held-out: 0.9826 → 0.9852.
  - A per-S1 "has-a-match" gating model (AUC 0.9993) added nothing: gated 0.9826 = top1_plus 0.9826.
  - — [README](https://github.com/rikkicoder/amazonMLchlng), [commit 567b34b477](https://github.com/rikkicoder/amazonMLchlng/commit/567b34b477), [commit ae8d3ac0d3](https://github.com/rikkicoder/amazonMLchlng/commit/ae8d3ac0d3)
- **Precision-only transfer:**
  - Tony-AJ: "Public stayed at 0.966 from v107 to v122 while the mock rose: only precision gains reached the leaderboard". Earlier, a "leaderboard-calibrated, test-shaped mock fold took public F0.5 from 0.954 to 0.966". — [commit 4408d7edaa](https://github.com/Tony-AJ/business_entity_resolution/commit/4408d7edaa), [docs/Documentation_template.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/docs/Documentation_template.md)
  - INFINITY-ISHANT probe: plain threshold t = 0.9 gave "Val −0.003 but LB +0.005 → test has many more near-copy distractors than train". — [submissions_log.md](https://github.com/INFINITY-ISHANT/byte-builders-entity-resolution/blob/main/submissions_log.md)
- **Validation scale matters:**
  - rishang19dx: a cross-fit score of 0.986 on a ~9× smaller universe "did not carry over to the leaderboard (US/India ≈ 0.944)". Candidates there were less alike than on test: median address TF-IDF cosine 0.32 vs 0.46, token-set ratio 0.57 vs 0.72. The team built a full-scale `scale_val`. — [README](https://github.com/rishang19dx/AMZN_MLC2026)
  - Epic021: "Held-out gains have matched leaderboard gains except France". The same line is in [rajesh08114 HANDOFF.md](https://github.com/rajesh08114/business_entity_resolution/blob/master/HANDOFF.md).
- **India normalisation bug found by TarunSitaraman** (commit at 05:51 UTC / 11:21 IST): "~10% of India S2 names (Telugu, Kannada, Tamil, Bengali, Gujarati, Malayalam, Oriya) normalised to an empty string; only Devanagari was transliterated". The fix shifts these ISCII-layout scripts onto Devanagari at a fixed offset. — [repo commits](https://github.com/TarunSitaraman/amazon-ml-challenge/commits/main)
- **Negative results GozonLegendent says not to repeat:**
  - Choosing thresholds by matching link-count histograms "always prefers the lowest threshold".
  - Rule-matched "silver" test pairs "agree ~1.0 in every band even where true precision is 0.77", so they cannot judge a probability band.
  - — [TEAM_HANDOFF.md §5](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)

### Inferences
- Any feature computed from candidate-group density (cluster counts, sibling counts, competition counts) should be checked for **test vs validation decoy acceptance**, as GozonLegendent did, before trusting validation. Three teams saw such features raise validation but fail on the LB: GozonLegendent, and harshgitty58 and mayankgoplani431 from the prior notes.
- The implied-precision-per-band diagnostic is label-free for US/India and gave a large, LB-confirmed threshold shift (0.36 → 0.75). If our current threshold was tuned only on TUNE, the same check on our test probabilities could be a cheap test.
- It is worth checking whether our India normaliser handles all nine Indic scripts. A silent empty-name bug on ~10% of India S2 records would be a recall loss that validation shares with test.

### Gaps
- None of these numbers can be checked against our data without our own run. GozonLegendent's implied-precision figures depend on its own model's calibration.

---

## 4. Score progressions from commit/README histories (what change produced each jump)

### Takeaway
The largest documented single LB jump today is **+0.0173 (0.969 → 0.986306)**. It came not from a new model but from **removing test-fragile cluster-count features and raising the threshold from the validation-optimal 0.36 to 0.75** after label-free test diagnostics. Other documented jumps:
- blending or stacking two teams' models: +0.004 to +0.009;
- dense e5 blocking plus a cross-encoder: +0.010;
- French self-training plus house-number features: +0.005.

### Cited Findings
- **GozonLegendent** (LB, [Documentation_template.md §5](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md)):

  | Version | Change | LB |
  |---|---|---|
  | dense baseline | bi-encoder top-1 | 0.753 |
  | v0 | + LightGBM pruner | 0.972 |
  | v2 | + cross-encoder, decoy + cluster features | 0.969 (val 0.99264; cluster features "broke on test") |
  | v3 | − cluster-count features, t = 0.75 | **0.986306** |
  | v5pf | + e5-base cross-encoder + post-rules | 0.986451 |
  | v6nc | + bge-reranker-v2-m3, post-rules **without** `fr_swap` | 0.986201 |
  | v6pf | + `fr_swap` | **0.987326** |

  The team's own summary: "Validation gains shrink on test. +0.0005 on validation became +0.0001–0.0009 on the board." — [TEAM_HANDOFF.md §4](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)
- **GarvRajput17:** "Progression: 0.968 → 0.978 (blocking + cross-encoder) → 0.983 (+ adaptive cleaning, self-training, house-number features) leaderboard F0.5". The handoff gives 0.983362. — [commit 67ffd09e37](https://github.com/GarvRajput17/MLChallenge/commit/67ffd09e37), [commit 7678c5d690](https://github.com/GarvRajput17/MLChallenge/commit/7678c5d690)
- **Epic021:** B1 0.952 → B2 0.970 → B3 0.970 → BLEND-B4 **0.979** → STACK-B5 **0.983** (teammate alone 0.975).
  - The 0.50/0.50 isotonic blend with the teammate gave holdout +0.0029.
  - The learned stage-2 stacker ("when to trust which model") gave another +0.0031 holdout, +0.004 LB and implied France 0.94 → 0.950.
  - — [STRATEGY.md §4](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md)
- **rikkicoder:** 0.9744 → 0.9793 (training density matched to test) → **0.9811** (v4 name-uniqueness and fuzzy house-number features). — [README](https://github.com/rikkicoder/amazonMLchlng)
- **rajesh08114:** v4 0.974 → v5 0.9754 → (+ French hand rules 0.9738, reverted) → v5 + neural v1 **0.9775**. — [HANDOFF.md](https://github.com/rajesh08114/business_entity_resolution/blob/master/HANDOFF.md)
- **INFINITY-ISHANT:** 0.955 (baseline) → … → **0.977** (two-stage XGBoost, cluster and sibling features, cross-encoder features, one-owner decoding, t = 0.9). — [README](https://github.com/INFINITY-ISHANT/byte-builders-entity-resolution)
- **Tony-AJ:** 0.954 → 0.966 (test-shaped mock fold), then flat at 0.966 through v122. — [docs/Documentation_template.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/docs/Documentation_template.md)
- **Other LBs posted today:**
  - Siva402-ai 0.970441 ([README](https://github.com/Siva402-ai/amazon-ml-challenge-2026));
  - ayushmanjha52 ~0.970 → 0.9709 ([EXPERIMENT_LOG.md](https://github.com/ayushmanjha52/business-entity-resolution/blob/main/docs/EXPERIMENT_LOG.md));
  - Abhigyanv23 best 0.9683 ([Documentation_template.md](https://github.com/Abhigyanv23/Amazon-ML-Challenge-2026/blob/main/Documentation_template.md));
  - rishang19dx 0.933414 ([README](https://github.com/rishang19dx/AMZN_MLC2026)).

### Inferences
- The pattern across teams: late-stage gains are **decision-layer and removal rules**, and **model blending/stacking**. Stacking two independently built models gave Epic021 its largest LB gain.
- In GozonLegendent's ladder, the 0.969 → 0.986 recovery is almost entirely the threshold and feature-drop change. That means a strong model with a validation-tuned threshold may sit ~0.01–0.017 below its LB potential if the test density shift is not handled.

### Gaps
- None of the ≥0.990 teams published a progression. No commit found shows a jump from 0.98x to 0.99x.

---

## 5. Organiser rules: test-data use, pseudo-labels, LLM size, upload limits, deadline

### Takeaway
**No primary organiser statement dated 26–27 Sep was found.** The Unstop FAQ/discussion page shows "Updated On: 25 Sep 26, 09:07 AM CUT", and its content could not be crawled. All rule statements below are **teams' readings**, and several conflict:
- submission limits: 5/day and 15 total vs "unlimited";
- LB display precision;
- whether candidate-set size affects ranking;
- the deadline time;
- whether the public LB counts in the final evaluation.

Several teams, including the top public one, openly use **unlabelled test inputs** (statistics, label-free selection, removal rules learned from their own test predictions, and in some cases pseudo-labels). One team explicitly avoids pseudo-labels pending written confirmation.

### Cited Findings
- **Upload limits (conflicting):**
  - "At most 5 submissions per day for the 3 days (15 in total). The submit button is disabled afterwards." — [Tony-AJ docs/GUIDELINES.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/docs/GUIDELINES.md)
  - "Submissions: 5 per day, 15 in total", citing "Guidelines.pdf". — [Mveen3 docs/Pipeline_Architecture.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Pipeline_Architecture.md)
  - "The team has **one** leaderboard attempt left" (~11:45 IST, day 3), consistent with a cap; also "the board keeps the max". — [GozonLegendent TEAM_HANDOFF.md](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)
  - **Contradicted by** "Submitted anyway (unlimited submissions)". — [Epic021 STRATEGY.md](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md)
  - The prior notes also recorded "uploads are unlimited" (harshgitty58).
- **LB precision:**
  - "The LB shows 3 decimals, so a change needs ≈ +0.001 to be visible" — [Epic021 STRATEGY.md](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md).
  - Several teams nevertheless quote 6-decimal public scores: GozonLegendent 0.987326, GarvRajput17 0.983362, Mveen3 0.984819, Siva402-ai 0.970441. The submission page may show more digits than the board. Unresolved.
- **Deadline:**
  - "Deadline 23:59 IST today" — [GozonLegendent TEAM_HANDOFF.md](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md)
  - Window "25 Sep 00:00 IST → 27 Sep 23:59 IST (Guidelines.pdf)" — [Mveen3 docs/Pipeline_Architecture.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Pipeline_Architecture.md)
  - The prior notes cited an Internshala listing ending at 21:00 IST. Conflict.
- **Public vs private LB:**
  - "Two leaderboards, public and private. Evaluation and shortlisting use performance on both … shortlisting is based on the submitted [version history]" — [Tony-AJ docs/GUIDELINES.md](https://github.com/Tony-AJ/business_entity_resolution/blob/main/docs/GUIDELINES.md)
  - The Unstop page says the private LB is "based on your final solution submission and validation on the complete test dataset", and the top 10 are invited to the finale "based on the leaderboard results and the solution presented in the document". — [Unstop](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)
- **`candidate_pairs.tsv` size (unverified, repeated claim):**
  - "The organisers announced that `candidate_pairs.tsv` is reviewed and that a **smaller candidate set per Source-1 entity ranks higher** beyond the leaderboard score" — [SmithaSimon HANDOFF.md](https://github.com/SmithaSimon/AmazonML/blob/main/HANDOFF.md)
  - "at a given score, fewer candidates per S1 ranks higher in the final evaluation" — [itsmeaks200 docs/HANDOFF_PROMPT.md](https://github.com/itsmeaks200/er-lab/blob/main/docs/HANDOFF_PROMPT.md)
  - The official README mirror says the file is "**not scored on the leaderboard**; we use it to analyse blocking quality" — [ritzy15 HF mirror card](https://huggingface.co/datasets/ritzy15/business-entity-resolution).
  - GozonLegendent's handoff rule: "`candidate_pairs.tsv` must be exactly the set the final model scores; matches ⊆ candidates". The team shrank candidates to ~1.08 per record and added a "shrink" commit at 06:32 UTC. — [TEAM_HANDOFF.md](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md), [commit e5645855d2](https://github.com/GozonLegendent/amazon-ml/commit/e5645855d2)
- **Test-data use and pseudo-labels (no organiser ruling found):**
  - **GozonLegendent's disclosed practice.** Its rule reads "No test labels exist; nothing is fitted on test. Unlabeled test statistics are allowed and disclosed". It discloses:
    - test IDF and locality/name frequencies;
    - threshold and feature-drop choices made from test score distributions;
    - `fr_swap`'s vocabulary learned "from the French links of the test prediction file";
    - "Public-leaderboard scores were used to compare submitted versions … including `fr_swap` on/off".
    - — [Documentation_template.md, Appendix A Compliance](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md)
  - **Mveen3's caution:** "Pseudo-labelling the unlabelled French test records — Not used unless organisers confirm in writing — A grey area under 'only the provided training data'"; "We ask on day 1". No answer is recorded. — [docs/Pipeline_Architecture.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Pipeline_Architecture.md)
  - **Epic021:** "The organizers **re-run the code**, so nothing may be tuned on this test file or on the LB." That is its own reading, yet it ran France LB probes as "diagnostic only". — [STRATEGY.md](https://github.com/Epic021/amazon-ml-challenge-2026/blob/HEAD/STRATEGY.md)
  - **Teams using test pseudo-labels or self-training openly:** GarvRajput17 ([commit](https://github.com/GarvRajput17/MLChallenge/commit/67ffd09e37)), Siva402-ai, INFINITY-ISHANT, Yashika2211, ayushmanjha52 (links in §2).
- **LLM size and licence:**
  - Teams restate MIT/Apache-2.0 and ≤8B parameters.
  - Mveen3 adopted a "Strict 8B reading; drop the Qwen-7B LoRA … A disqualification risk is not worth it. Budget ≈1B." That treats the cap conservatively, apparently as a total across models. — [docs/Pipeline_Architecture.md](https://github.com/Mveen3/amazon-ml-challange/blob/main/docs/Pipeline_Architecture.md)
  - GozonLegendent lists each model "well under the 8B limit" (118M, 278M, 568M), i.e. a per-model reading. — [Documentation_template.md](https://github.com/GozonLegendent/amazon-ml/blob/main/Documentation_template.md)
- **LLM use in public repos:**
  - None has an LB result. Namballa-Srivansh committed "LLM jury integration" (05:23 UTC) with no description or score. — [commits](https://github.com/Namballa-Srivansh/Amazon-ML-Challenge/commits)
  - itsmeaks200 only plans "an LLM judge (Qwen2.5, Apache, ≤8B, LoRA) on the uncertain pairs". — [docs/RESEARCH_SYNTHESIS.md](https://github.com/itsmeaks200/er-lab/blob/main/docs/RESEARCH_SYNTHESIS.md)
- **Validator:** "organisers' validator arrived as a 0-byte file" for one team, who re-implemented it. — [TarunSitaraman EXPERIMENTS.md](https://github.com/TarunSitaraman/amazon-ml-challenge/blob/main/EXPERIMENTS.md)

### Inferences
- The candidate-set-size claim has now appeared in three independent repos (HarpalKalankar in the prior notes, SmithaSimon, itsmeaks200). The official README text says the file is not scored. This is **unverified**. A cheap hedge that costs no F0.5 is to keep `candidate_pairs.tsv` exactly equal to the final scored set and prune it where recall is unaffected, as GozonLegendent does at ~1.08 per record.
- The top public team discloses LB-based and test-internal selection of removal rules. If reproducibility audits accept that, test-internal French vocabulary rules are a precedent, not a rules breach. This is my inference, not an organiser ruling.
- If the 5/day, 15-total cap is real, every remaining upload should be a planned probe. "The board keeps the max" (GozonLegendent) means a failed probe does not lower the public score.

### Gaps
- No primary organiser text (Unstop FAQ answers, Discord, email or Guidelines.pdf) could be retrieved. The limits, deadline, LB precision, candidate-size weighting and pseudo-label permissibility are all second-hand.

---

## 6. Updates on previously known leads, Hugging Face, and social channels

### Takeaway
Among the previously known leads, only **rikkicoder** (LB 0.9793 → **0.9811**) and **abbinavv** (overnight commits, no new LB) changed materially. nithinfilo, harshgitty58, NFG, mayankgoplani431 and HarpalKalankar had no new commits after 26 Sep 18:00 UTC, and the HF leads are unchanged. **No relevant LinkedIn, X, Reddit, Medium, YouTube or Discord posts from 26–27 Sep were found.**

### Cited Findings
- **nithinfilo** ([repo](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution)), **harshgitty58** ([repo](https://github.com/harshgitty58/Amazon_ML_Challenge)), **mayankgoplani431** ([repo](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution)) and **HarpalKalankar** ([repo](https://github.com/HarpalKalankar/business-entity-resolution)): 0 commits since 26 Sep 18:00 UTC (GitHub API, checked 27 Sep ~12:10 IST).
- **NFG / aditya-bheke:** one commit at 26 Sep 18:23 UTC ("Business entity resolution pipeline with docs, tests and CI"). The README still reports "public leaderboard 0.9666 → **0.9810**". — [repo](https://github.com/aditya-bheke/business-entity-resolution)
- **rikkicoder:** v4 LB **0.9811**, plus a has-a-match gate that brought no gain (see §3). — [README](https://github.com/rikkicoder/amazonMLchlng)
- **abbinavv:** 16 commits between 26 Sep 20:18 and 23:53 UTC, including:
  - "Log sub22 leaderboard score 0.963841";
  - a prior-corrected second stage for US/India;
  - "prior_stack --apply-to: score an unlabelled country with the model (France as US)";
  - cohesion features;
  - rival/list-context features.

  Scores given for sub28–sub33 are **predicted** (0.968–0.9702), not LB. — [commits](https://github.com/abbinavv/ml_challenge_2026/commits)
- **Hugging Face (card metadata only):**
  - [akshatbakshi](https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026): lastModified 2026-09-24T19:01Z, gated.
  - [vc940](https://huggingface.co/datasets/vc940/business-entity-resolution-normalized): 2026-09-24T23:00Z.
  - [rishavk77](https://huggingface.co/datasets/rishavk77/amc26-entity-resolution-embeddings): 2026-09-25T23:33Z.

  None has changed since the prior notes. New challenge-named HF repos from 26–27 Sep have **empty cards**, so nothing about methods can be read from them:
  - vaibhav343/Amazon_ML_Challenge_2026;
  - rayaanoidpr/amlc-2026-artifacts;
  - Oddproblem/amazon-ml-pipeline;
  - Krut27/amazon-ml-2026-candidates;
  - Ujjansh/ber-er-encrypted-backups;
  - [guldasta/france-test-data](https://huggingface.co/datasets/guldasta/france-test-data), created 27 Sep 05:11 UTC. Not opened, per the constraints.
- **Social and web:**
  - Exa date-filtered search (from 26 Sep) returned no challenge posts: 23 + 20 results, all unrelated.
  - Parallel/Web search found only GitHub repos, a Kaggle dataset mirror ([satwiksps](https://www.kaggle.com/datasets/satwiksps/amazon-ml-challenge-2026)) and a basic Kaggle notebook ([mobeenfatimah](https://www.kaggle.com/code/mobeenfatimah/amazon-ml-challenge-2026-entity-resolution), 25 Sep).
  - Unstop FAQ shows "Updated On: 25 Sep 26, 09:07 AM CUT" with 89,403 registered. — [Unstop](https://unstop.com/hackathons/crp-amazon-ml-challenge-2026-amazon-1743604)
- **Other new repos that were checked and found low value:**
  - [vigneshkomma](https://github.com/vigneshkomma/amazon-ml-challenge): a basic LightGBM template, no LB.
  - [Akash-bardia](https://github.com/Akash-bardia/amazon-ml-challenge-2026): val 0.9761, last push 26 Sep.
  - [AvinashMalladi](https://github.com/AvinashMalladi/amazon-ml-challenge-2026-entity-resolution): "F_0.5 > 0.98" in the description only, last push 01:57 UTC.
  - md-yahya1: pairwise F1 only.

### Inferences
- Public write-ups keep arriving near the deadline, mostly as final "Documentation_template.md" fills. A second sweep in the last hours (or after 2 Oct results) may surface more ≥0.985 teams. The top ≥0.990 teams have shown no sign of publishing before results.

### Gaps
- The sweep covers pushes up to ~06:55 UTC (12:25 IST) on 27 Sep. Later pushes, private repos and deleted repos are not covered; `Amazon-ML-Challenge-2026/business-entity-resolution` now returns 404.
- LinkedIn and X content is poorly indexed by the available search tools for same-day posts, so absence there is weak evidence.
