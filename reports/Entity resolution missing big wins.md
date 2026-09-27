# Fix France's decision layer before anything else

The biggest win we are probably missing is in France, and it is about how French links are decided rather than a new scorer. The 0.990 teams almost certainly score about **0.97 or better in France**. Our figure is about 0.937. Their French score is roughly equal to their US/India score, which points to one language-agnostic, claimant-aware pipeline run the same way in every country. Our French rows are still chosen by a transformer blend with a threshold and one owner. That is essentially the US/India setup from before the rival/rank tree, and adding that tree gave US/India about +0.006. The second gap is retrieval: about half of what US/India still loses comes from true records that are never retrieved. Reverse search, reverse-rank features and a cross-encoder check on newly retrieved pairs are what public teams measured as their largest remaining gains. Every other lever in the notes is worth tenths of a point or less, or has already failed here or for public teams (self-training, adversarial domain adaptation, global assignment solvers, graph post-processing).

Realistic public target today: **0.982–0.986**. First place would need France at about 0.97 or better *and* US/India at about 0.990 or better, and no measured lever reaches both. Claims are tagged throughout:

| Tag | Meaning |
|---|---|
| **[M-ours]** | measured by us |
| **[M-public]** | measured and published by another team in this competition |
| **[R]** | from research or other competitions |
| **[I]** | inferred; not measured |

## Leaders must score about 0.97 or better in France

Blanking the French rows splits the public score into its two parts. For three teams, blanking France gives a French weight of about 15%:

| Team | Public score | Score with France blanked | Source |
|---|---|---|---|
| Us | 0.978287 | 0.846352 | [probe brief](../docs/briefs/PROBES_FRANCE_20260926.md) |
| NFG | 0.9810 | 0.8459 | [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md) |
| harshgitty58 | 0.969 | 0.836 | [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md) |

[M-public, M-ours]

Our decode gives **US/India 0.98559 against arena B_REST 0.985511**, a difference under 0.0001. That agreement is the main reason to trust the formula below. It still assumes France has the train no-match rate of 5.58%.

**public = 0.850248 · US/India + 0.149752 · France**

The leaders' French score follows from this formula:

- The public top three on 26 Sep were **0.9906 / 0.9894 / 0.9888** ([harshgitty58](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md)) [M-public].
- A reviewer measured the US/India ceiling at **about 0.995**: a system that is perfect except for the shared-name empty-address copies ([plan, §A](../docs/briefs/PLAN_TO_099_20260927.md)) [C5-M].
- Even at that ceiling, a 0.9906 team needs France ≥ 0.966, and a 0.9915 team needs France ≥ 0.972 [I].
- Neither NFG (France ≈ 0.948) nor we (≈ 0.937) are close.

A search of GitHub, Hugging Face and social posts found nothing published by any team at 0.990 or above, so how they got there is unknown. The ~0.9915 leader figure and "~21 teams ≥ 0.990" in our brief are unconfirmed by any public source.

| Scenario | US/India | France | Public (formula) |
|---|---|---|---|
| Today's measured public | 0.9856 | 0.937 | 0.9783 [M-ours] |
| Current US/India model, France unchanged | 0.9869 | 0.937 | ≈0.9794 [I] |
| France at NFG level | 0.9869 | 0.948 | ≈0.9811 [I] |
| France 0.97, US/India +0.0006 | 0.9875 | 0.970 | ≈0.9849 [I] |
| Minimum pair for 0.9906 | 0.990 | 0.994 | 0.9906 [I] |
| Leader-like pair | 0.993 | 0.983 | 0.9915 [I] |

A perfect France with our current US/India gives 0.9889. US/India at its ceiling with our current France gives 0.9863. So neither half alone reaches first place.

In expected public points, France is the larger lever. Raising France from 0.937 to 0.97 is worth +0.0049. Raising US/India from 0.9869 to 0.990 is worth +0.0026, and that US/India headroom is shrinking against a hard ceiling [I].

## France is decided by the model least likely to transfer

The measured French symptoms all point at the cross-encoder, not at the candidates or the threshold [M-ours]:

- **The cross-encoder misfires on legal forms.** C2 scores "Club du Cours" vs "Club du Cours SARL" at about 0.0003. This class is **99.85–99.99% true** in US/India.
- **Tree-yes / C2-no pairs are 28× more common in France**: 0.545 per S1, against 0.016–0.026 in US/India.
- **Moving the French threshold will not help.** Probe A dropped 43,517 uncertain French links and scored 0.97826 (Δ −0.000027). That puts the band at about **70% true**, which is almost exactly break-even. Probe C′ puts confident non-legal-form links at 92–96% true.
- NFG found a stricter French threshold helped them, but their scorer is different. For us, cutting at a different place is neutral; the gain has to come from ordering the pairs better.

Sources: [probe brief](../docs/briefs/PROBES_FRANCE_20260926.md), [plan §2](../docs/briefs/PLAN_TO_099_20260927.md).

The literature explains why the transformer is the weak link:

- Fine-tuned PLM matchers moved to unseen entities lose **22–61 F1 (RoBERTa) and 36–56 F1 (Ditto)** ([Peeters, Steiner & Bizer](https://arxiv.org/html/2310.11244v4)) [R].
- Fine-tuning small LLMs "hurt[s] cross-domain transfer" ([Steiner et al.](https://arxiv.org/html/2409.08185)) [R]. Our Qwen3-Reranker-0.6B is in that class, so its French behaviour will likely resemble a fine-tuned PLM rather than a robust zero-shot LLM [I].

Our language-agnostic string GBDT has one asset nothing else has: **its mean scores match both probe truth rates**, about 0.67 against 64–69% true and 0.93 against 92–96% true [M-ours]. The caveats:

- Two bags and two parameters leave zero degrees of freedom.
- The probe truth rates are themselves estimates from our pricing model.
- Calibration is necessary, not sufficient: a model can be calibrated and still rank pairs poorly.

**For combining the three French models, simple aggregation is the evidence-backed default.** The WRENCH benchmark and its 2024 follow-up found majority vote "one of the most consistent methods" ([WRENCH](https://arxiv.org/abs/2109.11377); [NeurIPS 2024 D&B](https://arxiv.org/pdf/2501.07727)) [R]. Label models such as Dawid–Skene assume the sources are conditionally independent. Our Frenchified cross-encoder and our Qwen variant were trained on the same US/India pairs, so they break that assumption and would double-count their shared errors [I].

The recommended rule [I]:

- **Anchor on the GBDT.** Recalibrate the cross-encoder and Qwen on France with a two-parameter Platt fit to the two probe bags.
- **Rescue:** a confident GBDT positive overrides a cross-encoder negative when the name core and address agree. This targets the legal-form misfire.
- **Veto:** transformers may veto only where they are confident and the GBDT is unsure.
- **Fallback:** majority vote over the three families. `k_france_vote.py` already implements most of this.
- **Falsifier:** sum the calibrated GBDT probabilities over French candidates to estimate the expected French link count, and reject any file whose mean list length contradicts it. The generator's 3.46 links per S1 and 5.58% no-match rate are country-independent in train [M-ours].

Two methods should be dropped explicitly:

- **Self-training on France.** NFG's US→India simulation lost 0.012 **despite 99.8%-accurate pseudo-labels** ([NFG](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)) [M-public]. Our strict-pseudo-label fine-tune shrank French lists to 3.19 per S1 [M-ours].
- **DADER-style feature alignment.** An independent re-run found MMD alignment can *lose* 16–25 F1 on some source→target pairs ([MFSN re-run](https://openreview.net/pdf?id=1GVyE9J021)). The published pipeline also uses target validation labels, which we do not have ([dader on PyPI](https://pypi.org/project/dader/)) [R].

## The larger French win is a country-free claimant model

In US/India, rival and rank features over each record's claimant set moved us from the C2 blend to the pair tree, **about +0.006** [M-ours, [probe brief §6](../docs/briefs/PROBES_FRANCE_20260926.md)]. France never received that layer. The likely missing piece is the same record-centric GBDT we use for US/India, **restricted to features that do not depend on language**:

- string-GBDT probability;
- K3 edit-operation features: legal-form relation, closed decoy-word list, δ ∈ {1,3,5,7,9,11,13,21}, initials/acronyms;
- claimant-set counts, ranks and margins.

It would be fit on US/India and applied to France with no C2 input. The rationale [I]:

- The generator's statistics are the same in every country [M-ours]. The generator notes argue its operators come from finite tables that a model can learn once and reuse.
- Every such operator (legal forms, the δ set, decoy words, templated names) is language-independent once text is folded. A model that sees *which operation* explains a difference, rather than a transformer's reading of French words, should transfer.
- This matches the MARLIN/Fellegi–Sunter argument that generator-style noise is best modelled per operation ([Bilenko & Mooney](https://www.cs.utexas.edu/~ml/papers/marlin-kdd-03.pdf); [Splink TF guide](https://moj-analytical-services.github.io/splink/topic_guides/comparisons/term-frequency.html)) [R].
- It matches NFG's pattern: they reached France ≈ 0.948 with a language-agnostic LightGBM, a stacked multilingual-e5 cross-encoder, one-owner decoding and an acronym rule that was right on 434/435 held-out pairs ([NFG README](https://github.com/aditya-bheke/business-entity-resolution)) [M-public].

**The risk is transfer.** In our earlier hard-pool runs, cross-country transfer was asymmetric: India→US 0.932, US→India 0.826 [M-ours, [decisions](../learnings/EXPERIMENT_DECISIONS.md)]. rikkicoder, by contrast, found the source-country threshold near-optimal in the target country, 0.9793 vs 0.9794 ([rikkicoder](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md)) [M-public].

**Gate for this model:**

- a leave-one-country-out score (US→India and India→US), which `k3_model.py` already supports;
- the probe-bag calibration check;
- a France-only upload whose US/India rows are byte-identical, so the change in France reads directly from the public score.

## Half of the US/India loss is never retrieved

On B_REST, **37,810 true links are never retrieved**. That is about 50% of what remains, ≈0.0072 of US/India macro F0.5. The rest is 37% retrieved-but-rejected (80% empty-address) and 13% false links [M-ours, [plan §2](../docs/briefs/PLAN_TO_099_20260927.md)]. Our new address and fuzzy-name routes raise the best-possible B_REST score from **0.9946 to 0.99675** [M-ours]. Public teams report oracles of 0.9947–0.9987.

Public teams converge on three retrieval-side results:

- **Reverse search** (each record retrieves its top-3 S1s). It lifted recall 97.0% → 99.2% for nithinfilo on a 14-state sample ([HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md)) and 96.4% → 97.5% for NFG, where the reverse rank became "the single strongest feature" ([NFG](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)) [M-public].
- **Reverse search holds up in a denser pool.** Forward-only top-10 loses 0.011 recall per 10× pool growth, against 0.004 with reverse-3 added ([rikkicoder](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md)) [M-public]. The test pool is 23% denser than train ([harshgitty58](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md)).
- **Newly retrieved pairs must be verified by a cross-encoder before they are accepted.** NFG's dense rescue kept 97% precision under a LightGBM + cross-encoder check and added +0.0010–0.0014 on validation ([NFG](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)) [M-public].

**The gap in our pipeline is the third point.** Our route candidates are scored without the lost cross-encoder, so they reach the GBDT with blank CE columns. Scoring them with Qwen-0.6B (or the Frenchified cross-encoder) takes only GPU time and restores the pattern that transferred for NFG.

Reverse-rank and reverse-margin features belong in the record-centric model as soon as `reverse_retrieval.py` finishes. They must be computed against the **full competing S1 pool**, not only the claimants already retrieved. rikkicoder attributes 39–50% of GBM gain to `rev_rank` [M-public]. The earlier C2-scored route estimate was about +0.0004 on B_REST [I, [probe brief](../docs/briefs/PROBES_FRANCE_20260926.md)].

Two retrieval ideas are lower value today [I]:

- **A contrastively fine-tuned e5-small retriever.** It reached R@10 0.9965 on a 5% sample ([rikkicoder](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md)) [M-public], but it only pays if our oracle after reverse search stays below about 0.995, and it would force a feature rebuild.
- **The listwise cross-encoder.** Listwise "select" beats pairwise by about 16–17.6% F1, but only against *zero-shot* pairwise LLMs ([ComEM](https://arxiv.org/html/2405.16884v3)). No competition winner used a listwise transformer as the final matcher. They fed listwise information to a GBDT, as our claimant-set features already do ([last-mile notes, Foursquare 1st](https://www.kaggle.com/competitions/foursquare-location-matching/writeups/re-waiwai-1st-place-solution)) [R].

## Levers ranked by expected gain per hour

Timing is from ~08:50 IST, with upload count and the 19:00 freeze as constraints. All gain figures are [I] estimates unless marked. The ranking weighs expected public change against hours and the risk that a validation gain does not transfer.

| Rank | Lever | Window | Effort | Expected public Δ | Basis |
|---|---|---|---|---|---|
| 1 | **French rows from a GBDT-anchored cross-family vote**: probe-bag Platt calibration, GBDT rescue of CE legal-form negatives, CE/Qwen veto only when confident, K3 veto limited to δ-set + legal-swap | before 10:00 | ~1 h CPU on existing scores, 1 France-only upload | +0.0005 to +0.002 | GBDT calibration [M-ours]; NFG France 0.948 [M-public]; transfer literature [R] |
| 2 | **Per-(S1, source) caps at the TRAIN maxima**, as a precision guard | before 10:00 | 15–30 min | ≤ +0.0001 | NFG removed ~670 impossible matches, one S1 with 146 [M-public] |
| 3 | **Country-free record-centric model on France** (string GBDT + K3 + claimant-set features, no C2), gated by leave-one-country-out and probe calibration | before 15:00 | 2–3 h CPU, 1 upload | +0.001 to +0.004 (high variance) | US/India rival tree +0.006 [M-ours]; leaders' France ≥ 0.97 [I] |
| 4 | **Cross-encoder scores for route and reverse candidates** (Qwen-0.6B), fed as features | before 15:00 | 1–2 h GPU | +0.0003 to +0.0006 | NFG verified rescue [M-public]; route estimate +0.0004 [I] |
| 5 | **Reverse-rank and reverse-margin features from full-pool reverse retrieval**, recomputed candidate features | before 15:00 | 2–4 h | +0.0004 to +0.0013 | nithinfilo, NFG, rikkicoder [M-public]; 0.0072 never-retrieved loss [M-ours] |
| 6 | **K1 integration** with the ensemble weights and threshold tuned *jointly* on B_REST (5-fold) | before 19:00 | 2 h | +0.0003 to +0.0008 | Heterogeneous ensembles +0.004 to +0.010 in [Foursquare](https://www.kaggle.com/c/foursquare-location-matching/discussion/336415) and [Shopee](https://www.kaggle.com/competitions/shopee-product-matching/writeups/lyaka-tkm-2nd-place-solution-matching-prediction-b) [R]; Qwen +0.00039 [M-ours] |
| 7 | **Density check** on every new claimant/group feature (adversarial AUC B_REST vs test) before promotion | before 19:00 | 30 min | protects against −0.008 | mayankgoplani431: val +0.003, LB 0.964 → 0.956 ([README](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution)) [M-public] |
| 8 | **One HOLDOUT-2 read** of the frozen finalist, then the final validated upload | 17:00–19:00 | 1–2 h | 0 (protects the choice) | Plan E8 |

Items 1 and 2 are the only ones that fit before 10:00. Item 1 carries the best gain per hour because it needs no training and targets the largest loss. Its gain range is wide because the GBDT's ranking quality in France is untested; only its calibration is known.

**By 15:00:**

- Item 3 is the high-variance, high-ceiling bet. It is the most plausible route to the ≥ 0.97 France the leaders appear to have.
- Items 4 and 5 are the solid US/India additions.

**By 19:00:** integrate (item 6), guard (item 7) and confirm (item 8).

**Upload rule:** keep France-only uploads with byte-identical US/India rows until the last file, so that each French change reads directly from the public score.

**Stop list.** Each of these is measured negative or not worth the hours today:

| Stop | Why |
|---|---|
| French self-training | [M-public] NFG −0.012 despite 99.8%-accurate pseudo-labels; [M-ours] S2 shrank French lists to 3.19 per S1 |
| Adversarial or MMD domain adaptation | [R] unstable (MFSN re-run) and needs target validation labels |
| Hungarian / min-cost-flow over greedy one-owner | [M-ours] +0.000037; [M-public] nithinfilo: greedy vs global differs on 14 pairs; [R] [MSR one-to-one report](https://www.microsoft.com/en-us/research/wp-content/uploads/2011/08/msr-report-1to1.pdf) and [Papadakis et al.](https://link.springer.com/article/10.1007/s00778-023-00791-3) find greedy on par with max-weight |
| Graph / cluster post-processing | [M-ours] cluster voting +0.000032 |
| Raw expected-F0.5 decoding | [M-ours] −0.0045 |
| Loosening the French threshold | [M-ours] probe A is at break-even |
| The 4B LoRA | cannot train, score and gate before the freeze [I] |

## Conclusion

Leaders appear to score France about as well as US/India. That makes most sense if they never treated France as a separate problem, and ran one language-agnostic, generator-aware, claimant-aware model everywhere. We have built every part of that model (the string GBDT, K3 edit operations, claimant-set features) but still decide France with the component that measurably misfires there. So the missing big win is probably reuse of existing parts. It does not need another scorer.

Two things remain unknown. First, whether the calibrated GBDT *ranks* French pairs well enough. Second, whether the public subset really carries 15% France; the decode's agreement with the arena supports it but does not prove it. Only one of today's uploads can answer the first question cleanly. It should be a France-only file built from the country-free model. Even so, a realistic finish is 0.982–0.986, since first place also needs US/India gains that no lever measured so far has shown.
