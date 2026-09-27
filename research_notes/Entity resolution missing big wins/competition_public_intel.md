# Public intel from other Amazon ML Challenge 2026 (entity resolution) participants

Scope and method: searched GitHub (repo + code search, about 480 candidate repos created after 20 Sep 2026; READMEs and selected docs read via the GitHub API), Hugging Face (dataset/model/space search plus dataset cards only), Kaggle, LinkedIn/Reddit/X/Unstop via web search. **Only public README/doc text was read. No other team's data, embeddings, checkpoints or code was downloaded or run.** "Date" is the repo's last push (UTC) unless stated otherwise.

Headline: **no team at or above 0.990 has published anything I could find.** The most detailed public write-ups come from teams scoring 0.967 to 0.981 on the public leaderboard. Their negative results and diagnostics are still useful. Nothing turned up for CDS_Team_iisc, GG, Androids, Banana, The Potato Gang, Confusion Matrix, Team X, Deliveryagents or CiSTUP: GitHub repo searches returned nothing, and no LinkedIn, Reddit or X posts from 24–27 Sep appeared. The LinkedIn and Reddit hits were about the 2024 and 2025 challenges.

---

## 1. Blocking / candidate generation and reported recall / oracle

### Takeaway
Every detailed public pipeline uses **country-partitioned sparse TF-IDF top-k plus reverse search** (each S2/S3 record retrieves its top-3 S1 records), and several add exact keys, extra channels or fine-tuned e5 retrieval. Reverse search is the most consistently reported recall gain (+1 to +2.2 pts). The best reported candidate recall is about 99.2–99.6%, with oracle F0.5 at 0.9947–0.9987. The only large retrieval improvement reported is **contrastively fine-tuned multilingual-e5-small**, which beat TF-IDF at R@10.

### Cited Findings
- **nithinfilo** (val 0.9747; the HANDOFF puts the leaderboard at rank 4–5 = 0.988 at 26 Sep 03:00 IST): partitions are (country, state). S2/S3 records with no state go into every partition of their country, and S1 records with no state are compared against the whole country. It has two channels, a name channel on char-3-gram TF-IDF of `name_compact` (top 20, threshold 0.25) and an address channel on word TF-IDF (top 15, threshold 0.20). Each runs forward and reverse, with reverse returning the top-3 S1 records per S2/S3 record. It sets `max_df=0.1`. "The reverse channels are what lift recall from 97.0% to 99.2%". Capping at 30 candidates per entity costs 1.5 recall points, so nothing is capped. Recall is 99.24% at 56.3 candidates per S1 on a 14-state sample, with **oracle F0.5 0.9974**. Retrieval uses `sparse_dot_topn`. — [nithinfilo HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md) (pushed 2026-09-25)
- **NFG (aditya-bheke; LB 0.9810)** uses IDF-weighted sparse vectors in three views (name+address, name, address), searched forward and reverse. "Reverse search lifted recall from 96.4% to 97.5%, and the reverse rank became the single strongest feature." A larger K gave "~0.25% extra recall for many more candidates". A char-trigram rescue found 9% of blocking misses. **Dense e5 rescue** found the true reference at rank 1 for 67% of sparse misses, and a LightGBM plus cross-encoder verifier kept 97% precision on the rescued pairs. The rescue added +0.0010 to +0.0014 on validation and **helped France on the leaderboard (0.9790 without the French rescue vs 0.9796 with it)**. — [NFG README](https://github.com/aditya-bheke/business-entity-resolution), [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md) (pushed 2026-09-26)
- **rikkicoder (LB 0.9793)** on a 5% sample:
  - TF-IDF with L4-canonical word tokens reaches R@10 0.989.
  - **Fine-tuned multilingual-e5-small (one epoch, 228k triplets, 12 min) reaches R@10 0.9965 and R@50 0.9991**, with India at 0.9953 and native-script names at 0.9971.
  - Off-the-shelf e5 (0.970), LaBSE (0.922) and MiniLM (0.747) all trail TF-IDF.
  - Forward-10 alone loses 0.011 recall per 10× pool growth; forward-10 plus reverse-3 loses only 0.004. Extrapolated to the full pool, recall is 0.988 (F0.5 ceiling 0.996).
  - Dense SVD and exact-key blocking alone are clearly worse (0.92–0.94).
  - Word tokens beat char 2–4-grams on L4 text (0.993 vs 0.987) and run 9× faster.
  - Phonetic skeletons hurt (0.974).
  - A cheap LightGBM pre-ranker keeping p≥0.003 cuts the list to 4.2 candidates per S1 at recall 0.9951.
  
  — [rikkicoder LAB_RESULTS.md](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md) (pushed 2026-09-27)
- **mayankgoplani431-del (LB 0.964)** adds a name-as-one-token TF-IDF pass and exact-key passes (name+house number, unique name, sorted-name+number, no-space name+number) to four TF-IDF passes and a reverse pass. This raised blocking recall from 95.4 to 97.5% and the leaderboard from 0.953 to 0.964: "recall gains transferred ~1.5× to the leaderboard". A stage-0 LightGBM ranker then keeps the top 20 per S1. — [mayankgoplani431 README](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution) (pushed 2026-09-26)
- **harshgitty58 (LB 0.969)** adds an address-only top-6 channel, address skeletons, name char-4-grams and name numbers, taking raw recall from 97.74 to 98.58%. It also uses **sibling expansion**: candidates are added through S2/S3 records already confidently matched, via shared keys such as exact address, core name, skeleton, or number+street. That added 0.90M train and 1.30M test pairs, and the oracle on its candidates is 0.9947. — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md) (2026-09-26)
- **rishavk77 (HF dataset card, embeddings only)** reports full-scale recall for off-the-shelf multilingual-e5-base at top-50 of 94.99% for name+address alone and 98.25% for the name+address plus name union, rising to about 99% with all three channels and exact keys. It recommends top-100. Its warning: "`max(cosine)` across channels is not comparable — z-score per channel before combining". Frequency-capped string blocking loses 20–38 pts when scaled up, while ranked top-K loses about 0.8 pt. — [rishavk77/amc26-entity-resolution-embeddings card](https://huggingface.co/datasets/rishavk77/amc26-entity-resolution-embeddings) (2026-09-25)
- Dataset EDA from a gated HF mirror card: there are **0 cross-country matches** among 7.64M GT pairs. Every S2/S3 record links to at most one S1, and about 25–27% of S2/S3 records are unmatched distractors. — [akshatbakshi/amazon-ml-challenge-2026 card](https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026) (2026-09-24; gated, only the card was read)

### Inferences
- Our retrieval family (E5/BGE plus trees) is already in the same class. The one concrete, measured retrieval idea we may lack is a **contrastively fine-tuned small e5 retriever trained on TRAIN pairs**, unioned with TF-IDF forward+reverse and with per-channel z-scored cosines. It is only worth adding if our retrieval ceiling on the full pool is still below about 0.995.
- The reverse-rank and reverse-margin features were the dominant GBM features for two teams: NFG ("single strongest") and rikkicoder (`rev_rank` 39–50% of gain). If our matcher lacks reverse-rank features computed against the **full competing pool**, that is a cheap check.
- The public teams put the oracle at 0.995–0.999, so blocking is not the main gap for them. Their loss is in the model and decision, and in France.

### Gaps
- The nithinfilo claim of "97.0% → 99.2%" is on a 14-state training sample, not the full pool. rikkicoder's e5 numbers are on a 5% sample with a 10.3M-pool extrapolation. Neither is a full-pool test measurement.
- No ≥0.990 team published recall or oracle numbers.

---

## 2. France handling (no training labels)

### Takeaway
No public team reports a France method that clearly works. Documented France effects:
- Stricter France thresholds help. NFG found France is precision-limited and the model is overconfident there.
- The dense rescue helped France slightly.
- **Self-training / pseudo-labelling hurt even at 99.8% pseudo-label accuracy** in a US→India simulation.
- Several France-specific rules that "looked right" failed on labelled proxies.

Leaderboard blanking probes by two teams put France at about 15% of the public-LB weight, with France scores of about 0.94–0.95 for 0.97–0.98 teams.

### Cited Findings
- NFG (LB 0.9810): blanking France gave **0.8459**, which splits the score into **US+India ≈ 0.985 and France ≈ 0.948**. "The whole gap between validation and leaderboard comes from the country with no training labels." France thresholds of 0.6 and 0.9 gave 0.970 and 0.971, so stricter was better. With France at 0.75, the score dropped 0.9796 → 0.9794, "the model is overconfident there". — [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)
- NFG's France-specific findings:
  - S1 "twins" (same name and street, different number) are **1.1% of French entities vs 0.02% elsewhere**, and copies that dropped the number cannot be attributed.
  - Dotted legal forms with no final dot ("S.A.S") were misread by their normaliser (about 1% of French records).
  - French names are much more generic ("Roubaix Ecole SARL").
  - An **acronym rule** ("Dietetique Amis SAS" → "DA" at the identical address, 10× more common in France) got 434/435 held-out pairs correct.
  
  — [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md), [NFG README](https://github.com/aditya-bheke/business-entity-resolution)
- NFG negative results for France:
  - "Self-training on the unseen country: simulated with US → India: pseudo-labels 99.8% correct, yet the score fell 0.927 → 0.915."
  - "France: drop pairs whose street numbers differ: US true copies change numbers just as often."
  - "France: accept uncertain pairs with the same number and street: no lift on labelled data."
  
  — [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)
- harshgitty58 (LB 0.969) blanking probes: blanking France gave 0.836, India 0.580, US 0.578. Swapping in prototype 1's France predictions changed nothing. The author notes the US drop "would need a US score above 1" at its 38.3% share of test S1. The author's conclusion: **the public LB subset is not proportional to the test country mix**, or the scoring differs. The implied singleton rate is about 5.6%. France gets matches for more records (63.5% of records get a parent vs 58–60%). — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md)
- rikkicoder's France check with no labels, on the full test pool: with L4 canonicalisation, France's top-1 cosine median reaches 1.000, top-1 house-number agreement is 0.877 (US 0.845), and France has **more near-duplicate distractors (6.7 neighbours with cosine ≥0.7 vs about 4.5–5.4 elsewhere)**. In their country-transfer test, the source-country threshold was near-optimal on the target country (0.9793 vs 0.9794). — [rikkicoder LAB_RESULTS.md](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md)
- abbinavv (LB 0.967148, 27 Sep): France is **re-blocked without the common-word (max_df) cut-off**. The decision layer uses a `--word-veto` (word-swap veto), a `--neighbour-veto US France`, and a "label-free test calibration" with group rules per (country, score band, number relation, name change), derived from validation-vs-test pair densities. — [abbinavv README](https://github.com/abbinavv/ml_challenge_2026) (pushed 2026-09-26)
- nithinfilo added French legal forms (sarl, sas, sasu, sci, eurl) and region/department/city → state lookups for France partitions, "never tuned on labels". Predicted France singleton rate is 4.8%. — [nithinfilo HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md)
- The vc940 normalised mirror card lists these French rules: French street abbreviations (R → rue, AV → avenue), French departments → regions, and a typo-fixing step that replaces S2/S3 words not in the same-country S1 vocabulary with the most frequent S1 word within 1–2 edits. That is a **label-free, test-internal vocabulary correction usable for France**. — [vc940/business-entity-resolution-normalized card](https://huggingface.co/datasets/vc940/business-entity-resolution-normalized) (2026-09-24; card only)
- mayankgoplani431 uses a leave-one-country-out proxy (train US → evaluate India) to simulate France and tried monotone constraints ("worse"). Its stated rules include "No leaderboard probing". — [mayankgoplani431 README](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution)

### Inferences
- **Arithmetic on the probes (inference):**
  - Each country's blanking drop ≈ its weight × (its score − its singleton rate), and the three drops sum to the score minus the overall singleton rate. harshgitty58's drops (0.133 / 0.389 / 0.391; total 0.913 at score 0.969) imply public-LB weights of about **France 15%, India 43%, US 43%**, compared with 15.0 / 46.7 / 38.3% of test S1.
  - NFG's France drop (0.1337) gives the same ~15% France weight.
  - At 15% France weight, a 0.9915 leader needs roughly: with US/India at 0.990, France ≈ 0.999; with US/India at 0.993, France ≈ 0.983. Under the assumption that 15% weight holds, **the leaders' France score must be about 0.98+**, versus our 0.937 and NFG's 0.948.
  - So most of the gap to first place is France (≈0.04–0.05 × 0.15 ≈ 0.006–0.007), with about 0.004–0.007 more from US/India (0.986 → ~0.992). This is derived from probe arithmetic, not a published claim.
- NFG's France at 0.948 beats ours (0.937) with a pipeline whose France-relevant pieces are:
  - a stricter France threshold (0.9 vs 0.6);
  - per-S1 match caps (5 S2 / 6 S3);
  - one-owner assignment;
  - an acronym rule;
  - a dense rescue verified by a cross-encoder;
  - address-less record assignment;
  - multilingual-e5 cross-encoders stacked with the GBM.

  Our France may be more over-merged or under-capped. A stricter France-only threshold probe is a cheap test that NFG found positive.
- The self-training negative (pseudo-labels 99.8% correct yet −0.012) is a strong public caution against naive France pseudo-labelling, at least without a proxy check.

### Gaps
- No public ≥0.99 team explained France. The probe inconsistency harshgitty58 raised (the public-LB country mix) is unresolved publicly.
- I found no public use of LLMs, synthetic French data ("Frenchify") or translated training data that reports a leaderboard result. Repos mentioning "synthetic" use fake fixtures for smoke tests only (e.g. [AbhijitK20](https://github.com/AbhijitK20/Business_entity_resolution), [buildby-anish](https://github.com/buildby-anish/business_entity_resolution)).

---

## 3. Matching model and decision rule

### Takeaway
The public 0.97–0.98 pipelines share a common core:
- LightGBM on about 50–64 language-agnostic string, number and legal-form features;
- blocking-context features (reverse rank, cosine, margin);
- name/address ambiguity counts;
- competition/group features;
- often a **second-stage stacker** over the candidate group.

The decision is **one-owner** (each S2/S3 goes to at most one S1), plus either an F0.5-tuned threshold or per-S1 expected-F0.5 subset selection, with per-S1 caps. Cross-encoders (multilingual-e5, MIT) help as stacked features, not standalone.

### Cited Findings
- NFG validation ladder:

  | Step | Validation F0.5 |
  |---|---|
  | LightGBM baseline | 0.9682 |
  | + name/address ambiguity counts | 0.9733 |
  | + reverse blocking/rank | 0.9774 |
  | + e5-small cross-encoder stacked (uncertain-pair AUC 0.924 → 0.964) | 0.9817 |
  | + e5-base and a second e5-small, band [0.002, 0.998] | 0.9827 |
  | + address-less record assignment | 0.9844 |
  | + CE on 2–4× data | 0.9852 |

  It uses five fine-tuned multilingual-e5 CEs (small and base) on about 4M uncertain pairs, with a 2-fold stacker. The GBM on 2× data gave 0 gain after stacking, and a second-stage group/sibling model gave +0.0004 (not kept). — [NFG README](https://github.com/aditya-bheke/business-entity-resolution), [EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)
- NFG decision: one-to-one ("a property of the data"), a threshold tuned on the exact metric, and **per-S1 caps at the training maximum (5 S2 / 6 S3)**, which "removes ~670 impossible test matches; one S1 had received 146 matches". The cap plus CE-verified rescue gave the best LB, 0.9810 (from 0.9796). — [EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)
- NFG **address-less assignment**: "decoys almost never lack an address (0.3% vs 4.4% of true copies), so an address-less record is a real copy ~98% of the time; a per-record model picks *which* reference it belongs to". This gave +0.0017. Stricter rules for shared-name address-less records all scored lower. — [NFG README](https://github.com/aditya-bheke/business-entity-resolution)
- rikkicoder on a 5% sample:

  | Setup | Validation F0.5 |
  |---|---|
  | LightGBM CTX+NAME+ADDR+NUM | 0.9890 |
  | + embeddings + CE features | 0.9917 |
  | second-stage model with group features, threshold 0.7 | **0.9934** |

  The second stage cut wrong matches on no-match S1s from 1.8% to 0.7%. One-to-one added nothing after stage 2, and the CE alone scored 0.9864. The team's own LB went 0.9744 → 0.9793 when training density matched the test pool. — [rikkicoder LAB_RESULTS.md](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md), [README](https://github.com/rikkicoder/amazonMLchlng)
- harshgitty58:

  | Step | Validation F0.5 |
  |---|---|
  | Stage 1: two half-split LightGBMs, out-of-fold | 0.9783 |
  | Stage 2: probability competition + sibling-similarity features | 0.9839 |
  | Per-S1 expected-F0.5 subset selection | 0.9842 |

  **The LB moved only 0.967 → 0.969.** Suggested next steps were cluster-level collective resolution over an S2/S3 record graph, assigning whole clusters to one S1 (estimated +0.002–0.004, untested). — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md)
- mayankgoplani431: its S3 variant (density-sensitive group features, candidate-vs-candidate features, entity decision model) **raised validation 0.9776 → 0.9806 but dropped the LB 0.964 → 0.956**. — [mayankgoplani431 README](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution)
- nithinfilo error breakdown at val 0.9747:
  - F0.5 is 0.932 for 1-true-match entities vs ≥0.981 for 4+.
  - **41% of missed records have no address** and a generic or truncated name.
  - **88% of FPs are near-copy distractors** (same name and street, house number changed, e.g. 2246/b → 2250/b, or one name word swapped).
  - Greedy vs global assignment affects only 14 pairs.
  
  — [nithinfilo HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md)
- Rutuja200630 (val 0.9761): source-specific thresholds (S2 0.75, S3 0.85) gave a small gain over a single threshold. — [Rutuja200630 README](https://github.com/Rutuja200630/amazon-ml-challenge-2026)

### Inferences
- These are candidate "missing wins" we should check against our pipeline, because they carried over to the leaderboard for NFG:
  1. **address-less record assignment** as a separate per-record model;
  2. **per-S1 match caps** at the training maximum per source;
  3. **CE-verified dense rescue** of blocking misses;
  4. a **group-aware second-stage stacker** that lowers FPs on singletons.
- The two teams whose validation rose while their LB stalled or fell (harshgitty58, mayankgoplani431) both relied on density- or sibling-sensitive group features. Group features appear to transfer poorly if training density differs from test density.

### Gaps
- No public LLM matcher (Qwen etc.) with an LB result was found. HF has unrelated entity-resolution LLM checkpoints (vivekdse, abicyclerider) dated before the challenge, not tied to it.

---

## 4. Generator insights, leaks/tricks, and organiser rules

### Takeaway
Public participants converge on these facts about the generator:
- **no cross-country links**;
- **each S2/S3 record has at most one S1 parent**;
- about 25–27% of pool records are distractors, planted as near-copies with a changed house/unit number, a swapped legal form or a swapped name word;
- **decoys almost never lack an address**;
- invented or DBA-style names appear at the correct address;
- **test has about 23% more pool records per S1 than train** (5.75 vs 4.67).

No leak exploitation is claimed publicly. The organiser rules forbid external lookups, cap models at MIT/Apache and ≤8B, and use a public subset / private remainder split.

### Cited Findings
- **Train/test density shift:** "Test has 5.75 S2/S3 records per S1 against 4.67 in train (+23%, the same in every country)" — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md). "Test has ~2× more distractors per S1 than train (5.8 vs 4.67 pool records per S1)." S3's extra accepted test pairs (~230k) had a different house number from the S1 in 60–90% of cases. — [mayankgoplani431 README](https://github.com/mayankgoplani431-del/amazon-ml-2026-entity-resolution). rikkicoder trained at "test-like density" and gained +0.0049 on the LB (0.9744 → 0.9793). — [rikkicoder README](https://github.com/rikkicoder/amazonMLchlng)
- **Orphan copies hypothesis** (test S1 missing about 19% of entities whose copies remain): proposed by [harshgitty58](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md). **Contradicted** by [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md): "Hiding 20% of S1 reproduced the test structure; neither the main model nor the rescue lost accuracy."
- **House-number noise vs decoys:** real noise truncates, pads or adds a suffix letter (7402 → 740, 882 → 00882, 10216 → 10216b), while distractors substitute a different number (315 → 320). The suggestion is digit-edit-type features. — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md) (proposed, untested). NFG notes that US true copies change street numbers as often as French ones, so a blanket number-mismatch veto fails. — [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)
- **Legal-form conflict** (LLC → Corp, SAS → SA) is "the decoy signature". Other noise types: acronyms, website-style names ("prairieexpressultra.com"), digit/letter swaps ("Ve1ma"), invented brand names, nine Indic scripts. **38% of reference names are shared** by several businesses ("Primary Care Group" ×253). — [NFG README](https://github.com/aditya-bheke/business-entity-resolution)
- **Invented names at the right address** (e.g. `veogild`, `korhalovantage`) make up 8–12% of misses. **61% of missed true pairs have an empty address.** Sibling copies share typos, spacing and unit numbers. — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md). NFG contradicts the sibling point: "Copies agree no more than chance (name equality 38.7% vs 36%)" — [NFG EXPERIMENTS.md](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md)
- Transliteration: a learned word-level Indic → Latin dictionary from matched training pairs (about 1,343–1,347 entries, 96–98% token coverage) is used by nithinfilo, NFG, rikkicoder, mayankgoplani431 and vc940. It raised native-script R@10 from 0.86 to 0.99 in rikkicoder's test. — [rikkicoder LAB_RESULTS.md](https://github.com/rikkicoder/amazonMLchlng/blob/main/student_resource/LAB_RESULTS.md), [nithinfilo HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md)
- Data-quirk normalisations: Telangana → AP and DC → WA are merged "because the data mixes them" — [nithinfilo HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md). Also alias splitting on DBA/formerly/aka, glued-word splitting and leet fixes — [vc940 card](https://huggingface.co/datasets/vc940/business-entity-resolution-normalized).
- **Rules (official problem statement, mirrored):**
  - External databases, APIs, geocoding and "any external data augmentation from internet sources" are strictly prohibited.
  - The final model must be MIT/Apache 2.0 and ≤8B parameters.
  - The public LB is a subset of test; the private LB is the remainder and decides final rankings.
  - `candidate_pairs.tsv` is not scored, but it is used to audit blocking (recall ceiling, reduction ratio).
  - Top teams' zips are reproduced and audited.
  
  — [nithish16 mirror of README](https://huggingface.co/datasets/nithish16/amazon-ml-challenge-2026), [logicalguy mirror](https://huggingface.co/datasets/logicalguy/amazon-ml-challenge-2026). The challenge window was 25 Sep 09:00 to 27 Sep 21:00 IST, with results and the top 50 on 2 Oct and the finale on 7 Oct. — [Internshala listing](https://internshala.com/competitions/amazon-ml-challenge-2026-win-%E2%82%B9225000/)
- Submission limits conflict: "The Portal allows 5 submissions a day" — [HarpalKalankar README](https://github.com/HarpalKalankar/business-entity-resolution); contradicted by "uploads are unlimited" — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md).
- One README claims `candidate_pairs.tsv` "is part of the final ranking; a smaller candidate set per entity ranks higher" — [HarpalKalankar README](https://github.com/HarpalKalankar/business-entity-resolution). This is **unverified and not in the official text**, which says the file is not scored but is used for blocking analysis.

### Inferences
- The +23% test pool density is the most concrete publicly documented generator shift. Two teams saw validation gains fail to transfer, and one saw a +0.005 LB gain from density-matched training. That supports **density-stressed tuning**: tune thresholds and group features with extra distractor negatives injected into TRAIN/TUNE pools.
- Address-less records being almost never decoys (0.3% vs 4.4%) is a strong structural prior that a pairwise matcher alone may not use.

### Gaps
- No organiser clarification posts (Unstop discussion/Discord) could be retrieved. The Unstop pages returned only boilerplate to the crawler.
- Whether using unlabelled test text (vocabulary, IDF, typo maps) is allowed is not addressed explicitly in the official text. mayankgoplani431 states it uses test "unlabeled (normalization, blocking statistics) exactly as at inference" as its own rule reading.

---

## 5. Leaderboard facts and top-team posts

### Takeaway
Public snapshots of the public LB top: **0.9884** (leader, 26 Sep, HarpalKalankar), **0.9906 / 0.9894 / 0.9888** (top 3 on 26 Sep, harshgitty58), and rank 4–5 at 0.988 with ranks 19–25 at 0.985 at 26 Sep 03:00 IST (nithinfilo). I found no public posts by any named top team, and no source confirming the ~0.9915 first place or ~21 teams ≥0.990 on 27 Sep morning.

### Cited Findings
- "Leaderboard on 26 Sep, 03:00 IST: rank 4–5 at 0.988, ranks 6–12 at 0.987, 13–18 at 0.986, 19–25 at 0.985." — [nithinfilo HANDOFF.md](https://github.com/nithinfilo/ml-challenge-2026-entity-resolution/blob/main/HANDOFF.md)
- "Top of the leaderboard on 2026-09-26: 0.9906, 0.9894, 0.9888." — [harshgitty58 PROGRESS.md](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md); the README says "top of the leaderboard was 0.9906 at handoff" — [harshgitty58 README](https://github.com/harshgitty58/Amazon_ML_Challenge)
- "Leader at the time of writing: 0.9884" (26 Sep) — [HarpalKalankar README](https://github.com/HarpalKalankar/business-entity-resolution)
- Public team LBs with methods: NFG 0.9810 ([link](https://github.com/aditya-bheke/business-entity-resolution)), rikkicoder 0.9793 ([link](https://github.com/rikkicoder/amazonMLchlng)), harshgitty58 0.969, Abhigyanv23 0.9683 ([link](https://github.com/Abhigyanv23/Amazon-ML-Challenge-2026)), abbinavv 0.967148, HarpalKalankar 0.967, mayankgoplani431 0.964.
- A README "targeting 0.998–0.9998" ([naveencmy](https://github.com/naveencmy/business_entity_resolution)) is aspirational, not a measured score. md-yahya1's 0.9987 is pairwise F1, not entity F0.5 ([link](https://github.com/md-yahya1/business-entity-resolution)).

### Inferences
- The leader rose from about 0.9884–0.9906 on 26 Sep to about 0.9915 (per the brief) by 27 Sep. Top scores were tightly packed, so small France or precision gains matter for rank.

### Gaps
- No LinkedIn/X/Reddit/Medium/YouTube post from 24–27 Sep 2026 by the named top teams was found. Such posts are likely after results on 2 Oct. The akshatbakshi HF dataset is gated; only its public card was read. vc940 has no GitHub repo for this challenge, only the HF normalised dataset card. I did not find a harshgitty58 post outside the repo, and the "per-country blanking probes" lead is confirmed as coming from that repo's PROGRESS.md.
