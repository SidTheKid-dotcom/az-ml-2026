# External second-opinion brief (27 Sep 2026, ~05:00 IST)

> **Note for the user. Do not paste this box.**
>
> Sending competition details to a third-party AI service may raise fair-play or data-use questions. The official
> README (`data/raw/README.md`) says:
>
> - "Participants are **STRICTLY NOT ALLOWED** to use external databases, APIs, or services to look up business
>   identities or resolve entities."
> - It prohibits "Any external data augmentation from internet sources".
> - "**Fair Play:** This challenge is designed to test your machine learning and data science skills using only the
>   provided training data."
> - "Any evidence of external data lookup will result in **immediate disqualification**".
> - "Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters."
>
> The README neither allows nor forbids asking an outside assistant for strategy, and the portal's terms may say
> more. So:
> - Use the service **only for strategy advice**.
> - Never send records, IDs or files, never let it score or resolve pairs, and never use its output as labels or
>   training data.
>
> The brief below contains no raw data; its examples are synthetic.
>
> **Service settings:**
> - Model: `abliterated-model-large-v2`, reasoning effort `max`, temperature ~1.0.
> - Leave web search off. It is the one path the service says is not zero-retention.

---
**Paste everything below this line.**

# Second-opinion request: entity resolution at 0.978, target > 0.99, ~19 hours left

You are advising a team in a supervised entity-resolution competition. This brief is self-contained.

**Evidence tags:**
- **[measured]:** measured directly (by the team unless a source is named);
- **[inferred]:** derived from measurements plus a stated assumption;
- **[hypothesis]:** untested.

Unmarked statements in §1 and §8 are official rules or fixed constraints. All examples are synthetic.

## 1. Problem and metric

- **Data:** three sources of business records (id, name, address, country).
  - Source 1 (S1) is deduplicated. For each S1, list the Source-2/3 records of the same business (zero, one or many).
  - Train: US and India, labelled. Test: US, India and **France, which has no labels anywhere**.
  - Test has 1,732,544 S1s: US 38%, India 47%, France 15%.
- **Metric:** F0.5 = 1.25·P·R / (0.25·P + R) per S1, macro-averaged over all S1s.
  - A no-match S1 scores 1 with an empty list and 0 if anything is predicted.
  - A matched S1 with an empty list scores 0 [inferred].
- **Price of one link** [inferred from the formula]: dropping a false link from an n-link list gains 4/(5n−1); dropping a
  true one loses 1/(5n−4). A false link costs ~3–4× a missed one.
- **Label structure** [measured, train]:
  - each S2/S3 record has at most one owning S1 (the "one-owner" rule);
  - true list sizes 0/1/2/3/4/5/6+ occur 5.6/5.4/17.0/24.1/21.9/14.6/11.4% of the time, maximum 11.
- **Leaderboards:** the public board scores a test subset. The private board ("the remaining portion") decides the final
  ranking.
- **`candidate_pairs.tsv`** (exactly what the final model scores) is reviewed for recall ceiling and reduction ratio.

## 2. Data and generator facts

**Base rates** [measured]:
- 3.46 true links per S1 and 5.58% no-match S1s, identical in US and India;
- pool density: train 4.68 records per S1 (74% true); test 5.75 (France 5.53);
- no test S1 or record exactly matches train, and 33% of test S1 names recur in train at other addresses.

**Generator signatures** [measured]:
- **House numbers:**
  - A decoy's house number is the S1's number plus δ ∈ {1,3,5,7,9,11,13,21}, never minus.
  - Copies truncate, zero-pad or add a letter (7402→740, 882→00882, 10216→10216b).
  - Substituted digits: that class is 45% true.
- **Legal forms:** LLC→INC and PVT_LTD→PUB_LTD swaps are 0% copies; PC/LP swaps ~60%. Copies change the legal form in US
  2%, India 21%.
- **Decoy types:** same-name decoys change the house number (73–80%) and the legal form (45–70%); same-address decoys
  substitute a name word (79–98%).
- **Decoy-only words:**
  - US: "group" (53,500 decoy vs 0 copy occurrences), "holdings" (52,773 vs 3), 21 location words;
  - India: overseas, industries, ventures, exports, infratech, public, private.

  dba/aka/née/formerly mark copies (~98%).
- **Label-free decoy test:** a word's house-number-shift rate predicts its decoy share, r = 1.000 (US) and 0.998
  (India).
- **Legal-form class:** same core name + same address digits + fuzzy address is 99.85–99.99% true (216k pairs).
- **Empty-address records:** almost never decoys (≤ 0.5%). But when several S1s share the name, each such pair is ~4%
  true, so this is an assignment problem.
- **Script and source:**
  - India: 24% of S2 names are in native script.
  - US S2 addresses are 89% upper-case (S3: 0%); S3 addresses have 53% word typos. The models do not see the source.
- **Leakage:** none by ID or row order.

**Synthetic illustration**, S1 "Harbor Point Dental LLC, 4410 Elm St":
- copies: "HARBOR POINT DENTAL, 4410 ELM STREET", "Harbor Point Dental dba HP Smiles, 441 Elm St";
- decoys: "Harbor Point Dental Holdings, 4421 Elm St", "Harbor Point Dental INC, 4415 Elm St".

**France (label-free):**
- the same +δ decoy numbers and mirrored decoy words (groupe, holding, développement as pure additions) [measured];
- Cie, services, associés and et fils behave as copy suffixes [measured];
- so the generator is probably the same code [inferred].

**The "arena"** (US/India labelled evaluation) [measured]:
- 2,072,821 train S1s × dense top-50 over the full train pool = 103.6M labelled pairs.
- It is split by connected claim components:
  - A: 1.45M S1s, training;
  - B_REST: 521,846 S1s, cross-fit evaluation, 60% US;
  - SLICE: 100k S1s, never read;
  - a sealed 50k holdout.

## 3. Current system and scores

**Pipeline** (B_REST unless noted):

| Stage | What | Result |
|---|---|---|
| Retrieval | multilingual-e5-small (fine-tuned), country-blocked top-50 | Link recall 0.9835; best-possible F0.5 0.9946 [measured] |
| Scorer C2 | bge-reranker-v2-m3 (568M) cross-encoder, "name \| address", 5.1M pairs from A | Plain blend 0.97958 [measured] |
| Scorer Q | Qwen3-Reranker-0.6B, 416k pairs | 0.98011: matches C2 on 12× less data [measured, one run] |
| Decision, US/India | CatBoost pair tree, ~20 features (scores, rank/gap in list, list counts, best rival claim on the record, exact-name/key flags, address kind), cross-fit on B_REST halves; threshold 0.725, then one owner per record, cap 11 | **0.985511** (US 0.9874, India 0.9826), P 0.99789, R 0.96096 [measured] |
| Variants not uploaded | Add Qwen features; or add collective (sibling/alias) features | 0.985900; 0.985971, CI > 0 [measured] |
| France | 0.9·C2 + 0.1·tree ≥ 0.10, plus a unique exact-name rule for empty-address records; one owner, cap 11 | No labels. The US/India tree removes more French links than it adds, unlike on labelled data [measured] |

**Public scores** [measured]: best file **0.978287**; the same file with French rows emptied: 0.846352.

**Decode:** assuming France's no-match rate is 5.58%, **US/India ≈ 0.9856** (the arena predicted 0.98551) and **France ≈
0.937 ± 0.006** [inferred]. Arena→test US/India transfer holds within ~±0.0005 after country-mix adjustment [inferred].

**What 0.99 needs** [inferred]: public ≈ 0.850·US/India + 0.150·France. A perfect France with today's US/India gives
0.98775.

| US/India | 0.9886 | 0.9896 | 0.9906 | 0.9916 | 0.9926 |
|---|---|---|---|---|---|
| France needed for 0.99 | 0.998 | 0.992 | 0.987 | 0.981 | 0.975 |

**Leaderboard** [measured, latest snapshot]: top ≈ 0.9906, 10th ≈ 0.9898. The leaders' country split is unknown.

## 4. Measured loss decomposition

**US/India** (B_REST, 1 − 0.985511 = 0.0145) [measured]:

| Source | Share | F0.5 points | Links | Notes |
|---|---|---|---|---|
| True record never retrieved | ~50% | 0.0072 | 37,810 | 6,732 have no record in the arena pool at all |
| Retrieved but rejected | ~37% | 0.0053 | 32,232 | 80% empty-address |
| False links | ~13% | 0.0019 | ~3.7–3.9k | 51% empty-address |

**Missed true records by type** [measured; a separate audit, so the totals don't reconcile exactly]:

| Record type | Missed | Recall |
|---|---|---|
| Empty-address copies | 39.7k | ~50% |
| Invented brand name at the correct address | 10.8k | 73–80% |
| Native-script name with an address | 5.4k | 95.5% |
| Ordinary names | ~6k | 99.5%+ |

**France (test, no labels):**
- **Smear** [measured], per S1, France vs US/India:
  - uncertain band (0.05–0.5): 1.94 vs 0.15–0.26;
  - candidates ≥ 0.5: 3.75 vs ~3.37;
  - pairs with tree ≥ 0.9 but C2 < 0.05: 0.545 vs 0.016–0.026.
- **C2 misfire:** C2 scores ~0.0003 for "Atelier Morel" vs "Atelier Morel SARL" at the same address. That class is
  99.85–99.99% true in US/India.
- **Probes** (public deltas [measured]; truth rates via a pricing model [inferred]):
  - dropping 43,517 uncertain non-legal-form links: −0.000027, i.e. ~70% true (break-even);
  - dropping one confident non-legal-form link from each of 139,543 lists: −0.005317, i.e. ~96% true.
- **Bill** [inferred]:
  - France loses ~16.4k S1-points, ≈ 43% of lost public points from 15% of S1s.
  - ~25k wrong links (precision ≈ 0.97) explain only 5–6k of them. The rest is missed links: recall 0.82–0.92,
    depending on assumptions.
  - French lists average 3.26 links against a 3.46 prior.
- **Ruled out as large losses** [measured]:
  - never-retrieved French records: mostly decoy-like, ≤ +0.0005;
  - house-number look-alikes: 89.5% of French links share all digits with their S1, against 60–66% in US/India;
  - street-token-disjoint links: one-character typos of true links.

## 5. What failed (do not re-propose without a new reason) [measured; some on older scores]

**US/India:**
- **Generator-signature rules:**
  - vetoes/rescues: ≤ +0.00002;
  - house-number-shift veto: −0.0039 (shifted links at the boundary are 97.6–98.8% true).
- **Name rules:**
  - assigning empty-address records by name: −0.0002 to −0.0006;
  - name-only on top of the tree: −0.0003;
  - name-only on digitless addresses: −0.0070;
  - unique-name same-street adds: −0.019.
- **Decoding:**
  - raw expected-F0.5 (uncalibrated): −0.0045;
  - filling S1s left wrongly empty: ≤ +0.0001;
  - mutual top-1: −0.24;
  - responsibility decode: −0.019;
  - emptying lists with a small top-1/top-2 gap: catastrophic.
- **Other rules:**
  - exact-key moves are 1.4% true and unowned adds 27.6%;
  - cluster voting: +0.00003 (CI includes 0);
  - count fill: +0.00044, but it hurts no-match S1s;
  - a second resolver tree: +0.0002;
  - threshold moves: +0.0004;
  - a C2 floor: −0.00004.
- **Retrieval extras:**
  - anyascii native-script branch: +3 true links;
  - target clustering: best-possible score +0.00001;
  - India branch beyond the union: best-possible score +0.0005.
- **Re-tuning for test density** (withholding S1s): null.

**France:**
- dropping the uncertain band;
- loosening the threshold (0.05–0.1 holds 1.23 pairs per S1 of look-alikes);
- recovering never-retrieved records;
- dropping street-disjoint links;
- a strict pseudo-label fine-tune (lists shrank to 3.19 per S1);
- vetoing French corporate words wholesale (≈ −0.008 France [inferred]).

**Pattern** [inferred]: every post-hoc rule since 0.978 moved the score by ≤ ±0.0005. The remaining gains need better
scores, candidates or inputs.

## 6. What worked [measured]

- **Public trajectory:** 0.9564 → 0.9646 (one-owner assignment) → 0.9783 (C2 + pair tree).
- **C2 training:** 0.97787 at 5k steps → 0.97872 at 10k → 0.97958 at 40k (diminishing returns).
- **Pair tree with rival-claim features:** +0.006 over the C2 blend (+0.0029 over the blend with name-only). The best
  rival claim carried 31% of importance in an earlier version.
- **Unique exact-name rule for empty-address records:** +0.0029 on weaker scores (adds 93% precise). Now France only.
- **Qwen-0.6B:** C2-level on 12× less data; stacked, +0.00039.
- **Collective features:** +0.00046, CI [0.00040, 0.00052].
- **Extra retrieval routes** (union; address and empty-address branches) raise the best-possible score 0.9946 → 0.99675
  (recall 0.9900). But new pairs are ≥ 99% negatives: C2 ≥ 0.3 keeps 1,180, of which 1,122 are true (≈ +0.0004,
  estimated).

## 7. Candidate strategies (neutral; none measured end-to-end)

Our draft plan runs A→G. Treat that as one option, not the answer. Gains are our estimates [hypothesis] and overlap.

**A. French scorer transfer via synthetic data.**
- **Positives:** French test S1 text transformed by the measured copy operations:
  - legal form added, dropped or abbreviated; Cie/services/associés; dba/formerly; initials;
  - invented brand names at the same address;
  - département↔région; R./Av./Bd.; N°/bis/ter; zero-padding; accent-dropping.
- **Negatives:** +δ house numbers, legal-form swaps, decoy-word additions, template rivals.
- **Training:** continue C2 with ≥ 50% US/India replay. Gate: B_REST may drop by at most 0.0005.
- **Then:** rescore 0.5–2.4M French pairs and measure with a France-only upload.
- **Estimate:** France 0.937 → 0.965–0.985, i.e. +0.004–0.007 public.
- **Against:**
  - the synthetic data may not match the real generator;
  - France is measurable only by upload;
  - the last French fine-tune failed;
  - using test text must be disclosed.

**B. US/India features for the missed types**, in the pair tree with 5-fold grouped cross-fit:
- an invented-name score (a character LM on S1 names);
- address exactness that ignores the name;
- transliteration similarity (an Indic→Latin dictionary learned from train pairs, plus anyascii);
- 2-hop links via accepted siblings;
- listwise and competition features.

Estimate +0.001–0.002 US/India. **Against:** all measured US/India levers so far sum to ~+0.001.

**C. Reverse retrieval** (record → top-k S1s) plus an address block, scored by C2 and B.
- Another team reports that forward + reverse blocking took recall from 97.0% to 99.2% [unverified].
- **Against:** new pairs are mostly decoys; realized gains ≪ best-possible gains; the reduction ratio is reviewed.

**D. Qwen3-Reranker-4B** (Apache-2.0, 4.02B), LoRA.
- A ~45-minute screen against 0.6B, then 1–2M hard pairs plus French synthetic data.
- ~1.5–2.5 h training and ~1.5–3 h scoring [inferred].
- MMTEB-R: 72.7, against 66.4 for 0.6B [measured, Qwen model card].
- Estimate +0.0005–0.002 US/India; France unknown.
- **Against:** it takes most of the GPU time, and LoRA tooling is not installed yet.

**E. Rich GBDT (150–250 features) with a corrected out-of-fold protocol.**
- Level 1 on A uses only features that are out-of-sample there. Never the reranker scores, which were trained on A.
- Level 2 on B_REST: 5-fold grouped OOF over the reranker scores and their listwise/rival features.
- Test: the mean of the 5 fold models.
- Estimate +0.001–0.003. **Against:** features for ~50M pairs, and B_REST has already been reused heavily.

**F. Metric-aware decoding.**
- Per-country OOF isotonic calibration: US is over-confident in the mid band (predicted 0.706, observed 0.679); India is
  under-confident (0.288 vs 0.325) [measured].
- An S1-level count model, then per-S1 expected-F0.5 over k = 0..11 with greedy one-owner resolution.
- **Theory:** later links need p ≈ 0.73–0.8, so a single 0.725 cut is suboptimal [inferred].
- **Against:** the measured gain for S1s left wrongly empty is ≤ +0.0001.

**G. France quick fixes** (CPU, minutes):
- link same-address records that only add SASU/EURL/SARL (~+0.0001);
- veto legal-form swap + house-number shift (2,864 links; +0.00014 to +0.00044).

**Also untested:**
- the source (S2/S3) as a model input;
- typo repair by projecting record words onto the S1 vocabulary;
- numeric features fused into the cross-encoder;
- checkpoint averaging.

**Precedent** [measured by others; transfer is a hypothesis]:
- **Foursquare 2022 winners:** listwise-feature GBDTs, cross-encoders with numeric features, and 2-hop recall, which
  lifted the ceiling from 0.978 to 0.9935.
- **Shopee:** set-size-aware decoding, +0.02.
- **Jigsaw:** pseudo-labels on unlabelled languages, 0.937 → 0.954.

**Uncertainties and contradictions:**
1. France's level assumes a 5.58% no-match rate (±0.005 moves France by ±0.005) and the full-test French share.
2. After the probes, the French loss split flipped from "mostly wrong links" to "mostly missed links". Both readings rest
   on a pricing model.
3. The denser test pool is not orphan copies: test has fewer empty-address records per S1 (US 0.166 vs 0.178, India 0.135
   vs 0.145) [measured]; the extras look like easy decoys [inferred]. Yet another team's +0.010 validation gain became
   +0.002 on the leaderboard.
4. The public subset looks US-heavier than the full test [inferred], so the private board is likely India-heavier.
   India is the weakest labelled country.
5. B_REST has been reused for many selections. SLICE is unread and can be read once.
6. Decoding theory (F) and measurement disagree.
7. Same-address template-word swaps are 15× more common among accepted French links (1.9% vs 0.13% US). They may be
   decoys or a small-vocabulary artefact; untested.
8. Uploads that also change US/India blur France by ~±0.003. France-only uploads measure the French delta exactly.

## 8. Constraints and resources

- **Deadline:** 27 Sep 23:59 IST (~19 h). At most 5 uploads remain today, and the last upload counts. The final ZIP
  (code, both TSVs, methodology) is due by ~23:00.
- **Models:** the final models must be MIT or Apache-2.0 and ≤ 8B parameters.
- **Data:** no external data, lookups or geocoding, and no other teams' files. Synthetic data from test S1 text is
  treated as allowed if disclosed.
- **Compute:**
  - one H100 80 GB. C2 trains at ~5.6 steps/s at batch 128 (40k steps ≈ 2 h), and concurrent scoring halves that. The
    Qwen-0.6B trainer ran at ~10% of peak [measured].
  - CPU: a 15-core/2 TB pod and a 32-vCPU/247 GB box.
- **Assets:**
  - C2 scores for arena and test candidates; Qwen-0.6B scores wherever C2 ≥ 0.01;
  - out-of-fold tree predictions on B_REST;
  - extra retrieval candidates;
  - a draft synthetic-record generator (fidelity not established);
  - a validated fallback file (0.978287).

## 9. The ask

Propose the highest-probability training/modeling strategy to exceed 0.99 within ~19 hours: what to train, on what
data, with what validation, expected gains, and what to stop doing. Challenge our assumptions.

Please return:
1. A ranked plan: an hour-by-hour allocation of the GPU and both CPU machines, a kill criterion for each step, and how to
   spend the uploads.
2. For each step: the expected public gain as a range, your probability that it lands, and a test that could falsify it
   within 2 hours.
3. Which assumptions above you think are wrong, and why.
4. If > 0.99 is unrealistic, say so, and give the plan that maximizes the expected *private*-leaderboard score.
5. A label on each of your claims (established, inferred or speculative), and a flag on anything that could breach the
   rules in §1 or §8.
