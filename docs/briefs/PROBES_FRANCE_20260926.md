# France probes tonight: design, evidence and feedback wanted (26 Sep 2026, ~23:20 IST)

## TL;DR

- **The probe decode is consistent with the arena.** US/India on the test scores **0.98559**, against arena B_REST 0.985511, a difference under 0.0001. France scores **0.9368** (about ±0.006).
  - This agreement assumes France's no-match rate equals train's 5.58%. It supports that assumption and arena transfer *jointly*; it does not prove either on its own.
- **The arena is now a trusted stand-in for US/India**, validated for methods like the current one. Uploads are spent only on France.
  - Changes that are sensitive to pool density may transfer less exactly: new retrieval, or rules that react to unclaimed records. The test pool has 5.8 records per S1, against 4.67 in train.
- **0.99 is out of reach with anything we have measured so far.** A perfect France caps the public score at **0.98775**. Reaching 0.99 needs US/India ≥ **0.98824**, i.e. +0.0027 on the arena, and every US/India lever measured so far adds up to about **+0.001**.
- **Tonight: two France-only probes** built on the 0.978287 file. They share no links, and each score converts into a truth rate:
  - **A:** drop 43,517 uncertain French links (blend 0.1–0.7), excluding legal-form pairs.
  - **B:** drop 9,033 confident French template-word-swap links.
- **Colleague 2's U1 and U2 were changed.** 58% of U1's links and 34% of U2's are legal-form pairs, which are about 99.9% true on US/India. That dilution would push the readings toward break-even whatever the rest holds.

## 1. What has been measured

### Public scores

| File | What it is | Public score |
|---|---|---|
| S1 | US/India: `pair_tree_c2` using C2 step-40k features, threshold 0.725.<br>France: 0.9·C2 + 0.1·tree ≥ 0.10, plus the empty-address name-only rule.<br>One owner per record, lists capped at 11. | 0.978287 |
| Probe | S1 with all 259,452 French rows left empty | 0.846352 |

### Decode

The weights come from test S1 counts: France has 259,452 of 1,732,544 businesses (0.149752), US/India 0.850248. The French no-match rate *s* is assumed to be 0.0558, the train rate (US 5.58%, India 5.59%).

- **US/India** = (0.846352 − 0.149752·*s*) / 0.850248 = **0.98559**. Arena B_REST cross-fit gives 0.985511.
- **France** = (0.978287 − 0.846352) / 0.149752 + *s* = **0.9368**.
- **Sensitivity:**
  - *s* ± 0.005 moves France by ±0.005 and US/India by ∓0.0009.
  - An uncertainty of ±0.0005 in the public subset's French share moves France by about ±0.003.
- **Rounding matters.** Decoding with 0.846 instead of the exact 0.846352 gives 0.98521 and 0.939.

### What 0.99 requires

The public score is 0.850248·US/India + 0.149752·France.

| If France reaches | US/India needed for 0.99 |
|---|---|
| 1.000 | 0.98824 |
| 0.9852 (parity with US/India) | 0.99084 |
| 0.960 | 0.99528 |

A perfect France with the current US/India gives 0.98775.

### US/India on arena B_REST (free, and now trusted)

**Where the 0.985511 model loses:**

| Loss source | Share of loss | Links | Detail |
|---|---|---|---|
| True record never retrieved | ~50% | 37,810 | Invented aliases at the same address, native-script names, empty-address names with typos |
| Retrieved but rejected | ~37% | 32,232 | 80% have an empty address |
| False links | ~13% | 3,865 | 51% have an empty address |

Empty-address records whose name is shared by several S1s are only 4.3–4.6% true, and pool structure cannot recover them.

**Measured gains:**

| Lever | Gain on B_REST |
|---|---|
| Qwen3-Reranker-0.6B ensemble | +0.000389 (0.985900) |
| Collective features (name siblings, alias recurrence, "dba" bridges) | +0.00046, CI [0.000398, 0.000519] |
| New address and fuzzy-name retrieval, scored by C2 at ≥ 0.3 | 1,180 new pairs, 1,122 true, about +0.0004 (an estimate; no full re-decode yet) |
| Union k=3, ranker only | +0.000275 (from an earlier agent report, not re-verified today) |

Together that is about **+0.001**.

### France (label-free, on test)

**List sizes are close to India's.**

| Predicted list size | France | US | India | Train truth |
|---|---|---|---|---|
| Empty | 5.55% | 5.84% | 6.06% | 5.58% |
| 1 link | 8.08% | 6.16% | 7.00% | 5.40% |
| Mean links per S1 | 3.26 | 3.36 | 3.29 | 3.46 |

**The loss is probably mostly wrong links (an inference, not a measurement).**
- France loses about 16,400 business-points.
- A missing link costs about 0.05–0.09; a false link about 0.17–0.29, or 1.0 on a no-match business.
- With list sizes near India's, most of the loss is probably false or swapped links, not missing ones.
- This assumes French truth follows the generator statistics measured on US/India (3.46 links, 5.58% no-match). France itself is unverified.

**Checks that ruled things out:**
- **Different house numbers:** 89.5% of accepted French links share every address digit with their S1, against 66.4% for US and 59.6% for India. The first number differs in only 2.4% of French links, against 8.5% in US/India, where such links are 99.7% true on the arena. So there is no excess of different-number look-alikes.
- **Different streets:** 7.05% of French links have no street token in common with their S1 (US 1.83%). Inspection of about 30 examples shows these are one-character street typos of true links, not an error class.
- **Never-retrieved records:** 59,340 French test records never reach any candidate list, and a proxy test suggests they are mostly decoys (the reference groups used for comparison contain errors of their own). Only 60.1% share a fuzzy street with a plausible owner, against 88.9% for accepted links and 63.7% for rejected candidates. Recovering them is worth at most +0.0005.

**Tree-yes / C2-no pairs** (tree ≥ 0.9, C2 < 0.05):
- France has 0.545 per S1; US/India have 0.016–0.026, a 28× difference.
- A fresh rescore matches the stored C2 scores, so this is not a scoring bug.

**Legal-form class:**
- **Definition:** same core name (legal forms and corporate suffix words removed), same address digits, fuzzy address match ≥ 70, and no corporate suffix word added.
- **B_REST:** 99.85–99.99% true across 216k pairs, for every kind of legal-form change.
- **France:** 603,075 pairs, of which 562,587 are already accepted. Boosting the rest adds only 7,856 links (the F1 file), because most of them lose the one-owner contest to a stronger claimant.

**Template-word swap class:**
- **Definition:** same address digits, and a core word replaced by a word that appears in ≥ 20 S1 names of the country (not a typo).
- **B_REST truth rate of accepted links:** US 98.5% (2,332 links), India 99.9% (1,734).
- **Accepted on test:** US 2,923 (0.13%), India 8,100 (0.30%), France **15,871 (1.9%, about 15× the US rate)**.
- **Caution:** on labelled US/India, accepted swaps are 98.5–99.9% *true*. Reading the French ones as decoys is a hypothesis, which Upload B tests. The 15× excess could also partly reflect France's small template vocabulary (see Q5).
- **French examples, all at the same address** (a random sample; judging them as decoys is subjective):
  - Team Ecole SARL → Team Fetes SARL
  - WG Club → WG College
  - Humanisme Maison → Humanisme Ecole
  - Cailloux Sport → Cailloux Loisirs
  - General Institut → General Élémentaire

## 2. Anatomy of the 846,833 French links in the 0.978287 file

| Blend band | Links |
|---|---|
| < 0.1 (name-only adds) | 12,074 |
| 0.1–0.3 | 65,920 |
| 0.3–0.7 | 6,262 |
| 0.7–0.9 | 18,124 |
| ≥ 0.9 | 744,453 |

- **Legal-form pairs among accepted links:** 562,587.
- **T1** (tree ≥ 0.9, C2 < 0.05): 39,217 links, of which **22,614 are legal-form pairs**. **T1r**, the same set without them, has 16,603.
- **T2r** (blend 0.1–0.7, minus the 28,665 legal-form pairs in that band): **43,517**. It breaks down as:

  | Part of T2r | Links |
  |---|---|
  | Tree-override links (T1r) | 16,603 |
  | Other in-band word swaps | 2,640 |
  | Other moderate-C2 links (acronyms, aliases, typo'd names, …) | ~24,274 |
- **Template swaps, 15,871 in total:** 6,838 fall in the 0.1–0.7 band (all inside T2r), and **9,033 sit at ≥ 0.7 (T3conf)**.
- **Contested records:** 32,022 French records have ≥ 2 French claimants at blend ≥ 0.1.

## 3. How a probe score converts into a truth rate

A probe is a rule edit that drops a set of links. For one dropped link in a list of predicted length *n*:

| If the dropped link is… | Change for that business (*n* ≥ 2) | When *n* = 1 |
|---|---|---|
| False | +4 / (5*n* − 1) | +0.5 |
| True | −1 / (5*n* − 4) | −1 |

This assumes the other links in the list are correct and complete, which holds well enough for rule edits and breaks for model re-scoring. The *n* = 1 values are heuristics, and several dropped links in the same list are treated independently. **Use the readouts for direction and rough size, not as exact truth rates.**

Δpublic(*p*) = [G·(1 − *p*) − L·*p*] / 1,732,544, where G and L are the summed gains and losses and *p* is the truth rate of the dropped set.

| Set | Links | Δ if 10% true | Δ if 50% true | Δ if 90% true | Break-even *p* |
|---|---|---|---|---|---|
| T1 (Colleague 2's U1) | 39,217 | +0.0047 | +0.0016 | −0.0016 | 0.70 |
| T1r | 16,603 | +0.0020 | +0.0006 | −0.0007 | 0.69 |
| T2 (Colleague 2's U2) | 84,256 | +0.0100 | +0.0033 | −0.0033 | 0.70 |
| **T2r (Upload A)** | 43,517 | +0.0051 | +0.0017 | −0.0018 | 0.695 |
| Template swaps, all | 15,871 | +0.0018 | +0.0005 | −0.0008 | 0.66 |
| **T3conf (Upload B)** | 9,033 | +0.0010 | +0.0003 | −0.0004 | 0.666 |
| Name-only adds | 12,074 | +0.0014 | +0.0005 | −0.0004 | 0.72 |

## 4. Tonight's two uploads

**What both files share:**
- Only French rows change; US/India rows are byte-identical to the 0.978287 file.
- They were built by code that reproduces the 0.978287 file byte-for-byte (SHA `559f37b9…` parity check).
- Both pass the official validator, use LF line endings only, claim no record twice, and cap lists at 11.

### Upload A: drop T2r (uncertain French links that aren't legal-form pairs)

- **File:** `probe-drop_T2r/matching_results.tsv`, SHA256 `41993a2fb0d830fe3b6b8c337aef7e744e4ae1a7ce240802895bcfeef58396a4`.
- **Change:** 43,517 links removed from 39,671 French rows.
- **Readout:** *p* = (0.00599 − Δ) / 0.00862, where Δ = score − 0.978287. Break-even is at *p* = 0.695.
- **Why this probe:**
  - It is the largest single region that could hold France's excess loss.
  - Legal-form pairs are excluded because they are known to be about 99.9% true and would dilute the reading.

### Upload B: drop T3conf (confident French template-word swaps)

- **File:** `probe-drop_T3conf/matching_results.tsv`, SHA256 `d6add04eadb0bff6f488abc5dedaaa7687ba5d22a77e38c234a0d15b52a61491`.
- **Change:** 9,033 links removed from 8,851 French rows.
- **Readout:** *p* = (0.00122 − Δ) / 0.00184. Break-even is at *p* = 0.666.
- **Why this probe:**
  - It tests whether the multilingual reranker confidently accepts generated same-address word-swap decoys, where the swapped words are semantically related French words.
  - It shares no links with Upload A.

## 5. Decision rules for tomorrow (pre-registered)

| Upload A result | Meaning | Tomorrow |
|---|---|---|
| Δ ≥ +0.0017 (*p* ≤ 0.5) | The uncertain band is mostly wrong | The first file drops T2r, and the French fine-tune focuses on negatives (collisions, swaps) |
| −0.0005 < Δ < +0.0017 | Mixed | Test T1r alone (file ready, SHA `2fc6bd82…`) |
| Δ ≤ −0.0005 (*p* ≥ 0.75) | The band is mostly true | Keep it; France's errors are in confident links or in recall |

| Upload B result | Meaning | Tomorrow |
|---|---|---|
| Δ ≥ +0.0003 | Confident swaps are mostly decoys | Drop them, and add template-swap hard negatives to the French fine-tune |
| Δ ≤ −0.0003 | Swaps are true copies in France | Never use swaps as negatives |
| In between | Inconclusive | No change |

## 6. What we did not pick, and why

- **Colleague 2's U1 and U2, as written:**
  - 58% of U1's links (22,614 of 39,217) are legal-form pairs, which are about 99.9% true on US/India.
  - U2 contains 28,665 legal-form pairs (34%) and 12,074 name-only adds (14%).
  - Both readings would therefore be pushed toward break-even. T1r and T2r are the cleaned versions.
- **Colleague 1's Upload 4** (fill France up to 3.464 links per S1 with 51,890 rejected pairs):
  - The target is too high. US/India themselves predict 3.29–3.36 links at about 96% recall, so a US/India-like France would predict about 3.3.
  - The pairs it adds come from the templated look-alike band, which both of us expect to be about 30% true, so the result would tell us little.
- **Colleague 1's Upload 5** (reassign records by an address tie-break, keeping the link count fixed):
  - Record-level rules applied after scoring failed on US/India (−0.019, −0.24, −0.0001).
  - 89.5% of French links already share their S1's digits, so a digit tie-break cannot separate same-address claimants.
  - House-number, city and name-overlap tests do not separate French copies from decoys.
  - Volume is not the problem: there are 32,022 contested French records.
- **U3** (add the 0.05–0.10 band): Colleague 1 predicts the similar Upload-4 set is about 30% true, and so do I. Confirming that would have little decision value. This is a judgment about information value, not a measured fact.
- **U4** (name-only in France): 12,074 links, worth about ±0.001. Secondary.
- **U5** (exact French no-match rate): it would tighten France's level (±0.005) but would not change any decision.
- **S2** (the French fine-tune, +12,056 / −30,743 links) and **pair_tree_c2 applied to France** (+8,649 / −16,619):
  - Both are whole-model changes that mostly move the same uncertain band as Upload A, so their results cannot be attributed to anything specific.
  - Both files are validated and kept as candidates for tomorrow.
- **Corrections to figures in circulation:**
  - **Probe decode:** 0.846352 (not 0.846) gives US/India 0.98559 and France 0.9368.
  - **Pair tree vs blend:** both figures are correct for their own baselines. The pair tree beats the C2 blend without name-only by +0.006, and the blend *with* name-only by +0.0029. The second is the relevant comparison for France, which already uses name-only.
  - **C2 < 0.005 truth rate:** accepted links with C2 < 0.005 are 73–84% true on B_REST. The 12–25% figure describes the whole tree-yes/C2-no cluster with no better claimant, which is mostly rejected pairs.

## 7. Tomorrow (27 Sep: 5 uploads, and only the last counts)

- **US/India** (gated on the arena, so no uploads needed):
  - collective features;
  - C2-scored new retrieval pairs at C2 ≥ 0.3;
  - union k=3;
  - Qwen, if a stacked gain clears the gate.

  About **+0.001** in total.
- **France:**
  - **Probe-validated edits** from tonight.
  - **Small recall adds:** +5,418 legal-form links and +2,026 unclaimed high-C2 new pairs.
  - **French fine-tune v2**, only if the probes point to learnable decoys. It would continue from C2 step 40k:
    - **Positives:** synthetic French copies built with the measured copy noise.
    - **Negatives:** only the measured decoy types (same name with a different house number; same address with a swapped word).
    - **Replay:** US/India data, so it doesn't forget.
    - **Gate:** B_REST may drop by at most 0.0005.
- **Realistic target: 0.982–0.985.** 0.99 would need a US/India breakthrough (+0.0027 on the arena) that nobody has shown yet.

## Caveats: assumptions and judgment calls in this brief

**Measured:**
- public scores and test country counts;
- all test-side link counts, bands, overlaps and histograms;
- B_REST truth rates, from arena labels;
- file SHAs and validator results.

**Assumed:**
- **France's no-match rate is 5.58%.** The whole decode depends on this.
- **The public subset has the full test's French share.**
- **France follows the generator statistics measured on US/India.** They were measured on US and India only.
- **US/India truth rates transfer to France.** In particular, legal-form pairs are ~99.9% true, which is why they are excluded from both probes.
- **The readout pricing is right.** It is an approximation: the other links are assumed correct, *n* = 1 values are heuristics, and dropped links are treated independently.

**Judgment (mine):**
- choosing A and B over the other proposals, using expected information value;
- the decision thresholds;
- the template-swap definition (≥ 20 S1 names, typo cutoff);
- reading the French swap examples as decoys;
- estimating the new-retrieval gain.

**Not re-verified today:**
- union +0.000275;
- the collective-features CI, which an agent reported.

The author also designed probes A and B, so treat the arguments for them with that in mind.

## 8. Feedback we want

1. **Pricing model.** Is "the other links are correct and complete" acceptable for converting Δ into *p*? As a rough guess (not measured), France has about 6% false and about 5% missing links. Should the per-list gains and losses be corrected for imperfect lists?
2. **Excluding legal-form pairs.** This relies on the US/India truth rate (99.85–99.99%). Is there any reason French legal-form variants (SARL, SAS, EURL, SASU, SCI) would behave differently?
3. **T2r mixes mechanisms:** 16.6k tree-override links, 2.6k other swaps and about 24k moderate-C2 links. Would you rather spend Upload A on the cleaner, smaller T1r (±0.001–0.002) and test the rest of T2r tomorrow?
4. **Second probe.** Is T3conf the right one, or would S2 or pair-tree-on-France tell us more for tomorrow's training?
5. **Template-swap definition.** It relies on "the word appears in ≥ 20 S1 names of the country". French templates have a small vocabulary. Could true French copies land on template words more often than US/India copies do, so that the class is inflated by vocabulary size rather than by decoys?
6. **US/India.** Is there any lever with a plausible +0.0027 on the arena that we haven't measured?

## 9. Reproducibility

**Scripts:**
- `code/business_entity_resolution/diagnostics/20260926_france/`:
  - `france_probe_sets.py`, `drop_swap_confident.py`, `template_swap.py`
  - `france_legal_file.py`, `digit_relation_rates.py`
  - `brest_error_audit_s1.py`, `brest_retrieval_misses.py`
- `code/business_entity_resolution/diagnostics/20260926_clarity/`:
  - `d1_unreached_france.py`, `d3_name_only_orphans.py`

**Pod outputs:**
- `/workspace/az-ml-new/eureka/probes/`: `probe_sets.json`, `drop_T1`, `drop_T1r`, `drop_T2r`, `drop_T3conf`
- `/workspace/az-ml-new/eureka/template_swap_fr.json`
- `/workspace/az-ml-new/distractor_sim/`
- `/workspace/az-ml-new/eureka/clarity/`

**Ledgers:**
- `learnings/c2_h100_run.md`
- `learnings/submissions.json`

**Data used:**
- Test: no labels exist, and none were read.
- Arena: B_REST labels only. SLICE and HOLDOUT-2 were not read.
