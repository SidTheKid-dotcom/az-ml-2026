# Negative-Side Analysis Manual
## Amazon ML Challenge 2026: how we read a submission, price every link, and decide what is true

**Team:** Techquilla Shots · **Written:** 26 Sep 2026 · **Status:** living reference for the experiment loop

This manual explains the method that predicted our leaderboard scores (0.9637 predicted vs 0.964232 actual; 0.9653 predicted vs 0.964563 actual) and turned exact score changes into counts of right and wrong links. It is meant to be fed to the experiment agents and engineers as `learnings/` material.

It does not change model weights by itself. What it gives the pipeline is:

1. Rules the decision layer should obey.
2. Training targets and sample weights for the matcher and the veto model.
3. A pre-upload checklist and prediction procedure.
4. An attribution procedure after every upload.

Every number here comes from the provided data or our own leaderboard results. No external data was used.

---

## 0. The ten learning points (read these first)

| # | Learning point | Evidence | How to use it in the model |
|---|---|---|---|
| L1 | **Every Source 2/3 record belongs to at most one Source 1 entity.** A record predicted for two entities is at least one guaranteed false link. | Train: 7,638,365 links, 0 records shared. Test: enforcing it took us from 0.956390 to 0.964563 (+14,160 entity-points). | Hard constraint in the final assignment step. Never ship a file with a contested ID. |
| L2 | **Local evaluation on a small sample of entities cannot see conflicts.** Only when every entity competes at once do contests appear. | TUNE/holdout scored 12k of 2.2M train entities. Test scored all 1.73M at once and produced 48,927 contested records. | Build a competitor-complete out-of-fold (OOF) world: all train entities scored against the full train pool, in folds grouped by entity. Tune every decision rule there. |
| L3 | **Removing a link that no other entity claims only pays if it is wrong more than ~22–27% of the time.** Blanket removals failed. | The 30,745 US/India removals were 61–79% correct links, and cost −528 entity-points. | Remove only with link-specific evidence (veto model, exact-key conflict, duplicate inconsistency). No global post-hoc thresholds. |
| L4 | **Adding a link only pays if it is correct more than ~73–79% of the time.** Short lists are the most dangerous place to add. | Adding one wrong link to a list of length 1 costs −0.444; adding a right one gains +0.167. | Additions need calibrated probability ≥ 0.8. Be stricter still on lists of length 0–2. |
| L5 | **Which claimant wins a contested record is worth up to +0.23 points.** Our current winner choice is right only ~71% of the time. | 85.2% of the 95,502 removed losing claims were wrong, so about 14.1k were the true owner. | Train a dedicated contest resolver (Section 7). Drop the record from every claimant when no claimant clears ~0.76 probability. |
| L6 | **France is the weakest country and has no labels.** | Simulated France v1 ≈ 0.889 against ≈ 0.968 for US/India. France had 19.9% of entities holding a contested record. | Use exact-key pseudo-labels (Section 8). Check France label-free diagnostics after every change. |
| L7 | **Most lost recall is the scorer rejecting candidates, not retrieval missing them.** | TUNE: dense K50 retrieval recall 0.984, final recall 0.939. | Retrain the reranker on far more data (only 36k of 2.2M entities were used). Re-score the pairs the gate skipped. |
| L8 | **The exact key is a 99% precise, label-free signal.** | Train precision 0.9912. Test: 76.5% of French entities have a key hit. | Use it as a feature, a tie-breaker in contests, and a source of French pseudo-labels. |
| L9 | **Changes to disjoint entities add up exactly on the leaderboard.** One upload can therefore test several changes if they touch different entities (factorial design). | #2 → #3 = France effect + US/India effect: 0.000026 + 0.000305 = 0.000331. | Design uploads as factorial crosses (Section 5). |
| L10 | **Always read the five-decimal score.** One point of the third decimal (0.001) equals ~1,733 entity-points, about the size of the effects we are testing. | 0.964 vs 0.965 hid a 0.000331 difference. | Record exact scores in `learnings/submissions.json`. |

---

## 1. The metric at entity level

For each Source 1 entity, let:

- **k** = number of IDs we predict
- **T** = number of true matches
- **TP** = number of predicted IDs that are true

Then:

```
F0.5 = 5·TP / (5·TP + 4·FP + FN) = 5·TP / (4·k + T)
```

Special cases:

- T = 0 and k = 0 → 1.0 (a singleton correctly left empty)
- T = 0 and k > 0 → 0.0 (false merge on a singleton)
- T > 0 and k = 0 → 0.0 (abstention)
- TP = 0 with k > 0 → 0.0

The leaderboard score is the mean of this value over all Source 1 entities in the scored subset. Because every entity contributes independently, a submission's score changes only through the entities whose lists changed.

**Entity-point.** One entity-point is one entity's score moving by 1.0. With N = 1,732,544 test entities:

```
Δ leaderboard × N = Σ (per-entity score change)      [entity-points]
0.001 of leaderboard ≈ 1,733 entity-points
```

This assumes the public subset has the same composition as the full test set. The public share is unknown, so treat entity-point counts as full-test equivalents.

---

## 2. Structural facts measured from train (the "invariants")

These are the facts the method treats as true about the data. All were measured on train ground truth and hold for US and India. France is assumed to follow them because the generator appears shared: France's noise patterns and key coverage look alike.

| Fact | Value | Used for |
|---|---|---|
| Records shared between two entities | 0 of 7,638,365 links | One-owner rule; every contest has at most one true owner |
| Singleton rate | 5.58% (US 5.58, India 5.59) | Expected empty rate; singleton prior |
| True matches per entity | mean 3.46, max 11; distribution 0: 5.6%, 1: 5.4%, 2: 17.0%, 3: 24.1%, 4: 21.9%, 5: 14.6%, 6+: 11.4% | List-length prior; cap at 11 |
| Unmatched look-alike records | ~26–27% of Source 2/3 records | Base rate of decoys |
| Country agreement | 100% of links | Country blocking is safe |
| Records per entity in the pool | train 4.68, test 5.75 | Test has more competition than train |
| Matched entities that match both S2 and S3 | ~80.5% | Label-free shape check |

**Rule for using an invariant.** An invariant can justify a hard constraint (L1, the cap) or a prior. It cannot justify removing individual links without per-link evidence (L3).

---

## 3. Label-free diagnostics for any submission file

Run these on every candidate file before upload. None needs test labels.

| Diagnostic | How to compute | Healthy value | What a deviation means |
|---|---|---|---|
| Contested records | Count IDs that appear in more than one entity's list | 0 | Each extra claim is a guaranteed false link |
| Longest list | Maximum k | ≤ 11 | Lists longer than the train maximum contain wrong IDs |
| Empty rate by country | Share with k = 0 | ≈ 5.6–6.0% | Too low means over-merging (France was 4.7% in v1); too high means over-abstaining |
| Mean IDs per entity by country | mean k | ≈ 3.24–3.30 at our precision level | France above US/India means France dilution |
| Exact-key agreement | Share of key links (Section 8) present; count of key links given to a different entity | Present ≥ 97%; conflicts falling | Conflicts point to wrong winners; missing links point to recall misses |
| S2+S3 co-occurrence | Share of non-empty lists with both S2 and S3 IDs | ≈ 80% | Low in one country means one-source dilution or misses |

v1 profile, for reference:

| Country | Entities | Empty | Mean IDs | Entities holding a contested ID | Lists ≥ 9 |
|---|---|---|---|---|---|
| France | 259,452 | 4.7% | 3.55 | 19.9% | 3,638 |
| India | 809,986 | 5.9% | 3.26 | 2.3% | 1,253 |
| US | 663,106 | 5.7% | 3.30 | 1.3% | 962 |

---

## 4. The price of a single link

This is the core of how the method decides whether a change should help.

### 4.1 Formulas

Take an entity whose other links are all correct.

| Action | Before | After | Change |
|---|---|---|---|
| Remove a **true** link from a perfect list of length k (T = k) | 1.0 | 5(k−1)/(4(k−1)+k) | negative |
| Remove a **wrong** link from a list of length k with one wrong ID (T = k−1) | 5(k−1)/(4k+k−1) | 1.0 | positive |
| Add a **true** link to a perfect list of length k missing one (T = k+1) | 5k/(4k+k+1) | 1.0 | positive |
| Add a **wrong** link to a perfect list of length k (T = k) | 1.0 | 5k/(4(k+1)+k) | negative |

### 4.2 Table by list length

| k (list length before the change) | Remove true | Remove wrong | Add true | Add wrong | Removal pays if P(wrong) > | Addition pays if P(true) > |
|---|---|---|---|---|---|---|
| 1 | −1.000 (entity becomes empty) | +1.0 if singleton | +0.167 | −0.444 | — | 0.727 |
| 2 | −0.167 | +0.444 | +0.091 | −0.286 | 0.273 | 0.759 |
| 3 | −0.091 | +0.286 | +0.062 | −0.211 | 0.241 | 0.771 |
| 4 | −0.062 | +0.211 | +0.048 | −0.167 | 0.229 | 0.778 |
| 5 | −0.048 | +0.167 | +0.038 | −0.138 | 0.222 | 0.782 |
| 6 | −0.038 | +0.138 | +0.032 | −0.118 | 0.218 | 0.785 |
| 7 | −0.032 | +0.118 | +0.028 | −0.103 | 0.215 | 0.787 |
| 8 | −0.028 | +0.103 | +0.024 | −0.091 | 0.213 | 0.788 |

Break-even formulas:

- A **removal** pays when `P(wrong) > |remove true| / (|remove true| + remove wrong)`.
- An **addition** pays when `P(true) > |add wrong| / (add true + |add wrong|)`.

### 4.3 What this means for the model

- **Removal is cheap to get right, and addition is expensive.** A link we are unsure of is worth removing once it is more than ~25% likely to be wrong. It is worth adding only when it is more than ~75% likely to be right.
- **Removing the only link of a list is a special case.** If the entity really is matched, removing its only link zeroes it (−1.0). If it is a singleton, the same removal restores it to +1.0. This decision must use the probability that the entity is a singleton, not the link score alone.
- **Averages we measured on our real files**, using actual list lengths:

| Set of links | Average value per link if true | Average value per link if wrong |
|---|---|---|
| 95,502 losing claims removed by the one-owner rule | −0.107 | +0.231 |
| 30,745 US/India links removed by blanket rules | −0.086 | +0.206 |
| 4,963 US/India links added | +0.161 | −0.389 |

**Training implication (cost-weighted learning).** Weight each training example by the value at stake:

- For a negative pair, weight = |add wrong| for the list it would join. Pairs attached to singletons and short lists should weigh 0.3–1.0; pairs on long lists 0.09–0.14.
- For a positive pair, weight = add true for its list.

This teaches the matcher that the dangerous mistakes are wrong links on short lists and on singletons.

---

## 5. Turning a leaderboard change into link accuracy

### 5.1 Additivity

If two changes touch disjoint sets of entities, their effects on the leaderboard add exactly. Countries never share records (100% country agreement), so changes limited to one country never interact with changes in another.

This lets one upload test two ideas: apply idea A to France and idea B to US/India, then compare with uploads that cross them.

**Worked example (our four uploads):**

| | France = the #2 rules | France = pure one-owner |
|---|---|---|
| US/India = the #2 rules | #2: 0.964232 | #4: 0.964258 |
| US/India = pure one-owner | not uploaded | #3: 0.964563 |

- France effect: #4 − #2 = **+0.000026**
- US/India effect: #3 − #4 = **+0.000305**
- Check: #3 − #2 = 0.000331 = 0.000026 + 0.000305 ✓

### 5.2 Inverting a change into the precision of the changed set

Let a change remove a set R of links and add a set A. For each link, the Section 4 formulas give its value if it was true (v_true) and if it was wrong (v_wrong), using that entity's actual list length. If a fraction p_R of R was true and a fraction p_A of A was true:

```
Δ leaderboard × N = Σ_R [ p_R·v_true + (1−p_R)·v_wrong ] + Σ_A [ p_A·v_true + (1−p_A)·v_wrong ]
```

With one unknown, this solves exactly. With two unknowns, it gives a line of possible combinations.

**Example 1: the one-owner rule (v1 → #3).**

- Observed: +0.008173 × 1,732,544 = **+14,160** entity-points.
- If all 95,502 removed claims had been wrong, the gain would have been +18,104. If some were true, each true one costs 0.107 relative to that.
- Solving: 85.2% of removed claims were wrong, so **14.8% (~14,100) were the true owner**.
- Of the 48,927 contests, the kept winner was therefore wrong about 29% of the time, and right about 71%.
- Headroom from perfect winner choice: 18,104 − 14,160 = 3,944 entity-points ≈ **+0.23 points**.

**Example 2: the US/India blanket rules (#4 → #3).**

- Observed: −0.000305 × N = **−528** entity-points for having the rules.
- Two unknowns (precision of the 30,745 removals and of the 4,963 additions) give this trade-off:

| If additions were correct | Then removed links were correct |
|---|---|
| 20% | 61% |
| 50% | 70% |
| 80% | 79% |

Removals only pay when the removed links are at most about 78% correct (removal break-even ≈ 22% wrong). Every row is at or past that line, so the removals were at best neutral and more likely harmful. Do not repeat blanket removals.

### 5.3 Assumptions in the inversion (read before trusting a number)

- **The other links in each changed entity are assumed correct.** Real lists have some wrong IDs and missed matches, so true per-link values are slightly smaller. The implied precisions are approximate, typically within about ±5 points.
- **The public subset is assumed to mirror the full test set.** If it over-represents one country, the entity-point scale shifts, but the sign does not change.
- **Each contest is assumed to have at most one true owner** (L1).

---

## 6. The simulator that predicts a score before upload

### 6.1 What it models

It builds a plausible hidden truth, many times over, that is consistent with everything we know, and scores both the baseline file and the candidate file against it.

1. **Undisputed links.** Each is true with a country-specific precision: a_US, a_IN, a_FR.
2. **Contested records.** Each has a true owner among its claimants with probability b. If it has one, the claimant kept in the candidate file is that owner with probability w.
3. **Missed matches.** Each entity with at least one true link gets extra missed links drawn from Poisson(λ), with λ = 0.21 to match holdout recall 0.939.
4. **Entities with links but no true link.** Each is a singleton with probability σ.
5. **Entities predicted empty.** Each is a singleton with probability s_e ≈ 0.9. Otherwise it has at least one missed match.
6. **Removed and added links.** Links removed from undisputed records are true with probability r. Added links are true with probability d.
7. **Scoring.** Score every entity in both files with the exact metric. Report the mean overall and per country.

### 6.2 Calibration anchors

| Anchor | Value | Fixes |
|---|---|---|
| Sealed holdout, US/India | ≈ 0.968 (precision 0.987, recall 0.939) | a_US = a_IN ≈ 0.99, λ ≈ 0.21 |
| v1 leaderboard | 0.956390 | a_FR ≈ 0.95, which implies France v1 ≈ 0.889 |

### 6.3 Sweep and track record

The unknowns are w, r, d, b and σ. Sweep each over a plausible range and report the central value and the range. Do not report a single number.

| Upload | Predicted (central, range) | Actual |
|---|---|---|
| #2 | 0.9637 (0.959–0.968) | 0.964232 |
| #3 | 0.9653 (after recalibration) | 0.964563 |

The simulator runs slightly optimistic, by about 0.0005–0.0007. Subtract ~0.0006 from future central predictions until more uploads recalibrate it.

### 6.4 Error bill for #3 (simulated, ≈ 3.5 points lost)

| Bucket | France | India | US | Total |
|---|---|---|---|---|
| Dilution: wrong IDs on matched entities | 0.68 | 0.38 | 0.31 | **1.37** |
| Partial miss: true IDs missing | 0.19 | 0.60 | 0.48 | 1.26 |
| Abstention: matched entity predicted empty | 0.14 | 0.30 | 0.23 | 0.67 |
| Singleton false merge + all-wrong lists | 0.09 | 0.04 | 0.03 | 0.16 (holdout suggests ≤ 0.37) |

This bill depends on the model. Use it to rank targets, not as measured fact.

---

## 7. Contest resolution (the +0.23-point target)

### 7.1 The decision

For a record claimed by entities E1…Em, the options are to give it to one claimant or to drop it from all of them.

- Keeping it with claimant Ei is worth p_i·(add true for Ei's list) − (1 − p_i)·|add wrong for Ei's list|.
- Dropping it from all claimants is worth 0 relative to that.

So keep the best claimant only if its probability p_i is above the addition break-even for its list, **≈ 0.73–0.79** (Section 4.2). Otherwise drop the record from everyone.

Our current top-blend-score rule is right about 71% of the time on average. That is below break-even, but the average hides wide variation: confident contests are probably above 90% and near-ties probably near 50%.

### 7.2 What to train

Train a **contest resolver** on the competitor-complete OOF world (L2):

- **One example per (contest, claimant).** Target: is this claimant the true owner?
- **Features:**
  - blend score, tree score, and rank among claimants
  - gap to the runner-up claimant
  - number of claimants
  - exact-key match to this claimant (Section 8)
  - whether the record's near-duplicates in the same source are already assigned to this claimant
  - the claimant's list length and score distribution
  - country
- **Output:** calibrated p_i per claimant. Apply the rule in 7.1 with the list-specific break-even.

**Measure:** OOF macro F0.5 on the competitor-complete world, and the number of winners changed on test.

**Keep if:** OOF rises and the leaderboard beats 0.964563.

---

## 8. The exact-key channel

**Definition.** Within a country, key = the name (lowercase, accents folded to ASCII, punctuation removed, legal suffixes removed) plus all digits of the address concatenated in order. Only Source 1 keys that occur exactly once are used.

The legal suffixes removed are: the, inc, incorporated, corp, corporation, co, company, llc, ltd, limited, pvt, private, llp, plc, sarl, sas, sasu, sa, sci, eurl, gmbh.

**Measured on train:**

- 2,195,877 key links
- precision **0.9912** (India 0.9842, US 0.9955)
- covers 28.5% of all true links; 57.3% of entities have at least one hit

**On test, compared with submission #2:**

| Country | Key links | Entities with a hit | Share of entities | In #2 | Missing from #2 | Key says a different owner | Hits on entities #2 left empty |
|---|---|---|---|---|---|---|---|
| France | 463,220 | 198,476 | **76.5%** | 97.3% | 12,389 | 3,561 | 502 |
| India | 758,744 | 434,706 | 53.7% | 97.8% | 16,958 | 5,913 | 1,768 |
| US | 686,272 | 396,232 | 59.8% | 99.7% | 2,277 | 1,038 | 144 |

**Uses:**

1. **Feature** for the matcher, the veto model and the contest resolver.
2. **Additions.** A missing key link that no other entity claims can be added, since its ~99% precision is far above the ~0.77 addition break-even.
3. **French pseudo-labels.** About 463k French pairs at an expected ~99% precision. This precision was measured on US/India only, so spot-check a sample of French key pairs by eye before training on them.
4. **France pseudo-holdout.** Recall on the French key pairs is an upper bound on true French recall, because key pairs are the easy ones.

---

## 9. What the method treats as true, ranked by strength

| Level | Type of claim | Example | How much to trust it |
|---|---|---|---|
| 1 | Arithmetic identity | Additivity across disjoint entities; the per-link formulas | Exact |
| 2 | Measured on train labels | One owner per record; 5.58% singletons; key precision 0.9912 | Holds for US/India; assumed for France |
| 3 | Inverted from leaderboard scores | 85.2% of losing claims wrong; US/India removals 61–79% true | Approximate (Section 5.3); sign is reliable |
| 4 | Simulated | Error bill; France v1 ≈ 0.889; score predictions | Model-dependent; use ranges and recalibrate after every upload |
| 5 | Hypothesis | Size of gain from contest resolver, retraining or French pseudo-labels | Unproven until measured on competitor-complete OOF and the leaderboard |

Never promote a level-4 or level-5 claim to a decision rule without a measurement at level 2 or 3.

---

## 10. Protocol for every upload

1. **Diff** the candidate against the current best, per country: links removed, links added, owner changes, entities newly empty or filled, contested records, longest list.
2. **Classify** every changed link: contested or not, exact-key or not, list length.
3. **Price** the change with the Section 4 table at the break-even and at the expected precision.
4. **Predict** with the simulator: central value and range. Subtract ~0.0006 for its current optimism.
5. **Design factorially.** When testing two ideas, confine each to a different country so a follow-up cross can separate them.
6. **Upload** and record the **five-decimal** score in `learnings/submissions.json`.
7. **Attribute** the result with Section 5 and solve for the implied precision of the changed sets.
8. **Recalibrate** the simulator and update the error bill.
9. **Log** the result as kept or rejected, with its measured change. Rejected ideas stay in the log.

---

## 11. How to feed this into training

| Component | Target or signal | Weight or threshold from this manual |
|---|---|---|
| Matcher (reranker) | True link or not | Sample weight = value at stake (4.3); hard negatives: same name with same digits but a different unit, same address with a different business, current wrong contest winners |
| Veto model | "Net-harmful": removing the link raises the entity's score in OOF | Veto when P(harmful) is above the removal break-even for the list (~0.22–0.27) |
| Contest resolver | Is this claimant the true owner? | Keep only when p > the list's addition break-even (~0.73–0.79); otherwise drop from all |
| List builder | Choose each entity's list to maximise expected F0.5 | Uses calibrated link probabilities, the list-length prior (Section 2) and the singleton probability |
| Singleton model | Is this entity a singleton? | Features from the whole candidate list: top score, gap between first and second, count, spread, whether the top candidate is claimed elsewhere |
| France | Pseudo-labels from exact keys; noise synthesis measured from key pairs | Check France label-free diagnostics after every change |
| Evaluation | Competitor-complete OOF on all 2.2M train entities | Keep a change only if OOF rises without lowering singleton retention, and the leaderboard does not contradict it |

---

## 12. Known limits

- The public leaderboard is a subset of test; the private board decides the final rank. Prefer changes whose OOF gain is stable across folds.
- The inversion assumes the other links in a changed entity are correct (5.3).
- France invariants are assumed, not measured.
- Exact-key precision in France is unverified.
- The simulator has been checked against two uploads. Its ranges matter more than its central value.
- Using pseudo-labels derived from test data is transductive. It uses only provided data, but confirm it is acceptable and describe it in the methodology document.

---

## Appendix A: upload history

| # | File | Contents | Predicted | Leaderboard |
|---|---|---|---|---|
| 1 | v1 | Flat blend threshold 0.775; no one-owner rule | — | 0.956390 |
| 2 | (1) | v1 + one-owner (542 contests dropped from all) + 34,797 undisputed removals + 5,672 additions + cap at 11 | 0.9637 | 0.964232 |
| 3 | (2) | **v1 + one-owner only** | 0.9653 | **0.964563** |
| 4 | (3) | France from #3 + US/India from #2 | — | 0.964258 |

## Appendix B: glossary

| Term | Meaning |
|---|---|
| Entity | A Source 1 business |
| Link | One (entity, Source 2/3 ID) pair in a submission |
| Contested record | A Source 2/3 ID predicted for two or more entities |
| Entity-point | One entity's score changing by 1.0; 0.001 of leaderboard ≈ 1,733 entity-points |
| Break-even | The link precision at which a removal or addition has zero expected effect |
| Competitor-complete OOF | Out-of-fold scores for all train entities computed together against the full train pool, so contests occur as on test |
| Exact key | Normalised name plus address digits, within country (Section 8) |
| Dilution | Wrong IDs on an entity that has true matches |
| Abstention | A matched entity predicted empty |
