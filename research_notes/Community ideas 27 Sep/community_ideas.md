# Community ideas scan: 27 Sep 2026, about 13:35–14:00 IST

**Scope.** I searched the public web: Reddit (r/Btechtards, r/learnmachinelearning), Unstop, LinkedIn/X (via search), Telegram registers, Kaggle and Hugging Face card text, and GitHub READMEs. For GitHub, I listed repos updated on 26–27 Sep and read about 190 README files as text only.

**What I did not do.** I did not clone, download or run any code, notebook, dataset, model or submission. I did not contact anyone and did not follow any payment links.

**Relation to earlier notes.** Most of this material is already in two earlier notes:
- `research_notes/Entity resolution missing big wins/competition_public_intel.md` (08:40)
- `research_notes/Proven wins for business matching/competition_updates.md` (12:29)

This note is the **delta** plus a ranked shortlist. Items marked *(prior note)* are already written up there but are **not** in our "what we already do" list.

**Headline.**
- No team at ≥0.990 has published its method anywhere I could reach.
- No public post reveals a single "France trick".
- The best public evidence on France is two independent leaderboard results (GozonLegendent 0.987, Epic021 0.983). Both point to over-accepted **same-address, one-word "qualifier" swaps** in French.

---

## 1. Ranked ideas not in our current pipeline list

| # | Idea | Source | Claimed effect | Credibility | < 1 h test on our pipeline |
|---|---|---|---|---|---|
| 1 | **French qualifier-swap removal.** Drop a French link when S1 and the record share the normalised street and house number, and the names differ by exactly one non-legal-form token. This is broader than our fixed decoy-word list. | [GozonLegendent TEAM_HANDOFF](https://github.com/GozonLegendent/amazon-ml/blob/main/TEAM_HANDOFF.md), [Epic021](https://github.com/Epic021/amazon-ml-challenge-2026) *(prior note)* | **+0.0011 on the public board** (0.986201 → 0.987326; 8,703 links removed). Epic021's implied France score went ~0.94 → ~0.95. | **High**: two independent board measurements; fits the decoy generator. | Count links in our 0.981022 France output that match the pattern. Remove them and diff. Check the removal set against our decoy-word signatures. Upload only if the removal set is mostly new (>2k links). |
| 2 | **Address-less record assignment.** Decoys almost never lack an address (0.3% vs 4.4% of true copies). Assign each address-less record per record (argmax S1), with a looser threshold. | [NFG README](https://github.com/aditya-bheke/business-entity-resolution) *(prior note)* | +0.0017 on validation; part of NFG's 0.9810 board lineage. | **Med-high** | On TUNE, measure the decoy rate among address-less pool records. If it is ≤1%, sweep a separate threshold for address-less records on TUNE and compare the paired F0.5 difference. |
| 3 | **Density-stressed threshold tuning.** Test has about 23% more pool records per S1 (5.75 vs 4.67). One team estimates about 40% distractors in test vs 26% in train. Retune thresholds after hiding about 20% of TUNE S1s, so their copies become orphan distractors, or after adding near-copy decoys. | [harshgitty58 PROGRESS](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md); **new**: [Sid-techweb README](https://github.com/Sid-techweb/AmazonML-New/blob/main/README.md) (40% vs 26%); rikkicoder *(prior note)* | rikkicoder: **+0.0049 on the board** from density-matched training. NFG: hiding 20% of S1 changed nothing. | **Medium** (conflicting evidence) | Reuse the saved TUNE probabilities. Drop 20% of S1 and re-score, with no retraining. If the best threshold moves by more than 0.02, try one upload at the stressed threshold. |
| 4 | **Per-S1 expected-F0.5 subset selection.** Sort an S1's candidates by probability and keep the top-k that maximises expected F0.5, instead of a fixed threshold. | [harshgitty58 PROGRESS](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md), [tharvin-byte README](https://github.com/tharvin-byte/amazon-ml-challenge-2026) | +0.0003 on validation. No board measurement. | **Medium-low** (small gain; needs calibrated probabilities) | CPU only, on the saved TUNE probabilities. Needs well-calibrated probabilities, so calibrate first (isotonic on TRAIN out-of-fold). |
| 5 | **Digit edit-type features.** Real noise truncates, pads or adds a suffix letter (7402→740, 882→00882, 10216→10216b). Decoys substitute a different number (315→320). This would complement our fixed decoy offsets {1,3,5,…,21}. | [harshgitty58 PROGRESS](https://github.com/harshgitty58/Amazon_ML_Challenge/blob/main/business_entity_resolution/PROGRESS.md) *(prior note)* | Proposed only (+0.0005–0.001 estimated). | **Low-med** | On our false links in TUNE, tabulate each house-number relation (truncation / padding / suffix / substitution / offset) for true copies vs decoys. Proceed only if one relation clearly separates them. |
| 6 | **French acronym rule.** For example, "Dietetique Amis SAS" → "DA" at an identical address. Such acronyms are 10× more common in France. | [NFG EXPERIMENTS](https://github.com/aditya-bheke/business-entity-resolution/blob/main/docs/EXPERIMENTS.md) *(prior note)* | 434/435 held-out pairs correct. | **Medium** | Count French test records whose name is the initials of an S1 name at the same address. Check whether our current output already accepts them. |
| 7 | **Label-free vocabulary repair.** Replace S2/S3 tokens that are absent from the same-country S1 vocabulary with the most frequent S1 token within 1–2 edits. This is usable for French typos. | [vc940 HF card](https://huggingface.co/datasets/vc940/business-entity-resolution-normalized) *(prior note; card text only)* | No score stated. | **Low-med** | Apply only to France and compute the change in top-1 retrieval agreement. It is a retrieval change, so candidate-dependent features would have to be recomputed. That is probably over 1 h, so skip unless the others fail. |
| 8 | **Stricter France-only threshold.** | NFG *(prior note)* | 0.6 → 0.9 improved the board score by ~0.001. Another team saw no gain from threshold changes. | **Mixed** | Cheap probe if uploads remain: raise the France threshold for our vote by one step. |

### Evidence against ideas (do not adopt without a density-stressed check)
- **Sibling/group/cluster features failed to transfer to the board for three teams:**
  - **new**: Sid-techweb's sibling-feature run scored **0.933** on the board and was withdrawn;
  - harshgitty58: validation +0.010 but board only +0.002;
  - mayankgoplani431: validation up, board 0.964 → 0.956.

  This matters for our record-centric GBDT over claimant sets: check it under the density stress described in idea 3.
- **France self-training is mixed:**
  - NFG's US→India simulation: 99.8% of pseudo-labels were correct, yet the score **fell** 0.927 → 0.915;
  - GarvRajput17 reports an estimated France ~0.917 → ~0.945 (prior note);
  - the purvanshjoshi [France doc](https://github.com/purvanshjoshi/business-entity-resolution/blob/main/docs/09_FRANCE_DOMAIN_ADAPTATION.md) (τ_pos 0.92 / τ_neg 0.08, per-country feature z-scoring, "use the India threshold for France") contains **only projected numbers**. Low credibility.

## 2. France-specific findings
- **Test mix: France is 15.0% of test S1s (259,452).** Blanking probes by two teams imply France carries ~15% of the public-board weight, with India and US ~43% each. For 0.97–0.98 teams, France scores about 0.94–0.95 (prior notes). ([HF card, akshatbakshi](https://huggingface.co/datasets/akshatbakshi/amazon-ml-challenge-2026), card only.)
- **Dominant French error:** over-accepted same-address siblings with one qualifier word swapped (idea 1).
- **Other documented French traits:**
  - S1 "twins" (same name and street, different number) are 1.1% of French S1s vs 0.02% elsewhere (NFG);
  - French names are more generic;
  - dotted legal forms without a final dot ("S.A.S") break normalisers;
  - France has more near-duplicate neighbours (6.7 vs 4.5–5.4, rikkicoder);
  - 63.5% of French records receive a parent vs 58–60% elsewhere (harshgitty58).
- **Common normalisations:**
  - accent folding;
  - SARL/SAS/SASU/SCI/EURL/EI/EARL, and Ets/Cie;
  - street abbreviations R/AV/BD/IMP/ALL/CH;
  - `N°`, `bât`;
  - department/region → region code.

  None of these reports a leaderboard gain on its own.
- A **stricter France threshold** and a **CE-verified dense rescue** each helped NFG by ~0.0006–0.001 on the board.

## 3. Leaderboard / rules facts (all second-hand; confirm in the portal)
- **Deadline conflict:**
  - 27 Sep 23:59 IST: [Pratheesh-555](https://github.com/Pratheesh-555/amazon-ml-challenge-2026), Team-AWWS and sikeniga READMEs;
  - 27 Sep 21:00 IST: the Internshala listing (prior note).
  
  **Plan for 21:00 IST.**
- **Upload limits conflict:** "5 submissions per day" ([Pratheesh-555](https://github.com/Pratheesh-555/amazon-ml-challenge-2026), Ganeshbeldar015, HarpalKalankar) vs "unlimited" (harshgitty58, Epic021). GozonLegendent reported "one leaderboard attempt left" at about 11:45 IST.
- **Final ranking:** the public board is a subset of test; the private board is the remainder (mirrored problem statement). Unstop says the private board is "based on your **final solution submission** and validation on the **complete test dataset**" ([Unstop](https://unstop.com/o/1743604)). **Treat the last upload as the one that counts, not the best.**
- **Timeline:** top-50 results on 2 Oct; finale on 7 Oct. Top-10 selection uses the board plus the documentation.
- **Candidate set size:** "Smaller, high-quality candidate sets are reviewed for ranking beyond the score" (Pratheesh-555, Pushkar0997 READMEs, unverified). The official text says `candidate_pairs.tsv` is not scored but is audited.
- **Portal practicalities** ([r/Btechtards thread](https://www.reddit.com/r/Btechtards/comments/1wpo5eb/amazon_ml_challenge_2026_submission_issue/)):
  - only the **team leader** can submit;
  - uploads have hung for 20+ minutes;
  - one user reports the first submission needed the TSV plus the code zip, and later ones only the TSV, with failures on both.
  
  **Upload the final package early.**

## 4. Sources to avoid (not followed)
- **Code sellers:** I found no explicit "selling code" offer for this challenge. Telegram register searches returned only unrelated channels.
- **Fair-play risks: do not use any of these:**
  - public mirrors of the official dataset (HF: akshatbakshi, nithish16, logicalguy, vc940; Kaggle: satwiksps);
  - a public Kaggle notebook (mobeenfatimah, 100+ copies);
  - about 190 public GitHub repos with full pipelines, some shipping pre-trained models and outputs (e.g. Akash-bardia).
