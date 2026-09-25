# Amazon ML Challenge 2026: problem-statement video (narration and slide text)

Source: the official 5:56 problem-statement video. The narration was transcribed and cleaned against the slides; the slide text was transcribed from video frames. Where anything differs, the written statement (`kb/2026/statement.md`) is authoritative.

## Complete cleaned spoken transcript

Welcome to the Amazon ML Challenge 2026. This year’s problem is **Business Entity Resolution**, a fundamental and widely encountered problem in real-world data. You will be given business records that arrive from three independent sources, each noisy and inconsistent, and the goal is to determine which of them describe the same real-world business. In this video, I will walk you through the problem statement, the dataset, the scoring, the submission format, and how entries are judged.

Let’s start with where this data comes from. Consider a business signing up on Amazon Business. At sign-up, we capture its core details, such as the business name and address. To build a richer picture of that same business, we pull in additional information from other data providers. Each of these sources typically comes from a different data vendor, with its own formats and conventions. The difficulty is that these external sources share no common identifier with ours, so the only fields we can rely on are the business name and address. For this challenge, we have deliberately limited the data to names and addresses.

Here is the core difficulty: the same real-world business is described differently by each source. One vendor may write Acme Robotics Incorporated, another abbreviates the address, and a third references a nearby landmark. Source 1 is our clean, deduplicated reference list. Sources 2 and 3 are the noisy fragments that must be reconciled against it, and there is no shared identifier linking them. Entity resolution is the task of linking these records together, establishing that differently written records all refer to the same business. For every entity in Source 1, the goal is to find all of its matching records in Sources 2 and 3. A Source 1 entity may match many records, exactly one, or none at all.

Comparing every Source 1 record against every Source 2 and Source 3 record would be far too expensive at scale, so we first apply blocking, sorting records into buckets using a cheap key built from both the name and the address, so that records likely to match land in the same bucket. Each color here is one such block. Notice that every record carries its own ID, a noisy name, and an address, and records can group either through a similar name or through a shared address. We follow the orange block, built around Acme Robotics, and the records in it become the candidate matches for that entity. Blocking favors recall, so the bucket also pulls in lookalikes: a business with a similar name at a different address and a different business that happens to share an address. The matching model removes those in the next step. Blocking narrows an enormous number of possible comparisons down to a manageable set of candidate pairs.

Finally, a matching model scores each candidate pair and keeps only the true matches, discarding the rest. That produces the two outputs of this challenge: the candidate pairs from your blocking stage and the final matching results, which is the file scored on the leaderboard.

You are provided with two datasets. The training set contains records across all three sources, together with the ground-truth labels, a file that specifies for each Source 1 business exactly which Source 2 and Source 3 records it matches. The test set contains the same three sources but no labels, and this is what you generate predictions for. A note on the label format: the ground truth is stored as one row per Source 1 entity. Its ID maps to a comma-separated list of all of its matching IDs, so a single row carries that entity’s complete match set, and the list is empty when the entity matches nothing. This mirrors exactly what you submit. One practical reminder: all files are tab-separated, so read them with an explicit tab separator; otherwise, the columns will not parse correctly.

There are two deliverables. Throughout the challenge, you upload a single file, `matching_results.tsv`, containing one row per Source 1 entity with its predicted matches. This is the only file scored on the leaderboard. At the close of the challenge, every team also submits a single archive. It contains the final matches, along with `candidate_pairs.tsv`, the candidate set your blocking stage produced before the model narrowed it down. This file is not scored, but it is used to audit the quality of your blocking. The archive also includes your complete runnable pipeline and a methodology document describing your approach. The packages of the top teams are reviewed in detail before final rankings are confirmed. One reminder: run the provided validation script before submitting so a simple formatting error does not cost you a submission.

Finally, how entries are judged. Submissions are scored using the **macro F0.5** metric, which weights precision twice as heavily as recall. In practical terms, incorrectly merging two different businesses is penalized roughly twice as much as missing a true match, so when in doubt, it is safer not to merge.

A few recommendations. First, understand singletons. A singleton is a Source 1 entity that has no matching record in Source 2 or Source 3. If you correctly predict an empty list for it, you earn a full score of one on that entity. If you predict any match, you score zero. So identifying the businesses with no match is just as important as finding the ones that do match. Second, your blocking strategy sets the ceiling on the recall you can achieve, so invest in it first, because you cannot match a record you never consider. And third, pay attention to region-specific patterns in both names and addresses.

One firm rule: this is a pure machine learning challenge, so external databases, APIs, and lookups are strictly prohibited. Use only the provided data. That is the challenge. Build something you are proud of, resolve those entities, and enjoy the process. We are excited to see what you create. All the best.

## On-screen slide transcription (all six slides and distinct builds)

### Slide 1/6 — Business Entity Resolution

> AMAZON ML CHALLENGE 2026  
> Business Entity Resolution  
> Determine which business records, arriving from three independent sources with no shared identifier, describe the same real-world business.

### Slide 2/6 — The problem: From sign-up to matched records

This slide changes through several illustrative diagrams:

1. **Amazon Business, create account.** BUSINESS NAME: “Acme Robotics Inc”; BUSINESS ADDRESS: “500 Market St, San Jose, CA”. PHONE and EMAIL are blank. Caption: “Phone and email exist in reality. This challenge uses name and address only.” Below: “A business signs up. We capture its name and address.”
2. **Three source representations.** Source 1, “reference, deduplicated”: Acme Robotics Inc.; 500 Market St, San Jose. Source 2, “vendor A format”: Acme Robotics Incorporated; 500 Market Street, San Jose CA. Source 3, “vendor B format”: Acme Robotics; Nr. City Hall, San Jose. Caption: “One business, three representations, each from a different data vendor, with no shared ID.”
3. **Entity links.** Source 1 reference Acme Robotics Inc. links to Source 2 match Acme Robotics Incorporated and Source 3 match Acme Robotics, City Hall; Source 2 look-alike Acme Bakery LLC is rejected. Caption: “Entity resolution links every Source 1 entity to its matches: zero, one, or many.”
4. **Blocking example.** Three columns labeled SOURCE 1, SOURCE 2, SOURCE 3. Their records have IDs, names and addresses. Orange Acme block: `S1-732914` Acme Robotics Inc (500 Market St, San Jose); `S2-118820` Acme Robotics Inc (500 Market Street, San Jose); `S2-540221` Acme Robotix (12 Elm Rd, San Jose); `S3-905477` Acme Robotics (Nr City Hall, San Jose); `S3-063118` Acme Bakery (500 Market St, San Jose). Green Delta examples: `S1-889301` Delta Foods (8 Oak Ave, Austin), `S2-397155` Delta Foods Co (8 Oak Avenue, Austin), `S3-651230` Delta Foods Ltd (Oak Ave, Austin). Blue Bright Cafe examples: `S1-410562` Bright Cafe LLC (22 Pine St, Reno), `S2-663049` Bright Cafe (22 Pine Street, Reno). Other examples: `S1-205774` Zen Traders (4 Hill Rd, Boise); `S3-472088` Kappa Motors (90 Lake Dr, Fargo). Acme block `candidate_pairs.tsv`: `S1-732914` maps to `S2-118820`, `S2-540221`, `S3-905477`, `S3-063118` — “one block, one shortlist.” Caption: “Blocking buckets records by a cheap key built from name and address, so similar names or shared addresses group together, matches and look-alikes alike.”
5. **Matching stage.** Four candidate pairs for `S1-732914` flow through a matching model; `S2-540221` and `S3-063118` are rejected. The scored `matching_results.tsv` example is `S1-732914 → S2-118820, S3-905477`, “one row per Source 1 entity.” Caption: “The model filters candidates and keeps only true matches, producing matching_results.tsv.”

### Slide 3/6 — The dataset: Train to learn, test to prove it

> **Training set:** Records across all 3 sources. Ground-truth labels: `train_ground_truth.tsv`. Use it to build and evaluate your model.  
> **Test set:** Same 3 sources, no labels. Produce a prediction for every Source 1 entity. This is what the leaderboard scores.  
> Ground truth is **one row per Source 1 entity**: its ID maps to a **comma-separated list** of all matching Source 2 and Source 3 IDs, so a single row holds the entity’s full match set (empty when it matches nothing). All files are **tab-separated (`.tsv`)**; read with `sep="\t"`.

### Slide 4/6 — What you submit: A file, then a package

> **During the challenge:** Upload `matching_results.tsv`. The only file scored on the leaderboard. One row per Source 1 entity; empty list means no match.  
> **Final package (`.zip`):** `output/matching_results.tsv` (final matches); `output/candidate_pairs.tsv` (blocking set, audited); `code/…` (runnable pipeline); `Documentation_template.md` (methodology).  
> Run `utils/validate_submission.py` to catch formatting errors before submitting.

### Slide 5/6 — Judging & tips: Scored on macro F0.5, precision-weighted

> `F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`  
> A false merge costs about twice as much as a miss, so when unsure, do not merge.  
> Singletons are Source 1 entities with no match in Source 2 or 3. Predict an empty list for them and you earn a full 1.0; predict any match and you score 0.  
> Blocking sets your recall ceiling, so invest there first. You cannot match a record you never consider.  
> Account for region-specific patterns in both names and addresses.  
> **No external data lookup. External databases, APIs, and geocoding services are strictly prohibited. Use only the provided data.**

### Slide 6/6 — All the best

> AMAZON ML CHALLENGE 2026  
> All the best  
> Build something you are proud of, resolve those entities, and enjoy the challenge. We are excited to see what you create.

## Covered elsewhere

The video omits exact TSV headers, the upload limit, model limits and leaderboard details. Those are in `kb/2026/statement.md` (official README) and `kb/2026/guidelines.md` (official guidelines).
