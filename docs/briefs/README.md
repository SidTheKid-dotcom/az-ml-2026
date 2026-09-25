# Shared context for external CPU agents (read fully before your brief)

You are helping team Techquilla Shots in the Amazon ML Challenge 2026
(business entity resolution). Several agents work in parallel; your brief gives
you one question that nobody else is working on. Answer it with measurements.

## The task

- Source 1 (S1): clean, deduplicated businesses. Sources 2 and 3 (S2/S3): noisy
  records. For every S1, predict the S2/S3 records that are the same business.
- Metric: for each S1, F0.5 = 5·TP / (4·k + T), with k = predicted count, T =
  true count; T = 0 and k = 0 scores 1; T = 0 and k > 0 scores 0. The score is
  the mean over all S1s. Precision counts twice as much as recall.
- Train: US + India with labels. Test: US + India + **France (no labels)**.
- One point of the third decimal (0.001) ≈ 1,733 entity-points on test.

## Facts measured on train labels

- Every S2/S3 record belongs to **at most one** S1 (7,638,365 links, none shared).
- 5.6% of S1s have no match. Matches per S1: mean 3.46, max 11.
- 26% of S2/S3 records are unowned look-alikes or records of absent businesses.
- Country always agrees between an S1 and its matches.
- Test has 5.75 S2/S3 records per S1; train has 4.68.
- Empty-address records (blank, `<NULL>`, `##`) are 4.4% of owned records but
  0.3% of unowned records: an empty-address record is ~98% owned by someone.
- Look-alikes mostly substitute digits at the same length (a different unit or
  house number); true copies mostly drop or insert digits.
- About half of unmatched records share a normalized name with some S1; half
  of those also share its full digit set.
- France: names are templated ("Nantes Club EURL", "Nantes Conseil EURL"),
  house numbers repeat, and the street name is the discriminator.

## Current system (v1) and scores

E5-small dense retrieval (top-50 per S1, country-blocked) → CatBoost tree
(gate) → BGE cross-encoder reranker → blend ≥ 0.775 → one owner per record.
Public leaderboard: our best **0.964563**; leaders 0.988; ranks 4–14 at
0.984–0.985. On 12,000 tuning S1s: F0.5 **0.968992**, precision 0.987, recall
0.939, 494 false links; 655 true links never retrieved (69% India); 1,885
retrieved but rejected.

## Who is doing what (do not duplicate)

- **GPU controller (Claude):** the GPU pod; the "arena" (all 2.07M training S1s
  scored at once); a larger cross-encoder; ClaimNet (a model that sees a record
  and all its claimant S1s at once); the final resolver; uploads.
- **agent-D:** the name-only rule (already measured: +0.003962 on the tuning
  set), the upload file for it, and the decision library (expected-F0.5 list
  decoding, one-owner assignment, bootstrap).
- **agent-F:** features for digit relations and edit types, street tokens,
  exact name+digits key, name/address multiplicity, French normalization;
  French pseudo-labels; label-free diagnostics of submission files.
- **Brief A (Cursor + Grok):** generator forensics.
- **Brief B (opencode GPT-6 sol):** no-match businesses and list size; train/test shift.
- **Brief C (opencode GPT-6 astra):** India native-script recall.

## Things already tried that failed (do not repeat)

Blanket link removals or additions; country-specific thresholds; fuzzy
name-only additions (−0.00093); "unrelated name at a matching address" rule
(only 51% precise); cluster-mate retrieval (+3 links); composite-key override
on existing candidates (0 links); source-balance bonus (interval crosses 0);
averaging trees; pure cross-encoder without the tree; stricter abstain gates;
evaluating on a sampled target pool (scores looked like 0.98, were 0.83);
choosing decision rules on a 12k sample where contests are invisible.

## Data (local machine)

- Raw files: `/home/sidvish/Documents/Work/Personal/az-ml-new/data/raw/dataset/train/`
  (`train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`,
  `train_ground_truth.tsv`) and `.../dataset/test/` (`test_source{1,2,3}.tsv`).
  Columns: `entity_id, business_name, business_address, country`; truth:
  `source1_entity_id, matched_entity_ids` (comma-separated, empty = no match).
  Read with tab separator; all columns as strings.
- v1 scores on 12,000 tuning S1s (development data):
  `/home/sidvish/Documents/Work/Personal/az-ml-v2/artifacts/v2_p1_decision/tune_pairs.parquet`
  (600,000 pairs: `left_idx, right_idx, source1_entity_id, candidate_entity_id,
  candidate_source, original_tree_probability, original_blend_probability,
  was_ce_scored, is_truth_for_scoring_only, baseline_accept, …`) and
  `tune_references.parquet` (`source1_entity_id, country, true_count, …`).
  Baseline rule: accept `original_blend_probability >= 0.775`.
- v1 scores on all 86,627,200 test pairs (label-free):
  `/home/sidvish/Documents/Work/Personal/az-ml-new/artifacts/v1_test_scores/`
  (`scored_pairs.parquet` with `left_idx, right_idx`; `probabilities.npy` blend,
  0 where not reranked; `base_probabilities.npy` tree) and `dense_test/`
  (`reference.parquet` rows = left_idx, `pool.parquet` rows = right_idx).
- Best submission file: `/home/sidvish/Downloads/assignment-only-lf/matching_results.tsv`.
- Deeper reading: `az-ml-new/learnings/negative_side_analysis_manual.md`,
  `az-ml-new/docs/PLAN_v2.md` §1 and §5. Earlier analysis scripts:
  `/tmp/claude-1000/-home-sidvish-Documents-Work-Personal-az-ml-new/41da6ea4-f15c-42cd-93af-ae21d9c57893/scratchpad/*.py`.

## Rules (hard)

1. **CPU only.** Never SSH anywhere. Never touch the GPU pod.
2. **Resources:** this machine has 8 cores / 14 GB and other agents run on it.
   Use at most **2 threads** (`OMP_NUM_THREADS=2`, `n_jobs=2`) and at most
   **2.5 GB RAM**; run under `nice -n 10`. Load only the columns you need;
   filter early; process in chunks. If `free -g` shows < 3 GB available, wait.
3. **Read-only** on `az-ml-new` and `az-ml-v2`. Never run git commands there.
   Write only under `/home/sidvish/Documents/Work/Personal/ext-experiments/<your-brief-letter>/`.
4. Python: `/home/sidvish/Documents/Work/Personal/az-ml-new/.venv/bin/python`
   (pandas 3, pyarrow, numpy, scipy, sklearn, rapidfuzz, catboost, unidecode).
   Do not pip-install into it; make your own venv under your folder if needed.
5. **No external data**: no internet lookups of businesses, places or
   addresses; no downloaded datasets. Libraries are fine. Vocabularies must be
   derived from the supplied files.
6. **Sealed holdout:** never use ground truth for the S1 IDs listed in
   `/home/sidvish/Documents/Work/Personal/az-ml-v2/artifacts/v2_holdout2/reserved_ids.txt`.
   Drop them before any label-based analysis.
7. **No leakage in features:** a feature may use scores, text and label-free
   population statistics (for example, how many S1s share a name). It may not
   use who owns a record or any other label.
8. Report the population and sample size behind every number. The tuning set is
   reused development data; say so. Do not claim French accuracy.

## Deliverable

In your folder:
- `report.md`: the question; what you tried (including failures); a table of
  findings with numbers, population and n; your recommendation; how confident
  you are and why; exactly what it would change in our system.
- `results.json`: the key numbers.
- Scripts that reproduce every number (runnable end to end).

Keep going through your brief's directions in order of expected value; stop a
direction as soon as the numbers say it is weak, and record it as a failure.
