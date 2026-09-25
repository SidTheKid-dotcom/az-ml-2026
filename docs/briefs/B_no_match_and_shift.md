# Brief B — No-match businesses, list size, and train→test shift (opencode GPT-6 sol)

Read `docs/briefs/README.md` first. Your folder:
`/home/sidvish/Documents/Work/Personal/ext-experiments/B/`.

## Part 1 — No-match businesses (main)

**Question:** which label-free, business-level signals tell that an S1 has no
true match even though its best candidate looks good? Wrongly merged no-match
businesses cost v1 4.7% (tuning) to 6.7% (holdout) of singletons; top teams
must be at ≤ 2%. Each 1% ≈ 0.056 points.

**Data:** `tune_pairs.parquet` + `tune_references.parquet` (12,000 S1s, 600k
pairs with v1 blend/tree scores, labels, `true_count`), raw text for those S1s
and candidates, and label-free population statistics you compute from the full
train S1 file (for example, how many S1s share a normalized name).

**Method:**
1. Describe the no-match S1s that v1 merges (T = 0, v1 accepts ≥ 1) against
   no-match S1s v1 correctly leaves empty and against matched S1s with one
   accepted link. Features per S1 (all label-free): top blend, second blend,
   gap, number of candidates above 0.5 / 0.775 / 0.9, tree scores, whether the
   top candidate's digits differ from the S1's at the same length (look-alike
   fingerprint), whether support comes from one source only (S2 or S3), address
   kind of the top candidate, S1 name multiplicity, S1 address multiplicity,
   candidate list length, text-similarity of top candidate (rapidfuzz).
2. Business-level classifier (logistic regression and small CatBoost), 5-fold
   cross-validation grouped by S1, no feature using any label other than the
   target. Report AUC, and the precision/recall of "predict empty" at several
   thresholds.
3. Simulate: set predicted-no-match S1s to empty (out-of-fold), compute the
   macro F0.5 delta against the baseline 0.968992 with an entity bootstrap.
   Report the no-match false-merge rate before/after.
4. **List size:** predict T (true match count, 0–11) from the same features;
   report calibration. This feeds the expected-F0.5 decoder, which currently
   loses when given raw pair probabilities (a lone strong candidate at p ≈ 0.55
   is true only 9% of the time).

**Leakage trap:** do not use "the record belongs to another S1" or any other
label except the target. On test that information does not exist.

## Part 2 — Train→test shift (label-free, short)

Test has 5.75 S2/S3 records per S1; train 4.68. **Question:** are the extra
test records extra copies or extra look-alikes? Compare train vs test per
country (US, India; France separately): records per S1, empty-address rate,
share of pool records whose normalized name matches some S1, share whose
normalized name + digit set matches some S1, near-duplicate rate within the
pool, and (using v1 test scores vs the tuning scores) the distribution of each
record's best score. Conclude what it implies for thresholds (stricter or
looser on test) and quantify it.

## Deliverable

`report.md`, `results.json`, scripts, and `no_match_features.py` computing
your features from arrays + text (so it can run on 2M S1s).
