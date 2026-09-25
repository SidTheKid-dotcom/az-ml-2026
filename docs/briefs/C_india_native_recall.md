# Brief C — India native-script recall (opencode GPT-6 astra)

Read `docs/briefs/README.md` first. Your folder:
`/home/sidvish/Documents/Work/Personal/ext-experiments/C/`.

## Question

Dense E5 retrieval (top-50 per S1, country-blocked) never retrieves 655 of
41,476 true links on the 12,000 tuning S1s; 69% of those misses are Indian and
194 have native-script names (Devanagari, Tamil, Telugu, Kannada, Bengali,
Gujarati, Malayalam, Odia, Gurmukhi). About 23.5% of Indian S2 names and 13.2%
of S3 names are in native scripts. **Which cheap CPU retrieval branch recovers
the most missed true links per extra candidate pair?**

## Method

1. Rebuild the miss list: for the 12,000 tuning S1s (`tune_references.parquet`),
   true links from `train_ground_truth.tsv` minus the pairs present in
   `tune_pairs.parquet`. Profile them: script, source, address kind, digits.
2. Pool: all Indian train S2/S3 records (≈ 4.1M) — the full competing pool,
   not a sample. Queries: the 4,787 Indian tuning S1s. Keep memory < 2.5 GB:
   hashed features (`HashingVectorizer`, float32 sparse), chunked sparse
   matrix products, top-k per query kept incrementally.
3. Branches to try, in order:
   - **Transliteration + char n-grams:** transliterate native-script text to
     Latin (Unidecode; or an Indic transliteration library in your own venv),
     then character 3–4-gram TF-IDF on name, top-k by cosine.
   - **Phonetic keys** on transliterated name tokens (Double Metaphone or a
     simple consonant skeleton) + blocking on address digits or city/state.
   - **Address-first:** records whose name is unusable (native script, trade
     name) but whose address digits and locality tokens match.
   - Combinations with a small k (5–20).
4. For each branch: recall of the 655 misses (and of all true links), extra
   pairs per S1 not already in the dense candidates, overlap with dense, and
   runtime. The useful number is **new true links per 1,000 extra pairs** at
   k ≤ 20.
5. Estimate the cost for test: 809,986 Indian test S1s against ≈ 4.7M Indian
   test records. It must run in ≤ 2 CPU-hours on 2 threads (or say what it
   needs).

## Out of scope

Dense retrieval, the cross-encoder and anything on the GPU (the controller);
feature engineering for matching (agent-F). You only produce candidates.

## Deliverable

`report.md`, `results.json`, scripts, and `native_branch.py` with a function
`candidates(s1_frame, pool_frame, k) -> DataFrame[s1_entity_id, record_entity_id, score]`
that runs on any country subset, so the controller can run it on test.
