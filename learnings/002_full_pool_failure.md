# Small-pool validation overstated precision

The initial CatBoost depth-8 model and its threshold of 0.825 were frozen.
The same 391 Indian tuning businesses were then tested against a much larger
candidate pool. No threshold was retuned for this comparison.

| Metric | Quick sampled pool | Full India pool |
| --- | ---: | ---: |
| Macro F0.5 | 0.972676 | 0.790510 |
| Precision | 0.990236 | 0.814631 |
| Recall | 0.950039 | 0.895394 |
| False-positive links | 12 | 261 |
| Correct singletons | 30/31 | 9/31 |
| Candidate link recall | 0.975800 | 0.912568 |

The full India pool contained 3,722,528 records. The initial quick pool contained
201,536 records across both countries. The reference set was identical, so the
18.22-point macro score drop is a scale effect rather than different query sampling.

## Decision

Do not promote any model based on quick-pool scores alone. The initial matcher is
not submission-ready. Training must include near-identical negative businesses
retrieved from the full pool, especially address-number and unit conflicts.

Retrieval also needs improvement: its full-pool oracle macro F0.5 was only 0.958889.
Classifier improvements alone cannot recover missing true matches. Combine lexical,
native-script phonetic, and supervised multilingual retrieval, then test the frozen
pipeline on a fresh holdout against the full candidate pool.

## Follow-up experiments

- Full-pool hard-negative mining for the existing training/tuning businesses.
- Address-number conflict features and negative-weighted tree training.
- Full training-pool embeddings from the fine-tuned multilingual E5 retriever.
- Cross-encoder blends evaluated for both macro F0.5 and false-positive count.

The corresponding experiment directory is `artifacts/full_pool_stress_v1/`.
