# Frozen matcher: full-pool diagnosis and uncertainty

The same 1,000 tuning references were scored with the initial depth-8 CatBoost
model and frozen threshold 0.825. No medium evaluation labels were used.

| Metric | Sampled pool | Full training pool |
| --- | ---: | ---: |
| Macro F0.5 | 0.981500 | 0.832719 |
| Link precision | 0.993289 | 0.852155 |
| Link recall | 0.962744 | 0.923714 |
| Correct singletons | 65/67 | 20/67 |
| Candidate link recall | 0.986103 | 0.945003 |
| Oracle macro F0.5 | 0.996166 | 0.977659 |

The full pool contained 9,290,818 training targets, processed by country. Its
542 false-positive links and 258 missed links reveal both classification and
retrieval problems. Of the missed links, 186 never became candidates and 72
were retrieved but rejected. The high-confidence false-positive sample is a
diagnostic selection, not a frequency estimate of error types.

Examples include RRZ versus RLZ Technologies at the same road number;
Star Raj Infotech versus Star Raj Solutions with door 47-10-20 versus 47-10-27;
North Properties versus North Properties Foods with house 224 versus 237;
and Panini Welfare Society at flat 903 versus 914. Pranansh Automation was a
true singleton, yet Prananex Automation at unit 10A rather than 6A scored 0.998.
Long shared addresses and shared secondary numbers conceal identity-bearing
name changes and unit differences. Hard numeric rejection is also unsafe: one
true Fortune Solutions variant uses F-103/2 against the reference F-103.

Ten thousand paired bootstrap replicates over Source 1 references give a 95%
interval of **[0.817223, 0.847584]** for the full-pool macro score. The paired
full-minus-sampled difference is **−0.148781**, interval
**[−0.163940, −0.134188]**. These intervals document scale loss on development
references; they do not establish fresh-holdout performance.

`ber.uncertainty` can reuse the same analysis for frozen models on a final
holdout. It aligns identical unique reference IDs, includes singletons and
zero-candidate entities, and resamples the same entities for both scores.
An optional stratum column preserves a fixed country mix.

Choose model blends and operating thresholds only on the hard tuning split.
Inspect the score/precision plateau across thresholds and across tuning
subgroups before freezing the choice. Do not select the threshold with the
best fresh evaluation score, and do not present tuning-selected intervals as
unbiased evaluation uncertainty. The first stress run did not preserve every
candidate probability; exact threshold robustness is deferred to the larger
hard-candidate experiment, avoiding another costly index build.

Evidence is in `artifacts/full_pool_stress_v1/`: `summary.json`,
`quick_pool_comparison.json`, `false_positive_diagnosis.json`,
`missed_links.tsv`, `entities.tsv`, `entities_quick_pool.tsv`, and
`paired_bootstrap_ci.json`.
