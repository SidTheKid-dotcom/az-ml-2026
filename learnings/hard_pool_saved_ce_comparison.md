# Saved cross-encoder scores with the residual-name tree

The residual-name tree and the saved 300-step cross-encoder are complementary on the 2,400 hard-pool tuning references. The highest tuning macro F0.5 is **0.950115** for a 50% blend at threshold 0.60. A 25% blend at threshold 0.65 scores **0.950051**, only 0.000064 lower, with **31 fewer false-positive links**. The metric winner remains the 50% blend; the 25% blend is a useful higher-precision near-tie to retain for independent validation.

| Variant | Threshold | Macro F0.5 | Precision | Recall | False-positive links | India F0.5 | US F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Residual tree | 0.675 | 0.947112 | 0.981706 | 0.896407 | 139 | 0.920458 | 0.965161 |
| Residual +25% CE | 0.650 | 0.950051 | 0.983195 | 0.900012 | 128 | 0.926800 | 0.965795 |
| Residual +50% CE | 0.600 | 0.950115 | 0.979297 | 0.903858 | 159 | 0.929517 | 0.964063 |
| Residual +75% CE | 0.775 | 0.942590 | 0.985370 | 0.874174 | 108 | 0.924944 | 0.954539 |

No model was trained in this comparison. The residual checkpoint was reloaded and its scores recomputed from its 122 aligned features. All 598,828 candidate rows were verified to have the same order as the CE run, and the reference and pool text/ID digests matched. Only the 149,230 tune candidate pairs were scored for model selection; all 2,400 tune references remain in the macro denominator.

The CE scores exist on the original-tree probability gate of 0.01, covering 11,755 tune pairs. That same gate was preserved, rather than assuming CE scores existed outside it. Every match accepted by the residual tree at its saved threshold had a CE score, and the gated residual baseline exactly matched the unrestricted baseline. Applying this saved procedure to future references must preserve the original gate or explicitly score any additional candidate pairs.

For the earlier original-tree blends, the 50% CE weight scored 0.948804 versus 0.948617 for 25%, a difference of **0.000187**. A paired bootstrap over all 2,400 complete references (10,000 replicates, seed 20260925) gives a 95% interval for that difference of **[-0.002536, 0.002939]**. The 50% blend produced 27 additional true-positive links and 14 additional false-positive links. It improved 88 references, worsened 81, and tied on 2,231. That small global tuning difference does not demonstrate a clear winner.

The corresponding country differences for 50%-minus-25% CE were India +0.003458 (95% interval [-0.002531, 0.009612]) and US -0.002028 ([-0.004235, -0.000138]). These are exploratory subgroup diagnostics on a reused tuning set. All intervals condition on selected models, gates, blend weights and thresholds; none is an estimate from untouched holdout data. Eval and test labels were not accessed.

Results, threshold traces, and complete per-reference score tables are in `artifacts/hard_ce_residual_comparison/`. `ber.cross_encoder_compare` reproduces this CPU-only comparison and refuses a dataset containing eval references.
