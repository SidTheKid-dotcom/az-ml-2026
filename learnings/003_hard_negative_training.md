# Full-pool hard negatives substantially improve precision

These comparisons use the same 2,400 quick tuning references and candidates
retrieved from all 10,320,219 original training targets. The models fit only
7,199 training references. No final evaluation labels were used.

| Matcher | Macro F0.5 | Precision | Recall | False links |
| --- | ---: | ---: | ---: | ---: |
| Original sampled-pool tree, frozen threshold | 0.832257 | 0.845120 | 0.922005 | 1406 |
| Same original tree, threshold retuned to 0.99 | 0.902935 | 0.970346 | 0.821896 | 209 |
| Best of 12 full-pool tree variants | 0.944533 | 0.979831 | 0.893282 | 153 |
| Distinctive-name residual features | 0.947112 | 0.981706 | 0.896407 | 139 |
| Hard-trained cross-encoder, 25% blend with tree | 0.948617 | 0.985367 | 0.890157 | 110 |
| Hard-trained cross-encoder, 50% blend with tree | 0.948804 | 0.983594 | 0.893402 | 124 |

The gain from hard-negative training is larger than the gain from merely making
the original threshold stricter. Numeric-conflict features and name-token
residuals help separate near twins from true spelling variants. The cross-encoder
helps especially on India; using it alone is worse (0.927328). Keep blends for
fresh validation. The 25% and 50% blend scores are very close, so the higher
precision alternative deserves comparison rather than treating a tiny tune
advantage as established superiority.

The separate maximum-candidate-confidence abstention rule did not improve over
the ordinary pair threshold. Discard that additional rule for this model.

## Retrieval remains a limit

Lexical plus native-script phonetic retrieval achieved 0.952289 link recall and
0.982442 oracle macro F0.5. India recall was only 0.925346, including 473/600
true links with Indic-script target names. The phonetic branch recovered 14
additional true links over lexical retrieval, but it is not sufficient at scale.
Full-pool dense retrieval and its union are the next comparison.

## Transfer warning

The fixed country-transfer experiment fit each model and selected its threshold
using that country's training/tuning references only. India-trained -> US tuning
scored 0.932335, while US-trained -> India tuning scored 0.826255. These are
development stress tests, not France accuracy estimates. More diverse training
and the pretrained multilingual models remain priorities.

The next dataset uses 35,999 fresh training references and 12,000 fresh tuning
references. All exclude the 12,000 original quick references. A separate 12,001
reference final holdout remains sealed.

Sources: `artifacts/hard_tabular_sweep/results.json`,
`artifacts/hard_residual_sweep/results.json`, `artifacts/hard_ce_v1/results.json`,
`artifacts/hard_decision_sweep/results.json`, and
`artifacts/hard_country_transfer/results.json`.
