# Completed model and decision comparisons

107 historical scored model configurations, including ensembles/blend weights and hyperparameter variants; 1 dense-only CatBoost compatibility screen; 4 P1 decision-layer comparisons on saved v1 scores. Decision variants are not new models. Repeated baselines and failed runs are not counted.

**Development scores do not establish submission readiness.** The initial model fell from 0.981500 to 0.832719 on the same 1,000 businesses when evaluated against 9,290,818 possible targets. Subsequent hard-pool experiments use all 10,320,219 original training targets. HOLDOUT-2 is reserved and unopened for v2 selection.

## V2 P1 decision layer — reused TUNE only

Same 12,000 medium TUNE references and 600,000 dense candidates retrieved against the full 10,320,219-target pool. The saved v1 threshold baseline is 0.968992 / 494 false links; calibrated variants use entity-disjoint cross-fitting within TUNE and choose a per-entity prefix (null scalar threshold in CSV/JSON). No fresh holdout or new model training.

| Decision variant | Macro F0.5 | Precision | Recall | False links | Singleton accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| assignment_delta_0.00 | 0.969029 | 0.987522 | 0.938760 | 492 | 0.952654 |
| calibrated_expected_f05_no_source_oof | 0.970062 | 0.988488 | 0.935722 | 452 | 0.959828 |
| calibrated_expected_f05_no_source_plus_assignment_0 | 0.970099 | 0.988538 | 0.935722 | 450 | 0.959828 |
| calibrated_expected_f05_nested_source_plus_assignment_0 | 0.970222 | 0.988251 | 0.936903 | 462 | 0.959828 |

No-source rule + pure assignment improves paired TUNE macro F0.5 by 0.001107 (95% entity-bootstrap interval [0.000378, 0.001860]); the nested source bonus adds only 0.000122 ([-0.000118, 0.000337]) and 12 more false links. Keep the higher-precision no-source candidate for a fresh holdout. The retrieved composite-key override adds zero accepted links and is not a duplicate scored row.

## V2 P3 dense-only tree compatibility screen

This is a scored CatBoost model fitted on only 1,200 medium TRAIN S1s; it is not a full-scale P3 tree or a union-candidate matcher. The 12,000 TUNE S1s were retrieved from all 10,320,219 targets.

| Setup | Matcher | Macro F0.5 | Precision | Recall | False links | Scope |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| v2_tree_dense_compat | catboost_1200_train_dense_only_screen | 0.937623 | 0.976618 | 0.878146 | 872 | full pool; 12000 reused medium TUNE references; 1200 sampled medium TRAIN; dense-only compatibility screen |

## P0 runtime and compatibility checks (not model scores)

Scratch had 15.09 GiB free after cleanup; the bounded GPU-kind supervisor and workstation CatBoost/parquet compatibility smokes passed. Selective BF16 with FP32 rescoring within ±0.04 of the fixed threshold preserves the fresh FP32 accept set on 2,000 probe pairs and all 12,000 TUNE references (43,511 CE pairs; 1,065 rescored), taking 34.11s versus 106.53s. This fresh FP32 run differs by one accepted link from the original saved v1 score; the runtime comparison is not a model gain. Batch256 reaches 1,414–1,474 pairs/s on the fixed benchmark, below the 1,700 adoption gate; retain batch128. GPU reports: `/workspace/az-ml-new/v2/amp_selective/report.json`; local runner/CatBoost smoke: `artifacts/v2_p0_runner/`.

Historical report paths in CSV/JSON resolve under `--artifacts` (`/home/sidvish/Documents/Work/Personal/az-ml-new/artifacts` for this regeneration).

## Full-pool development

| Setup | Matcher | Macro F0.5 | Precision | Recall | False links | Scope |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| hard_union_bge_ce600 | blend_0.5 | 0.969812 | 0.987683 | 0.934743 | 97 | full pool; 2400 tuning references |
| hard_union_bge_ce600 | blend_0.75 | 0.969290 | 0.986712 | 0.937027 | 105 | full pool; 2400 tuning references |
| medium_dense_bge_ce1800 | blend_0.75 | 0.968992 | 0.987471 | 0.938760 | 494 | full pool; 12000 tuning references |
| medium_dense_bge_ce1800 | blend_0.5 | 0.968374 | 0.986135 | 0.939700 | 548 | full pool; 12000 tuning references |
| medium_dense_bge_ce1800 | blend_1 | 0.967452 | 0.985953 | 0.935818 | 553 | full pool; 12000 tuning references |
| hard_union_bge_ce600 | blend_1 | 0.966240 | 0.983946 | 0.935464 | 127 | full pool; 2400 tuning references |
| medium_dense_bge_ce600 | blend_0.5 | 0.966067 | 0.986475 | 0.933793 | 531 | full pool; 12000 tuning references |
| medium_dense_bge_ce600 | blend_0.75 | 0.965763 | 0.988073 | 0.928754 | 465 | full pool; 12000 tuning references |
| medium_dense_bge_ce1800 | blend_0.25 | 0.962778 | 0.989600 | 0.915349 | 399 | full pool; 12000 tuning references |
| medium_dense_bge_ce600 | blend_0.25 | 0.962476 | 0.989666 | 0.914336 | 396 | full pool; 12000 tuning references |
| hard_union_bge_ce600 | blend_0.25 | 0.961914 | 0.984629 | 0.923807 | 120 | full pool; 2400 tuning references |
| medium_dense_bge_ce600 | blend_1 | 0.961729 | 0.983240 | 0.927886 | 656 | full pool; 12000 tuning references |
| medium_dense_residual_sweep | residual_d8_neg1 | 0.958655 | 0.985494 | 0.914023 | 558 | full pool; 12000 tuning references |
| medium_dense_residual_sweep | residual_d8_neg4 | 0.958124 | 0.988383 | 0.906669 | 442 | full pool; 12000 tuning references |
| medium_dense_residual_sweep | residual_d8_neg2 | 0.957716 | 0.990062 | 0.903173 | 376 | full pool; 12000 tuning references |
| medium_dense_sweep | d10_neg2_numeric1 | 0.957134 | 0.988471 | 0.903366 | 437 | full pool; 12000 tuning references |
| medium_dense_sweep | d10_neg1_numeric1 | 0.957044 | 0.986855 | 0.906862 | 501 | full pool; 12000 tuning references |
| medium_dense_sweep | d8_neg1_numeric1 | 0.956587 | 0.987362 | 0.904137 | 480 | full pool; 12000 tuning references |
| hard_union_sweep | d8_neg4_numeric1 | 0.956008 | 0.988752 | 0.897969 | 85 | full pool; 2400 tuning references |
| medium_dense_sweep | d8_neg2_numeric1 | 0.955768 | 0.986215 | 0.905560 | 525 | full pool; 12000 tuning references |
| hard_union_sweep | d6_neg2_numeric1 | 0.955432 | 0.988352 | 0.897368 | 88 | full pool; 2400 tuning references |
| hard_union_sweep | d8_neg2_numeric1 | 0.955061 | 0.982372 | 0.910828 | 136 | full pool; 2400 tuning references |
| hard_bge_ce600 | blend_0.75 | 0.954969 | 0.983827 | 0.906502 | 124 | full pool; 2400 tuning references |
| hard_union_sweep | d8_neg1_numeric1 | 0.954885 | 0.979332 | 0.916837 | 161 | full pool; 2400 tuning references |
| hard_dense_sweep | d8_neg2_numeric1 | 0.954765 | 0.984243 | 0.908304 | 121 | full pool; 2400 tuning references |
| hard_bge_ce600 | blend_0.5 | 0.954626 | 0.980438 | 0.915515 | 152 | full pool; 2400 tuning references |
| hard_union_sweep | d6_neg1_numeric1 | 0.954598 | 0.984475 | 0.906862 | 119 | full pool; 2400 tuning references |
| hard_dense_sweep | d8_neg4_numeric1 | 0.954487 | 0.984553 | 0.903858 | 118 | full pool; 2400 tuning references |
| hard_union_sweep | d6_neg4_numeric1 | 0.954344 | 0.988056 | 0.894724 | 90 | full pool; 2400 tuning references |
| medium_dense_sweep | d6_neg1_numeric1 | 0.954336 | 0.986710 | 0.900400 | 503 | full pool; 12000 tuning references |
| hard_dense_sweep | d8_neg1_numeric1 | 0.954152 | 0.985327 | 0.903858 | 112 | full pool; 2400 tuning references |
| medium_dense_sweep | d6_neg2_numeric1 | 0.954146 | 0.986784 | 0.900087 | 500 | full pool; 12000 tuning references |
| medium_dense_sweep | d10_neg1_numeric0 | 0.954071 | 0.984745 | 0.904282 | 581 | full pool; 12000 tuning references |
| medium_dense_sweep | d10_neg2_numeric0 | 0.953823 | 0.984313 | 0.904692 | 598 | full pool; 12000 tuning references |
| hard_dense_sweep | d6_neg1_numeric1 | 0.953497 | 0.985288 | 0.901454 | 112 | full pool; 2400 tuning references |
| hard_dense_sweep | d6_neg2_numeric1 | 0.952998 | 0.984422 | 0.903738 | 119 | full pool; 2400 tuning references |
| medium_dense_sweep | d8_neg1_numeric0 | 0.952634 | 0.984578 | 0.900497 | 585 | full pool; 12000 tuning references |
| medium_dense_sweep | d8_neg2_numeric0 | 0.952608 | 0.983135 | 0.903727 | 643 | full pool; 12000 tuning references |
| hard_dense_sweep | d6_neg4_numeric1 | 0.952441 | 0.985014 | 0.900493 | 114 | full pool; 2400 tuning references |
| hard_bge_ce600 | blend_0.25 | 0.951631 | 0.985114 | 0.898690 | 113 | full pool; 2400 tuning references |
| hard_union_sweep | d8_neg4_numeric0 | 0.951354 | 0.978269 | 0.908905 | 168 | full pool; 2400 tuning references |
| medium_dense_sweep | d6_neg1_numeric0 | 0.951313 | 0.985704 | 0.892685 | 537 | full pool; 12000 tuning references |
| hard_union_sweep | d8_neg2_numeric0 | 0.951123 | 0.980550 | 0.908785 | 150 | full pool; 2400 tuning references |
| medium_dense_sweep | d6_neg2_numeric0 | 0.951009 | 0.985672 | 0.892371 | 538 | full pool; 12000 tuning references |
| hard_union_sweep | d8_neg1_numeric0 | 0.950452 | 0.976532 | 0.915154 | 183 | full pool; 2400 tuning references |
| hard_union_sweep | d6_neg4_numeric0 | 0.950420 | 0.982309 | 0.900853 | 135 | full pool; 2400 tuning references |
| hard_union_sweep | d6_neg1_numeric0 | 0.950409 | 0.977755 | 0.908545 | 172 | full pool; 2400 tuning references |
| hard_ce_residual_comparison | alternative_ce_0.5 | 0.950115 | 0.979297 | 0.903858 | 159 | full pool; 2400 tuning references |
| hard_ce_residual_comparison | alternative_ce_0.25 | 0.950051 | 0.983195 | 0.900012 | 128 | full pool; 2400 tuning references |
| hard_dense_sweep | d8_neg4_numeric0 | 0.949851 | 0.983676 | 0.897969 | 124 | full pool; 2400 tuning references |
| hard_bge_ce600 | blend_1 | 0.949739 | 0.984249 | 0.893643 | 119 | full pool; 2400 tuning references |
| hard_union_sweep | d6_neg2_numeric0 | 0.949583 | 0.983762 | 0.895565 | 123 | full pool; 2400 tuning references |
| hard_dense_sweep | d8_neg1_numeric0 | 0.949537 | 0.980203 | 0.904459 | 152 | full pool; 2400 tuning references |
| hard_dense_sweep | d8_neg2_numeric0 | 0.949282 | 0.980318 | 0.903858 | 151 | full pool; 2400 tuning references |
| hard_dense_sweep | d6_neg2_numeric0 | 0.949147 | 0.981489 | 0.898450 | 141 | full pool; 2400 tuning references |
| hard_dense_sweep | d6_neg1_numeric0 | 0.948880 | 0.981498 | 0.898930 | 141 | full pool; 2400 tuning references |
| hard_ce_v1 | blend_0.5 | 0.948804 | 0.983594 | 0.893402 | 124 | full pool; 2400 tuning references |
| hard_ce_v1 | blend_0.25 | 0.948617 | 0.985367 | 0.890157 | 110 | full pool; 2400 tuning references |
| hard_dense_sweep | d6_neg4_numeric0 | 0.948070 | 0.987643 | 0.883668 | 92 | full pool; 2400 tuning references |
| hard_residual_sweep | residual_d8_neg1 | 0.947112 | 0.981706 | 0.896407 | 139 | full pool; 2400 tuning references |
| hard_residual_sweep | residual_d8_neg2 | 0.946630 | 0.982308 | 0.894123 | 134 | full pool; 2400 tuning references |
| hard_residual_sweep | residual_d8_neg4 | 0.945732 | 0.979748 | 0.895325 | 154 | full pool; 2400 tuning references |
| hard_tabular_sweep | d8_neg1_numeric1 | 0.944533 | 0.979831 | 0.893282 | 153 | full pool; 2400 tuning references |
| hard_tabular_sweep | d8_neg2_numeric0 | 0.944209 | 0.977632 | 0.892922 | 170 | full pool; 2400 tuning references |
| hard_tabular_sweep | d8_neg4_numeric1 | 0.944208 | 0.981393 | 0.887393 | 140 | full pool; 2400 tuning references |
| hard_tabular_sweep | d6_neg1_numeric1 | 0.943268 | 0.975773 | 0.895445 | 185 | full pool; 2400 tuning references |
| hard_tabular_sweep | d8_neg2_numeric1 | 0.943096 | 0.985073 | 0.880303 | 111 | full pool; 2400 tuning references |
| hard_ce_v1 | blend_0.75 | 0.942636 | 0.985753 | 0.873092 | 105 | full pool; 2400 tuning references |
| hard_ce_residual_comparison | alternative_ce_0.75 | 0.942590 | 0.985370 | 0.874174 | 108 | full pool; 2400 tuning references |
| hard_tabular_sweep | d8_neg1_numeric0 | 0.942533 | 0.982303 | 0.880543 | 132 | full pool; 2400 tuning references |
| hard_tabular_sweep | d6_neg2_numeric1 | 0.941804 | 0.984998 | 0.875856 | 111 | full pool; 2400 tuning references |
| hard_tabular_sweep | d6_neg4_numeric1 | 0.941775 | 0.986946 | 0.872251 | 96 | full pool; 2400 tuning references |
| hard_tabular_sweep | d8_neg4_numeric0 | 0.941746 | 0.972381 | 0.897008 | 212 | full pool; 2400 tuning references |
| hard_tabular_sweep | d6_neg2_numeric0 | 0.940050 | 0.968507 | 0.898089 | 243 | full pool; 2400 tuning references |
| hard_tabular_sweep | d6_neg1_numeric0 | 0.939907 | 0.977666 | 0.883788 | 168 | full pool; 2400 tuning references |
| hard_tabular_sweep | d6_neg4_numeric0 | 0.938307 | 0.978584 | 0.878620 | 160 | full pool; 2400 tuning references |
| hard_country_transfer | hard_country_transfer | 0.932335 | 0.956303 | 0.910582 | 208 | full pool; 1431 tuning references; transfer from India |
| hard_ce_v1 | blend_1 | 0.927328 | 0.971926 | 0.865401 | 208 | full pool; 2400 tuning references |
| hard_country_transfer | hard_country_transfer | 0.826255 | 0.931921 | 0.741722 | 180 | full pool; 969 tuning references; transfer from US |

## Small-pool screening

| Setup | Matcher | Macro F0.5 | Precision | Recall | False links | Scope |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| e5_cross200 | blend_0.5 | 0.986490 | 0.992067 | 0.976926 | 65 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d8_neg2_numeric1 | 0.986408 | 0.993028 | 0.975604 | 57 | sampled pool; 2400 tuning references |
| e5_small_ft200 | e5_small_ft200 | 0.986225 | 0.990873 | 0.978488 | 75 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d8_neg1_numeric1 | 0.986215 | 0.991843 | 0.979089 | 67 | sampled pool; 2400 tuning references |
| e5_cross200 | blend_0.25 | 0.986209 | 0.992421 | 0.975604 | 62 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d6_neg1_numeric0 | 0.986129 | 0.989701 | 0.981613 | 85 | sampled pool; 2400 tuning references |
| e5_cross200 | blend_0.75 | 0.985988 | 0.992888 | 0.973080 | 58 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d6_neg1_numeric1 | 0.985821 | 0.988768 | 0.983896 | 93 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d8_neg1_numeric0 | 0.985774 | 0.990755 | 0.978849 | 76 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d6_neg2_numeric1 | 0.985746 | 0.989350 | 0.982454 | 88 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d8_neg4_numeric1 | 0.985636 | 0.993130 | 0.972840 | 56 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d8_neg2_numeric0 | 0.985407 | 0.992286 | 0.973921 | 63 | sampled pool; 2400 tuning references |
| bge_m3 | bge_m3 | 0.985402 | 0.990861 | 0.977286 | 75 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d6_neg4_numeric1 | 0.985139 | 0.994687 | 0.967552 | 43 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d6_neg2_numeric0 | 0.985029 | 0.992397 | 0.972599 | 62 | sampled pool; 2400 tuning references |
| e5_small | e5_small | 0.984941 | 0.992287 | 0.974042 | 63 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d6_neg4_numeric0 | 0.984765 | 0.990717 | 0.974763 | 76 | sampled pool; 2400 tuning references |
| e5_ft_tabular_sweep | d8_neg4_numeric0 | 0.984466 | 0.988620 | 0.981372 | 94 | sampled pool; 2400 tuning references |
| phonetic_v1 | catboost_deep | 0.984147 | 0.991067 | 0.973321 | 73 | sampled pool; 2400 tuning references |
| phonetic_v1 | catboost_ensemble | 0.983526 | 0.989868 | 0.974522 | 83 | sampled pool; 2400 tuning references |
| e5_cross200 | blend_1 | 0.982912 | 0.988521 | 0.972840 | 94 | sampled pool; 2400 tuning references |
| phonetic_v1 | catboost | 0.982753 | 0.990320 | 0.971277 | 79 | sampled pool; 2400 tuning references |
| quick_v1 | catboost_deep | 0.982352 | 0.993783 | 0.960461 | 50 | sampled pool; 2400 tuning references |
| quick_v1 | catboost_ensemble | 0.981867 | 0.991505 | 0.967792 | 69 | sampled pool; 2400 tuning references |
| quick_v1 | catboost | 0.981712 | 0.990312 | 0.970436 | 79 | sampled pool; 2400 tuning references |
| phonetic_v1 | logistic | 0.979592 | 0.989215 | 0.959019 | 87 | sampled pool; 2400 tuning references |
| quick_v1 | logistic | 0.977753 | 0.988827 | 0.957217 | 90 | sampled pool; 2400 tuning references |
| quick_v1 | heuristic | 0.931796 | 0.986575 | 0.847855 | 96 | sampled pool; 2400 tuning references |


Small-pool rows use 2,400 tuning businesses and 201,536 targets. Full-pool rows indicate their reference population; different populations and transfer checks are not direct model comparisons. Thresholds use tuning labels only. The official test set has hidden labels; no entire-test F0.5 or F1 is available.

Keep for independent validation: full-pool E5 plus lexical/phonetic retrieval, numeric-conflict features, and tree/BGE cross-encoder blends. Negative class weights depend on the candidate set: weight 1 won the initial lexical tree, while weight 4 won the quick-reference union tree. Prefer measured blends over the pure reranker in paired tests; heuristic-only matching and simple tree averaging remain weak leading options.

BGE600 on the lexical hard pool improved on MiniLM300 by 0.006166 at matched 19,200 pair presentations; paired 95% tune interval [0.002883, 0.009499]. Union candidates plus 50% BGE reached 0.969812 versus the union tree 0.956008. These are tuning results; larger-data and fresh v2 holdout comparisons remain necessary. The earlier residual-tree 25% CE near-tie had 31 fewer false links than its 50% blend. Original-tree 50%-minus-25% CE has a tune-only paired 95% interval of [-0.002536, 0.002939]. P2 retrieval oracles, P4 cluster/ByT5 text probes, the frozen full-pool stress test, and anchor checks are documented in EXPERIMENT_DECISIONS.md rather than counted as new model configurations.

The first E5 fine-tuning attempt failed on non-tensor tokenizer metadata. Tensor-only device transfer fixed it, and the successful retry is recorded above. No result is attributed to the failed run.
