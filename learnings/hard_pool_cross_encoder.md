# Hard-pool cross-encoder experiment

The 300-step multilingual MiniLM reranker improved the selected hard-pool tree on the same 2,400 tuning references. The best tuning blend used equal tree and cross-encoder probabilities and scored macro F0.5 **0.948804**, compared with **0.944533** for the tree. These scores select experiments and operating thresholds; they are not a fresh holdout estimate. No eval or test labels were used.

| Variant | Tune threshold | Macro F0.5 | Micro precision | Micro recall | False-positive links | India F0.5 | US F0.5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selected tree | 0.675 | 0.944533 | 0.979831 | 0.893282 | 153 | 0.916633 | 0.963424 |
| 25% cross-encoder | 0.700 | 0.948617 | 0.985367 | 0.890157 | 110 | 0.924669 | 0.964832 |
| 50% cross-encoder | 0.650 | 0.948804 | 0.983594 | 0.893402 | 124 | 0.928128 | 0.962804 |
| 75% cross-encoder | 0.775 | 0.942636 | 0.985753 | 0.873092 | 105 | 0.925059 | 0.954538 |
| Cross-encoder alone | 0.825 | 0.927328 | 0.971926 | 0.865401 | 208 | 0.904944 | 0.942486 |

The equal blend gained 0.004271 macro F0.5, reduced false-positive links by 29, and kept overall recall essentially level. Most of its gain was in India, while the 25% blend improved both countries and had fewer false-positive links. The two leading blend scores differ by only 0.000187 on this tuning set, so that ordering needs independent validation. The weaker pure reranker supports using it as complementary evidence rather than replacing the tree in this experiment.

The candidate mining pool contained 10,320,219 source records: 4,133,346 India and 6,186,873 US. The exact saved tree candidate rows and feature bundle were reused, and probabilities were recomputed from the selected checkpoint. There were 149,230 tuning candidate pairs with link retrieval recall 0.952289 and oracle macro F0.5 0.982442. A label-free tree probability gate of 0.01 sent 11,755 pairs to the reranker; its link recall was 0.950847. Missing true links remain in the score denominator.

Training used only TRAIN labels, with 23,670 available positive pairs and 71,010 selected hard negatives. Half of the negative pool was reserved for high-tree-score numeric or name conflicts. Each batch contained 32 positives and 32 negatives. Across 300 steps, the model saw 9,600 distinct positives and 9,600 distinct negatives. Blend weights and thresholds were selected only on TUNE. TRAIN inference was skipped.

The model was `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 0.118B parameters), loaded from the existing offline cache. Sequence length was 256, learning rate 2e-5, training batch size 64, and inference batch size 128. It used two CPU threads and a 22% PyTorch GPU allocator cap. Runtime was 74.2 seconds, including 56.1 seconds for checkpoint loading/training and 3.9 seconds for tune inference. A first launch failed before loading data or training because the allocator API needed an explicit CUDA device index; the corrected retry completed successfully.

Reproducible configuration, results, selected blend, threshold traces, and training trace are in `artifacts/hard_ce_v1/`. The large checkpoint and pair arrays remain at `/tmp/az-ml-new/artifacts/hard_ce_v1/` on the authorized GPU host. The adapted implementation supports both earlier HF bundles and completed tabular-sweep bundles. All 16 metric, adapter, and bootstrap tests pass.
