# Lead analysis/build scripts (27 Sep 2026)

Label-free test analyses and train-label (US/India) rule checks behind FINAL_C, FINAL_C2 and FINAL_C3.
Paths to data and candidate TSVs are absolute and machine-specific; outputs (TSV/parquet) are kept outside Git.

- `sig_rates.py` — shared helpers (row reader, name folding, core name); truth rates of name/number signature buckets on train and in FINAL_C French links.
- `num_ops.py` — classifies house-number edit operations on true exact-core-name train pairs whose numbers differ.
- `sig2.py` — same-core-name pairs by number-edit class x street agreement x name uniqueness (train truth, test pool/FINAL_C links).
- `fr_numfix.py` — number-noise copy rule: train precision per country/edit class; emits French rule pairs for test.
- `fr_addword.py` — France test: record core = S1 core + one added token; same-number vs decoy-offset share per token.
- `link_mix.py` — per-country link mix of a submission by address/name category (label-free).
- `swap_train.py` — train precision (US/India) of FINAL_C's same-number one-word-swap removal rule.
- `build_a2.py` — builds FINAL_A2 = FINAL_A with the French one-word-swap removal applied.
- `admin_check.py` — France admin component (région) of addresses and exact-copy pairs left unlinked by a submission.
- `samenum.py` — exact core name + first house number matches, bucketed by S1 count and street match (train truth / test link state).
- `sibling_check.py` — exact name+number records with 2+ S1s and one street match: right, wrong-sibling or unlinked.
- `fr_exact_add.py` — builds FINAL_C3 = FINAL_C2 + French exact-name + same-house-number copies (caps 11, S2 <= 5, S3 <= 6).
- `build_c2.py` — builds FINAL_C2 = FINAL_C + U's India native-script dictionary links (u8t) under the same caps.
- `sim_check.py` — near-name variants at the same address: truth per similarity band (train) and link state (test).
