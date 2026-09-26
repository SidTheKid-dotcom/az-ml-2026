# Union retrieval beyond dense top-50 (26 Sep 2026, agent U)

**Hypothesis:** exact-key branches find true links that the dense E5 top-50
misses, and a ranker fitted on partition A keeps them within a few extra pairs
per S1.

**Method:** `ber.union_mine` matches each S1 against the full pool of its split,
country-blocked, on eight exact keys: normalized name, phonetic name, address
digit sequence, digit set, core name + digit sequence or set, and first core
token + digit sequence or set. A posting over its cap (32–256) is skipped for
that S1. Dense pairs are removed. `ber.union_rank` fits a CatBoost ranker
on A union pairs (16.5M pairs, 12,721 positives, A truth only). It ranks the
new pairs per S1, and the top k are kept. No truth goes into retrieval.

## B_REST (521,846 S1s; 1,806,858 true links; SLICE and HOLDOUT-2 unread)

Dense: recall 0.983687, oracle 0.994595. Δ is the oracle macro F0.5 change
(accept exactly the retrieved true links), with a 95% claim-component bootstrap.

| k | Extra pairs | New true (learned rank) | per 1,000 | Recall | Oracle Δ [CI] | New true (text rank) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 345,602 | 3,900 | 11.3 | 0.985845 | +0.000781 [+0.000739, +0.000826] | 1,242 |
| 2 | 653,359 | 4,289 | 6.6 | 0.986060 | +0.000848 [+0.000803, +0.000898] | 1,573 |
| 3 | 934,014 | 4,381 | 4.7 | 0.986111 | +0.000864 [+0.000818, +0.000914] | 1,805 |
| 5 | 1,431,055 | 4,413 | 3.1 | 0.986129 | +0.000867 [+0.000820, +0.000918] | 2,183 |
| 10 | 2,418,034 | 4,421 | 1.8 | 0.986133 | +0.000868 [+0.000822, +0.000919] | 2,884 |

Test pairs for each k: 1.18M, 2.26M, **3.25M**, 5.03M and 8.67M. Most new true
links come from the address-number branches (digit sequence 2,356; digit set
2,029 at k = 10), and 74% are in India.

**Decision:** keep the learned rank at **k = 3**. It holds 99% of the k = 10
gain with 3.25M test pairs. The text rank is discarded.

**Matcher without the GPU:** the new-pair score is the ranker probability. It is
added to the C2 10k dense decision (0.9·C2 + 0.1·tree ≥ 0.10, one owner, cap 11)
with `ber.union_merge`, and t is cross-fitted on component halves. At k = 3
(t = 0.9 / 0.925) the macro F0.5 goes 0.978722 → 0.978997,
**Δ +0.000275 [+0.000240, +0.000310]**. That adds 1,960 links: 1,764 true
(precision 0.90), and 20 go to singletons. The C2-scored version still needs
the GPU scoring of `set_b_rest_k3`.

**Test dry run (dense = the NOT_FOR_SUBMISSION c2_blend_name_only file,
ranker-only t = 0.9):** +9,003 links: India 5,848, US 806, France 2,349. The
files validate: 89.87M candidate pairs, zero contested, max list 11. The US
and India rates per S1 match B_REST. **France is unvalidated**: 0.9% of French
S1s get a link, against 0.12% in the US. Pass `--countries India US` unless
France is checked separately.

Numbers: `learnings/union_retrieval.json`. Pod outputs:
`/workspace/az-ml-new/union/`.

**Limitations:** the threshold is chosen on B_REST, which is reused
development data. The oracle is an upper bound. France has no labels.
