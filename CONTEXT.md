# Operating context

Public leaderboard leaders are at **0.988**. Our best scored file is **0.965**.
Local medium-tune F0.5 for frozen v1 is **0.968992**. Holdout, opened once, is
**0.968292**. Do not retune on that holdout. HOLDOUT-2 (50,000 S1s, seed
20260929) stays sealed.

## Scored files (user, 2026-09-26)

| File | Public F0.5 |
| --- | ---: |
| `Downloads/matching_results.tsv` | 0.956 |
| `Downloads/v2-decision/matching_results.tsv` | 0.964 |
| `Downloads/assignment-only-lf/matching_results.tsv` | **0.965** |
| `Downloads/final-hybrid-lf/matching_results.tsv` | 0.964 |

Assignment-only beat the France/US-India hybrid. Stop stacking decision rules
on the same v1 scores. Hashes are in `learnings/submissions.json`.

## What moved the score

Full-pool hard negatives. A sampled pool scored ~0.98 and fell to ~0.83 on the
same businesses once the real pool was used. Dense retrieval then BGE reranking
is the matcher: tree ~0.957, BGE1800 blend 0.969 on 12,000 medium tune S1s
against all 10,320,219 targets. 600→1800 steps gained +0.003. One-to-one
assignment gained about +0.008 on the public board and +0.001 locally. LF
uploads scored. A CRLF upload was reported FAILED.

## What did not

Averaging trees. Pure cross-encoder. Stricter abstain gates. Cluster-mate
retrieval (+3 true links). A lexical-only tree on the pairs dense missed (it
abstains). Composite-key override on candidates we already had (0 links).
Source-bonus (interval crosses zero). Country-specific thresholds. CE batch
256 (slower than 128). One CatBoost fit on ~2.05M S1s: it crossed the 50 GB
cgroup floor and the old queue then quit, which is how the GPU went idle.

## Where the 0.023 is

On the candidates dense already returns, the oracle is about 0.994 and the
matcher is 0.969. Most of the gap is pairs we retrieve and then get wrong,
plus France and a test set with more targets per business than train. Union
retrieval lifts the ceiling a little (tune link recall 0.984 → 0.993) but
those extra pairs still need a reranker, not another giant tree.

Historical competition notes in `amazon-ml-challenge-2026/kb` say the same
thing in one line: first place is the metric, more training signal, and a
second model family. The archive itself stays out of this repo. Shopee-style
metric learning is the family we have not trained. Qwen3-Reranker-0.6B is
cached and allowed, and the current trainer cannot run it: it requires one
classification logit, and Qwen scores yes/no as a language model.

## Rules that stay in the code

- `run_gpu_queue.py` runs one GPU job. A failure does not cancel the next job.
- A job whose `--output` is under `/tmp` does not start when the overlay has
  under 4 GiB free. Write checkpoints to `/workspace/az-ml-new`.
- Matching TSVs use LF (`lineterminator='\n'`).
- Do not promote a sampled-pool score. The pool is 10,320,219 rows.
- CatBoost: 750k pairs fit in memory; 1.5M swapped; the full union fit died.
  Do not launch an unbounded tree fit.
- Two threads for OpenMP/BLAS. The host CPU count is not the cgroup limit
  (7.65 cores, 50 GB RAM).
- Compare models on the same S1 ids only.

## Running now

The A40 is training BGE for 14,400 steps on the medium dense set, then a
7,200-step field-format run. Both reuse the 36k training businesses. Keep a
result only if medium-tune F0.5 beats 0.968992 and the paired interval does
not cross zero. When that queue exits, `keep_gpu_busy.sh` starts
`experiments/queue.json`: the same setup with conflict negatives only
(the false links are same-name, same-digits, different unit). Output is on
`/workspace`, so it does not fill the 30 GB overlay.

## Next code, in this order

1. Let the 14,400 and field-format runs finish and read their tune scores.
2. Read the conflict-only tune score the same way.
3. Score the union's extra pairs with the frozen BGE. Do not refit CatBoost
   on all 2M businesses in one process.
4. Add a yes/no adapter and train Qwen3-Reranker-0.6B on the same medium
   pairs. That is the second family. Longer BGE on the same 36k rows is not it.
