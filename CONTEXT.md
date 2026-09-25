# Operating context

Public leaderboard leaders are at **0.988**. Our best scored file is **0.964563**.
Local medium-tune F0.5 for frozen v1 is **0.968992**. Holdout, opened once, is
**0.968292**. Do not retune on that holdout. HOLDOUT-2 (50,000 S1s, seed
20260929) stays sealed.

How to price a link, read an upload, and train the next model:
[learnings/negative_side_analysis_manual.md](learnings/negative_side_analysis_manual.md).
Record every public score to five decimals. 0.001 of the board is about 1,733
entity-points.

## Scored files

| # | File | Public F0.5 |
| --- | --- | ---: |
| 1 | `Downloads/matching_results.tsv` | 0.956390 |
| 2 | `Downloads/v2-decision/matching_results.tsv` | 0.964232 |
| 3 | `Downloads/assignment-only-lf/matching_results.tsv` | **0.964563** |
| 4 | `Downloads/final-hybrid-lf/matching_results.tsv` | 0.964258 |

#3 is v1 plus the one-owner rule only. #2 added blanket removals and additions
on top of that and lost. France’s share of #2→#3 was +0.000026; US/India’s
share was +0.000305. Hashes are in `learnings/submissions.json`.

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

The detached R0 follow-ups are complete. Field-format 7,200 scored medium TUNE
**0.973686** (422 false links); plain-text 14,400 scored **0.974288** (392 false
links), the current reranker checkpoint leader and +0.005296 over v1. Because
step count and format both changed, this does not isolate a format effect. The
competitor-complete arena score is running over 2,072,821 eligible S1s; labels
are complete with zero HOLDOUT-2 rows read, and a detached watcher will preserve
the completed aligned arrays under `/workspace/az-ml-new/arena/v1/`. v1 test
score arrays remain hash-verified at
`/workspace/az-ml-new/releases/submission_v1_scores/`.

## The plan

**[docs/PLAN_v2.md](docs/PLAN_v2.md) is the executable final plan** (arena,
ownership decision layer, matcher v2, France, GPU schedule, uploads). Its §1
records the 26 Sep measurements: empty-address records are ~98% owned and v1
accepts only 25% of their true links; the name-only rule gains +0.0040 on
TUNE; France holds 74.5% of test contests. The list below is the earlier order,
kept for its manual references.

## Earlier next-code order

The manual’s priced headroom on the current files, if every contest winner
were right, is 3,944 entity-points (about 0.0023 of the full test). The
larger miss is the scorer: retrieval recall 0.984 versus final recall 0.939,
and a simulated France v1 near 0.889.

1. Let the 14,400 and field-format runs finish and read their tune scores.
2. Read the conflict-only tune score the same way. Those negatives are the
   same-name, same-digits, different-unit pairs in the manual.
3. Train the contest resolver on a competitor-complete out-of-fold world
   (manual section 7). Keep a claimant only above that list’s addition
   break-even, about 0.73–0.79. Otherwise drop the record from everyone.
4. Weight matcher examples by the entity-point value of a wrong short list
   (manual section 4.3). Score the union’s extra pairs with the frozen BGE.
5. Add a yes/no adapter and train Qwen3-Reranker-0.6B. French exact-key
   pairs are the pseudo-label source (manual section 8); spot-check them
   before training.
