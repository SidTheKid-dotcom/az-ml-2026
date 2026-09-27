# Operating context

## Current — 27 Sep: public best 0.985207

**FINAL_C3 scored 0.985207 on the public platform (user-reported)**, +0.004185
over 0.981022. File `~/Downloads/u4-final/FINAL_C3_usmixFC_nat_frCexact.tsv`,
SHA256 `47aad8ebfe1e52f1b6abbde29762e8aee6b84aa54b4ca77a7be0df4024328698`.
LF and official matching-file `--check-ids` validation passed. Its earlier score
estimate is obsolete. This is the earlier K/native/French-rule candidate, not a
result of the subsequent takeover training.

**Read next:** [CURRENT_STATE](learnings/CURRENT_STATE.md) for the running GPU
jobs, working SSH access, raw-data audit and next file; [submission ledger](learnings/submissions.json)
for scored files; [takeover evidence](learnings/takeover_20260927.json) for the new
development results. Use SSH for Runpod, as the user requested; do not use its MCP.

The raw audit verifies a broader current candidate union, but also **8,334 true
dense candidates lost by its retained-pair filter** on B_REST. New E5 control
retrieval adds 742 true links on the fixed 30k screen (308 beyond unpruned old
routes), raising the current-union oracle to 0.998672. This is not matcher F0.5.
The extra BGE training is near-tied with its initial checkpoint, and the byte
screen is below its +0.0005 target. The next useful work is scoring new candidates
and validating an actual TSV, not claiming those ceilings as submission scores.

## Historical context — 26 Sep (superseded by the live sources above)

**Full factual handoff (26 Sep 14:30 IST):**
[docs/briefs/HANDOFF_20260926_1430.md](docs/briefs/HANDOFF_20260926_1430.md).

**Live sources, read next:** [learnings/CURRENT_STATE.md](learnings/CURRENT_STATE.md)
(preferred file, fallbacks, do-not-repeat list),
[docs/RUNBOOK.md](docs/RUNBOOK.md) (the AWS CPU box; the RunPod pod is dead),
[learnings/EXPERIMENT_DECISIONS.md](learnings/EXPERIMENT_DECISIONS.md). Dated
briefs under `docs/briefs/` are historical snapshots.

**Preferred unuploaded file (26 Sep 14:15 IST):**
`output/pair_tree_us_india/matching_results.tsv`, SHA256
`b7bc5a3d71d04a69002e4cf5f602fe400dcbf59fa657c2fbcbfc1c4dac3f83f7`. It uses the
pair model for US/India and keeps name-only rows for France. Its priced public
score is about 0.970, which is an estimate, not a measured score. Details and
fallbacks are in CURRENT_STATE.

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
  under 4 GiB free. On any new GPU, write checkpoints to a network volume.
- Matching TSVs use LF (`lineterminator='\n'`).
- Do not promote a sampled-pool score. The pool is 10,320,219 rows.
- CatBoost: 750k pairs fit in memory; 1.5M swapped; the full union fit died.
  Do not launch an unbounded tree fit.
- Check the real CPU/RAM limit before choosing threads. The dead pod's cgroup
  was 7.65 cores / 50 GB (two threads); the AWS box is 32 vCPU / 247 GiB.
- Compare models on the same S1 ids only.

## Compute now (26 Sep afternoon IST)

The RunPod A40 pod is terminated and the C2 run died with it. No GPU exists:
every checked AWS GPU quota is 0. The only machine is the AWS `r7i.8xlarge` CPU
box in `docs/RUNBOOK.md`. The arena lives there under
`/opt/az-ml-new/artifacts/arena` (partitions A 1,450,975 / B_REST 521,846 /
SLICE 100,000). Arena v1 + one-owner is 0.972382 on the full arena. Paths under
`/workspace/az-ml-new` or `/tmp/az-ml-*` referenced below were on the pod and
are gone unless a copy is named in CURRENT_STATE.

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
