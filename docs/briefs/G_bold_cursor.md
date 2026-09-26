# Brief G — bold CPU experiments for Cursor agents (one agent per task)

Read `docs/briefs/README.md` (rules, data) and `docs/briefs/HANDOFF_claude_code.md`
first. Folder per task: `/home/sidvish/Documents/Work/Personal/ext-experiments/G<n>/`.
CPU only, **≤ 2 threads, ≤ 2 GB RAM** (the machine already runs 4 agents), never
SSH, read-only on the repo, never use HOLDOUT-2 labels, no external data.

**New data since the README:** a competitor-complete arena is local, where all
2.07M train S1s compete for all records. Files:
- `az-ml-new/artifacts/arena/v1/`: `pairs.parquet` (103.6M rows, `left_idx`,
  `right_idx`), `reference.parquet`, `pool.parquet`, `probabilities.npy`
  (v1 blend), `base_probabilities.npy` (tree).
- `artifacts/arena/labels/`: `label.npy`, `true_count.npy`, `rec_owner.npy`.
- `artifacts/arena/finalize_v1/s1_partition.npy`: 0 = A (train), 1 = B_REST,
  2 = SLICE. Fit only on A or B_REST; report on SLICE once.

Baseline on the arena: v1 + one-owner F0.5 0.972382; + name-only 0.975361.
Always use memory maps and column projection; never load all 103M rows as
pandas objects.

Goal: find anything that moves the arena score or France by a lot. Report the
delta vs the baseline on SLICE, with a component bootstrap
(`ber.arena_decision.component_bootstrap`, components in
`finalize_v1/component_ids.npy`).

## G1 — France ownership solver (highest priority)

French names are templated ("Nantes Club EURL" vs "Nantes Conseil EURL"),
house numbers repeat, and the street decides. Build a label-free
**French key**: normalized street tokens + house number + city + the name's
distinctive word(s) (drop city, legal form and generic words, derived from
French S1 frequencies). Validate the same construction on US/India arena
labels (precision/coverage). Then apply it on test France (v1 test scores in
`artifacts/v1_test_scores/`) and produce a France-only change set:
- links added where the key is unique and both texts agree;
- links dropped where the S1's key contradicts the record's street;
- contested records resolved by key.

Price it with the manual's per-link values. Deliver a France-only matching
file variant built from `/home/sidvish/Downloads/assignment-only-lf/matching_results.tsv`.

## G2 — Record-cluster voting

Copies of one business across S2/S3 look like each other; look-alikes differ
systematically. On the arena, cluster records (exact normalized name + digit
set, then fuzzy) and let a cluster's confident owner claim uncertain members
(or veto members whose digits disagree with the cluster). Measure it as a
decision-layer change on SLICE. (Cluster-mate *retrieval* failed before; this
is a *decision* test.)

## G3 — No-match detector at arena scale

Brief B, part 1, but on the arena (2M S1s, 120k no-match): a business-level
classifier (list-level features only, fitted on A) predicting no match. Set
predicted no-match S1s to empty. Report the change in SLICE F0.5 and in the
no-match false-merge rate (now ~4.8%; target ≤ 2%).

## G4 — Count and source structure

Each S1 has 0–11 matches (mean 3.46), usually 1–2 from S2 and 1–2 from S3.
Build a per-S1 predicted count (label-free features, fitted on A) and a
source-balance check. Prune lists that exceed the prediction, and add a
second-source link when the list is one-sided and a candidate is close. Measure
on SLICE.

## Deliverable (each)

`report.md` (numbers, population, n, failures), `results.json`, runnable
scripts, and for any win, a function that applies it to arrays so the
controller can plug it into the final stack.
