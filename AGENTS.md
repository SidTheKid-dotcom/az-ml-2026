# Experiment loop — read before working

Read `README.md`, `learnings/CURRENT_STATE.md`, `docs/RUNBOOK.md`, and
`learnings/EXPERIMENT_DECISIONS.md`. Do not rediscover setup or rerun completed
experiments unless a concrete failure or new hypothesis requires it.

## Working objective

Run many **small, decisive experiments**, retain verified gains, record losses,
and produce a reproducible submission. Aim for first place without claiming an
unmeasured score or guaranteed rank. Keep explanations and new infrastructure lean.

## The loop

1. Inspect live jobs/results before launching anything. Reuse candidates, text,
   embeddings and checkpoints; do not launch duplicate queues or rebuild indexes.
2. State one hypothesis and the changed variable: retrieval recall, difficult
   negatives, model family, field representation, training size, or decision rule.
3. Keep training/tuning businesses fixed for direct comparisons. For fast tests,
   reduce query businesses or training steps; retain the **full competing target
   pool**. A sampled pool is only an initial compatibility/speed screen.
4. Run a bounded screen, then extend promising approaches. Prefer cached licensed
   Hugging Face models. Stop weak variants; preserve their config and measured score.
5. Compare official macro F0.5, precision, recall, false links, singleton accuracy,
   country/script breakdown, retrieval ceiling and runtime. The tiny numerical
   winner on a reused tune set is not necessarily better; check paired differences
   and retain a higher-precision near-tie when useful.
6. Save config, seed, split IDs, data/model identities, complete candidate pairs,
   checkpoint, threshold, probabilities and results. Add one short keep/discard
   explanation to `learnings/`; regenerate the ledger and update current state.
7. Freeze finalists and thresholds before opening the final holdout. Confirm on
   the whole held-out reference set against the full target pool, including every
   retrieval miss and singleton. Then generate and validate the official test files.

## Guardrails that protect useful results

- Split by S1 business, never by pair. Fit on TRAIN; choose thresholds/blends on
  TUNE. Quick eval is development data. Medium eval was opened once for the frozen
  v1 model (F0.5 0.968292); never use it to tune subsequent models. Reserve a new
  entity-disjoint holdout from unused training S1s before the next selection cycle.
- Do not inject true edges into retrieval. Retaining missed truth *records* for
  accounting is allowed; missed links must still lower the reported score.
- Preserve one row for every scored S1, even without candidates. Follow the exact
  singleton convention. Reject duplicate pairs/IDs and mismatched feature order.
- France has no training labels. Do not call US/India performance French accuracy.
- Use only supplied business records. No external business lookup/geocoding or
  external task labels. Respect MIT/Apache licensing and the final parameter cap.
- Never label a sampled/tuning score an entire-test score. Test labels are hidden;
  the public leaderboard covers only a subset. Record every public submission.
- Official files override conflicting inherited controller notes. Existing user
  authorization covers GPU experimentation and suitable Hugging Face models.

## Avoid anchoring on the current winner

The chosen E5/tree/BGE pipeline is a baseline, not a required architecture. After
v1 is safely released, test different licensed HF models, retrieval unions,
training sizes, negative sampling or calibration when a bounded comparison can
answer a concrete question. The inherited `models.yaml` is a checked inventory,
not an organizer-mandated model list; verify license/size and use an explicit
updated `--allowlist` for additions. New architectures may need a new adapter;
the current CE trainer supports only its tested single-logit model families.

Keep failed results and their scope; discard weak compute runs, not evidence.
Do not compare different reference samples as if they were a controlled test.
Changing retrieval requires recomputing candidate-dependent features and checking
the matcher again. Hypotheses are not proven by one favorable tuning score.
Country and script differences, test distractor shift, and repeated selection
remain limitations; no report establishes absence of bias or overfitting.

## Release and handoff

Use `learnings/CURRENT_STATE.md` for mutable job status; frozen manifests are
historical selection snapshots. Never edit the source/checkpoints used by an
active release. Develop a new experiment in a separate output/code checkout.
`code/business_entity_resolution/README.md` covers portable replay, training and
the validated final ZIP. An agent needs the supplied data and authorized GPU/key
access as well as Git; secrets, large data and weights are deliberately external.
Update the decision table, current state and release hashes; commit small useful
changes. Do not add a dashboard or another overlapping status document.

## GPU use

Keep useful computation queued and overlap CPU preparation with GPU training.
Optimize completed informative experiments per hour, not process count or VRAM
occupancy. Increase batch size or run compatible jobs when utilization is low;
avoid memory exhaustion, CPU oversubscription and slowing the critical validation
path. Verify actual cgroup limits rather than the host's advertised resources.

Keep `/workspace/amlc` unchanged. Use this project's scratch/durable directories,
preserve successful checkpoints, and keep secrets out of logs and Git. Model
experiments take precedence over dashboards, large refactors, and speculative
infrastructure. Update these entry-point documents rather than adding long diaries.
