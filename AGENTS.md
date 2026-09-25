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
  TUNE. The initial quick eval is development data; fresh medium eval stays sealed.
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
