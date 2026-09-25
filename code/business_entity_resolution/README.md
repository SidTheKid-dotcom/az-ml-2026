# Business entity resolution experiments

The pipeline retrieves candidate S2/S3 businesses within each observed country,
then scores pairs using name/address features and supervised matchers. All scores
use the official **mean of per-S1 F0.5**, including singleton and missing-candidate
cases. Predictions never use the numerical part of identifiers.

## Environment

Use Python 3.12 and install `requirements.txt`. The current snapshot pins the
locally verified environment. GPU experiments also require the server's existing
CUDA-enabled PyTorch installation; that runtime will be recorded separately.

Run commands below from the repository root with
`PYTHONPATH=code/business_entity_resolution/src` and the environment's Python.

```bash
python -m ber.prepare \
  --archive 6ab10eb3b23ba_student_resource.zip \
  --output data/quick --references 12000 --distractors 160000

python -m ber.experiment \
  --data data/quick --output artifacts/quick_v1 \
  --models heuristic,logistic,catboost,catboost_deep \
  --top-k 20 --iterations 350 --threads 4

python -m unittest discover -s code/business_entity_resolution/tests -v
```

Use `--device GPU` for CatBoost on the available GPU. Every model, probability
vector, threshold sweep, retrieval report, and experiment summary is saved under
the output directory. `selected_by_tune` is the winner based only on the tuning
split. Baseline runs retain unsuccessful approaches as evidence rather than
silently overwriting experiment history.

## Validation design and limits

The quick experiment samples S1 businesses uniformly, splits them 60/20/20 into
training/tuning/evaluation groups stratified by country, retains all their true
matches, and adds random unmatched candidate records. S1 entities never cross
splits. Candidate text is available without labels for pool-only TF-IDF indexing,
as it will be at inference. All true links count in the metric even if retrieval
misses them. Model training uses only the training labels. Threshold selection
uses tuning labels.

**This sampled pool is easier than the full release.** Quick scores are development
signals, not leaderboard predictions. Full-pool stress tests and a fresh final
holdout are required before promoting a final model. Training has no French
labels; these experiments do not measure French matching accuracy.

The official constraints remain in force: use only provided business data,
no external identity/geocoding lookup, MIT/Apache-2.0 pretrained models within the
parameter limit, every test S1 in the final TSVs, and final matches contained in
the exact candidate set passed to the matching model. Validate final outputs
with the supplied validator including `--check-ids`.
