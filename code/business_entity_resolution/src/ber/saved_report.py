"""Audit a completed development tree's saved predictions without re-training.

The sweep's pairs and probabilities must be from the same completed directory.
Final holdout scoring belongs to the frozen inference path, not this audit tool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .experiment import load_data, json_write
from .metrics import pair_labels, score_pairs
from .uncertainty import paired_bootstrap


def run(data, artifact):
    artifact = Path(artifact)
    reference, pool, truth = load_data(data)
    pairs = pd.read_parquet(artifact / 'pairs.parquet')
    probability = np.load(artifact / 'best_probabilities.npy', mmap_mode='r')
    checkpoint = artifact / 'best.joblib'
    bundle = joblib.load(checkpoint)
    threshold = float(bundle['threshold'])
    if probability.shape != (len(pairs),) or not np.isfinite(probability).all():
        raise ValueError('Saved probabilities do not align with completed pair rows')
    if reference.entity_id.duplicated().any() or pool.entity_id.duplicated().any():
        raise ValueError('Duplicate record IDs')
    if pairs.duplicated(['left_idx', 'right_idx']).any():
        raise ValueError('Duplicate candidate edges')
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    if len(pairs) and (left.min() < 0 or left.max() >= len(reference)
                       or right.min() < 0 or right.max() >= len(pool)):
        raise ValueError('Candidate indices are outside the supplied records')
    selected = reference.split.eq('tune').to_numpy()
    if not selected.any() or not set(reference.loc[selected, 'entity_id']) <= truth.keys():
        raise ValueError('Truth must explicitly cover every tuning reference')
    positive = pair_labels(reference, pool, pairs, truth).astype(bool)
    accepted = np.asarray(probability, dtype=np.float64) >= threshold
    predicted = np.bincount(left[accepted], minlength=len(reference))
    tp = np.bincount(left[accepted & positive], minlength=len(reference))
    actual = np.array([len(truth.get(key, set())) for key in reference.entity_id])
    frame = pd.DataFrame({'source1_entity_id': reference.entity_id, 'country': reference.country,
                          'true_links': actual, 'predicted_links': predicted,
                          'true_positive_links': tp, 'false_positive_links': predicted-tp,
                          'false_negative_links': actual-tp})
    for name, numerator, denominator in [('macro_f05', 5*tp, 4*predicted+actual),
                                          ('macro_f1', 2*tp, predicted+actual)]:
        values = np.divide(numerator, denominator, out=np.zeros(len(reference)), where=denominator > 0)
        values[(actual == 0) & (predicted == 0)] = 1
        frame[name] = values
    frame = frame.loc[selected]
    score = score_pairs(reference, pool, pairs, probability, truth, threshold, selected)
    expected = bundle['record']['tune']['macro_f05']
    if not np.isclose(score['macro_f05'], expected, atol=1e-12, rtol=0):
        raise ValueError('Saved bundle/predictions/data do not reproduce the selected score')
    score.update(macro_f1=float(frame.macro_f1.mean()),
                 false_positive_links=int(frame.false_positive_links.sum()),
                 false_negative_links=int(frame.false_negative_links.sum()))
    report = {'score': score, 'scope': 'tuning only; not an independent holdout',
              'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'threshold_changed': False,
              'uncertainty': paired_bootstrap(frame.macro_f05, strata=frame.country)}
    frame.to_csv(artifact/'tune_entities.tsv', sep='\t', index=False)
    json_write(artifact/'saved_prediction_report.json', report)
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--artifact', required=True)
    args = parser.parse_args()
    run(args.data, args.artifact)
