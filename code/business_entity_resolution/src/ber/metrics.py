"""Official macro F0.5, including the challenge's singleton convention."""
from __future__ import annotations

import numpy as np


def entity_f05(true_ids, predicted_ids):
    truth, prediction = set(true_ids), set(predicted_ids)
    if not truth:
        return float(not prediction)
    tp = len(truth & prediction)
    return 5.0 * tp / (4.0 * len(prediction) + len(truth))


def score_pairs(reference, pool, pairs, probabilities, truth, threshold, mask=None):
    """Score every selected reference, even if retrieval returned zero candidates."""
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    selected = np.ones(len(reference), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    keep = (np.asarray(probabilities, dtype=np.float64) >= threshold) & selected[left]
    predicted_count = np.bincount(left[keep], minlength=len(reference))
    pool_ids = pool.entity_id.to_numpy()
    ref_ids = reference.entity_id.to_numpy()
    positive = np.fromiter(
        (pool_ids[r] in truth.get(ref_ids[l], set()) for l, r in zip(left, right)),
        dtype=bool, count=len(pairs),
    )
    tp = np.bincount(left[keep & positive], minlength=len(reference))
    true_count = np.asarray([len(truth.get(x, set())) for x in ref_ids])
    denominator = 4 * predicted_count + true_count
    scores = np.divide(5.0 * tp, denominator, out=np.zeros(len(reference)), where=denominator > 0)
    scores[(true_count == 0) & (predicted_count == 0)] = 1.0
    total_tp, total_pred, total_true = tp[selected].sum(), predicted_count[selected].sum(), true_count[selected].sum()
    result = {
        'macro_f05': float(scores[selected].mean()),
        'micro_precision': float(total_tp / total_pred) if total_pred else 0.0,
        'micro_recall': float(total_tp / total_true) if total_true else 0.0,
        'reference_count': int(selected.sum()),
        'predicted_links': int(total_pred),
        'true_links': int(total_true),
        'threshold': float(threshold),
    }
    for country in sorted(reference.loc[selected, 'country'].unique()):
        cm = selected & reference.country.eq(country).to_numpy()
        result[f'f05_{country}'] = float(scores[cm].mean())
    sm = selected & (true_count == 0)
    result['singleton_accuracy'] = float((predicted_count[sm] == 0).mean()) if sm.any() else None
    return result


def pair_labels(reference, pool, pairs, truth):
    ids1, ids2 = reference.entity_id.to_numpy(), pool.entity_id.to_numpy()
    return np.fromiter(
        (ids2[r] in truth.get(ids1[l], set()) for l, r in zip(pairs.left_idx, pairs.right_idx)),
        dtype=np.uint8, count=len(pairs),
    )


def candidate_report(reference, pool, pairs, truth, mask=None):
    labels = pair_labels(reference, pool, pairs, truth)
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    selected = np.ones(len(reference), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    found = np.bincount(left, weights=labels, minlength=len(reference))
    true_count = np.asarray([len(truth.get(x, set())) for x in reference.entity_id])
    nonempty = selected & (true_count > 0)
    result = {
        'pairs': int(selected[left].sum()),
        'pairs_per_reference': float(selected[left].sum() / selected.sum()),
        'link_recall': float(found[selected].sum() / true_count[selected].sum()),
        'complete_match_set_recall': float((found[nonempty] == true_count[nonempty]).mean()),
    }
    oracle = score_pairs(reference, pool, pairs, labels, truth, 0.5, selected)
    result['oracle_macro_f05'] = oracle['macro_f05']
    return result


def tune_threshold(reference, pool, pairs, probabilities, truth, mask, thresholds=None):
    """Tune only on the tuning split; held-out evaluation is never used here."""
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    labels = pair_labels(reference, pool, pairs, truth)
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        raise ValueError('Threshold tuning requires at least one reference')
    if thresholds is None:
        tuning_scores = np.asarray(probabilities)[selected[left]]
        maximum = float(tuning_scores.max()) if len(tuning_scores) else 1.0
        reject_all = np.nextafter(maximum, float('inf'))
        thresholds = np.unique(np.r_[0.0, np.linspace(0.05, 0.9, 35), .925, .95, .975, .99, .995, reject_all])
    true_count = np.asarray([len(truth.get(x, set())) for x in reference.entity_id])
    best = (-1.0, 0.5)
    trace = []
    for threshold in thresholds:
        keep = np.asarray(probabilities, dtype=np.float64) >= threshold
        count = np.bincount(left[keep], minlength=len(reference))
        tp = np.bincount(left[keep & labels.astype(bool)], minlength=len(reference))
        denominator = 4 * count + true_count
        values = np.divide(5.0 * tp, denominator, out=np.zeros(len(reference)), where=denominator > 0)
        values[(true_count == 0) & (count == 0)] = 1.0
        score = float(values[selected].mean())
        trace.append({'threshold': float(threshold), 'macro_f05': score})
        if (score, float(threshold)) > best:
            best = (score, float(threshold))
    return best[1], trace
