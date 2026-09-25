"""Paired bootstrap uncertainty over complete Source 1 evaluation entities.

Models, retrieval settings, and thresholds must already be fixed. Resampling is
over reference businesses, never individual candidate pairs. The same bootstrap
indices are used for both models, including references with no candidates and
singletons. This utility does not train, tune, or open any dataset labels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def paired_bootstrap(left, right=None, *, n_bootstrap=10_000, confidence=0.95,
                     seed=20260925, strata=None, batch_size=200):
    """Percentile CIs for macro scores and their paired left-minus-right difference.

Inputs contain one score per reference, in the same order. Optional ``strata``
preserve each stratum's observed reference count in every replicate, e.g. for a
fixed country mix. Without strata this is ordinary paired entity resampling.
The returned interval is conditional on the supplied fixed scoring procedure;
it does not account for selecting models/thresholds on these same references.
"""
    left = np.asarray(left, dtype=np.float64)
    if left.ndim != 1 or len(left) < 2 or not np.isfinite(left).all():
        raise ValueError("left must contain at least two finite per-reference scores")
    arrays = [left]
    if right is not None:
        right = np.asarray(right, dtype=np.float64)
        if right.shape != left.shape or not np.isfinite(right).all():
            raise ValueError("right must contain one finite score per left reference")
        arrays.append(right)
    if n_bootstrap < 100 or batch_size < 1 or not 0 < confidence < 1:
        raise ValueError("Require >=100 replicates, positive batch size, and confidence in (0,1)")
    values = np.column_stack(arrays)
    if strata is None:
        groups = [np.arange(len(left))]
    else:
        labels = np.asarray(strata, dtype=str)
        if labels.shape != left.shape:
            raise ValueError("strata must contain one label per reference")
        groups = [np.flatnonzero(labels == key) for key in sorted(set(labels))]
    rng = np.random.default_rng(seed)
    replicates = np.empty((n_bootstrap, values.shape[1]), dtype=np.float64)
    for start in range(0, n_bootstrap, batch_size):
        stop = min(start + batch_size, n_bootstrap)
        totals = np.zeros((stop - start, values.shape[1]), dtype=np.float64)
        for group in groups:
            positions = group[rng.integers(0, len(group), size=(stop - start, len(group)))]
            totals += values[positions].sum(axis=1)
        replicates[start:stop] = totals / len(left)
    names = ["left_mean"]
    estimates = [float(left.mean())]
    if right is not None:
        names += ["right_mean", "left_minus_right"]
        estimates += [float(right.mean()), float((left - right).mean())]
        replicates = np.column_stack((replicates, replicates[:, 0] - replicates[:, 1]))
    tail = (1 - confidence) / 2
    limits = np.quantile(replicates, [tail, 1 - tail], axis=0)
    result = {
        "reference_count": len(left), "n_bootstrap": int(n_bootstrap),
        "confidence": float(confidence), "seed": int(seed),
        "sampling_unit": "source1_entity", "paired": right is not None,
        "stratum_sizes": [int(len(group)) for group in groups] if strata is not None else None,
        "method": "percentile bootstrap",
        "estimates": dict(zip(names, estimates)),
        "confidence_intervals": {name: [float(limits[0, i]), float(limits[1, i])]
                                 for i, name in enumerate(names)},
        "bootstrap_standard_errors": {name: float(replicates[:, i].std(ddof=1))
                                      for i, name in enumerate(names)},
        "limitation": "Conditional on fixed models, retrieval and thresholds; tuning-set intervals do not establish held-out generalization.",
    }
    return result


def aligned_scores(left_frame, right_frame=None, *, id_column="source1_entity_id",
                   score_column="macro_f05", stratum_column=None):
    """Align complete reference populations by ID; reject drops and duplicates."""
    for label, frame in (("left", left_frame), ("right", right_frame)):
        if frame is None:
            continue
        if id_column not in frame or score_column not in frame:
            raise ValueError(f"{label} needs {id_column} and {score_column}")
        if frame[id_column].isna().any() or frame[id_column].duplicated().any():
            raise ValueError(f"{label} reference IDs must be present and unique")
    left = left_frame[score_column].to_numpy(dtype=np.float64)
    right = None
    strata = None
    if stratum_column is not None:
        if stratum_column not in left_frame or left_frame[stratum_column].isna().any():
            raise ValueError("left is missing a stratum label")
        strata = left_frame[stratum_column].astype(str).to_numpy()
    if right_frame is not None:
        if set(left_frame[id_column]) != set(right_frame[id_column]):
            raise ValueError("Paired comparison requires identical reference ID sets; no inner-join omissions")
        aligned = right_frame.set_index(id_column).loc[left_frame[id_column]]
        right = aligned[score_column].to_numpy(dtype=np.float64)
        if stratum_column is not None and stratum_column in aligned:
            if not np.array_equal(strata, aligned[stratum_column].astype(str).to_numpy()):
                raise ValueError("Paired reference stratum labels disagree")
    return left, right, strata


def read_table(path):
    path = Path(path)
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, sep="\t")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True, help="One score per reference, TSV or Parquet")
    parser.add_argument("--right", help="Optional paired model/situation scores on identical reference IDs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--id-column", default="source1_entity_id")
    parser.add_argument("--score-column", default="macro_f05")
    parser.add_argument("--stratify-column")
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    left_frame = read_table(args.left)
    right_frame = read_table(args.right) if args.right else None
    left, right, strata = aligned_scores(left_frame, right_frame, id_column=args.id_column,
                                         score_column=args.score_column, stratum_column=args.stratify_column)
    result = paired_bootstrap(left, right, n_bootstrap=args.replicates,
                              confidence=args.confidence, seed=args.seed, strata=strata)
    result.update(left_file=str(Path(args.left).resolve()),
                  right_file=str(Path(args.right).resolve()) if args.right else None,
                  score_column=args.score_column, stratify_column=args.stratify_column)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()
