"""Compare saved cross-encoder scores with another frozen tree on TUNE only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .cross_encoder_experiment import (
    _thresholds, blend_probabilities, detailed_score, load_base_bundle,
)
from .experiment import json_write, load_data
from .hf_experiment import frame_digest
from .metrics import pair_labels, tune_threshold
from .uncertainty import paired_bootstrap


def entity_scores(reference, pool, pairs, probabilities, truth, threshold, mask):
    """One complete score per selected S1, including singletons and missed truth."""
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    chosen = np.asarray(mask, dtype=bool)
    labels = pair_labels(reference, pool, pairs, truth).astype(bool)
    keep = (np.asarray(probabilities, dtype=np.float64) >= threshold) & chosen[left]
    predicted = np.bincount(left[keep], minlength=len(reference))
    tp = np.bincount(left[keep & labels], minlength=len(reference))
    actual = np.asarray([len(truth.get(entity, set())) for entity in reference.entity_id])
    denominator = 4 * predicted + actual
    scores = np.divide(5.0 * tp, denominator, out=np.zeros(len(reference)), where=denominator > 0)
    scores[(actual == 0) & (predicted == 0)] = 1.0
    return pd.DataFrame({
        "source1_entity_id": reference.entity_id.to_numpy()[chosen],
        "country": reference.country.to_numpy()[chosen],
        "macro_f05": scores[chosen], "true_links": actual[chosen],
        "predicted_links": predicted[chosen], "true_positive_links": tp[chosen],
        "false_positive_links": (predicted - tp)[chosen],
        "false_negative_links": (actual - tp)[chosen],
    })


def run(args):
    data, cross_dir, alternative_dir, output = map(
        Path, (args.data, args.cross_dir, args.alternative_dir, args.output))
    reference, pool, truth = load_data(data)
    if not reference.split.isin(["train", "tune"]).all():
        raise ValueError("This diagnostic accepts a TRAIN/TUNE-only dataset")
    config = json.loads((cross_dir / "experiment_config.json").read_text())
    original_report = json.loads((cross_dir / "results.json").read_text())
    if config["evaluation_used"] or original_report["evaluation_used"]:
        raise ValueError("Expected the tune-only cross-encoder experiment")
    for key, frame, columns in (
        ("reference_digest", reference, ["entity_id", "business_name", "business_address", "country", "split"]),
        ("pool_digest", pool, ["entity_id", "business_name", "business_address", "country"]),
    ):
        if frame_digest(frame, columns) != config[key]:
            raise ValueError(f"Cross-encoder {key} does not match this dataset")
    pairs = pd.read_parquet(cross_dir / "pairs.parquet")
    if frame_digest(pairs, ["left_idx", "right_idx"]) != config["pair_digest"]:
        raise ValueError("Saved cross-encoder pair digest changed")
    alternative, other_pairs, alternative_scores, _, provenance = load_base_bundle(alternative_dir, "tabular")
    columns = ["left_idx", "right_idx"]
    if not np.array_equal(pairs[columns].to_numpy(), other_pairs[columns].to_numpy()):
        raise ValueError("Alternative tree candidate pairs are not in the identical order")
    if pairs.duplicated(columns).any():
        raise ValueError("Duplicate candidate links would bias entity scoring")
    cross = np.load(cross_dir / "cross_probabilities.npy")
    original = np.load(cross_dir / "base_probabilities.npy")
    selected = np.load(cross_dir / "reranked_mask.npy")
    for label, values in (("cross", cross), ("original", original), ("selected", selected)):
        if values.shape != (len(pairs),):
            raise ValueError(f"{label} scores/gate do not align with saved candidates")
    if selected.dtype != np.bool_:
        raise ValueError("Cross-encoder gate must be boolean")
    if not np.isfinite(cross).all() or np.any((cross < 0) | (cross > 1)):
        raise ValueError("Invalid saved cross-encoder scores")
    tune = reference.split.eq("tune").to_numpy()
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    tune_pairs = tune[left]
    report = {
        "evaluation_used": False, "training_performed": False,
        "alternative_dir": str(alternative_dir.resolve()),
        "cross_encoder_dir": str(cross_dir.resolve()),
        "pair_count": len(pairs), "pair_digest": config["pair_digest"],
        "candidate_order_verified": True, "data_digests_verified": True,
        "alternative_provenance": provenance,
        "cross_encoder_gate": {
            "source": "saved original-tree gate, not recomputed from alternative tree",
            "original_tree_cutoff": config["rerank_cutoff"],
            "tune_pairs": int(tune_pairs.sum()),
            "tune_pairs_scored": int((tune_pairs & selected).sum()),
            "alternative_saved_threshold_matches_without_ce_score": int(
                (tune_pairs & ~selected & (alternative_scores >= alternative["threshold"])).sum()),
        },
        "alternative_at_saved_threshold": detailed_score(
            reference, pool, pairs, alternative_scores, truth, alternative["threshold"], tune),
    }
    output.mkdir(parents=True, exist_ok=True)
    variants = [("alternative_baseline", 0., alternative_scores),
                ("gated_alternative", 0., blend_probabilities(alternative_scores, cross, selected, 0.))]
    variants.extend((f"alternative_ce_{weight:g}", weight,
                     blend_probabilities(alternative_scores, cross, selected, weight))
                    for weight in (.25, .5, .75))
    comparisons, traces = [], {}
    for label, weight, values in variants:
        threshold, trace = tune_threshold(reference, pool, pairs, values, truth, tune,
                                           thresholds=_thresholds(values[tune_pairs]))
        score = detailed_score(reference, pool, pairs, values, truth, threshold, tune)
        frame = entity_scores(reference, pool, pairs, values, truth, threshold, tune)
        if not np.isclose(frame.macro_f05.mean(), score["macro_f05"], rtol=0, atol=1e-12):
            raise AssertionError("Entity scores disagree with the official macro metric")
        frame.to_csv(output / f"{label}_entities.tsv", sep="\t", index=False)
        comparisons.append({"variant": label, "cross_encoder_weight": weight,
                            "threshold": threshold, "tune": score})
        traces[label] = trace
    report["comparisons"] = comparisons
    report["selected_by_tune"] = max(comparisons, key=lambda item: (
        item["tune"]["macro_f05"], item["variant"] == "alternative_baseline"))

    original_variants = {item["variant"]: item for item in original_report["comparisons"]}
    frames = {}
    for weight in (.25, .5):
        label = f"blend_{weight:g}"
        record = original_variants[label]
        values = blend_probabilities(original, cross, selected, weight)
        frame = entity_scores(reference, pool, pairs, values, truth, record["threshold"], tune)
        if not np.isclose(frame.macro_f05.mean(), record["tune"]["macro_f05"], rtol=0, atol=1e-12):
            raise AssertionError("Saved original-tree blend score is not reproducible")
        if int(frame.false_positive_links.sum()) != record["tune"]["false_positive_links"]:
            raise AssertionError("Saved original-tree blend FP count is not reproducible")
        frames[weight] = frame
        frame.to_csv(output / f"original_{label}_entities.tsv", sep="\t", index=False)
    half, quarter = frames[.5], frames[.25]
    delta = half.macro_f05.to_numpy() - quarter.macro_f05.to_numpy()
    bootstrap = paired_bootstrap(half.macro_f05, quarter.macro_f05,
                                 n_bootstrap=args.replicates)
    bootstrap.update({
        "left_label": f"original_tree_ce_0.5_at_{original_variants['blend_0.5']['threshold']:g}",
        "right_label": f"original_tree_ce_0.25_at_{original_variants['blend_0.25']['threshold']:g}",
        "scope": "Tune only; weights and thresholds selected on these references",
        "entities_better_with_half": int((delta > 0).sum()),
        "entities_worse_with_half": int((delta < 0).sum()),
        "entities_equal": int((delta == 0).sum()),
        "false_positive_delta_half_minus_quarter": int(
            half.false_positive_links.sum() - quarter.false_positive_links.sum()),
        "true_positive_delta_half_minus_quarter": int(
            half.true_positive_links.sum() - quarter.true_positive_links.sum()),
        "by_country": {},
    })
    for country in sorted(half.country.unique()):
        mask = half.country.eq(country)
        bootstrap["by_country"][country] = paired_bootstrap(
            half.loc[mask, "macro_f05"], quarter.loc[mask, "macro_f05"],
            n_bootstrap=args.replicates)
    report["original_tree_near_tie"] = bootstrap
    report["limitation"] = (
        "All comparisons and bootstrap intervals are TUNE diagnostics, conditional on models, "
        "selected gates, blend weights and thresholds. They do not establish held-out generalization.")
    json_write(output / "results.json", report)
    json_write(output / "thresholds.json", traces)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cross-dir", required=True)
    parser.add_argument("--alternative-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--replicates", type=int, default=10_000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
