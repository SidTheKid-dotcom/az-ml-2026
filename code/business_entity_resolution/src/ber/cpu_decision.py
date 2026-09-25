"""Replay the frozen medium TUNE matcher and diagnose bounded CPU decision rules.

Only the completed TRAIN/TUNE dense candidate artifact is accepted. The 12,001-S1
holdout and official test data are neither inputs nor selection populations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cross_encoder_experiment import blend_probabilities, detailed_score
from .experiment import json_write, load_data
from .hf_experiment import frame_digest
from .metrics import pair_labels
from .numeric_features import NAMES as NUMERIC_NAMES
from .uncertainty import paired_bootstrap


TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
TREE_FLOORS = (.25, .50)
DISAGREEMENT_TREE_FLOOR = .50
DISAGREEMENT_CE_FLOOR = .999


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_verified(data_dir: Path, scores_dir: Path, features_dir: Path):
    metadata = json.loads((data_dir / "metadata.json").read_text())
    complete = json.loads((data_dir / "complete.json").read_text())
    config = json.loads((scores_dir / "experiment_config.json").read_text())
    saved = json.loads((scores_dir / "results.json").read_text())
    if (complete.get("state") != "complete" or complete.get("metadata") != metadata
            or complete["config"]["labels_used_for_mining"] is not False
            or metadata["positive_candidate_injection"] is not False
            or complete["config"]["pool_embedding_identity"]["scope"] != "full_pool"):
        raise ValueError("Input candidates are not completed, label-free full-pool mining")
    if (config["evaluation_used"] or saved["evaluation_used"]
            or not config["tune_only_inference"] or config["rerank_cutoff"] != .01
            or config["full_pool_records"] != metadata["mining_full_pool_records"]
            or config["retrieval_checkpoint_sha256"] != complete["config"]["checkpoint_sha256"]):
        raise ValueError("Expected frozen medium, tune-only, full-pool reranking inputs")
    reference, pool, truth = load_data(data_dir)
    if (reference.entity_id.duplicated().any() or pool.entity_id.duplicated().any()
            or set(reference.split.unique()) != {"train", "tune"}):
        raise ValueError("Only unique TRAIN/TUNE reference and target IDs are allowed")
    if (frame_digest(reference, ["entity_id", "business_name", "business_address", "country", "split"])
            != config["reference_digest"]
            or frame_digest(pool, ["entity_id", "business_name", "business_address", "country"])
            != config["pool_digest"]):
        raise ValueError("Reference/pool row identities differ from the saved reranker")
    tune = reference.split.eq("tune").to_numpy()
    tune_ids = sorted(reference.entity_id[tune])
    if (len(tune_ids) != 12000 or hashlib.sha256("\n".join(tune_ids).encode()).hexdigest()
            != TUNE_ID_SHA256):
        raise ValueError("The expected untouched medium TUNE split is missing")
    train_ids = reference.entity_id[~tune]
    train_truth = {key: sorted(truth.get(key, set())) for key in train_ids}
    if hashlib.sha256(json.dumps(train_truth, sort_keys=True).encode()).hexdigest() != config["training_truth_digest"]:
        raise ValueError("Saved TRAIN truth identity disagrees")
    pairs = pd.read_parquet(scores_dir / "pairs.parquet")
    if (len(pairs) != metadata["candidate_pairs"]
            or frame_digest(pairs, ["left_idx", "right_idx"]) != config["pair_digest"]
            or pairs.duplicated(["left_idx", "right_idx"]).any()
            or pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
            or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool)):
        raise ValueError("Saved candidate pairs, row order, or index bounds disagree")
    feature_pairs = pd.read_parquet(features_dir / "pairs.parquet", columns=["left_idx", "right_idx"])
    if not np.array_equal(pairs[["left_idx", "right_idx"]].to_numpy(), feature_pairs.to_numpy()):
        raise ValueError("Numeric features are not aligned with the scored candidate order")
    numeric = pd.read_parquet(features_dir / "numeric_features.parquet")
    if len(numeric) != len(pairs) or list(numeric) != NUMERIC_NAMES:
        raise ValueError("Numeric feature row count/order/semantics differ from the matcher")
    arrays = {name: np.load(scores_dir / f"{name}.npy", mmap_mode="r") for name in (
        "base_probabilities", "cross_probabilities", "reranked_mask", "probabilities")}
    base, cross, gate, saved_blend = (arrays[name] for name in arrays)
    if any(array.shape != (len(pairs),) for array in arrays.values()) or gate.dtype != np.bool_:
        raise ValueError("Saved scores/gate are not aligned with every candidate pair")
    if any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in (base, cross, saved_blend)):
        raise ValueError("Saved probabilities must be finite in [0,1]")
    if hashlib.sha256(base.tobytes()).hexdigest() != config["base_probability_digest"]:
        raise ValueError("The original tree probabilities changed")
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    if not np.array_equal(gate, (base >= .01) & tune[left]) or np.any(cross[~gate] != 0):
        raise ValueError("Unscored CE pairs must not be treated as valid CE predictions")
    blend = blend_probabilities(base, cross, gate, .75)
    if not np.array_equal(saved_blend, blend):
        raise ValueError("The selected saved probabilities are not the specified gated 75% blend")
    input_paths = {
        "reference": data_dir / "reference.parquet", "pool": data_dir / "pool.parquet",
        "truth": data_dir / "truth.json", "data_metadata": data_dir / "metadata.json",
        "data_complete": data_dir / "complete.json", "pairs": scores_dir / "pairs.parquet",
        "base": scores_dir / "base_probabilities.npy", "cross": scores_dir / "cross_probabilities.npy",
        "gate": scores_dir / "reranked_mask.npy", "selected_blend": scores_dir / "probabilities.npy",
        "saved_config": scores_dir / "experiment_config.json", "saved_report": scores_dir / "results.json",
        "numeric": features_dir / "numeric_features.parquet",
        "numeric_pairs": features_dir / "pairs.parquet",
    }
    provenance = {name: {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
                  for name, path in input_paths.items()}
    provenance.update(reference_digest=config["reference_digest"], pool_digest=config["pool_digest"],
                      pair_digest=config["pair_digest"], tune_id_sha256=TUNE_ID_SHA256,
                      encoder_checkpoint_sha256=config["retrieval_checkpoint_sha256"],
                      reranker_model=config["model_id"], threshold=.775, tree_gate=.01)
    return reference, pool, truth, pairs, numeric, base, cross, gate, blend, tune, saved, provenance


def diagnostics(reference, pool, truth, pairs, numeric, base, cross, blend, tune):
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    labels = pair_labels(reference, pool, pairs, truth).astype(bool)
    keep = (blend >= .775) & tune[left]
    fp, tp = keep & ~labels, keep & labels
    missed = ~keep & labels & tune[left]
    counts = {}
    for name in ("num_unit_conflict", "num_name_conflict", "num_postcode_conflict",
                 "num_alpha_prefix_conflict_same_digits", "num_alpha_suffix_conflict_same_digits",
                 "num_short_small_offset", "num_shared_only_long"):
        conflict = numeric[name].to_numpy() > 0
        counts[name] = {"false_links": int((fp & conflict).sum()),
                        "true_links": int((tp & conflict).sum()),
                        "missed_retrieved_true_links": int((missed & conflict).sum())}
    out = {"accepted_false_links": int(fp.sum()), "accepted_true_links": int(tp.sum()),
           "retrieved_true_links_rejected": int(missed.sum()), "numeric_conflicts": counts,
           "score_quartiles": {label: {kind: [float(x) for x in np.quantile(a[mask], [.25, .5, .75])]
                                      for kind, a in (("tree", base), ("ce", cross), ("blend", blend))}
                               for label, mask in (("false_links", fp), ("true_links", tp))}}
    return out, labels


def entity_counts(reference, truth, left, labels, accepted):
    """Per-S1 official score, including every absent retrieval and singleton."""
    predicted = np.bincount(left[accepted], minlength=len(reference))
    tp = np.bincount(left[accepted & labels], minlength=len(reference))
    actual = np.fromiter((len(truth.get(key, set())) for key in reference.entity_id),
                         dtype=np.int32, count=len(reference))
    denominator = 4 * predicted + actual
    f05 = np.divide(5.0 * tp, denominator, out=np.zeros(len(reference)), where=denominator > 0)
    f1 = np.divide(2.0 * tp, predicted + actual, out=np.zeros(len(reference)),
                   where=predicted + actual > 0)
    singleton_correct = (actual == 0) & (predicted == 0)
    f05[singleton_correct] = f1[singleton_correct] = 1.0
    return {"f05": f05, "f1": f1, "predicted": predicted, "tp": tp, "actual": actual}


def script_breakdown(reference, tune, baseline, variant):
    """Non-ASCII names are a coarse observable script proxy, not country transfer."""
    name_non_ascii = np.fromiter((not str(name).isascii() for name in reference.business_name),
                                 dtype=bool, count=len(reference))
    groups = {}
    for country in sorted(reference.country[tune].unique()):
        for label, flag in (("ascii_name", False), ("non_ascii_name", True)):
            selected = tune & reference.country.eq(country).to_numpy() & (name_non_ascii == flag)
            if selected.any():
                groups[f"{country}/{label}"] = {
                    "references": int(selected.sum()), "baseline_f05": float(baseline[selected].mean()),
                    "variant_f05": float(variant[selected].mean()),
                    "delta_f05": float((variant[selected] - baseline[selected]).mean())}
    return groups


def compare_vetoes(reference, pool, truth, pairs, base, cross, blend, tune, labels, baseline, replicates):
    """Change only the label-free veto of an already-accepted v1 pair."""
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    original_keep = (blend >= .775) & tune[left]
    original_entities = entity_counts(reference, truth, left, labels, original_keep)
    if not np.isclose(original_entities["f05"][tune].mean(), baseline["macro_f05"], atol=1e-12, rtol=0):
        raise ValueError("Per-entity scores do not reproduce the official baseline")
    rules = [(f"tree_floor_{floor:.2f}", base >= floor) for floor in TREE_FLOORS]
    # The global false-link CE upper quartile is <.999; true-link lower quartile
    # is >.9999. Spare low-tree links only when the CE is near its score ceiling.
    rules.append(("weak_tree_unsaturated_ce", (base >= DISAGREEMENT_TREE_FLOOR)
                  | (cross >= DISAGREEMENT_CE_FLOOR)))
    variants = []
    for name, supported in rules:
        started = time.monotonic()
        # This changes the decision only. Gate, 75/25 blend, threshold and all candidates stay fixed.
        probabilities = np.where(supported, blend, 0)
        accepted = (probabilities >= .775) & tune[left]
        if np.any(accepted & ~original_keep):
            raise ValueError("A conservative veto cannot add a new link")
        removed = original_keep & ~accepted
        report = detailed_score(reference, pool, pairs, probabilities, truth, .775, tune)
        entities = entity_counts(reference, truth, left, labels, accepted)
        if not np.isclose(entities["f05"][tune].mean(), report["macro_f05"], atol=1e-12, rtol=0):
            raise ValueError("Per-entity scores disagree with the official score")
        paired = paired_bootstrap(entities["f05"][tune], original_entities["f05"][tune],
                                  n_bootstrap=replicates, seed=20260925)
        delta = entities["f05"][tune] - original_entities["f05"][tune]
        variants.append({"rule": name, "fixed_blend_threshold": .775,
                         "score": report, "macro_f1": float(entities["f1"][tune].mean()),
                         "removed_false_links": int((removed & ~labels).sum()),
                         "removed_true_links": int((removed & labels).sum()),
                         "entities_improved": int((delta > 0).sum()),
                         "entities_worsened": int((delta < 0).sum()),
                         "entities_unchanged": int((delta == 0).sum()),
                         "paired_tune_bootstrap": paired,
                         "country_and_name_script": script_breakdown(
                             reference, tune, original_entities["f05"], entities["f05"]),
                         "evaluation_seconds": time.monotonic() - started})
    return float(original_entities["f1"][tune].mean()), variants


def run(args):
    started = time.monotonic()
    data_dir, scores_dir, features_dir, output = map(Path, (args.data, args.scores, args.features, args.output))
    reference, pool, truth, pairs, numeric, base, cross, gate, blend, tune, saved, provenance = read_verified(
        data_dir, scores_dir, features_dir)
    baseline = detailed_score(reference, pool, pairs, blend, truth, .775, tune)
    expected = saved["selected_by_tune"]["tune"]
    for key, value in expected.items():
        if key not in baseline or not np.isclose(baseline[key], value, rtol=0, atol=1e-12):
            raise ValueError(f"Baseline reproduction differs for {key}: {baseline.get(key)} != {value}")
    observed, labels = diagnostics(reference, pool, truth, pairs, numeric, base, cross, blend, tune)
    report = {"population": "medium TRAIN/TUNE only; all 10,320,219 training targets competed in retrieval",
              "reference_count": int(tune.sum()), "candidate_pairs": len(pairs),
              "tune_pairs": int(tune[pairs.left_idx.to_numpy(dtype=np.int64)].sum()),
              "scored_ce_pairs": int(gate.sum()), "provenance": provenance,
              "baseline": baseline, "diagnostics": observed,
              "runtime_seconds": time.monotonic() - started,
              "limitation": "TUNE reuse; no medium eval/official test labels or French labels inspected."}
    output.mkdir(parents=True, exist_ok=True)
    json_write(output / "baseline_diagnostics.json", report)
    baseline_f1, variants = compare_vetoes(reference, pool, truth, pairs, base, cross, blend, tune,
                                          labels, baseline, args.replicates)
    report.update(baseline_macro_f1=baseline_f1, comparisons=variants,
                  retrieval_ceiling=json.loads((data_dir / "metadata.json").read_text())["retrieval"]["tune"],
                  runtime_seconds=time.monotonic() - started)
    checkpoints = json.loads(Path(args.checkpoints).read_text())
    report["provenance"]["reranker_checkpoint_sha256"] = checkpoints["rerankers"]["medium_dense_bge_ce1800"]["checkpoint_sha256"]
    report["provenance"]["tree_checkpoint_sha256"] = checkpoints["rerankers"]["medium_dense_bge_ce1800"]["tree_components"]["original_gate_and_tree.joblib"]["sha256"]
    config = {"schema": 1, "seed": 20260925, "device": "cpu", "threads": 2,
              "hypotheses": [
                  "Unconditional pair-level tree support may remove CE-overruled false links.",
                  "Where tree support is weak, requiring near-saturated CE may spare true links while "
                  "removing less-certain false links (chosen after baseline score-quartile diagnosis)."],
              "changed_variable": "additional label-free acceptance veto on otherwise accepted v1 pairs",
              "tree_floors": list(TREE_FLOORS),
              "conditional_rule": {"reject_if_tree_below": DISAGREEMENT_TREE_FLOOR,
                                   "and_ce_below": DISAGREEMENT_CE_FLOOR},
              "baseline_tree_gate": .01, "blend_ce_weight": .75,
              "fixed_blend_threshold": .775, "bootstrap_replicates": args.replicates,
              "train_references": 35999, "tune_references": 12000,
              "full_competing_target_pool": 10320219, "candidate_pairs": len(pairs),
              "data_source": "artifacts/full_dense_medium (completed TRAIN/TUNE candidate cache)",
              "score_source": "artifacts/medium_dense_bge_ce1800 (saved aligned gated CE/tree scores)",
              "feature_source": "artifacts/medium_dense_sweep (aligned numeric features)",
              "split_id_sha256": TUNE_ID_SHA256, "pair_digest": provenance["pair_digest"],
              "evaluation_used": False, "no_threshold_retuning": True,
              "name_script_proxy": "Non-ASCII in reference business_name; not a full Unicode script classifier"}
    json_write(output / "experiment_config.json", config)
    json_write(output / "results.json", report)
    print(json.dumps({"baseline": baseline, "comparisons": variants,
                      "runtime_seconds": report["runtime_seconds"]}, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", default="docs/checkpoints.json")
    parser.add_argument("--replicates", type=int, default=2000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
