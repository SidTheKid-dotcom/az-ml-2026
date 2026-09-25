"""TRAIN-CAL decision and complete medium TUNE scoring of newly scored BGE pairs.

Frozen v1 predictions are unchanged. The CE threshold uses only TRAIN-CAL truth;
TUNE truth is accessed after that choice and the score covers every TUNE S1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cpu_decision import entity_counts, file_sha256, read_verified
from .cpu_lexical_rescue import calibration_split, id_sha, score_added
from .experiment import json_write
from .metrics import pair_labels
from .retrieval_ablation import _script
from .uncertainty import paired_bootstrap


CE_THRESHOLDS = (.5, .75, .9, .95, .975, .99, .995, .999, .9995, .9999)


def choose_threshold(scores, labels):
    trace = []
    for threshold in CE_THRESHOLDS:
        selected = scores >= threshold
        tp = int(labels[selected].sum())
        fp = int(selected.sum()) - tp
        trace.append({"threshold": threshold, "true_links": tp, "false_links": fp,
                      "utility_tp_minus_5fp": tp - 5 * fp})
    best = max(trace, key=lambda row: (row["utility_tp_minus_5fp"], row["threshold"]))
    if best["utility_tp_minus_5fp"] <= 0:
        best = {"threshold": 1.01, "true_links": 0, "false_links": 0,
                "utility_tp_minus_5fp": 0}
    return best, trace


def run(args):
    started = time.monotonic()
    data_dir, score_dir, feature_dir, output = map(
        Path, (args.data, args.scores, args.features, args.output))
    shortlist, manifest_path, probs_path, gpu_report_path = map(
        Path, (args.shortlist, args.manifest, args.probabilities, args.gpu_report))
    manifest = json.loads(manifest_path.read_text())
    gpu_report = json.loads(gpu_report_path.read_text())
    if (file_sha256(shortlist) != manifest["shortlist_sha256"]
            or gpu_report["input_shortlist_sha256"] != manifest["shortlist_sha256"]
            or gpu_report["input_manifest_sha256"] != file_sha256(manifest_path)
            or gpu_report["checkpoint_identity_sha256"] != manifest["v1_reranker_checkpoint_sha256"]
            or file_sha256(probs_path) != gpu_report["probabilities_sha256"]
            or gpu_report["labels_read"] is not False):
        raise ValueError("The scored CE rows/checkpoint do not match the frozen label-free queue")
    queue = pd.read_parquet(shortlist)
    probabilities = np.load(probs_path)
    counts = {"cal": manifest["calibration_pairs"], "tune": manifest["tune_pairs"]}
    if (len(queue) != sum(counts.values()) or queue.split.value_counts().to_dict() != counts
            or gpu_report["calibration_rows"] != counts["cal"]
            or gpu_report["tune_rows"] != counts["tune"]
            or probabilities.shape != (len(queue),) or not np.isfinite(probabilities).all()
            or np.any((probabilities < 0) | (probabilities > 1))
            or queue.duplicated(["source1_entity_id", "target_entity_id"]).any()):
        raise ValueError("CE probabilities do not cover exactly the unique shortlisted pairs")
    reference, pool, truth, dense_pairs, numeric, base, cross, gate, blend, tune, saved, provenance = read_verified(
        data_dir, score_dir, feature_dir)
    del numeric, base, cross, gate
    fit_refs, cal_refs = calibration_split(reference)
    if (id_sha(reference.entity_id[cal_refs]) != manifest["cal_reference_id_sha256"]
            or provenance["tune_id_sha256"] != manifest["tune_reference_id_sha256"]):
        raise ValueError("Calibration or TUNE reference identities changed")
    left = pd.Index(reference.entity_id).get_indexer(queue.source1_entity_id)
    if (np.any(left < 0) or not reference.split.eq("tune").to_numpy()[left[queue.split.eq("tune")]].all()
            or not cal_refs[left[queue.split.eq("cal")]].all()):
        raise ValueError("CE shortlist crossed the saved reference split")
    pair_path = Path(args.new_pairs)
    pair_source = pd.read_parquet(pair_path, columns=["left_idx", "right_idx"])
    rescue_report = json.loads(Path(args.rescue_report).read_text())
    if file_sha256(pair_path) != rescue_report["new_pair_sha256"]:
        raise ValueError("New candidate pair artifact changed after lexical retrieval")
    positions = queue.new_pair_position.to_numpy(dtype=np.int64)
    if (len(set(positions)) != len(queue) or positions.min() < 0 or positions.max() >= len(pair_source)
            or not np.array_equal(pair_source.left_idx.to_numpy()[positions], left)):
        raise ValueError("Shortlist lost its original candidate pair order")
    lexical_pool_ids = pd.read_parquet(Path(args.lexical) / "pool.parquet", columns=["entity_id"]).entity_id.to_numpy()
    if not np.array_equal(lexical_pool_ids[pair_source.right_idx.to_numpy(dtype=np.int64)[positions]],
                          queue.target_entity_id.to_numpy()):
        raise ValueError("CE target IDs do not match the full-pool retrieved edges")

    cal = queue.split.eq("cal").to_numpy()
    # The threshold is fixed before opening TUNE labels for this matcher.
    cal_labels = np.fromiter((target in truth.get(source, set()) for source, target in
                              zip(queue.source1_entity_id[cal], queue.target_entity_id[cal])),
                             dtype=bool, count=int(cal.sum()))
    calibrated_choice, trace = choose_threshold(probabilities[cal], cal_labels)
    if args.fixed_threshold is None:
        choice = calibrated_choice
        choice_source = "TRAIN-CAL TP-5*FP optimization"
    else:
        matches = [row for row in trace if row["threshold"] == args.fixed_threshold]
        if len(matches) != 1:
            raise ValueError("Frozen CE threshold must be on the predeclared grid")
        choice = matches[0]
        choice_source = "frozen gate-0.75 TRAIN-CAL threshold; only the lexical gate changes"

    dense_left = dense_pairs.left_idx.to_numpy(dtype=np.int64)
    dense_labels = pair_labels(reference, pool, dense_pairs, truth).astype(bool)
    baseline = entity_counts(reference, truth, dense_left, dense_labels, (blend >= .775) & tune[dense_left])
    expected = saved["selected_by_tune"]["tune"]
    if not np.isclose(baseline["f05"][tune].mean(), expected["macro_f05"], atol=1e-12, rtol=0):
        raise ValueError("The complete v1 TUNE baseline was not reproduced")
    tune_rows = queue.split.eq("tune").to_numpy()
    tune_labels = np.fromiter((target in truth.get(source, set()) for source, target in
                               zip(queue.source1_entity_id[tune_rows], queue.target_entity_id[tune_rows])),
                              dtype=bool, count=int(tune_rows.sum()))
    labels = np.zeros(len(queue), dtype=bool)
    labels[cal], labels[tune_rows] = cal_labels, tune_labels
    accepted = tune_rows & (probabilities >= choice["threshold"])
    score, f05 = score_added(reference, baseline, left, labels, accepted, tune)
    paired = paired_bootstrap(f05[tune], baseline["f05"][tune], n_bootstrap=2000, seed=20260925)
    delta = f05[tune] - baseline["f05"][tune]
    groups = {}
    for row in np.flatnonzero(accepted):
        script = _script(queue.iloc[row].right_text.split(" | ", 1)[0])
        group = groups.setdefault(script, {"true_links": 0, "false_links": 0})
        group["true_links" if labels[row] else "false_links"] += 1
    gate_comparison = None
    if args.prior_decisions or args.prior_results:
        if not args.prior_decisions or not args.prior_results:
            raise ValueError("Supply both previous gate decisions and results")
        prior_path = Path(args.prior_decisions)
        prior_result = json.loads(Path(args.prior_results).read_text())
        if (file_sha256(prior_path) != prior_result["decisions_sha256"]
                or prior_result["train_calibration"]["choice"]["threshold"] != choice["threshold"]
                or not np.isclose(prior_result["baseline"]["macro_f05"], expected["macro_f05"], atol=1e-12)):
            raise ValueError("Prior gate is not a fixed-threshold comparison on the same v1 baseline")
        prior = pd.read_parquet(prior_path)
        if prior.duplicated(["source1_entity_id", "target_entity_id"]).any():
            raise ValueError("Prior shortlist decisions contain duplicate links")
        previous_left = pd.Index(reference.entity_id).get_indexer(prior.source1_entity_id)
        if np.any(previous_left < 0) or not reference.split.eq("tune").to_numpy()[
                previous_left[prior.accepted.to_numpy(dtype=bool)]].all():
            raise ValueError("Prior accepted links escaped the identical TUNE reference population")
        old_labels = np.fromiter((target in truth.get(source, set()) for source, target in
                                  zip(prior.source1_entity_id, prior.target_entity_id)),
                                 dtype=bool, count=len(prior))
        old_score, old_f05 = score_added(reference, baseline, previous_left, old_labels,
                                         prior.accepted.to_numpy(dtype=bool), tune)
        if not np.isclose(old_score["macro_f05"], prior_result["tune"]["macro_f05"], atol=1e-12):
            raise ValueError("Previous gate per-reference score was not reproduced")
        old_keys = set(zip(prior.loc[prior.accepted, "source1_entity_id"],
                           prior.loc[prior.accepted, "target_entity_id"]))
        new_keys = set(zip(queue.loc[accepted, "source1_entity_id"],
                           queue.loc[accepted, "target_entity_id"]))
        if not old_keys <= new_keys:
            raise ValueError("Wider lexical gate lost a previously accepted link")
        gate_delta = f05[tune] - old_f05[tune]
        gate_comparison = {"new_gate": manifest["gate"], "prior_gate": .75,
                           "shared_acceptances_preserved": True,
                           "incremental_true_links": score["added_true_links"] - old_score["added_true_links"],
                           "incremental_false_links": score["added_false_links"] - old_score["added_false_links"],
                           "f05_delta": float(gate_delta.mean()),
                           "paired_tune": paired_bootstrap(f05[tune], old_f05[tune],
                                                            n_bootstrap=2000, seed=20260925),
                           "entities_improved": int((gate_delta > 0).sum()),
                           "entities_worsened": int((gate_delta < 0).sum()),
                           "entities_unchanged": int((gate_delta == 0).sum())}
    decisions = queue[["split", "new_pair_position", "source1_entity_id", "target_entity_id"]].copy()
    decisions["ce_probability"] = probabilities
    decisions["accepted"] = accepted
    output.mkdir(parents=True, exist_ok=True)
    decisions.to_parquet(output / "decisions.parquet", index=False)
    config = {"schema": 1, "seed": 20260925, "threshold_grid": list(CE_THRESHOLDS),
              "threshold_source": choice_source,
              "lexical_gate": manifest["gate"], "v1_blend_weight_ce": .75,
              "v1_threshold": .775, "v1_tree_gate": .01,
              "dense_pair_digest": provenance["pair_digest"],
              "shortlist_sha256": manifest["shortlist_sha256"],
              "reranker_checkpoint_sha256": manifest["v1_reranker_checkpoint_sha256"],
              "reranker_weights_sha256": gpu_report["checkpoint_weights_sha256"],
              "train_cal_id_sha256": manifest["cal_reference_id_sha256"],
              "tune_id_sha256": manifest["tune_reference_id_sha256"],
              "target_pool_records": 10320219, "holdout_used": False,
              "no_true_candidate_injection": True,
              "ce_scores_only_for_shortlisted_pairs": True}
    report = {"baseline": expected, "train_calibration": {"choice": choice,
              "calibrated_choice": calibrated_choice, "grid": trace,
              "reference_count": int(cal_refs.sum()), "shortlisted_pairs": int(cal.sum()),
              "positive_shortlisted_pairs": int(cal_labels.sum())},
              "tune": score, "tune_shortlisted_pairs": int(tune_rows.sum()),
              "tune_shortlisted_true_pairs": int(tune_labels.sum()),
              "accepted_target_name_script": groups,
              "paired_gate_comparison": gate_comparison,
              "paired_tune": paired, "entities_improved": int((delta > 0).sum()),
              "entities_worsened": int((delta < 0).sum()),
              "entities_unchanged": int((delta == 0).sum()),
              "probabilities_sha256": gpu_report["probabilities_sha256"],
              "decisions_sha256": file_sha256(output / "decisions.parquet"),
              "gpu_score_report_sha256": file_sha256(gpu_report_path),
              "local_evaluation_seconds": time.monotonic() - started,
              "limitations": "TUNE was reused for earlier architecture choices; this CAL-selected rule is "
                             "a development comparison, not held-out or French/test accuracy. "
                             "No v1 holdout labels guide this choice; the newly reserved holdout remains sealed."}
    json_write(output / "experiment_config.json", config)
    json_write(output / "results.json", report)
    print(json.dumps({"train_calibration": report["train_calibration"], "tune": score,
                      "paired_delta": paired["estimates"]["left_minus_right"],
                      "paired_interval": paired["confidence_intervals"]["left_minus_right"],
                      "gpu_seconds": gpu_report["seconds"],
                      "local_seconds": report["local_evaluation_seconds"]}, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--lexical", required=True)
    parser.add_argument("--new-pairs", required=True)
    parser.add_argument("--rescue-report", required=True)
    parser.add_argument("--shortlist", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--gpu-report", required=True)
    parser.add_argument("--fixed-threshold", type=float,
                        help="Reuse the earlier TRAIN-CAL BGE threshold when only the lexical gate changes")
    parser.add_argument("--prior-decisions", help="Earlier gate's complete CAL/TUNE decisions")
    parser.add_argument("--prior-results", help="Earlier gate's complete official TUNE report")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
