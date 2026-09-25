"""Train a CPU-only lexical-only rescue branch and test actual TUNE macro F0.5.

The dense v1 predictions stay fixed. A new CatBoost sees only TRAIN lexical
edges absent from dense retrieval; its acceptance threshold is selected on an
entity-disjoint TRAIN calibration partition. TUNE is scored once at that rule.
All candidate edges were already mined against the complete training target pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd

from .cpu_decision import entity_counts, file_sha256, read_verified
from .experiment import json_write, load_data
from .features import build_features
from .hf_experiment import record_texts
from .metrics import pair_labels
from .models import fit_model, predict_proba
from .numeric_features import NAMES as NUMERIC_NAMES, build_numeric_features
from .retrieval_ablation import _script
from .uncertainty import paired_bootstrap


SEED = 20260925
LEXICAL_RANK = 20
CALIBRATION_THRESHOLDS = (.001, .003, .01, .03, .1, .3, .5, .75, .9, .97, .99, .995)
TEXT_COLUMNS = (
    "name_ratio", "name_partial", "name_token_sort", "name_token_set",
    "core_ratio", "core_token_sort", "core_token_set", "name_token_jaccard",
    "name_token_left_coverage", "name_token_right_coverage", "name_exact", "core_exact",
    "address_ratio", "address_partial", "address_token_sort", "address_token_set",
    "address_token_jaccard", "address_token_left_coverage", "address_token_right_coverage",
    "address_exact", "raw_address_ratio", "number_disjoint", "short_number_disjoint",
    "long_number_disjoint", "name_len_ratio", "address_len_ratio", "name_address_min",
    "core_address_min", "name_number_conflict", "address_name_conflict", "source3",
    "name_native_right", "address_native_right",
)
LEXICAL_COLUMNS = ("name_score", "address_score", "retrieval_score", "retrieval_rank",
                   "phonetic_score", "phonetic_hit", "phonetic_only", "lexical_scores_missing")
MODEL_PARAMS = {"iterations": 250, "depth": 5, "learning_rate": .05,
                "thread_count": 2, "l2_leaf_reg": 5.0}


def id_sha(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def calibration_split(reference):
    train = reference.split.eq("train").to_numpy()
    selected = np.fromiter((int.from_bytes(hashlib.blake2b(
        f"{SEED}\0{entity_id}".encode(), digest_size=8).digest(), "big") % 5 == 0
        for entity_id in reference.entity_id), dtype=bool, count=len(reference))
    return train & ~selected, train & selected


def new_top20_edges(reference, dense_pool, dense_pairs, lexical_pool, lexical_pairs):
    left = dense_pairs.left_idx.to_numpy(dtype=np.int64)
    right = dense_pairs.right_idx.to_numpy(dtype=np.int64)
    mapping = pd.Index(lexical_pool.entity_id).get_indexer(dense_pool.entity_id)
    mapped = mapping[right]
    dense_keys = pd.DataFrame({"left_idx": left[mapped >= 0], "right_idx": mapped[mapped >= 0]})
    selected = (lexical_pairs.retrieval_rank.to_numpy() <= LEXICAL_RANK) | lexical_pairs.phonetic_hit.to_numpy()
    lexical = lexical_pairs.loc[selected].copy().reset_index(names="source_pair_position")
    if lexical.duplicated(["left_idx", "right_idx"]).any() or dense_keys.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Candidate pairs must be unique within each branch")
    joined = lexical.merge(dense_keys, on=["left_idx", "right_idx"], how="left",
                           validate="one_to_one", indicator=True, sort=False)
    new_edges = joined.loc[joined._merge.eq("left_only")].drop(columns="_merge").reset_index(drop=True)
    if (new_edges.left_idx.min() < 0 or new_edges.left_idx.max() >= len(reference)
            or new_edges.right_idx.min() < 0 or new_edges.right_idx.max() >= len(lexical_pool)):
        raise ValueError("Additional candidate positions escaped the completed pool")
    return new_edges


def pair_local_features(reference, pool, edges, output):
    """Use pair-local text/numeric evidence; never reuse stale union context features."""
    batches = []
    stage = time.monotonic()
    for start in range(0, len(edges), 10000):
        batch = edges.iloc[start:start + 10000].reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The copy keyword is deprecated")
            text = build_features(reference, pool, batch)[list(TEXT_COLUMNS)]
        numeric = build_numeric_features(reference, pool, batch)[NUMERIC_NAMES]
        lexical = batch[list(LEXICAL_COLUMNS)].astype(np.float32)
        lexical.columns = ["lex_" + column for column in LEXICAL_COLUMNS]
        batches.append(pd.concat([text, numeric, lexical], axis=1))
        if start == 0 or (start // 10000 + 1) % 10 == 0:
            print(f"CPU pair features {min(start + 10000, len(edges)):,}/{len(edges):,}", flush=True)
    features = pd.concat(batches, ignore_index=True).astype(np.float32)
    if len(features) != len(edges) or features.isna().any().any() or not np.isfinite(features.to_numpy()).all():
        raise ValueError("Additional-candidate features contain missing or nonfinite values")
    features.to_parquet(output / "features.parquet", index=False)
    return features, time.monotonic() - stage


def fit_indices(features, labels, fit_mask):
    positive = np.flatnonzero(fit_mask & labels)
    negative = np.flatnonzero(fit_mask & ~labels)
    if len(positive) < 100:
        raise ValueError("Too few TRAIN positives for a bounded lexical-only screen")
    sample_size = min(len(negative), 100 * len(positive))
    hard_size = round(.8 * sample_size)
    priority = np.maximum.reduce((features.lex_name_score.to_numpy(),
                                  features.lex_address_score.to_numpy(),
                                  features.lex_phonetic_score.to_numpy(),
                                  features.core_token_sort.to_numpy()))
    rng = np.random.default_rng(SEED)
    randomized = rng.permutation(negative)
    hardest = randomized[np.argsort(-priority[randomized], kind="stable")]
    hard = hardest[:hard_size]
    remainder = rng.choice(hardest[hard_size:], size=sample_size - hard_size, replace=False)
    return np.sort(np.r_[positive, hard, remainder]), len(positive), len(negative)


def choose_calibration(scores, truth):
    comparisons = []
    for threshold in CALIBRATION_THRESHOLDS:
        selected = scores >= threshold
        tp = int(truth[selected].sum())
        fp = int(selected.sum()) - tp
        comparisons.append({"threshold": threshold, "true_links": tp, "false_links": fp,
                            "utility_tp_minus_5fp": tp - 5 * fp})
    winner = max(comparisons, key=lambda row: (row["utility_tp_minus_5fp"], row["threshold"]))
    if winner["utility_tp_minus_5fp"] <= 0:
        winner = {"threshold": 1.01, "true_links": 0, "false_links": 0,
                  "utility_tp_minus_5fp": 0}
    return winner, comparisons


def score_added(reference, baseline, left, labels, accepted, tune):
    extra_count = np.bincount(left[accepted], minlength=len(reference))
    extra_tp = np.bincount(left[accepted & labels], minlength=len(reference))
    predicted = baseline["predicted"] + extra_count
    tp = baseline["tp"] + extra_tp
    actual = baseline["actual"]
    denominator = 4 * predicted + actual
    f05 = np.divide(5.0 * tp, denominator, out=np.zeros(len(reference)), where=denominator > 0)
    f1 = np.divide(2.0 * tp, predicted + actual, out=np.zeros(len(reference)),
                   where=predicted + actual > 0)
    singleton = actual == 0
    f05[singleton & (predicted == 0)] = f1[singleton & (predicted == 0)] = 1.0
    result = {"macro_f05": float(f05[tune].mean()), "macro_f1": float(f1[tune].mean()),
              "precision": float(tp[tune].sum() / predicted[tune].sum()),
              "recall": float(tp[tune].sum() / actual[tune].sum()),
              "predicted_links": int(predicted[tune].sum()), "true_links": int(actual[tune].sum()),
              "false_links": int((predicted[tune] - tp[tune]).sum()),
              "singleton_accuracy": float((predicted[tune & singleton] == 0).mean()),
              "singleton_false_merges": int(np.sum((tune & singleton) & (predicted > 0))),
              "added_true_links": int(extra_tp[tune].sum()),
              "added_false_links": int((extra_count[tune] - extra_tp[tune]).sum())}
    for country in sorted(reference.country[tune].unique()):
        selected = tune & reference.country.eq(country).to_numpy()
        result[f"f05_{country}"] = float(f05[selected].mean())
        result[f"added_true_{country}"] = int(extra_tp[selected].sum())
        result[f"added_false_{country}"] = int((extra_count[selected] - extra_tp[selected]).sum())
    return result, f05


def run(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=\"\" before this local CPU experiment")
    started = time.monotonic()
    dense_dir, score_dir, feature_dir, lexical_dir, output = map(
        Path, (args.dense, args.scores, args.dense_features, args.lexical, args.output))
    output.mkdir(parents=True, exist_ok=True)
    reference, dense_pool, truth, dense_pairs, numeric, base, cross, gate, blend, tune, saved, provenance = read_verified(
        dense_dir, score_dir, feature_dir)
    del numeric, base, cross, gate
    dense_left = dense_pairs.left_idx.to_numpy(dtype=np.int64)
    dense_labels = pair_labels(reference, dense_pool, dense_pairs, truth).astype(bool)
    baseline = entity_counts(reference, truth, dense_left, dense_labels, (blend >= .775) & tune[dense_left])
    baseline_score = saved["selected_by_tune"]["tune"]
    if not np.isclose(baseline["f05"][tune].mean(), baseline_score["macro_f05"], rtol=0, atol=1e-12):
        raise ValueError("Saved v1 predictions did not reproduce their official TUNE score")
    del dense_labels
    previous = json.loads(Path(args.union_report).read_text())
    hashes = previous["input_sha256"]
    for name, path in (("dense_pairs", score_dir / "pairs.parquet"),
                       ("dense_reference", dense_dir / "reference.parquet"),
                       ("dense_pool", dense_dir / "pool.parquet"),
                       ("dense_truth", dense_dir / "truth.json"),
                       ("lex_reference", lexical_dir / "reference.parquet"),
                       ("lex_pool", lexical_dir / "pool.parquet"),
                       ("lex_pairs", lexical_dir / "candidate_pairs.parquet"),
                       ("lex_truth", lexical_dir / "truth.json")):
        if file_sha256(path) != hashes[name]:
            raise ValueError(f"Completed retrieval-union source changed: {name}")
    lex_ref, lex_pool, lex_truth = load_data(lexical_dir)
    if (not reference.equals(lex_ref) or truth != lex_truth
            or previous["full_competing_target_pool"] != 10320219):
        raise ValueError("The lexical branch differs from the exact saved medium reference/truth")
    lex_pairs = pd.read_parquet(lexical_dir / "candidate_pairs.parquet")
    if len(lex_pairs) != 2987950 or lex_pairs.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Lexical full-pool candidate identities changed")
    edges = new_top20_edges(reference, dense_pool, dense_pairs, lex_pool, lex_pairs)
    left = edges.left_idx.to_numpy(dtype=np.int64)
    if int(tune[left].sum()) != previous["rank_cap_screens"]["top20_or_phonetic"]["new_pairs_vs_dense"]:
        raise ValueError("Top20 new-pair count differs from the controlled retrieval screen")
    labels = pair_labels(reference, lex_pool, edges, truth).astype(bool)
    if int(labels[tune[left]].sum()) != 152:
        raise ValueError("Top20 retrieved true-link count differs from the original screen")
    fit_refs, cal_refs = calibration_split(reference)
    split = reference.split.to_numpy()[left]
    if not np.array_equal(tune[left], split == "tune"):
        raise ValueError("Candidate split alignment changed")
    edges.to_parquet(output / "new_pairs.parquet", index=False)
    del dense_pairs, dense_pool, lex_pairs
    features, feature_seconds = pair_local_features(reference, lex_pool, edges, output)
    fit_mask, cal_mask, tune_mask = fit_refs[left], cal_refs[left], tune[left]
    fit_rows, positive_count, negative_count = fit_indices(features, labels, fit_mask)
    stage = time.monotonic()
    model = fit_model(features.iloc[fit_rows], labels[fit_rows], kind="catboost", device="CPU",
                      seed=SEED, **MODEL_PARAMS)
    training_seconds = time.monotonic() - stage
    cal_rows, tune_rows = np.flatnonzero(cal_mask), np.flatnonzero(tune_mask)
    stage = time.monotonic()
    cal_probs = predict_proba(model, features.iloc[cal_rows])
    tune_probs = predict_proba(model, features.iloc[tune_rows])
    inference_seconds = time.monotonic() - stage
    choice, sweep = choose_calibration(cal_probs, labels[cal_rows])
    accepted = np.zeros(len(edges), dtype=bool)
    accepted[tune_rows] = tune_probs >= choice["threshold"]
    score, f05 = score_added(reference, baseline, left, labels, accepted, tune)
    delta = f05[tune] - baseline["f05"][tune]
    paired = paired_bootstrap(f05[tune], baseline["f05"][tune], n_bootstrap=2000, seed=SEED)
    selected_rows = tune_rows[tune_probs >= choice["threshold"]]
    scripts = {}
    for row in selected_rows:
        label = _script(lex_pool.business_name.iloc[int(edges.right_idx.iloc[row])])
        group = scripts.setdefault(label, {"true_links": 0, "false_links": 0})
        group["true_links" if labels[row] else "false_links"] += 1
    checkpoint = output / "lexical_only_catboost.joblib"
    joblib.dump({"model": model, "feature_columns": list(features), "threshold": choice["threshold"],
                 "seed": SEED, "fit_reference_id_sha256": id_sha(reference.entity_id[fit_refs])}, checkpoint)
    np.save(output / "tune_pair_positions.npy", tune_rows)
    np.save(output / "tune_probabilities.npy", tune_probs)
    config = {"schema": 1, "seed": SEED, "device": "CPU", "threads": 2,
              "hypothesis": "A TRAIN-fitted lexical-only classifier can recover top20 union truth links with "
                            "few enough false links to improve the fixed dense-v1 TUNE macro F0.5.",
              "changed_variable": "add independently scored top20/phonetic lexical-only links to frozen dense-v1 decisions",
              "retrieval_source": "completed full-pool medium dense and lexical/phonetic candidate artifacts",
              "full_competing_target_pool": 10320219, "dense_input_pair_digest": provenance["pair_digest"],
              "lexical_input_pair_sha256": hashes["lex_pairs"],
              "tree_checkpoint_sha256": json.loads((Path(args.decision_report)).read_text())["provenance"]["tree_checkpoint_sha256"],
              "reranker_checkpoint_sha256": json.loads((Path(args.decision_report)).read_text())["provenance"]["reranker_checkpoint_sha256"],
              "fit_reference_count": int(fit_refs.sum()), "cal_reference_count": int(cal_refs.sum()),
              "tune_reference_count": int(tune.sum()),
              "fit_reference_id_sha256": id_sha(reference.entity_id[fit_refs]),
              "cal_reference_id_sha256": id_sha(reference.entity_id[cal_refs]),
              "tune_reference_id_sha256": provenance["tune_id_sha256"],
              "lexical_rank_cap": LEXICAL_RANK, "retain_native_phonetic_hits": True,
              "feature_columns": list(features), "pair_local_text_numeric_features_only": True,
              "train_negative_sampler": "100 negatives per positive: 80% hardest by score, 20% random; FIT only",
              "model": "CatBoostClassifier", "model_params": MODEL_PARAMS,
              "calibration_thresholds": list(CALIBRATION_THRESHOLDS),
              "calibration_objective": "maximize TP - 5*FP on disjoint TRAIN-cal; reject all if nonpositive",
              "v1_blend_threshold": .775, "v1_tree_rerank_gate": .01,
              "tune_threshold_fit": False, "holdout_labels_used": False}
    report = {"baseline": baseline_score, "calibration": {"choice": choice, "grid": sweep,
              "train_fit_positive_candidates": positive_count, "train_fit_negative_candidates": negative_count,
              "cal_true_candidates": int(labels[cal_rows].sum()), "cal_candidates": len(cal_rows)},
              "new_branch_tune": score, "additional_candidate_counts": {"all": len(edges),
              "fit": int(fit_mask.sum()), "cal": len(cal_rows), "tune": len(tune_rows),
              "tune_true": int(labels[tune_rows].sum())},
              "accepted_target_name_script": scripts, "paired_tune": paired,
              "entities_improved": int((delta > 0).sum()), "entities_worsened": int((delta < 0).sum()),
              "entities_unchanged": int((delta == 0).sum()),
              "checkpoint_sha256": file_sha256(checkpoint),
              "new_pair_sha256": file_sha256(output / "new_pairs.parquet"),
              "features_sha256": file_sha256(output / "features.parquet"),
              "tune_probability_sha256": file_sha256(output / "tune_probabilities.npy"),
              "timing_seconds": {"feature_build": feature_seconds, "training": training_seconds,
                                 "scoring_cal_and_tune": inference_seconds,
                                 "total": time.monotonic() - started},
              "scope": "35,999 original TRAIN S1s split by entity into FIT/CAL; all 12,000 medium TUNE "
                       "S1s and all originally mined full-pool candidates retained for evaluation.",
              "limitations": "TUNE has already been reused; this is development evidence, not independent "
                             "confirmation or French/test accuracy. CE scores are valid only for dense candidates. "
                             "A separately trained lexical branch may shift precision on unseen distractors."}
    json_write(output / "experiment_config.json", config)
    json_write(output / "results.json", report)
    print(json.dumps({"calibration_choice": choice, "baseline_f05": baseline_score["macro_f05"],
                      "new_branch_tune": score, "paired_tune_delta": paired["estimates"]["left_minus_right"],
                      "paired_tune_ci": paired["confidence_intervals"]["left_minus_right"],
                      "runtime_seconds": report["timing_seconds"]}, allow_nan=False), flush=True)
    return report


def prepare_ce_shortlist(args):
    """Freeze a tiny label-free CAL/TUNE text queue for later BGE scoring."""
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("Set CUDA_VISIBLE_DEVICES=\"\" before local preparation")
    output = Path(args.output)
    destination = Path(args.shortlist_output) if args.shortlist_output else output
    if args.ce_gate not in (.5, .75):
        raise ValueError("This bounded shortlist comparison supports only gates 0.50 and 0.75")
    if destination != output and destination.exists() and any(destination.iterdir()):
        raise FileExistsError("Use a new empty output directory for the expanded shortlist")
    report = json.loads((output / "results.json").read_text())
    config = json.loads((output / "experiment_config.json").read_text())
    gate = args.ce_gate  # Chosen from TRAIN-CAL diagnostics; no TUNE labels select a pair.
    cal_result = next(row for row in report["calibration"]["grid"] if row["threshold"] == gate)
    expected_counts = {0.75: (35, 83), 0.5: (52, 300)}
    if (cal_result["true_links"], cal_result["false_links"]) != expected_counts[gate]:
        raise ValueError("The TRAIN-CAL lexical gate is not the screened artifact")
    for filename, expected in (("new_pairs.parquet", report["new_pair_sha256"]),
                               ("features.parquet", report["features_sha256"]),
                               ("lexical_only_catboost.joblib", report["checkpoint_sha256"]),
                               ("tune_probabilities.npy", report["tune_probability_sha256"])):
        if file_sha256(output / filename) != expected:
            raise ValueError(f"Cached rescue input has changed: {filename}")
    reference = pd.read_parquet(Path(args.lexical) / "reference.parquet").fillna("")
    pool = pd.read_parquet(Path(args.lexical) / "pool.parquet").fillna("")
    _, cal_refs = calibration_split(reference)
    if id_sha(reference.entity_id[cal_refs]) != config["cal_reference_id_sha256"]:
        raise ValueError("Calibration reference IDs changed")
    edges = pd.read_parquet(output / "new_pairs.parquet")
    features = pd.read_parquet(output / "features.parquet")
    bundle = joblib.load(output / "lexical_only_catboost.joblib")
    if list(features) != config["feature_columns"] or len(edges) != len(features):
        raise ValueError("Candidate ordering/feature semantics changed")
    left = edges.left_idx.to_numpy(dtype=np.int64)
    cal_rows = np.flatnonzero(cal_refs[left])
    tune_rows = np.load(output / "tune_pair_positions.npy")
    tune_probs = np.load(output / "tune_probabilities.npy")
    if (len(tune_rows) != len(tune_probs) or len(cal_rows) != report["calibration"]["cal_candidates"]
            or not reference.split.eq("tune").to_numpy()[left[tune_rows]].all()):
        raise ValueError("Saved calibration/tune score coverage changed")
    cal_probs = predict_proba(bundle["model"], features.iloc[cal_rows])
    selected_cal = cal_rows[cal_probs >= gate]
    selected_tune = tune_rows[tune_probs >= gate]
    if len(selected_cal) != sum(expected_counts[gate]):
        raise ValueError("TRAIN-CAL CE shortlist changed")
    positions = np.r_[selected_cal, selected_tune]
    selected = edges.iloc[positions]
    left_text = record_texts(reference.iloc[selected.left_idx.to_numpy(dtype=np.int64)])
    right_text = record_texts(pool.iloc[selected.right_idx.to_numpy(dtype=np.int64)])
    frame = pd.DataFrame({
        "split": ["cal"] * len(selected_cal) + ["tune"] * len(selected_tune),
        "new_pair_position": positions,
        "source1_entity_id": reference.entity_id.to_numpy()[selected.left_idx.to_numpy(dtype=np.int64)],
        "target_entity_id": pool.entity_id.to_numpy()[selected.right_idx.to_numpy(dtype=np.int64)],
        "lexical_probability": np.r_[cal_probs[cal_probs >= gate], tune_probs[tune_probs >= gate]],
        "left_text": left_text, "right_text": right_text})
    if frame.duplicated(["source1_entity_id", "target_entity_id"]).any():
        raise ValueError("CE shortlist has duplicate links")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "ce_shortlist.parquet"
    frame.to_parquet(path, index=False)
    manifest = {"schema": 1, "gate": gate, "rule": f"lexical-only tree probability >= {gate:.2f}",
                "calibration_pairs": len(selected_cal), "tune_pairs": len(selected_tune),
                "gate_chosen_from": f"TRAIN-CAL diagnostics ({cal_result['true_links']} true, "
                                    f"{cal_result['false_links']} false); not TUNE labels",
                "labels_used_to_select_individual_pairs": False, "holdout_used": False,
                "candidate_source_sha256": report["new_pair_sha256"],
                "lexical_checkpoint_sha256": report["checkpoint_sha256"],
                "v1_reranker_checkpoint_sha256": config["reranker_checkpoint_sha256"],
                "cal_reference_id_sha256": config["cal_reference_id_sha256"],
                "tune_reference_id_sha256": config["tune_reference_id_sha256"],
                "shortlist_sha256": file_sha256(path),
                "usage": "After v1 release, score every row with frozen BGE; choose the CE rule on "
                         "TRAIN-CAL, then compare official macro F0.5 on complete medium TUNE."}
    json_write(destination / "ce_shortlist_manifest.json", manifest)
    print(json.dumps(manifest, allow_nan=False), flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--dense-features", required=True)
    parser.add_argument("--lexical", required=True)
    parser.add_argument("--union-report", required=True)
    parser.add_argument("--decision-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prepare-ce-shortlist", action="store_true",
                        help="Reuse saved CPU scores to freeze a label-free BGE queue; no retraining")
    parser.add_argument("--ce-gate", type=float, default=.75)
    parser.add_argument("--shortlist-output", help="New output for a different CAL-chosen shortlist gate")
    args = parser.parse_args()
    prepare_ce_shortlist(args) if args.prepare_ce_shortlist else run(args)


if __name__ == "__main__":
    main()
