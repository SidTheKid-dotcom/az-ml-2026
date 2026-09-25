"""P6 label-free inference adapter; the full release is gated on a TUNE finalist.

This module does not choose a finalist, fit a calibrator, or read any truth file.
``run(..., fixture=True)`` is limited to tiny synthetic inputs and NEVER writes a
submission-ready marker. A real run needs a frozen JSON (schema 2), a completed
label-free TEST candidate directory, and a separate, post-freeze HOLDOUT-2
attestation. No benchmark prefix or partial-test option exists.

Frozen JSON contract (all paths may be relative to the JSON):

* selection: status="tune_finalist", finalist_id, tune_id_sha256, and evidence
  {path, sha256}. Evidence binds components_sha256(config) and attests TUNE
  selection without holdout labels.
* retrieval: kind="dense_k50_v1" or "test_union", encoder_checkpoint and
  encoder_sha256. Dense reuse requires the existing complete 86,627,200-pair
  artifact; test_union requires a NEW completed, label-free test candidate set.
* tree: checkpoint, sha256, feature_policy="as_saved" or "dense_key_shared".
* cross_encoder: checkpoint, sha256. Only tested one-logit classifiers work.
* blend: alpha and tree_gate; runtime: amp (fp32, bf16, selective_fp32),
  max_length, inference_batch_size, reference_batch_size, pad_to_multiple_of,
  threads and device. Selective FP32 also needs a TUNE parity artifact bound to
  this exact matcher, retrieval, decision and band.
* decision: kind="threshold_assignment" with threshold, or
  "p1_count_assignment" with calibration {kind, path, sha256[, review,
  review_sha256]}, source_weight, and assignment_delta=0. The calibration
  artifact is either the P1 experiment_config.json (kind="p1_saved", only for
  the original dense tree/CE score scale) or a new TUNE calibration artifact
  (kind="tune_refit", see _load_decision). Both retain TRAIN-only count priors;
  a refit includes prior_source {split, train_reference_ids_sha256,
  train_truth_sha256, full_competing_target_pool}.
* code_sha256: source_hashes() as frozen before the holdout. No v1 frozen
  pipeline/source hashes are rewritten.

After the chosen candidate has been scored and validated separately on the
complete HOLDOUT-2, the attestation binds this JSON's SHA256 to committed
prediction bytes. It records all 50,000 S1 IDs, three fixed output hashes
(matching, candidates, pair probabilities), the one-shot score and a promise
of no post-holdout selection. Only then may the full-test run create ready.json. The ready
marker is compatible with package_portal_submission.validated_release; call
``python -m ber.v2_release verify --release NEW_DIR`` before packaging the new
release path. Neither the adapter nor verification performs an upload.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import fcntl
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .cross_encoder_experiment import blend_probabilities, load_reranker
from .full_embeddings import atomic_json, checkpoint_identity, file_digest
from .models import predict_proba
from .predict import feature_batch
from .predict_streaming import accepted_mask, compact_batch, iter_reference_batches
from .rerank_inference import score_reranker_batched, selective_fp32_mask
from .v2_decision import (CountPosterior, EPS, TUNE_ID_SHA256, adjust_source,
                          predict_calibration)
from .v2_tree_train import feature_columns, union_feature_columns
from .validate_release import validate_release


TEST_REFERENCES = 1_732_544
TEST_TARGETS = 9_969_589
V1_DENSE_PAIRS = 86_627_200
TRAIN_TARGETS = 10_320_219
HOLDOUT2_IDS_SHA256 = "7c98dcaef35d5a392a051e0525f854a2a6ceb55c366616ff2184abd98aa4ae63"
ORIGINAL_TEST_SHA256 = {
    "test_source1.tsv": "3d4a32c54c2ca9c53fd7c2be105bf26f708f94c4d2f88eb370972a195665c2f5",
    "test_source2.tsv": "79d906c7497af2ace70aa277f6e334a652094909de99bd6c57b53420b6a7b2dd",
    "test_source3.tsv": "850942b11d2a4343486ed0834e28bce9f3b385f3fd497fd60ccf4ea3b8bda035",
}
OFFICIAL_VALIDATOR_SHA256 = "f96f59934383a15095914f507620c078c474e8f9c60e864b2d051ed173a22dfc"
RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
CLAIM_DTYPE = np.dtype([("left", "<i4"), ("right", "<i4"),
                        ("pair_row", "<i8"), ("score", "<f8")])
CODE_FILES = (
    "ber/v2_release.py", "ber/v2_decision.py", "ber/v2_tree_train.py",
    "ber/features.py", "ber/numeric_features.py", "ber/normalization.py",
    "ber/predict.py", "ber/predict_streaming.py", "ber/models.py",
    "ber/rerank_inference.py", "ber/cross_encoder_experiment.py",
    "ber/full_embeddings.py", "ber/validate_release.py",
    "package_portal_submission.py",
)


def source_hashes() -> dict[str, str]:
    """Pin inference/validation/packaging source without changing frozen v1."""
    root = Path(__file__).resolve().parents[1]
    return {name: file_digest(root / name) for name in CODE_FILES}


def components_sha256(config: dict) -> str:
    """Noncircular binding between TUNE selection evidence and frozen components."""
    components = {key: config[key] for key in
                  ("retrieval", "tree", "cross_encoder", "blend", "runtime", "decision")}
    return hashlib.sha256(json.dumps(components, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _resolved(root: Path, path: str) -> Path:
    value = Path(path)
    return (value if value.is_absolute() else root / value).resolve()


def _pinned(root: Path, item: dict, label: str, identities: dict[str, str]) -> tuple[Path, dict]:
    if not isinstance(item, dict) or not {"path", "sha256"} <= item.keys():
        raise ValueError(f"{label} needs a path and SHA256")
    path = _resolved(root, item["path"])
    if not path.is_file() or file_digest(path) != item["sha256"]:
        raise ValueError(f"{label} file changed or is missing: {path}")
    identities[str(path)] = item["sha256"]
    return path, _read_json(path)


def _numeric(value, label: str, *, minimum: float = 0., maximum: float = 1.) -> float:
    if (type(value) not in (float, int) or not np.isfinite(value)
            or not minimum <= value <= maximum):
        raise ValueError(f"{label} must be finite and within [{minimum}, {maximum}]")
    return float(value)


def _calibration_models(models: dict) -> None:
    if not isinstance(models, dict) or "__pooled__" not in models or "France" in models:
        raise ValueError("Country calibration needs a pooled, unlabeled-France fallback")
    for country, model in models.items():
        x, y = np.asarray(model["x"], dtype=np.float64), np.asarray(model["y"], dtype=np.float64)
        if (country not in {"__pooled__", "India", "US"} or x.ndim != 1 or len(x) < 2
                or x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all()
                or np.any(np.diff(x) <= 0) or np.any(np.diff(y) < -1e-12)
                or np.any((x < 0) | (x > 1)) or np.any((y <= 0) | (y >= 1))):
            raise ValueError(f"Invalid monotone, bounded {country} isotonic calibrator")
        _numeric(model["p_ungated"], f"{country} ungated probability", minimum=EPS, maximum=1-EPS)


def _score_identity(config: dict) -> dict:
    return {"tree_sha256": config["tree"]["sha256"],
            "cross_encoder_sha256": config["cross_encoder"]["sha256"],
            "retrieval_kind": config["retrieval"]["kind"],
            "alpha": config["blend"]["alpha"], "gate": config["blend"]["tree_gate"]}


def _load_decision(config: dict, root: Path, identities: dict[str, str], fixture: bool) -> dict:
    decision = config["decision"]
    if decision.get("assignment_delta") != 0:
        raise ValueError("The frozen release requires pure one-to-one assignment (delta=0)")
    kind = decision.get("kind")
    if kind == "threshold_assignment":
        threshold = _numeric(decision.get("threshold"), "Frozen threshold", minimum=EPS)
        return {"kind": kind, "threshold": threshold}
    if kind != "p1_count_assignment":
        raise ValueError("Unsupported frozen decision component")
    weight = _numeric(decision.get("source_weight"), "Source-balance weight", maximum=1.)
    calibration = decision["calibration"]
    path, artifact = _pinned(root, calibration, "TUNE calibrator", identities)
    if calibration.get("kind") == "p1_saved":
        _, review = _pinned(root, {"path": calibration["review"],
                                    "sha256": calibration["review_sha256"]},
                            "P1 TUNE review", identities)
        provenance = artifact["provenance"]
        if (artifact.get("tune_id_sha256") != TUNE_ID_SHA256
                or artifact.get("evaluation_or_old_holdout_used") is not False
                or artifact.get("france_labels_used") is not False
                or artifact.get("full_competing_target_pool") != TRAIN_TARGETS
                or artifact.get("tree_gate") != config["blend"]["tree_gate"]
                or artifact.get("blend_ce_weight") != config["blend"]["alpha"]
                or config["retrieval"]["kind"] != "dense_k50_v1"
                or provenance.get("tree_checkpoint_sha256") != config["tree"]["sha256"]
                or provenance.get("reranker_checkpoint_sha256") != config["cross_encoder"]["sha256"]
                or review.get("artifact_sha256", {}).get("experiment_config.json") != file_digest(path)
                or review["recommendation_for_new_holdout"]["final_calibrator"]
                != "experiment_config.json/outer_cross_fit/final_calibrator_fitted_on_all_tune"
                or review["recommendation_for_new_holdout"]["source_balance_weight"] != weight
                or review["recommendation_for_new_holdout"]["assignment_margin"] != 0):
            raise ValueError("P1 TUNE calibrator/review does not fit these frozen scores or choice")
        models = artifact["outer_cross_fit"]["final_calibrator_fitted_on_all_tune"]
        prior = artifact["count_model"]["training_only_prior"]
    elif calibration.get("kind") == "tune_refit":
        source = artifact.get("prior_source", {})
        if (artifact.get("schema") != 2 or artifact.get("fit_scope") != "TUNE"
                or artifact.get("tune_id_sha256") != TUNE_ID_SHA256
                or artifact.get("holdout_labels_used") is not False
                or artifact.get("full_competing_target_pool") != TRAIN_TARGETS
                or source.get("split") != "TRAIN"
                or source.get("full_competing_target_pool") != TRAIN_TARGETS
                or source.get("train_reference_ids_sha256") in (None, TUNE_ID_SHA256, HOLDOUT2_IDS_SHA256)
                or not source.get("train_truth_sha256")
                or artifact.get("score_identity") != _score_identity(config)
                or artifact.get("selection", {}).get("source_weight") != weight
                or artifact["selection"].get("assignment_delta") != 0):
            raise ValueError("Refitted TUNE calibration is unbound to the selected score scale")
        models, prior = artifact["calibrators"], artifact["training_only_prior"]
    else:
        raise ValueError("Only a verified saved-P1 or newly TUNE-refitted calibrator is supported")
    _calibration_models(models)
    if (not isinstance(prior.get("train_references"), int) or prior["train_references"] < 1
            or prior.get("train_true_links", 0) < 0):
        raise ValueError("Count prior must originate from TRAIN businesses")
    posterior = CountPosterior(np.asarray(prior["p_true_count"], dtype=np.float64),
                               prior["gate_true_link_discovery"], prior["pair_prevalence"])
    return {"kind": kind, "models": models, "posterior": posterior, "source_weight": weight,
            "artifact_sha256": file_digest(path)}


def _runtime(config: dict, root: Path, identities: dict[str, str], fixture: bool) -> dict:
    runtime = config["runtime"]
    if any(type(runtime.get(key)) is not int or runtime[key] < 1 for key in
           ("reference_batch_size", "inference_batch_size", "max_length", "threads", "pad_to_multiple_of")):
        raise ValueError("Frozen reference/CE batch, length, padding and thread counts must be positive integers")
    if runtime.get("length_bucketing") is not True:
        raise ValueError("Only the frozen, stable length-sorted CE inference adapter is supported")
    if runtime.get("amp") not in {"fp32", "bf16", "selective_fp32"}:
        raise ValueError("Unsupported frozen CE precision")
    if (not isinstance(runtime.get("device"), str) or not runtime["device"]
            or (not fixture and not runtime["device"].startswith("cuda"))):
        raise ValueError("A real release needs a frozen CUDA CE device")
    if config["decision"]["kind"] == "p1_count_assignment" and runtime.get("skip_provably_rejected_pairs", False):
        raise ValueError("The P1 count posterior requires CE scores for EVERY tree-gated candidate")
    if type(runtime.get("skip_provably_rejected_pairs", False)) is not bool:
        raise ValueError("Frozen upper-bound skip must be a boolean")
    if (type(runtime.get("max_pairs_per_batch", 200_000)) is not int
            or runtime.get("max_pairs_per_batch", 200_000) < 1):
        raise ValueError("A complete S1 candidate batch needs a positive pair limit")
    if runtime["amp"] == "selective_fp32":
        band = _numeric(runtime.get("fp32_band"), "FP32 band")
        anchor = _numeric(runtime.get("fp32_anchor"), "FP32 blend anchor", minimum=EPS)
        if not fixture:
            _, evidence = _pinned(root, runtime["tune_parity"], "TUNE precision parity", identities)
            if (evidence.get("scope") != "TUNE" or evidence.get("references") != 12_000
                    or evidence.get("full_competing_target_pool") != TRAIN_TARGETS
                    or evidence.get("passed") is not True
                    or evidence.get("full_fp32_decision_disagreements") != 0
                    or evidence.get("score_identity") != _score_identity(config)
                    or evidence.get("decision_kind") != config["decision"]["kind"]
                    or evidence.get("fp32_band") != band or evidence.get("fp32_anchor") != anchor):
                raise ValueError("Selective FP32 needs a full-TUNE, decision-level parity gate")
    elif runtime["amp"] == "bf16" and not fixture:
        if "fp32_band" in runtime or "fp32_anchor" in runtime:
            raise ValueError("BF16 scoring cannot use selective FP32 settings")
        _, evidence = _pinned(root, runtime["tune_parity"], "TUNE BF16 parity", identities)
        if (evidence.get("scope") != "TUNE" or evidence.get("references") != 12_000
                or evidence.get("full_competing_target_pool") != TRAIN_TARGETS
                or evidence.get("passed") is not True
                or evidence.get("score_identity") != _score_identity(config)
                or evidence.get("decision_kind") != config["decision"]["kind"]):
            raise ValueError("BF16 CE needs a decision-level TUNE parity gate")
    elif runtime["amp"] == "fp32" and any(key in runtime for key in ("fp32_band", "fp32_anchor", "tune_parity")):
        raise ValueError("Selective FP32 settings cannot leak into another CE precision")
    return runtime


def _test_sources(test_dir: Path, fixture: bool, identities: dict[str, str]) -> None:
    for name, expected in ORIGINAL_TEST_SHA256.items():
        path = test_dir / name
        observed = file_digest(path)
        if not fixture and observed != expected:
            raise ValueError(f"Official {name} differs from the verified supplied dataset")
        identities[str(path.resolve())] = observed


def _reference_matches_raw(reference: pd.DataFrame, path: Path, expected: int) -> None:
    if (len(reference) != expected or reference.entity_id.duplicated().any()
            or reference[list(RECORD_COLUMNS)].isna().any().any()):
        raise ValueError("Missing/duplicate/null Source 1 record in test candidates")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if next(reader) != list(RECORD_COLUMNS):
            raise ValueError("Original test Source 1 header changed")
        for index, (row, saved) in enumerate(zip_longest(
                reader, reference[list(RECORD_COLUMNS)].itertuples(index=False, name=None)), start=1):
            if (row is None or saved is None or row != list(saved)
                    or len(row) != len(RECORD_COLUMNS) or not row[0].startswith("S1-")):
                raise ValueError(f"Missing, reordered, duplicate, or mismatched Source 1 at row {index}")


def _pool_matches_raw(pool: pd.DataFrame, test_dir: Path, expected: int) -> None:
    if (pool.entity_id.duplicated().any() or pool.entity_id.isna().any()
            or pool[list(RECORD_COLUMNS)].isna().any().any() or pool.raw_row_idx.isna().any()):
        raise ValueError("Duplicate, missing or null compact target ID/raw row")
    raw_indices = pool.raw_row_idx.to_numpy()
    if (not np.issubdtype(raw_indices.dtype, np.integer)
            or (len(raw_indices) and (raw_indices[0] < 0 or raw_indices[-1] >= expected
                                     or np.any(raw_indices[1:] <= raw_indices[:-1])))):
        raise ValueError("Compact pool must preserve unique, increasing original raw target rows")
    selected = iter(pool[["raw_row_idx", *RECORD_COLUMNS]].itertuples(index=False, name=None))
    current = next(selected, None)
    seen_raw = set()  # Exact IDs, not hash-only uniqueness; freed before GPU scoring.
    count = 0
    for source, prefix in ((2, "S2-"), (3, "S3-")):
        with (test_dir / f"test_source{source}.tsv").open(encoding="utf-8", newline="") as stream:
            reader = csv.reader(stream, delimiter="\t")
            if next(reader) != list(RECORD_COLUMNS):
                raise ValueError(f"Original test Source {source} header changed")
            for row in reader:
                if len(row) != 4 or not row[0].startswith(prefix):
                    raise ValueError("Original target has an invalid ID or field count")
                if row[0] in seen_raw:
                    raise ValueError("Duplicate original test target ID")
                seen_raw.add(row[0])
                if current is not None and current[0] == count:
                    if list(current[1:]) != row:
                        raise ValueError(f"Candidate target ID/text/raw order differs at row {count}")
                    current = next(selected, None)
                count += 1
    if count != expected or current is not None:
        raise ValueError(f"Full test target pool/compact mapping is incomplete: {count} != {expected}")


@dataclass
class Prepared:
    config: dict
    config_sha256: str
    file_sha256: dict[str, str]
    checkpoint_identities: dict
    source_sha256: dict[str, str]
    reference: pd.DataFrame
    pool: pd.DataFrame
    pairs: Path
    pair_count: int
    decision: dict
    runtime: dict
    fixture: bool
    expected_references: int
    expected_targets: int


def prepare(config_path: Path, candidates: Path, test_dir: Path, validator: Path, *, fixture: bool = False) -> Prepared:
    """Read ONLY frozen configuration, label-free test records and TUNE artifacts."""
    config_path, candidates, test_dir, validator = (Path(p).resolve() for p in
                                                   (config_path, candidates, test_dir, validator))
    config = _read_json(config_path)
    if config.get("schema") != 2:
        raise ValueError("Expected a new v2 frozen config, not the historical v1 pipeline")
    if any(key in config for key in ("benchmark_references", "test_prefix", "truth_path", "holdout_truth")):
        raise ValueError("A release config cannot request a prefix or a truth file")
    if fixture:
        counts = config.get("fixture_counts", {})
        nref, ntarget = counts.get("references"), counts.get("targets")
        if (type(nref) is not int or type(ntarget) is not int
                or not 1 <= nref <= 32 or not 1 <= ntarget <= 64
                or config.get("selection", {}).get("scope") != "fixture"):
            raise ValueError("Fixture mode is limited to a tiny synthetic population")
    else:
        nref, ntarget = TEST_REFERENCES, TEST_TARGETS
        selection = config["selection"]
        if (selection.get("status") != "tune_finalist" or selection.get("scope") != "TUNE"
                or selection.get("tune_id_sha256") != TUNE_ID_SHA256
                or selection.get("components") != ["retrieval", "tree", "cross_encoder", "decision"]
                or not isinstance(selection.get("finalist_id"), str) or not selection["finalist_id"]):
            raise ValueError("P3/P4/P5 finalist and every supported component must be frozen on complete TUNE")
        if "fixture_counts" in config or selection.get("benchmark_only", False) is not False:
            raise ValueError("A full-test finalist cannot carry fixture or benchmark population limits")
    identities = {str(config_path): file_digest(config_path)}
    if not fixture:
        _, chosen = _pinned(config_path.parent, config["selection"]["evidence"],
                            "TUNE finalist evidence", identities)
        if (chosen.get("status") != "tune_finalist" or chosen.get("scope") != "TUNE"
                or chosen.get("tune_id_sha256") != TUNE_ID_SHA256
                or chosen.get("finalist_id") != config["selection"]["finalist_id"]
                or chosen.get("components") != config["selection"]["components"]
                or chosen.get("holdout2_labels_used") is not False
                or chosen.get("full_competing_target_pool") != TRAIN_TARGETS
                or chosen.get("components_sha256") != components_sha256(config)):
            raise ValueError("Selection evidence does not freeze this exact TUNE finalist")
    sources = source_hashes()
    if not fixture and config.get("code_sha256") != sources:
        raise ValueError("Frozen inference code differs from TUNE-finalist code hashes")
    _test_sources(test_dir, fixture, identities)
    if not fixture and file_digest(validator) != OFFICIAL_VALIDATOR_SHA256:
        raise ValueError("Only the unchanged official submission validator is accepted")
    identities[str(validator)] = file_digest(validator)
    tree_path = _resolved(config_path.parent, config["tree"]["checkpoint"])
    if file_digest(tree_path) != config["tree"]["sha256"]:
        raise ValueError("Frozen tree checkpoint changed")
    identities[str(tree_path)] = config["tree"]["sha256"]
    checkpoint_ids = {"tree": {"path": str(tree_path), "sha256": config["tree"]["sha256"]}}
    for key, entry in (("cross_encoder", config["cross_encoder"]), ("encoder", config["retrieval"])):
        directory = _resolved(config_path.parent, entry["checkpoint"] if key == "cross_encoder"
                              else entry["encoder_checkpoint"])
        actual = checkpoint_identity(directory)
        if actual["sha256"] != (entry["sha256"] if key == "cross_encoder"
                                else entry["encoder_sha256"]):
            raise ValueError(f"Frozen {key} checkpoint changed")
        checkpoint_ids[key] = actual
        for item in actual["files"]:
            identities[str(directory / item["path"])] = item["sha256"]
    alpha = _numeric(config["blend"]["alpha"], "CE blend weight", minimum=EPS)
    gate = _numeric(config["blend"]["tree_gate"], "Tree gate", minimum=EPS)
    if config["retrieval"].get("kind") not in ("dense_k50_v1", "test_union"):
        raise ValueError("Only fixed dense K50 or a new completed test union may be released")
    policy = "as_saved" if config["retrieval"]["kind"] == "dense_k50_v1" else "dense_key_shared"
    if config["tree"].get("feature_policy") != policy:
        raise ValueError("Tree feature policy is incompatible with the selected retrieval")
    decision = _load_decision(config, config_path.parent, identities, fixture)
    runtime = _runtime(config, config_path.parent, identities, fixture)

    metadata_path, complete_path = candidates / "metadata.json", candidates / "complete.json"
    meta, complete = _read_json(metadata_path), _read_json(complete_path)
    if (complete.get("state") != "complete" or complete.get("metadata") != meta
            or complete.get("config", {}).get("labels_accessed") is not False
            or complete["config"].get("positive_candidate_injection") is not False
            or meta.get("labels_accessed") is not False
            or meta.get("positive_candidate_injection") is not False
            or meta.get("all_input_references_retained") is not True
            or meta.get("references") != nref or meta.get("mining_full_pool_records") != ntarget
            or meta.get("pair_order") != ["left_idx", "right_idx"]
            or meta.get("benchmark_only", False) is not False
            or (candidates / "truth.json").exists()):
        raise ValueError("TEST candidates must be complete, label-free, full-pool and unprefixed")
    kind = config["retrieval"]["kind"]
    if kind == "dense_k50_v1":
        query = complete["config"].get("query_input", {})
        if (meta.get("mode") != "unlabeled_inference" or meta.get("top_k") !=
                (config["retrieval"].get("top_k") if fixture else 50)
                or (not fixture and meta.get("candidate_pairs") != V1_DENSE_PAIRS)
                or query.get("expected_rows") != nref
                or query.get("size_bytes") != (test_dir / "test_source1.tsv").stat().st_size
                or complete["config"].get("pool_embedding_identity", {}).get("checkpoint", {}).get("sha256")
                != config["retrieval"]["encoder_sha256"]
                or complete["config"].get("pool_embedding_identity", {}).get("target_rows") != ntarget):
            raise ValueError("Only the complete v1 dense test retrieval can be reused")
    elif (meta.get("mode") != "v2_test_union"
          or meta.get("retrieval_kind") != "test_union"
          or meta.get("feature_policy") != "dense_key_shared"
          or meta.get("candidate_groups_complete") is not True
          or meta.get("references_scanned") != nref
          or meta.get("no_test_prefix") is not True
          or meta.get("reference_input_sha256") != identities[str((test_dir / "test_source1.tsv").resolve())]
          or complete["config"].get("encoder_sha256") != config["retrieval"]["encoder_sha256"]):
        raise ValueError("P2 union finalist requires NEW, completed, full-test union candidates")
    paths = {name: candidates / name for name in ("reference.parquet", "pool.parquet", "pairs.parquet")}
    paths.update(metadata=metadata_path, complete=complete_path)
    for path in paths.values():
        identities[str(path)] = file_digest(path)
    reference = pd.read_parquet(paths["reference.parquet"], columns=list(RECORD_COLUMNS))
    pool = pd.read_parquet(paths["pool.parquet"], columns=[*RECORD_COLUMNS, "raw_row_idx"])
    _reference_matches_raw(reference, test_dir / "test_source1.tsv", nref)
    _pool_matches_raw(pool, test_dir, ntarget)
    pair_file = pq.ParquetFile(paths["pairs.parquet"])
    pair_names = set(pair_file.schema_arrow.names)
    forbidden = {"label", "training_label", "is_match", "truth", "ground_truth", "split"}
    if (forbidden & pair_names or not {"left_idx", "right_idx"} <= pair_names
            or pair_file.metadata.num_rows != meta.get("candidate_pairs")
            or len(pool) != meta.get("pool_records")):
        raise ValueError("TEST pair file has labels, missing indices or an incomplete candidate set")
    columns = (feature_columns(pair_names) if policy == "as_saved" else union_feature_columns(pair_names))
    bundle = joblib.load(tree_path)
    model = bundle["model"]
    if (list(bundle["feature_columns"]) != columns
            or (hasattr(model, "feature_names_") and list(model.feature_names_) != columns)
            or float(bundle.get("anchor_threshold", 0)) != 0
            or (not fixture and not type(model).__module__.startswith("catboost"))):
        raise ValueError("Frozen tree feature order/family/anchor differs from complete candidates")
    if (kind == "test_union" and not {"dense_score", "retrieval_score", "dense_hit",
                                       "phonetic_key_hit"} <= pair_names):
        raise ValueError("Union candidate-dependent features were not recomputed")
    if not 0 < alpha <= 1 or not 0 < gate <= 1:
        raise ValueError("Invalid frozen blend/gate")
    return Prepared(config, identities[str(config_path)], identities, checkpoint_ids, sources,
                    reference, pool, paths["pairs.parquet"], pair_file.metadata.num_rows,
                    decision, runtime, fixture, nref, ntarget)


def _tree_batch(prepared: Prepared, batch, model, columns: list[str]) -> np.ndarray:
    reference, pool, pairs = prepared.reference, prepared.pool, batch.pairs
    if pairs.empty:
        return np.empty(0, dtype=np.float32)
    left, right = (pairs[name].to_numpy(dtype=np.int64) for name in ("left_idx", "right_idx"))
    if not np.array_equal(reference.country.to_numpy()[left], pool.country.to_numpy()[right]):
        raise ValueError("A TEST candidate crossed the country block")
    if "right_raw_idx" in pairs and not np.array_equal(
            pairs.right_raw_idx.to_numpy(dtype=np.int64), pool.raw_row_idx.to_numpy()[right]):
        raise ValueError("Candidate pair lost its original target raw row")
    if "target_entity_id" in pairs and not np.array_equal(
            pairs.target_entity_id.to_numpy(), pool.entity_id.to_numpy()[right]):
        raise ValueError("Candidate pair lost its original target ID")
    left_records, right_records, local = compact_batch(reference, pool, batch)
    if prepared.config["tree"]["feature_policy"] == "dense_key_shared":
        if not np.array_equal(local.dense_score.to_numpy(dtype=np.float32),
                              local.retrieval_score.to_numpy(dtype=np.float32)):
            raise ValueError("Union retrieval_score must equal the recomputed dense cosine")
        rank = local.groupby("left_idx", sort=False).dense_score.rank(method="min", ascending=False)
        if not np.array_equal(local.retrieval_rank.to_numpy(dtype=np.int64),
                              rank.to_numpy(dtype=np.int64)):
            raise ValueError("Union candidate-dependent dense ranks were not recomputed")
        local["name_score"] = np.float32(0)
        local["address_score"] = np.float32(0)
        local["phonetic_score"] = np.float32(0)
        local["lexical_retrieval_score"] = np.float32(0)
        local["lexical_scores_missing"] = True
        local["phonetic_only"] = local.phonetic_key_hit.to_numpy(dtype=bool)
        local["dense_only"] = local.dense_hit.to_numpy(dtype=np.float32)
    features = feature_batch(left_records, right_records, local, columns)
    if list(features.columns) != columns or not np.isfinite(features.to_numpy()).all():
        raise ValueError("TEST matcher feature order or finite values differ from the tree")
    if type(model).__module__.startswith("catboost"):
        probability = np.asarray(model.predict_proba(features, thread_count=prepared.runtime["threads"]))
        if probability.shape != (len(pairs), 2):
            raise ValueError("CatBoost returned a partial candidate score vector")
        result = probability[:, 1].astype(np.float32)
    else:
        result = predict_proba(model, features)
    if result.shape != (len(pairs),) or not np.isfinite(result).all() or np.any((result < 0) | (result > 1)):
        raise ValueError("Tree scores are not aligned finite probabilities")
    return result


def _ce_batch(prepared: Prepared, batch, base: np.ndarray, score_fn) -> tuple[np.ndarray, np.ndarray, int]:
    blend, runtime, decision = prepared.config["blend"], prepared.runtime, prepared.decision
    selected = base >= blend["tree_gate"]
    if runtime.get("skip_provably_rejected_pairs", False):
        upper = blend_probabilities(base, np.ones(len(base), dtype=np.float32), selected, blend["alpha"])
        selected &= np.asarray(upper, dtype=np.float64) >= decision["threshold"]
    cross = np.zeros(len(base), dtype=np.float32)
    near = np.zeros(len(base), dtype=bool)
    if selected.any():
        left, right, local = compact_batch(prepared.reference, prepared.pool, batch)
        def score(mask: np.ndarray, amp: str) -> np.ndarray:
            value = np.asarray(score_fn(left, right, local, mask, amp), dtype=np.float32)
            if (value.shape != base.shape or not np.isfinite(value).all()
                    or np.any((value[mask] < 0) | (value[mask] > 1))
                    or np.any(value[~mask] != 0)):
                raise ValueError("CE scorer lost selected pair order, range or gate")
            return value
        precision = "bf16" if runtime["amp"] == "selective_fp32" else runtime["amp"]
        cross = score(selected, precision)
        if runtime["amp"] == "selective_fp32":
            near = selective_fp32_mask(base, cross, selected, alpha=blend["alpha"],
                                       threshold=runtime["fp32_anchor"], band=runtime["fp32_band"])
            if near.any():
                corrected = score(near, "fp32")
                cross[near] = corrected[near]
    return blend_probabilities(base, cross, selected, blend["alpha"]), cross, int(near.sum())


def _decide(prepared: Prepared, batch, base: np.ndarray, blended: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    decision, pairs = prepared.decision, batch.pairs
    if decision["kind"] == "threshold_assignment":
        accepted = accepted_mask(blended, decision["threshold"])
        return accepted, np.asarray(blended, dtype=np.float64)
    left = pairs.left_idx.to_numpy(dtype=np.int64) - batch.start
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    gate = base >= prepared.config["blend"]["tree_gate"]
    country = prepared.reference.country.to_numpy()[batch.start:batch.end]
    calibrated = predict_calibration(decision["models"], blended, gate, left, country)
    accepted = np.zeros(len(pairs), dtype=bool)
    scores = calibrated.copy()
    boundaries = np.searchsorted(left, np.arange(batch.end - batch.start + 1))
    pool_ids = prepared.pool.entity_id.to_numpy()
    for ref in range(batch.end - batch.start):
        start, end = boundaries[ref:ref+2]
        if start == end or not gate[start:end].any():
            continue
        g = gate[start:end]
        p = (calibrated[start:end] if decision["source_weight"] == 0 else
             adjust_source(calibrated[start:end], g,
                           np.asarray([str(x)[:3] for x in pool_ids[right[start:end]]]),
                           decision["source_weight"]))
        scores[start:end] = p
        eligible = np.flatnonzero(g) + start
        order = eligible[np.lexsort((right[eligible], -p[eligible - start]))]
        ungated = float(p[~g][0]) if (~g).any() else EPS
        best, _ = decision["posterior"].best_prefix(p[order - start], ungated, end - start)
        accepted[order[:best]] = True
    if np.any(accepted & ~gate):
        raise ValueError("Count decision selected a CE-unscored target")
    return accepted, scores


def _pure_assignment(claims: np.ndarray) -> tuple[np.ndarray, dict]:
    """Global P1 delta=0 ordering without millions of per-target Python groups."""
    keep = np.zeros(len(claims), dtype=bool)
    if not len(claims):
        return keep, {"conflicting_targets": 0, "removed_claims": 0, "unresolved_extra_claims": 0}
    order = np.lexsort((claims["pair_row"], claims["left"], -claims["score"], claims["right"]))
    targets = claims["right"][order]
    first = np.r_[True, targets[1:] != targets[:-1]]
    keep[order[first]] = True
    groups = int(first.sum())
    conflicts = int(np.count_nonzero(np.diff(np.r_[np.flatnonzero(first), len(claims)]) > 1))
    return keep, {"conflicting_targets": conflicts, "removed_claims": len(claims)-groups,
                  "unresolved_extra_claims": 0}


def _scorer(prepared: Prepared, config_path: Path):
    import torch

    torch.set_num_threads(prepared.runtime["threads"])
    model, tokenizer = load_reranker(str(_resolved(config_path.parent,
                                         prepared.config["cross_encoder"]["checkpoint"])),
                                   prepared.runtime["device"])
    options = SimpleNamespace(device=prepared.runtime["device"],
                              max_length=prepared.runtime["max_length"],
                              inference_batch_size=prepared.runtime["inference_batch_size"],
                              length_bucketing=True,
                              pad_to_multiple_of=prepared.runtime["pad_to_multiple_of"])
    def score(reference, pool, pairs, selected, amp):
        options.amp = amp
        return score_reranker_batched(model, tokenizer, reference, pool, pairs, selected, options)
    return score


def _commit_outputs(prepared: Prepared, output: Path, config_path: Path, *, score_fn=None) -> dict:
    checkpoint = _resolved(config_path.parent, prepared.config["tree"]["checkpoint"])
    bundle = joblib.load(checkpoint)
    columns = list(bundle["feature_columns"])
    scorer = score_fn if score_fn is not None else _scorer(prepared, config_path)
    pair_count = prepared.pair_count
    arrays = {name: np.lib.format.open_memmap(output / f"{name}.npy.tmp", mode="w+",
                                             dtype=np.float32, shape=(pair_count,))
              for name in ("base_probabilities", "cross_probabilities", "probabilities")}
    claims_tmp = output / "selected_claims.bin.tmp"
    references_scored = candidate_rows = near_count = selected_count = 0
    with claims_tmp.open("xb") as claims_stream:
        for batch in iter_reference_batches(prepared.pairs, prepared.expected_references,
                                            len(prepared.pool), prepared.runtime["reference_batch_size"],
                                            max_pairs_per_batch=prepared.runtime.get("max_pairs_per_batch", 200_000),
                                            require_sorted_right=True):
            base = _tree_batch(prepared, batch, bundle["model"], columns)
            blended, cross, near = _ce_batch(prepared, batch, base, scorer)
            accept, scores = _decide(prepared, batch, base, blended)
            if accept.shape != base.shape or scores.shape != base.shape or not np.isfinite(scores).all():
                raise ValueError("Decisions/scores lost the original candidate pair order")
            end = candidate_rows + len(base)
            if end > pair_count:
                raise ValueError("Scored more candidates than the complete input")
            arrays["base_probabilities"][candidate_rows:end] = base
            arrays["cross_probabilities"][candidate_rows:end] = cross
            arrays["probabilities"][candidate_rows:end] = blended
            chosen = np.flatnonzero(accept)
            if len(chosen):
                claims = np.empty(len(chosen), dtype=CLAIM_DTYPE)
                claims["left"] = batch.pairs.left_idx.to_numpy(dtype=np.int32)[chosen]
                claims["right"] = batch.pairs.right_idx.to_numpy(dtype=np.int32)[chosen]
                claims["pair_row"] = chosen + candidate_rows
                claims["score"] = scores[chosen]
                claims.tofile(claims_stream)
            references_scored, candidate_rows = batch.end, end
            near_count += near
            selected_count += len(chosen)
        claims_stream.flush()
        os.fsync(claims_stream.fileno())
    if references_scored != prepared.expected_references or candidate_rows != pair_count:
        raise ValueError("Full original Source 1/candidate scores were not committed")
    for name, array in arrays.items():
        array.flush()
        del array
        temporary = output / f"{name}.npy.tmp"
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        temporary.replace(output / f"{name}.npy")
    claims_tmp.replace(output / "selected_claims.bin")
    if (output / "selected_claims.bin").stat().st_size != selected_count * CLAIM_DTYPE.itemsize:
        raise ValueError("Global claim file lost a selected candidate")
    claims = (np.memmap(output / "selected_claims.bin", mode="r", dtype=CLAIM_DTYPE,
                        shape=(selected_count,)) if selected_count else np.empty(0, dtype=CLAIM_DTYPE))
    winners, conflicts = _pure_assignment(claims)
    if not np.array_equal(claims["pair_row"], np.sort(claims["pair_row"])):
        raise ValueError("Saved accepted claims are not in original pair order")
    match_tmp = output / "matching_results.tsv.tmp"
    candidate_tmp = output / "candidate_pairs.tsv.tmp"
    pool_ids = prepared.pool.entity_id.to_numpy()
    ref_ids = prepared.reference.entity_id.to_numpy()
    cursor = rows = links = 0
    pair_offset = 0
    with match_tmp.open("x", encoding="utf-8", newline="\n") as matching, \
            candidate_tmp.open("x", encoding="utf-8", newline="\n") as candidate:
        matching.write("source1_entity_id\tmatched_entity_ids\n")
        candidate.write("source1_entity_id\tcandidate_entity_ids\n")
        for batch in iter_reference_batches(prepared.pairs, prepared.expected_references,
                                            len(prepared.pool), prepared.runtime["reference_batch_size"],
                                            max_pairs_per_batch=prepared.runtime.get("max_pairs_per_batch", 200_000),
                                            require_sorted_right=True):
            left = batch.pairs.left_idx.to_numpy(dtype=np.int64)
            right = batch.pairs.right_idx.to_numpy(dtype=np.int64)
            boundaries = np.searchsorted(left, np.arange(batch.start, batch.end + 1))
            for local, ref in enumerate(range(batch.start, batch.end)):
                start, end = boundaries[local:local+2]
                candidate_ids = pool_ids[right[start:end]]
                selected_ids = []
                while cursor < len(claims) and claims["left"][cursor] == ref:
                    claim = claims[cursor]
                    local_row = int(claim["pair_row"]) - pair_offset
                    if not start <= local_row < end or right[local_row] != claim["right"]:
                        raise ValueError("A selected claim no longer maps to its original pair/ID")
                    if winners[cursor]:
                        selected_ids.append(str(pool_ids[claim["right"]]))
                    cursor += 1
                matching.write(f"{ref_ids[ref]}\t{','.join(sorted(selected_ids))}\n")
                candidate.write(f"{ref_ids[ref]}\t{','.join(sorted(candidate_ids))}\n")
                rows += 1
                links += len(selected_ids)
            pair_offset += len(batch.pairs)
        matching.flush()
        candidate.flush()
        os.fsync(matching.fileno())
        os.fsync(candidate.fileno())
    if rows != prepared.expected_references or cursor != len(claims) or pair_offset != pair_count or links != int(winners.sum()):
        raise ValueError("One-to-one output omitted an S1, candidate or selected pair")
    match_tmp.replace(output / "matching_results.tsv")
    candidate_tmp.replace(output / "candidate_pairs.tsv")
    return {"source1_rows": rows, "candidate_pairs": pair_count, "initial_claims": selected_count,
            "accepted_pairs": links, "selective_fp32_rescored_pairs": near_count,
            "assignment": conflicts, "singleton_convention": "no accepted links => empty second TSV field",
            "probability_order": "unchanged pairs.parquet row order", "labels_accessed": False,
            "benchmark_only": prepared.fixture, "official_test_score": None}


def _verify_inputs(prepared: Prepared) -> None:
    for path, sha in prepared.file_sha256.items():
        if file_digest(Path(path)) != sha:
            raise ValueError(f"Input changed during inference: {path}")
    if source_hashes() != prepared.source_sha256:
        raise ValueError("Inference/validation source changed during prediction")
    for name in ("encoder", "cross_encoder"):
        identity = prepared.checkpoint_identities[name]
        if checkpoint_identity(Path(identity["path"])) != identity:
            raise ValueError(f"{name} checkpoint changed during inference")


def _holdout_attestation(path: Path, config_sha: str, identities: dict[str, str]) -> dict:
    """Read the one-shot report ONLY after the full-test outputs are fixed."""
    report = _read_json(path)
    fixed = report.get("fixed_predictions", {})
    score = report.get("one_shot_score", {})
    if (report.get("state") != "complete" or report.get("frozen_pipeline_sha256") != config_sha
            or report.get("source1_rows") != 50_000
            or report.get("full_competing_target_pool") != TRAIN_TARGETS
            or report.get("holdout2_id_sha256") != HOLDOUT2_IDS_SHA256
            or report.get("prediction_files_committed_before_truth") is not True
            or report.get("no_post_holdout_selection") is not True
            or fixed.get("source1_rows") != 50_000
            or fixed.get("holdout2_id_sha256") != HOLDOUT2_IDS_SHA256
            or fixed.get("committed_before_truth") is not True
            or score.get("reference_count") != 50_000
            or score.get("full_competing_target_pool") != TRAIN_TARGETS
            or type(fixed.get("candidate_pairs")) is not int or fixed["candidate_pairs"] < 0):
        raise ValueError("One-shot HOLDOUT-2 proof must bind this config, all 50k S1s and a fixed score")
    _numeric(score.get("macro_f05"), "One-shot held-out macro F0.5")
    artifacts = {}
    for name in ("matching_results.tsv", "candidate_pairs.tsv", "probabilities.npy"):
        item = fixed.get("files", {}).get(name, {})
        if not isinstance(item, dict) or not {"path", "sha256"} <= item.keys():
            raise ValueError(f"HOLDOUT-2 fixed {name} needs a path and SHA256")
        prediction = _resolved(path.parent, item["path"])
        if not prediction.is_file() or not prediction.stat().st_size or file_digest(prediction) != item["sha256"]:
            raise ValueError(f"HOLDOUT-2 fixed {name} changed after truth was opened")
        identities[str(prediction)] = item["sha256"]
        artifacts[name] = item["sha256"]
    identities[str(path.resolve())] = file_digest(path)
    return {"attestation_path": str(path.resolve()), "sha256": identities[str(path.resolve())],
            "fixed_file_sha256": artifacts}


def run(config_path: Path, candidates: Path, test_dir: Path, output: Path, *,
        validator: Path | None = None, holdout_attestation: Path | None = None,
        fixture: bool = False, dry_run: bool = False, score_fn=None) -> dict:
    """Score ALL S1 groups, resolve global conflicts, then validate both TSVs."""
    config_path = Path(config_path).resolve()
    output = Path(output).resolve()
    test_dir = Path(test_dir).resolve()
    candidates = Path(candidates).resolve()
    validator = Path(validator or Path(__file__).resolve().parents[2] / "resources/validate_submission.py").resolve()
    if dry_run and not fixture:
        raise ValueError("Dry-run is intentionally restricted to a tiny synthetic fixture")
    if score_fn is not None and not fixture:
        raise ValueError("Injected test scorers cannot be used in a real release")
    if (any(part.startswith("submission_v1") for part in output.parts)
            or Path("/workspace/amlc") in output.parents
            or output == candidates or output == test_dir
            or candidates in output.parents or test_dir in output.parents):
        raise ValueError("Use a separate new v2 release directory, never a frozen/input directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("A partial/finished release needs inspection; choose a new output path")
    if not fixture and holdout_attestation is None:
        raise ValueError("A full release requires a post-freeze, one-shot HOLDOUT-2 attestation")
    prepared = prepare(config_path, candidates, test_dir, validator, fixture=fixture)
    if fixture and prepared.pair_count > 2_000:
        raise ValueError("Synthetic fixture must contain at most 2,000 candidate pairs")
    summary = {"source1_rows": prepared.expected_references, "target_rows": prepared.expected_targets,
               "candidate_pairs": prepared.pair_count, "pipeline_sha256": prepared.config_sha256,
               "retrieval_kind": prepared.config["retrieval"]["kind"],
               "fixture_only": fixture, "dry_run": dry_run, "labels_accessed": False,
               "ready_for_submission": False}
    if dry_run:
        return summary
    output.mkdir(parents=True, exist_ok=False)
    with (output / "inference.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / "frozen_pipeline.json").write_bytes(config_path.read_bytes())
        if file_digest(output / "frozen_pipeline.json") != prepared.config_sha256:
            raise ValueError("Frozen config changed while being copied")
        report = _commit_outputs(prepared, output, config_path, score_fn=score_fn)
        _verify_inputs(prepared)
        fixed_hashes = {name: file_digest(output / name) for name in
                        ("matching_results.tsv", "candidate_pairs.tsv", "base_probabilities.npy",
                         "cross_probabilities.npy", "probabilities.npy", "selected_claims.bin",
                         "frozen_pipeline.json")}
        report.update(pipeline_sha256=prepared.config_sha256,
                      retrieval_kind=prepared.config["retrieval"]["kind"],
                      decision_kind=prepared.decision["kind"], input_file_sha256=prepared.file_sha256,
                      checkpoint_identities=prepared.checkpoint_identities, source_sha256=prepared.source_sha256,
                      fixed_file_sha256=fixed_hashes, ready_for_submission=False)
        atomic_json(output / "report.json", report)
        atomic_json(output / "complete.json", {"state": "complete", "report_sha256": file_digest(output / "report.json"),
                                                "fixed_file_sha256": fixed_hashes, "labels_accessed": False,
                                                "benchmark_only": fixture})
        validation = validate_release(output / "matching_results.tsv", output / "candidate_pairs.tsv",
                                      test_dir, output / "validation.json", validator=validator,
                                      expected_rows=prepared.expected_references)
        if not validation["passed"] or validation["counts"]["candidate_links"] != prepared.pair_count or \
                validation["counts"]["matched_links"] != report["accepted_pairs"]:
            raise ValueError(f"Full official and strict candidate/subset/coverage validation failed: {validation['errors']}")
        if fixture:
            return {**summary, **report, "fixture_validation_passed": True,
                    "ready_for_submission": False}
        attestation = _holdout_attestation(Path(holdout_attestation).resolve(),
                                           prepared.config_sha256, prepared.file_sha256)
        _verify_inputs(prepared)
        files = {name: file_digest(output / name) for name in
                 (*fixed_hashes, "report.json", "complete.json", "validation.json",
                  "validation.validator.log", "inference.lock")}
        ready = {"ready_for_submission": True, "schema": 2,
                 "source1_rows": prepared.expected_references, "target_rows": prepared.expected_targets,
                 "pipeline_sha256": prepared.config_sha256, "file_sha256": files,
                 "input_file_sha256": prepared.file_sha256,
                 "checkpoint_identities": prepared.checkpoint_identities,
                 "source_sha256": prepared.source_sha256,
                 "holdout2": attestation, "candidate_pairs": prepared.pair_count,
                 "retrieval_kind": prepared.config["retrieval"]["kind"],
                 "validation_kind": "format_and_integrity_only", "official_test_score": None,
                 "platform_submission_performed": False,
                 "no_test_prefix": True, "duplicate_ids_or_pairs": False,
                 "predictions_subset_of_candidates": True, "full_original_order_coverage": True,
                 "labels_accessed_during_test_inference": False}
        atomic_json(output / "ready.json", ready)
        try:
            verify_ready(output)
        except BaseException:
            (output / "ready.json").unlink()
            raise
        return ready


def verify_ready(release: Path) -> dict:
    """Extended v2 check before handing a NEW release to the existing ZIP builder."""
    from package_portal_submission import validated_release

    release = Path(release).resolve()
    ready = validated_release(release)
    if (ready.get("schema") != 2 or ready.get("target_rows") != TEST_TARGETS
            or ready.get("no_test_prefix") is not True
            or ready.get("full_original_order_coverage") is not True
            or ready.get("predictions_subset_of_candidates") is not True
            or ready.get("duplicate_ids_or_pairs") is not False
            or ready.get("labels_accessed_during_test_inference") is not False
            or ready.get("official_test_score") is not None):
        raise ValueError("V2 readiness requires all original test IDs/pairs and zero test-label use")
    for name, sha in ready["file_sha256"].items():
        if file_digest(release / name) != sha:
            raise ValueError(f"Release file changed after complete validation: {name}")
    for name, sha in ready["input_file_sha256"].items():
        if file_digest(Path(name)) != sha:
            raise ValueError(f"Release input changed after validation: {name}")
    if source_hashes() != ready["source_sha256"]:
        raise ValueError("Release source differs from the frozen checkout")
    for name in ("encoder", "cross_encoder"):
        saved = ready["checkpoint_identities"][name]
        if checkpoint_identity(Path(saved["path"])) != saved:
            raise ValueError(f"Release {name} checkpoint changed")
    if file_digest(Path(ready["checkpoint_identities"]["tree"]["path"])) != ready["checkpoint_identities"]["tree"]["sha256"]:
        raise ValueError("Release tree checkpoint changed")
    report = _read_json(release / "report.json")
    validation = _read_json(release / "validation.json")
    if (report.get("benchmark_only") is not False or report.get("labels_accessed") is not False
            or report.get("candidate_pairs") != ready["candidate_pairs"]
            or validation.get("metric_score_computed") is not False
            or validation.get("validation_kind") != "format_and_integrity_only"
            or validation["counts"].get("candidate_links") != ready["candidate_pairs"]
            or validation["counts"].get("matched_links") != report.get("accepted_pairs")):
        raise ValueError("Ready marker, complete score stream and strict validator disagree")
    return ready


def _saved_pair_batches(path: Path, nref: int, npool: int, *, read_rows: int = 100_000):
    """Read only aligned positional columns; never materialize the 86M pairs."""
    parquet = pq.ParquetFile(path)
    if not {"left_idx", "right_idx"} <= set(parquet.schema_arrow.names):
        raise ValueError("Saved pair file has no positional edge columns")
    pending_left = pending_right = np.empty(0, dtype=np.int32)
    previous_left = previous_right = -1
    consumed = 0
    for record in parquet.iter_batches(batch_size=read_rows, columns=["left_idx", "right_idx"],
                                       use_threads=False):
        left, right = (record.column(i).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
                       for i in (0, 1))
        if len(left) == 0:
            continue
        if (left.min() < 0 or left.max() >= nref or right.min() < 0 or right.max() >= npool
                or left[0] < previous_left or (left[0] == previous_left and right[0] <= previous_right)
                or np.any(left[1:] < left[:-1])
                or np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1]))):
            raise ValueError("Saved candidates have an out-of-bounds, duplicate or reordered pair")
        previous_left, previous_right = int(left[-1]), int(right[-1])
        if len(pending_left):
            left = np.r_[pending_left, left]
            right = np.r_[pending_right, right]
        cutoff = int(np.searchsorted(left, left[-1], side="left"))
        if cutoff:
            yield consumed, left[:cutoff], right[:cutoff]
            consumed += cutoff
        pending_left, pending_right = left[cutoff:], right[cutoff:]
        if len(pending_left) > 200_000:
            raise ValueError("A complete Source 1 group exceeds the bounded candidate limit")
    if len(pending_left):
        yield consumed, pending_left, pending_right
        consumed += len(pending_left)
    if consumed != parquet.metadata.num_rows:
        raise ValueError("Saved pair reader dropped candidate rows")


def _saved_claims(pairs: Path, scores: np.ndarray, base: np.ndarray,
                  countries: np.ndarray, pool_count: int, decision: dict,
                  claims_path: Path, *, expected_pairs: int = V1_DENSE_PAIRS,
                  progress_every: int = 100_000, stop_reference: int | None = None) -> dict:
    """P1 on original pair order, with only one projected Parquet batch in RAM."""
    if (scores.dtype != np.float32 or base.dtype != np.float32
            or scores.shape != (expected_pairs,) or base.shape != scores.shape):
        raise ValueError("Saved v1 base/blend arrays must exactly cover all aligned pairs")
    posterior = decision["posterior"]
    models = decision["models"]
    total = accepted = skipped_gate = refs_seen = 0
    with claims_path.open("xb") as stream:
        for offset, left, right in _saved_pair_batches(pairs, len(countries), pool_count):
            truncated = False
            if stop_reference is not None:
                count = int(np.searchsorted(left, stop_reference))
                if not count:
                    break
                truncated = count < len(left)
                left, right = left[:count], right[:count]
            stop = offset + len(left)
            raw = np.asarray(scores[offset:stop])
            tree = np.asarray(base[offset:stop])
            if (not np.isfinite(raw).all() or not np.isfinite(tree).all()
                    or np.any((raw < 0) | (raw > 1)) or np.any((tree < 0) | (tree > 1))
                    or np.any(raw[tree < .01] != 0)):
                raise ValueError("Saved scores are nonfinite, out of range or violate the v1 0.01 gate")
            gate = tree >= .01
            skipped_gate += int(np.count_nonzero(gate & (raw == 0)))
            first_ref = int(left[0])
            calibrated = predict_calibration(models, raw, gate, left - first_ref,
                                             countries[first_ref:int(left[-1])+1])
            boundaries = np.r_[0, np.flatnonzero(np.diff(left)) + 1, len(left)]
            selected = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                refs_seen += 1
                eligible = np.flatnonzero(gate[start:end]) + start
                if not len(eligible):
                    continue
                p = calibrated[start:end]
                order = eligible[np.lexsort((right[eligible], -calibrated[eligible]))]
                ungated = float(p[~gate[start:end]][0]) if (~gate[start:end]).any() else EPS
                best, _ = posterior.best_prefix(calibrated[order], ungated, end-start)
                if best:
                    selected.append(order[:best])
            if selected:
                chosen = np.sort(np.concatenate(selected))
                claims = np.empty(len(chosen), dtype=CLAIM_DTYPE)
                claims["left"] = left[chosen]
                claims["right"] = right[chosen]
                claims["pair_row"] = offset + chosen
                claims["score"] = calibrated[chosen]
                claims.tofile(stream)
                accepted += len(claims)
            total = stop
            if refs_seen // progress_every > (refs_seen - len(boundaries) + 1) // progress_every:
                print(json.dumps({"provisional_scored_s1": refs_seen, "pairs": total,
                                  "claims": accepted}), flush=True)
            if truncated:
                break
        stream.flush()
        os.fsync(stream.fileno())
    if ((stop_reference is None and total != expected_pairs)
            or claims_path.stat().st_size != accepted * CLAIM_DTYPE.itemsize):
        raise ValueError("Saved score/pair stream or selected claims are incomplete")
    return {"candidate_pairs": total, "references_with_candidates": refs_seen,
            "pre_assignment_claims": accepted, "gated_zero_blend_pairs": skipped_gate}


def _provisional_inputs(v1: Path, data: Path, p1: Path, review: Path, test_dir: Path,
                        *, fixture: bool = False, expected_rows: int = TEST_REFERENCES,
                        expected_pairs: int = V1_DENSE_PAIRS) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """Bind complete v1 inference, not an arbitrary score vector or candidate subset."""
    ready = _read_json(v1 / "ready.json")
    complete = _read_json(v1 / "complete.json")
    report = _read_json(v1 / "report.json")
    pipeline = _read_json(v1 / "frozen_pipeline.json")
    if (ready.get("ready_for_submission") is not True or complete.get("state") != "complete"
            or complete.get("report") != report or report.get("scorer") != "frozen_tree_ce"
            or report.get("benchmark_only") is not False or report.get("labels_accessed") is not False
            or report.get("reference_count") != expected_rows
            or report.get("candidate_pairs") != expected_pairs
            or report.get("pair_artifact_mode") != "hardlink"
            or report.get("pair_order") != ["left_idx", "right_idx"]
            or pipeline.get("gate") != .01 or pipeline.get("alpha") != .75
            or pipeline.get("tree_checkpoint_sha256") != report.get("checkpoint_sha256")
            or pipeline.get("cross_encoder_sha256") != report.get("cross_encoder_sha256")
            or file_digest(v1 / "frozen_pipeline.json") != ready.get("pipeline_sha256")
            or file_digest(v1 / "complete.json") != ready.get("file_sha256", {}).get("complete.json")
            or file_digest(v1 / "report.json") != ready.get("file_sha256", {}).get("report.json")
            or file_digest(v1 / "candidate_pairs.tsv") != ready.get("file_sha256", {}).get("candidate_pairs.tsv")
            or Path(report.get("data", "")).resolve() != data.resolve()
            or not (v1 / "scored_pairs.parquet").samefile(data / "pairs.parquet")):
        raise ValueError("Saved v1 full-test score, pipeline or candidate identity is incomplete")
    identities = {}
    tree = Path(pipeline["tree_checkpoint"])
    if not fixture:
        if (file_digest(tree) != pipeline["tree_checkpoint_sha256"]
                or checkpoint_identity(Path(pipeline["cross_encoder_checkpoint"]))["sha256"]
                != pipeline["cross_encoder_sha256"]):
            raise ValueError("Frozen v1 tree/CE checkpoints changed")
        identities["tree_sha256"] = pipeline["tree_checkpoint_sha256"]
        identities["cross_encoder_sha256"] = pipeline["cross_encoder_sha256"]
        identities["encoder_sha256"] = pipeline["retrieval"]["checkpoint_sha256"]
    calibrated_config = {
        "tree": {"sha256": pipeline["tree_checkpoint_sha256"]},
        "cross_encoder": {"sha256": pipeline["cross_encoder_sha256"]},
        "retrieval": {"kind": "dense_k50_v1"}, "blend": {"alpha": .75, "tree_gate": .01},
        "decision": {"kind": "p1_count_assignment", "assignment_delta": 0, "source_weight": 0.,
                     "calibration": {"kind": "p1_saved", "path": str(p1), "sha256": file_digest(p1),
                                     "review": str(review), "review_sha256": file_digest(review)}}}
    decision = _load_decision(calibrated_config, p1.parent, {}, fixture)
    refs = pd.read_parquet(data / "reference.parquet", columns=["entity_id", "country"])
    pool = pd.read_parquet(data / "pool.parquet", columns=["entity_id"])
    if (len(refs) != expected_rows or refs.entity_id.duplicated().any()
            or len(pool) != report["pool_count"] or pool.entity_id.duplicated().any()
            or pq.ParquetFile(v1 / "scored_pairs.parquet").metadata.num_rows != expected_pairs):
        raise ValueError("Saved reference, compact targets or positional pairs changed")
    with (test_dir / "test_source1.tsv").open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if next(reader)[0] != "entity_id":
            raise ValueError("Original test Source 1 header changed")
        for row, entity in zip_longest(reader, refs.entity_id):
            if row is None or entity is None or row[0] != entity:
                raise ValueError("Saved v1 reference lost original Source 1 ID/order/coverage")
    scores = np.load(v1 / "probabilities.npy", mmap_mode="r")
    base = np.load(v1 / "base_probabilities.npy", mmap_mode="r")
    input_paths = [v1 / name for name in ("complete.json", "report.json", "ready.json",
                    "frozen_pipeline.json", "scored_pairs.parquet", "probabilities.npy",
                    "base_probabilities.npy", "candidate_pairs.tsv")]
    input_paths += [data / name for name in ("reference.parquet", "pool.parquet")]
    input_paths += [p1, review]
    evidence = {str(path.resolve()): {"bytes": path.stat().st_size, "sha256": file_digest(path)}
                for path in input_paths}
    return {"ready": ready, "pipeline": pipeline, "report": report, "file_identity": evidence,
            "checkpoint_identity": identities, "reference_ids": refs.entity_id.to_numpy(),
            "reference_country": refs.country.to_numpy(), "pool_ids": pool.entity_id.to_numpy(),
            "decision": decision}, scores, base, calibrated_config


def provisional_saved_v1(v1: Path, data: Path, p1: Path, review: Path, test_dir: Path,
                         output: Path, validator: Path, *, expected_rows: int = TEST_REFERENCES,
                         expected_pairs: int = V1_DENSE_PAIRS, fixture: bool = False,
                         benchmark_references: int | None = None) -> dict:
    """No CE inference, no test/holdout labels, no ready marker or release action."""
    if not fixture and (expected_rows != TEST_REFERENCES or expected_pairs != V1_DENSE_PAIRS):
        raise ValueError("A provisional official prediction cannot use a test prefix")
    if benchmark_references is not None and not 1 <= benchmark_references <= 20_000:
        raise ValueError("A saved-score timing screen is limited to 20,000 S1s and is NEVER submittable")
    started = time.monotonic()
    v1, data, p1, review, test_dir, output, validator = (Path(x).resolve() for x in
                                                     (v1, data, p1, review, test_dir, output, validator))
    if (output.exists() or output == v1 or output == data or v1 in output.parents
            or data in output.parents or any(part.startswith("submission_v1") for part in output.parts)):
        raise FileExistsError("Use a fresh, separate provisional v2 output directory")
    if not fixture and (file_digest(test_dir / "test_source1.tsv") != ORIGINAL_TEST_SHA256["test_source1.tsv"]
                        or file_digest(validator) != OFFICIAL_VALIDATOR_SHA256):
        raise ValueError("Official test Source 1 or validator bytes changed")
    inputs, scores, base, decision_config = _provisional_inputs(
        v1, data, p1, review, test_dir, fixture=fixture,
        expected_rows=expected_rows, expected_pairs=expected_pairs)
    preparation_seconds = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=False)
    claims_tmp = output / "claims.bin.tmp"
    scoring_started = time.monotonic()
    counts = _saved_claims(v1 / "scored_pairs.parquet", scores, base,
                           inputs["reference_country"], len(inputs["pool_ids"]),
                           inputs["decision"], claims_tmp, expected_pairs=expected_pairs,
                           stop_reference=benchmark_references)
    scoring_seconds = time.monotonic() - scoring_started
    claims_tmp.replace(output / "claims.bin")
    nclaim = counts["pre_assignment_claims"]
    claims = (np.memmap(output / "claims.bin", dtype=CLAIM_DTYPE, mode="r", shape=(nclaim,))
              if nclaim else np.empty(0, dtype=CLAIM_DTYPE))
    winners, assignment_report = _pure_assignment(claims)
    refs, pool = inputs["reference_ids"], inputs["pool_ids"]
    if benchmark_references is not None:
        refs = refs[:benchmark_references]
    temporary = output / ("benchmark_matching_results.tsv.tmp" if benchmark_references is not None
                          else "matching_results.tsv.tmp")
    claimed = 0
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write("source1_entity_id\tmatched_entity_ids\n")
        cursor = 0
        for i, ref in enumerate(refs):
            accepted = []
            while cursor < nclaim and claims["left"][cursor] == i:
                if winners[cursor]:
                    accepted.append(str(pool[claims["right"][cursor]]))
                cursor += 1
            stream.write(f"{ref}\t{','.join(sorted(accepted))}\n")
            claimed += len(accepted)
        if cursor != nclaim:
            raise ValueError("Global assignment left a claim outside the complete Source 1 rows")
        stream.flush()
        os.fsync(stream.fileno())
    matching_name = "benchmark_matching_results.tsv" if benchmark_references is not None else "matching_results.tsv"
    temporary.replace(output / matching_name)
    if benchmark_references is not None:
        report = {"state": "benchmark_only", "ready_for_submission": False,
                  "benchmark_only": True, "benchmark_source1_rows": len(refs),
                  "candidate_pairs_scored": counts["candidate_pairs"],
                  "pre_assignment_claims": nclaim, "assigned_links": claimed,
                  "preparation_seconds_including_all_source_and_checkpoint_hashing": preparation_seconds,
                  "scoring_seconds": scoring_seconds,
                  "full_projection_scoring_seconds": scoring_seconds * expected_rows / len(refs),
                  "full_projection_excluding_copy_and_validation_seconds":
                      preparation_seconds + scoring_seconds * expected_rows / len(refs),
                  "benchmark_matching_sha256": file_digest(output / matching_name),
                  "labels_accessed": False, "no_ready_marker": True}
        atomic_json(output / "benchmark.json", report)
        return report
    shutil.copyfile(v1 / "candidate_pairs.tsv", output / "candidate_pairs.tsv")
    if file_digest(output / "candidate_pairs.tsv") != inputs["ready"]["file_sha256"]["candidate_pairs.tsv"]:
        raise ValueError("Copied candidate TSV changed from the validated v1 file")
    result = validate_release(output / "matching_results.tsv", output / "candidate_pairs.tsv",
                              test_dir, output / "validation.json", validator=validator,
                              expected_rows=expected_rows)
    if (result["passed"] is not True or result["counts"]["candidate_links"] != expected_pairs
            or result["counts"]["matched_links"] != claimed):
        raise ValueError(f"Provisional official/strict validation failed: {result['errors']}")
    if file_digest(v1 / "probabilities.npy") != inputs["file_identity"][str(v1 / "probabilities.npy")]["sha256"]:
        raise ValueError("Saved v1 blended scores changed while applying the P1 rule")
    files = {name: {"bytes": (output / name).stat().st_size,
                    "sha256": file_digest(output / name)} for name in
             ("matching_results.tsv", "candidate_pairs.tsv", "claims.bin", "validation.json")}
    report = {"state": "provisional_validated", "ready_for_submission": False,
              "source1_rows": expected_rows, "candidate_pairs": expected_pairs,
              "full_competing_test_targets": TEST_TARGETS if not fixture else None,
              "counts": {**counts, "assigned_links": claimed, "assignment": assignment_report},
              "file_sha256": files, "saved_v1_input_identity": inputs["file_identity"],
              "v1_checkpoint_identity": inputs["checkpoint_identity"],
              "p1_calibration_sha256": file_digest(p1), "p1_review_sha256": file_digest(review),
              "gate": .01, "alpha": .75, "source_bonus": 0., "assignment_delta": 0.,
              "scorer": "saved_v1_blended_scores_only", "no_test_ce_or_retrieval_compute": True,
              "labels_accessed": False, "holdout2_truth_opened": False,
              "old_holdout_independent_confirmation_required": True,
              "saved_upper_bound_skipped_gate_pairs": counts["gated_zero_blend_pairs"],
              "official_validator_passed_without_warnings": result["original_validator"]["passed_without_warnings"],
              "strict_full_coverage_subset_passed": result["stream_complete"],
              "duration_seconds": time.monotonic() - started}
    atomic_json(output / "provisional.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Score a frozen finalist; release only with full holdout proof")
    for name in ("config", "candidates", "test-dir", "output"):
        predict.add_argument(f"--{name}", type=Path, required=True)
    predict.add_argument("--validator", type=Path)
    predict.add_argument("--holdout-attestation", type=Path)
    predict.add_argument("--fixture", action="store_true", help="tiny synthetic inputs; never submission-ready")
    predict.add_argument("--dry-run", action="store_true", help="fixture-only preflight, without output")
    verify = commands.add_parser("verify", help="check all v2 ready hashes before portal packaging")
    verify.add_argument("--release", type=Path, required=True)
    saved = commands.add_parser("provisional-saved", help="Full-coverage P1 from immutable v1 test scores; NEVER release-ready")
    for name in ("v1", "data", "p1-config", "p1-review", "test-dir", "output", "validator"):
        saved.add_argument(f"--{name}", type=Path, required=True)
    saved.add_argument("--benchmark-references", type=int,
                       help="10k–20k original S1 timing only; no candidate TSV, validator or ready marker")
    args = parser.parse_args()
    if args.command == "predict":
        result = run(args.config, args.candidates, args.test_dir, args.output,
                     validator=args.validator, holdout_attestation=args.holdout_attestation,
                     fixture=args.fixture, dry_run=args.dry_run)
    elif args.command == "verify":
        result = verify_ready(args.release)
    else:
        result = provisional_saved_v1(args.v1, args.data, args.p1_config, args.p1_review,
                                      args.test_dir, args.output, args.validator,
                                      benchmark_references=args.benchmark_references)
    print(json.dumps(result, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
