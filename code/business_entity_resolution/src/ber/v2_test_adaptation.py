"""P5: label-free test diagnostics and a gated pseudo-label CE adaptation.

``screen-v1`` reads the validated v1 matching TSV, while ``diagnose-v1-scores``
reads its saved *raw blend* scores. Neither can create pseudo-labels. Both use
supplied test records without opening any truth file. ``select`` instead needs
a separately frozen v2 BGE CE checkpoint, a hashed TUNE-only calibrator (pooled
for France), TUNE selection report, and a completed full-test *calibrated* score
bundle. Its contract is documented by ``checked_frozen`` and
``checked_scored_v2`` below. The bundle contains ordered reference/pool/candidate
Parquets, an aligned probability NPY for EVERY candidate (including tree-gated
ones), and a marker binding their SHA256s, the calibrator, frozen checkpoint,
full test pool, and original raw test file hashes. No test truth path is accepted.
For a controlled fit, the frozen manifest additionally supplies a fixed raw
tree+CE blend/threshold decision and the same-decision, all-S1 TUNE entity TSV.
Other decision families require their own paired-evaluation adapter.

Conflicting p>=.98 claims to a target are ALL discarded; p<=.02 negatives on
such ambiguous targets are discarded too. Negatives remain retrieved edges and
are capped per S1 by highest score within the fixed <=.02 band. Every S1,
including zero-candidate singletons, has a row in the non-submission audit.
Training uses separate TRAIN and test record tables and weights test examples
at <=.3 in the actual BCE loss. ``fit`` cannot run before the frozen v2/TUNE
and population-disjointness checks pass. This module never writes official TSVs.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .normalization import normalize_text


COUNTRIES = ("France", "India", "US")
COUNTRY_CODE = {country: index for index, country in enumerate(COUNTRIES)}
EXPECTED_REFERENCES = 1_732_544
EXPECTED_TARGETS = 9_969_589
TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
POSITIVE_MIN = .98
NEGATIVE_MAX = .02
PSEUDO_WEIGHT = .3
MAX_NEGATIVES_PER_S1 = 10
SAMPLE_SEED = 20260925
SCORE_BINS = (0., .001, .01, .02, .05, .1, .2, .5, .8, .9, .95, .98, .99, 1.0000001)
_DIGITS = re.compile(r"\d+")
_DEVANAGARI = re.compile(r"[\u0900-\u097f]")
RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: Path, report: dict) -> None:
    if path.exists() or path.with_suffix(path.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite an adaptation artifact: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _fresh_output(path: Path) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Use a new, adaptation-owned output directory: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def _verified_file(path: Path, digest: str, *, bytes_expected: int | None = None) -> dict:
    path = Path(path)
    if (not path.is_file() or (bytes_expected is not None and path.stat().st_size != bytes_expected)
            or len(digest) != 64 or sha256(path) != digest):
        raise ValueError(f"Input is absent or differs from its SHA256/size manifest: {path}")
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def _test_inputs(test_dir: Path, integrity: Path) -> dict[str, dict]:
    inventory = {Path(item["path"]).name: item for item in _json(integrity)["files"]}
    result = {}
    for source in ("source1", "source2", "source3"):
        name = f"test_{source}.tsv"
        item = inventory[name]
        result[source] = _verified_file(test_dir / name, item["sha256"], bytes_expected=item["bytes"])
    return result


def _check_count(actual: int, expected: int, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name}: expected {expected:,}, found {actual:,}; no partial-test result is valid")


def _raw_rows(path: Path, expected_header: tuple[str, ...]):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if tuple(next(reader)) != expected_header:
            raise ValueError(f"Unexpected supplied test/validated output columns: {path}")
        for row in reader:
            if len(row) != len(expected_header):
                raise ValueError(f"Malformed row in supplied test/validated output: {path}")
            yield row


def _rank(entity_id: str, seed: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{seed}\0{entity_id}".encode(), digest_size=16).digest(), "big")


def _offer_sample(heaps: dict[str, list], row: tuple, size: int, seed: int) -> None:
    country, entity_id = row[4], row[1]
    entry = (-_rank(entity_id, seed), entity_id, row)
    group = heaps[country]
    if len(group) < size:
        heapq.heappush(group, entry)
    elif entry > group[0]:
        heapq.heapreplace(group, entry)


def _script(name: str, address: str) -> str:
    text = name + " " + address
    return "Devanagari" if _DEVANAGARI.search(text) else "other_non_ASCII" if not text.isascii() else "ASCII"


def _resolve_targets(test_dir: Path, wanted: set[str], inputs: dict) -> dict[str, tuple]:
    """Check the sampled target IDs/text against BOTH supplied test source files."""
    targets: dict[str, tuple] = {}
    for source in ("source2", "source3"):
        for entity_id, name, address, country in _raw_rows(Path(inputs[source]["path"]), RECORD_COLUMNS):
            if entity_id in wanted:
                if entity_id in targets or not entity_id.startswith("S2-" if source == "source2" else "S3-"):
                    raise ValueError("Duplicate or cross-source sampled test target ID")
                targets[entity_id] = (name, address, country)
    if set(targets) != wanted:
        raise ValueError("A sampled candidate target is absent from the supplied test S2/S3 records")
    return targets


def _raw_roundtrip(name: str, address: str) -> dict:
    raw = f"{name} | {address}"
    parts = raw.split(" | ")
    rendered = " | ".join(parts) if len(parts) == 2 else None
    return {"format_roundtrip": rendered == raw and parts == [name, address],
            "name_normalization_idempotent": normalize_text(normalize_text(name)) == normalize_text(name),
            "address_normalization_idempotent": normalize_text(normalize_text(address)) == normalize_text(address),
            "name_nonempty_after_normalization": bool(normalize_text(name)),
            "address_digits": _DIGITS.findall(address), "script": _script(name, address)}


def _sample_text_diagnostics(sample: list[tuple], target_for_id: dict[str, tuple],
                             links: dict[int, list[str]], *, edge_scope: str) -> dict:
    """Observed exact-text clusters are a LOWER bound within this balanced probe."""
    keys = {entity_id: (country, name, address) for entity_id, (name, address, country)
            in target_for_id.items()}
    group_sizes = Counter(keys.values())
    groups = {}
    for country in COUNTRIES:
        selected = [row for row in sample if row[4] == country]
        record_rows = [(row[2], row[3]) for row in selected]
        target_ids = {entity_id for row in selected for entity_id in links[row[0]]}
        record_rows.extend((target_for_id[entity_id][0], target_for_id[entity_id][1])
                           for entity_id in sorted(target_ids))
        raw = [_raw_roundtrip(name, address) for name, address in record_rows]
        supported = sum(group_sizes[keys[entity_id]] > 1 for entity_id in target_ids)
        with_mates = sum(any(group_sizes[keys[entity_id]] > 1 for entity_id in links[row[0]])
                         for row in selected)
        within_s1 = sum(any(count > 1 for count in Counter(keys[entity_id] for entity_id
                                                          in links[row[0]]).values()) for row in selected)
        groups[country] = {"sampled_references": len(selected), "observed_target_records": len(target_ids),
                           "targets_in_observed_exact_duplicate_group": supported,
                           "references_with_observed_exact_duplicate_target": with_mates,
                           "references_with_two_observed_same_cluster_candidates": within_s1,
                           "raw_canonicalization": {
                               "records_checked": len(raw),
                               "lossless_name_address_format_roundtrips": sum(item["format_roundtrip"] for item in raw),
                               "name_normalization_idempotent": sum(item["name_normalization_idempotent"] for item in raw),
                               "address_normalization_idempotent": sum(item["address_normalization_idempotent"] for item in raw),
                               "nonempty_normalized_names": sum(item["name_nonempty_after_normalization"] for item in raw),
                               "script_counts": dict(Counter(item["script"] for item in raw))}}
    return {"scope": edge_scope,
            "cluster_definition": "exact (raw country, name, address), among balanced-probe targets only; full-pool coverage is unknown",
            "roundtrip_definition": "raw name | address serialization and language-independent normalization, not learned denoiser accuracy",
            "by_country": groups}


def screen_v1(test_dir: Path, integrity: Path, release: Path, output: Path,
              *, sample_per_country: int = 256, seed: int = SAMPLE_SEED,
              expected_references: int = EXPECTED_REFERENCES) -> dict:
    """Full v1 output coverage and a country-balanced raw-record sample; NO scores/labels."""
    if sample_per_country < 1 or seed < 0:
        raise ValueError("A positive country-balanced sample and nonnegative seed are required")
    inputs = _test_inputs(test_dir, integrity)
    saved = _json(release)
    matching = (release.parent.parent / saved["matching_results"]["path"]).resolve()
    _verified_file(matching, saved["matching_results"]["sha256"],
                   bytes_expected=saved["matching_results"]["bytes"])
    if saved.get("source1_rows") != expected_references:
        raise ValueError("Validated v1 release has a different Source 1 population")
    stream = iter(_raw_rows(matching, ("source1_entity_id", "matched_entity_ids")))
    heaps: dict[str, list] = {country: [] for country in COUNTRIES}
    counts = {country: Counter() for country in COUNTRIES}
    seen = set()
    for index, (entity_id, name, address, country) in enumerate(_raw_rows(
            Path(inputs["source1"]["path"]), RECORD_COLUMNS)):
        if country not in counts or not entity_id.startswith("S1-") or entity_id in seen:
            raise ValueError("Duplicate/malformed Source 1 ID or unexpected country")
        seen.add(entity_id)
        try:
            predicted_id, targets = next(stream)
        except StopIteration as error:
            raise ValueError("Validated v1 output omitted a Source 1 row") from error
        if predicted_id != entity_id:
            raise ValueError("Validated v1 output lost Source 1 order/coverage")
        links = targets.split(",") if targets else []
        if len(links) != len(set(links)) or any(not link.startswith(("S2-", "S3-")) for link in links):
            raise ValueError("Duplicate/malformed v1 accepted target ID")
        row = counts[country]
        row["references"] += 1
        row["accepted_links"] += len(links)
        row["empty_predictions"] += not links
        row["multiple_links"] += len(links) > 1
        _offer_sample(heaps, (index, entity_id, name, address, country, links), sample_per_country, seed)
    _check_count(len(seen), expected_references, "official-test reference rows")
    if next(stream, None) is not None:
        raise ValueError("Validated v1 output has an extra/duplicate Source 1 row")
    sample = sorted((entry[2] for group in heaps.values() for entry in group), key=lambda row: row[0])
    if any(len(heaps[country]) != sample_per_country for country in COUNTRIES):
        raise ValueError("A country has fewer references than the balanced diagnostic sample")
    links = {row[0]: row[5] for row in sample}
    targets = _resolve_targets(test_dir, {value for values in links.values() for value in values}, inputs)
    for row in sample:
        if any(targets[target][2] != row[4] for target in links[row[0]]):
            raise ValueError("Sampled v1 accepted link crossed a test country block")
    label_free = _sample_text_diagnostics(sample, targets, links, edge_scope="v1 accepted links only")
    prediction_report = release.parent.parent / "output/submission_v1/report.json"
    if (prediction_report.is_file() and sum(c["accepted_links"] for c in counts.values())
            != _json(prediction_report)["accepted_pairs"]):
        raise ValueError("V1 accepted-link accounting changed")
    report = {"schema": 1, "scope": "validated_v1_test_output_label_free", "labels_read": False,
              "pseudo_labels_created": False, "source_integrity_sha256": sha256(integrity),
              "test_inputs": inputs, "matching": _verified_file(matching, saved["matching_results"]["sha256"]),
              "release_manifest_sha256": sha256(release), "full_reference_count": len(seen),
              "full_by_country": {country: dict(counts[country]) for country in COUNTRIES},
              "balanced_sample_per_country": sample_per_country, "sample_seed": seed,
              "balanced_sample_ordered_id_sha256": hashlib.sha256(
                  "\n".join(row[1] for row in sample).encode()).hexdigest(),
              "balanced_text_diagnostics": label_free,
              "score_distribution_shift": {"status": "requires aligned saved score array; binary v1 TSV is not a probability"},
              "limitations": "No test truth, accuracy, French correctness, or pseudo-label fitness measured."}
    output = _fresh_output(output)
    _write_json(output / "screen_v1.json", report)
    return report


@dataclass(frozen=True)
class Scored:
    reference: Path
    pool: Path
    pairs: Path
    probabilities: Path
    nreference: int
    npool: int
    npairs: int
    source: str
    identity: dict


def checked_frozen(path: Path, *, allowlist: Path) -> dict:
    """Only a selected v2 BGE checkpoint with verifiable TUNE and licence/size can seed P5."""
    import yaml

    manifest = _json(path)
    tune = manifest.get("tune", {})
    if (manifest.get("schema") != 1 or manifest.get("version") != "v2" or manifest.get("state") != "frozen"
            or manifest.get("selected_on") != "TUNE" or manifest.get("holdout2_labels_used") is not False
            or manifest.get("test_labels_used") is not False or manifest.get("positive_candidate_injection") is not False
            or tune.get("references") != 12_000 or tune.get("id_sha256") != TUNE_ID_SHA256
            or tune.get("labels_used_only_for_selection") is not True):
        raise ValueError("P5 requires a frozen v2 selected on the fixed TUNE; holdout/test labels and injected edges are barred")
    tune_report = Path(tune["report_path"])
    _verified_file(tune_report, tune["report_sha256"])
    calibration = manifest.get("calibration", {})
    if (calibration.get("fit_scope") != "TUNE_US_India_only"
            or calibration.get("france_policy") != "pooled_TUNE_no_France_labels"):
        raise ValueError("Pseudo confidence needs a frozen US/India TUNE calibrator with no French tuning")
    _verified_file(Path(calibration["path"]), calibration["sha256"])
    checkpoint = Path(manifest["ce_checkpoint_path"])
    marker = checkpoint / "complete.json"
    _verified_file(marker, manifest["ce_checkpoint_complete_sha256"])
    state = _json(marker)
    files = state.get("files_sha256", {})
    if (state.get("state") != "complete" or not files or "training_config.json" not in files
            or not any(name.endswith((".safetensors", ".bin")) for name in files)
            or set(files) != {file.name for file in checkpoint.iterdir() if file.is_file() and file.name != "complete.json"}):
        raise ValueError("Frozen v2 CE checkpoint lacks its complete, hashed weights/config/tokenizer")
    for name, digest in files.items():
        _verified_file(checkpoint / name, digest)
    model_ids = manifest.get("model_ids", [])
    if (not isinstance(model_ids, list) or len(model_ids) != len(set(model_ids))
            or "BAAI/bge-reranker-v2-m3" not in model_ids
            or manifest.get("ce_model_id") != "BAAI/bge-reranker-v2-m3"):
        raise ValueError("Only the selected, supported BGE single-logit CE family can be adapted")
    inventory = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
    available = {row["id"]: row for row in inventory["models"]}
    if (any(model not in available or available[model]["licence"] not in {"mit", "apache-2.0"}
            for model in model_ids) or not 0 < sum(float(available[m]["params_b"]) for m in model_ids)
            <= float(inventory["budget_total_params_b"]) <= 8):
        raise ValueError("Frozen v2 ensemble violates MIT/Apache or the combined 8B parameter cap")
    if sha256(allowlist) != manifest.get("allowlist_sha256"):
        raise ValueError("Frozen model inventory changed after v2 selection")
    return {"path": str(Path(path).resolve()), "sha256": sha256(path),
            "checkpoint": str(checkpoint.resolve()), "checkpoint_complete_sha256": sha256(marker),
            "tune_report_sha256": sha256(tune_report), "tune_id_sha256": TUNE_ID_SHA256,
            "calibration_sha256": calibration["sha256"],
            "models": model_ids, "inventory_params_b": sum(float(available[m]["params_b"]) for m in model_ids),
            "allowlist_sha256": sha256(allowlist)}


def _marker_file(marker: Path, item: dict, *, rows: int | None = None) -> Path:
    if not {"path", "sha256", "rows"} <= set(item):
        raise ValueError("Completed v2 scored file must have path, SHA256 and row count")
    path = Path(item["path"])
    path = path if path.is_absolute() else marker.parent / path
    if rows is not None and item["rows"] != rows:
        raise ValueError("V2 scored file lost complete reference/candidate coverage")
    _verified_file(path, item["sha256"])
    return path


def checked_scored_v2(marker: Path, frozen: dict, inputs: dict, *, nreference: int,
                      full_targets: int) -> Scored:
    document = _json(marker)
    if (document.get("schema") != 1 or document.get("state") != "complete"
            or document.get("population") != "official_test" or document.get("labels_read") is not False
            or document.get("positive_candidate_injection") is not False
            or document.get("score_semantics") != "TUNE_calibrated_match_probability"
            or document.get("all_candidates_calibrated") is not True
            or document.get("frozen_v2_sha256") != frozen["sha256"]
            or document.get("calibration_sha256") != frozen["calibration_sha256"]
            or document.get("ce_checkpoint_complete_sha256") != frozen["checkpoint_complete_sha256"]
            or document.get("full_target_pool_rows") != full_targets
            or document.get("reference_count") != nreference or document.get("candidate_pairs", 0) < 0
            or document.get("test_input_sha256") != {key: item["sha256"] for key, item in inputs.items()}):
        raise ValueError("V2 score bundle needs frozen CE/TUNE, supplied full test, calibrated scores and no label/injection use")
    files = document["files"]
    reference = _marker_file(marker, files["reference"], rows=nreference)
    pool = _marker_file(marker, files["pool"])
    pairs = _marker_file(marker, files["pairs"], rows=document["candidate_pairs"])
    probability = _marker_file(marker, files["probabilities"], rows=document["candidate_pairs"])
    if (pq.ParquetFile(reference).metadata.num_rows != nreference
            or pq.ParquetFile(pool).metadata.num_rows != files["pool"]["rows"]
            or pq.ParquetFile(pairs).metadata.num_rows != document["candidate_pairs"]
            or files["pool"]["rows"] > full_targets):
        raise ValueError("Scored v2 candidate bundle is partial or its compact pool is invalid")
    if {"label", "training_label", "true_link", "is_truth_for_scoring_only"} & set(
            pq.ParquetFile(pairs).schema_arrow.names):
        raise ValueError("Official-test scored candidates may not contain any truth/label columns")
    retrieval = _marker_file(marker, files["retrieval_complete"])
    retrieved = _json(retrieval)
    if (retrieved.get("state") != "complete" or retrieved.get("config", {}).get("positive_candidate_injection") is not False
            or retrieved.get("config", {}).get("labels_accessed") is not False
            or retrieved.get("metadata", {}).get("mining_full_pool_records") != full_targets
            or retrieved.get("metadata", {}).get("candidate_pairs") != document["candidate_pairs"]
            or retrieved.get("metadata", {}).get("reference_sha256") != files["reference"]["sha256"]
            or retrieved.get("metadata", {}).get("pool_sha256") != files["pool"]["sha256"]
            or retrieved.get("metadata", {}).get("candidate_pairs_sha256") != files["pairs"]["sha256"]
            or document.get("retrieved_pair_sha256") != files["pairs"]["sha256"]):
        raise ValueError("Scored candidates must be identical to complete label-free full-pool retrieval, never truth-injected")
    probabilities = np.load(probability, mmap_mode="r", allow_pickle=False)
    if probabilities.shape != (document["candidate_pairs"],) or not np.issubdtype(probabilities.dtype, np.floating):
        raise ValueError("V2 calibrated score vector is not one real value per retrieved edge")
    return Scored(reference, pool, pairs, probability, nreference, files["pool"]["rows"],
                  document["candidate_pairs"], "v2_TUNE_calibrated", {"marker": str(marker.resolve()),
                  "marker_sha256": sha256(marker), "frozen": frozen, "test_inputs": inputs,
                  "files": files, "full_target_pool_rows": full_targets})


def checked_v1_scores(output: Path, inputs: dict, release: Path, *, nreference: int,
                      full_targets: int) -> Scored:
    """Saved v1 blend scores are diagnostic only; they are NOT calibrated pseudo probabilities."""
    completed = _json(output / "complete.json")
    report = completed.get("report", {})
    released = _json(release)
    candidate_dir = Path(report.get("data", ""))
    retrieval = _json(candidate_dir / "complete.json")
    if (completed.get("state") != "complete" or report.get("labels_accessed") is not False
            or report.get("benchmark_only") is not False or report.get("scorer") != "frozen_tree_ce"
            or report.get("reference_count") != nreference or released.get("source1_rows") != nreference
            or report.get("pipeline_sha256") != released.get("pipeline_sha256")
            or retrieval.get("state") != "complete"
            or retrieval.get("config", {}).get("positive_candidate_injection") is not False
            or retrieval.get("config", {}).get("labels_accessed") is not False
            or retrieval.get("metadata", {}).get("mining_full_pool_records") != full_targets
            or retrieval.get("metadata", {}).get("candidate_pairs") != report.get("candidate_pairs")
            or Path(retrieval["config"]["query_input"]["path"]).name != "test_source1.tsv"
            or retrieval["config"]["query_input"]["size_bytes"] != inputs["source1"]["bytes"]):
        raise ValueError("Only the completed, full-test, label-free and validated v1 diagnostic may be read")
    reference, pool, pairs, probabilities = (candidate_dir / "reference.parquet", candidate_dir / "pool.parquet",
                                             output / "scored_pairs.parquet", output / "probabilities.npy")
    if (pq.ParquetFile(reference).metadata.num_rows != nreference
            or pq.ParquetFile(pool).metadata.num_rows != report["pool_count"]
            or pq.ParquetFile(pairs).metadata.num_rows != report["candidate_pairs"]
            or not pairs.samefile(Path(report["pairs"]))):
        raise ValueError("V1 saved scored pair order or complete test records changed")
    for label, path in (("reference", reference), ("pool", pool), ("pairs", Path(report["pairs"]))):
        original = report["input_identities"][label]
        if path.stat().st_size != original["size_bytes"] or path.stat().st_mtime_ns != original["mtime_ns"]:
            raise ValueError("V1 frozen scored input identity changed")
    scores = np.load(probabilities, mmap_mode="r", allow_pickle=False)
    if scores.shape != (report["candidate_pairs"],) or not np.issubdtype(scores.dtype, np.floating):
        raise ValueError("Saved v1 score array is incomplete")
    return Scored(reference, pool, pairs, probabilities, nreference, report["pool_count"],
                  report["candidate_pairs"], "v1_raw_blend_not_calibrated",
                  {"pipeline_sha256": report["pipeline_sha256"], "complete_sha256": sha256(output / "complete.json"),
                   "score_sha256": sha256(probabilities), "pairs_sha256": sha256(pairs),
                   "retrieval_complete_sha256": sha256(candidate_dir / "complete.json"),
                   "release_sha256": sha256(release), "test_inputs": inputs, "full_target_pool_rows": full_targets})


def _raw_sample(source1: Path, nreference: int, size: int, seed: int) -> tuple[list[tuple], dict, str]:
    heaps: dict[str, list] = {country: [] for country in COUNTRIES}
    seen = set()
    counts: Counter = Counter()
    digest = hashlib.sha256()
    for index, (entity_id, name, address, country) in enumerate(_raw_rows(source1, RECORD_COLUMNS)):
        if country not in heaps or not entity_id.startswith("S1-") or entity_id in seen:
            raise ValueError("Raw test Source 1 contains a duplicate/malformed ID or country")
        seen.add(entity_id)
        digest.update((entity_id + "\n").encode())
        counts[country] += 1
        _offer_sample(heaps, (index, entity_id, name, address, country), size, seed)
    _check_count(len(seen), nreference, "official test Source 1 rows")
    if any(len(heaps[country]) != size for country in COUNTRIES):
        raise ValueError("Country-balanced diagnostic has too few supplied references")
    return sorted((entry[2] for group in heaps.values() for entry in group), key=lambda row: row[0]), dict(counts), digest.hexdigest()


def _reference_codes(scored: Scored, sample: list[tuple], raw_id_sha: str) -> tuple[np.ndarray, dict]:
    codes = np.empty(scored.nreference, dtype=np.uint8)
    expected = {row[0]: row for row in sample}
    digest = hashlib.sha256()
    seen = set()
    cursor = 0
    for batch in pq.ParquetFile(scored.reference).iter_batches(batch_size=25_000, columns=list(RECORD_COLUMNS)):
        frame = batch.to_pandas()
        for offset, row in enumerate(frame.itertuples(index=False, name=None)):
            entity, name, address, country = row
            if entity in seen or country not in COUNTRY_CODE:
                raise ValueError("Scored test references have duplicate IDs or an unexpected country")
            seen.add(entity)
            digest.update((entity + "\n").encode())
            if cursor + offset in expected and row != expected[cursor + offset][1:5]:
                raise ValueError("Scored reference ID/text/country differs from supplied test Source 1")
            codes[cursor + offset] = COUNTRY_CODE[country]
        cursor += len(frame)
    if cursor != scored.nreference or digest.hexdigest() != raw_id_sha:
        raise ValueError("Scored test references lost full raw Source 1 order or singleton coverage")
    return codes, {"raw_and_scored_ordered_ids_sha256": raw_id_sha, "unique_ids": len(seen)}


def _pool_codes(path: Path, rows: int) -> np.ndarray:
    codes = np.empty(rows, dtype=np.uint8)
    # Same collision-detecting uniqueness guard as the full dense test miner;
    # distinct positional rows may not silently denote the same target ID.
    hashes = np.empty(rows, dtype=np.uint64)
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=["entity_id", "country"]):
        ids, countries = (batch.column(i).to_pylist() for i in (0, 1))
        if any(country not in COUNTRY_CODE for country in countries):
            raise ValueError("Scored test target pool contains an unexpected country")
        if any(not isinstance(entity, str) or not entity.startswith(("S2-", "S3-")) for entity in ids):
            raise ValueError("Scored test target pool contains a malformed/empty S2/S3 ID")
        codes[offset:offset + len(countries)] = [COUNTRY_CODE[country] for country in countries]
        hashes[offset:offset + len(ids)] = pd.util.hash_array(np.asarray(ids, dtype=object))
        offset += len(countries)
    _check_count(offset, rows, "scored compact test pool")
    hashes.sort()
    if np.any(hashes[1:] == hashes[:-1]):
        raise ValueError("Scored test target pool contains duplicate IDs or an ID-hash collision")
    return codes


def _score_batches(scored: Scored, reference_codes: np.ndarray, pool_codes: np.ndarray,
                   *, batch_size: int = 100_000):
    """Strict global pair order also rejects duplicate pairs spanning row groups."""
    scores = np.load(scored.probabilities, mmap_mode="r", allow_pickle=False)
    if scores.shape != (scored.npairs,) or batch_size < 1:
        raise ValueError("Aligned probabilities and a positive reader batch are required")
    file = pq.ParquetFile(scored.pairs)
    if not {"left_idx", "right_idx"} <= set(file.schema_arrow.names):
        raise ValueError("Scored candidate file lacks positional pair IDs")
    previous = (-1, -1)
    offset = 0
    for batch in file.iter_batches(batch_size=batch_size, columns=["left_idx", "right_idx"], use_threads=False):
        left, right = (batch.column(i).to_numpy(zero_copy_only=False) for i in (0, 1))
        if (not np.issubdtype(left.dtype, np.integer) or not np.issubdtype(right.dtype, np.integer)
                or len(left) == 0 or left.min() < 0 or left.max() >= scored.nreference
                or right.min() < 0 or right.max() >= scored.npool):
            raise ValueError("Scored candidate index is null, noninteger or outside its complete test records")
        left, right = left.astype(np.int64), right.astype(np.int64)
        if ((int(left[0]), int(right[0])) <= previous
                or np.any(left[1:] < left[:-1])
                or np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1]))):
            raise ValueError("Scored candidate order changed or a duplicate pair crossed batch boundaries")
        previous = (int(left[-1]), int(right[-1]))
        if not np.array_equal(reference_codes[left], pool_codes[right]):
            raise ValueError("Scored test candidate crossed a country block")
        values = np.asarray(scores[offset:offset + len(left)], dtype=np.float64)
        if len(values) != len(left) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError("Scored probabilities are incomplete, nonfinite or outside [0,1]")
        yield left, right, values, offset
        offset += len(left)
    _check_count(offset, scored.npairs, "scored candidate pairs")


def _pool_probe(path: Path, requested: set[int]) -> dict[int, tuple]:
    records = {}
    ordered = np.asarray(sorted(requested), dtype=np.int64)
    offset = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=50_000, columns=list(RECORD_COLUMNS)):
        lower, upper = np.searchsorted(ordered, [offset, offset + batch.num_rows])
        if upper > lower:
            frame = batch.to_pandas()
            for position in ordered[lower:upper]:
                row = tuple(frame.iloc[int(position) - offset])
                if int(position) in records or not row[0].startswith(("S2-", "S3-")):
                    raise ValueError("Duplicate or malformed sampled compact target")
                records[int(position)] = row
        offset += batch.num_rows
    if set(records) != requested:
        raise ValueError("Sampled compact target is absent from the scored test pool")
    return records


def _distribution(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {"count": len(data), "histogram_edges": list(SCORE_BINS),
            "histogram_counts": np.histogram(data, bins=SCORE_BINS)[0].tolist(),
            "quantiles_0_25_50_75_90_98_100": np.quantile(data, [0, .25, .5, .75, .9, .98, 1]).tolist()
            if len(data) else None, "mean": float(data.mean()) if len(data) else None,
            "fraction_at_least_098": float((data >= POSITIVE_MIN).mean()) if len(data) else None,
            "fraction_at_most_002": float((data <= NEGATIVE_MAX).mean()) if len(data) else None}


def _ks(a: list[float], b: list[float]) -> float | None:
    if not a or not b:
        return None
    x, y = np.sort(a), np.sort(b)
    grid = np.unique(np.concatenate((x, y)))
    return float(np.max(np.abs(np.searchsorted(x, grid, side="right") / len(x)
                               - np.searchsorted(y, grid, side="right") / len(y))))


def diagnose_scores(scored: Scored, test_dir: Path, integrity: Path, output: Path,
                    *, sample_per_country: int = 256, seed: int = SAMPLE_SEED,
                    batch_size: int = 100_000) -> dict:
    """Country-balanced scores and exact observed clusters; no fit/decision tuning."""
    if sample_per_country < 1 or seed < 0:
        raise ValueError("Positive sample and fixed nonnegative seed required")
    inputs = _test_inputs(test_dir, integrity)
    if scored.identity["test_inputs"] != inputs:
        raise ValueError("Scored bundle is not from these exact supplied test files")
    sample, raw_counts, raw_id_sha = _raw_sample(Path(inputs["source1"]["path"]), scored.nreference,
                                                  sample_per_country, seed)
    ref_codes, ref_check = _reference_codes(scored, sample, raw_id_sha)
    pool_codes = _pool_codes(scored.pool, scored.npool)
    sampled = np.zeros(scored.nreference, dtype=bool)
    sampled[[row[0] for row in sample]] = True
    by_country = {country: {"pair_scores": []} for country in COUNTRIES}
    top = np.full(scored.nreference, -1., dtype=np.float64)
    candidate_counts = np.zeros(scored.nreference, dtype=np.int32)
    links: dict[int, list[int]] = {row[0]: [] for row in sample}
    for left, right, values, _ in _score_batches(scored, ref_codes, pool_codes, batch_size=batch_size):
        unique, count = np.unique(left, return_counts=True)
        candidate_counts[unique] += count.astype(np.int32)
        chosen = sampled[left]
        for position in np.flatnonzero(chosen):
            ref = int(left[position])
            top[ref] = max(top[ref], float(values[position]))
            by_country[COUNTRIES[ref_codes[ref]]]["pair_scores"].append(float(values[position]))
            links[ref].append(int(right[position]))
    # Group counts from all scored S1s include zero-candidate references by design.
    result = {}
    for country in COUNTRIES:
        indices = np.flatnonzero(ref_codes == COUNTRY_CODE[country])
        selected = [row for row in sample if row[4] == country]
        values = [float(top[row[0]]) for row in selected if top[row[0]] >= 0]
        result[country] = {"sampled_references": len(selected), "full_references": len(indices),
                           "full_candidate_pairs": int(candidate_counts[indices].sum()),
                           "full_zero_candidate_references": int((candidate_counts[indices] == 0).sum()),
                           "sampled_zero_candidate_references": int(sum(top[row[0]] < 0 for row in selected)),
                           "candidate_pair_scores": _distribution(by_country[country]["pair_scores"]),
                           "per_reference_maximum": _distribution(values)}
        if len(indices) != raw_counts[country]:
            raise ValueError("Scored references' full country distribution differs from raw test records")
    pool_probe = _pool_probe(scored.pool, {target for rows in links.values() for target in rows})
    if len({row[0] for row in pool_probe.values()}) != len(pool_probe):
        raise ValueError("The scored compact sample contained a duplicate test target ID")
    targets = _resolve_targets(test_dir, {pool_probe[index][0] for index in pool_probe}, inputs)
    for row in pool_probe.values():
        if row[1:] != (*targets[row[0]][:2], targets[row[0]][2]):
            raise ValueError("Scored compact target text/country differs from supplied test records")
    for row in sample:
        if any(pool_probe[i][3] != row[4] for i in links[row[0]]):
            raise ValueError("Sampled scored target crossed a test country block")
    named = {row[0]: [pool_probe[i][0] for i in links[row[0]]] for row in sample}
    text = _sample_text_diagnostics(sample, targets, named, edge_scope="all scored retrieved candidates")
    report = {"schema": 1, "scope": scored.source + "_balanced_test_label_free",
              "labels_read": False, "pseudo_labels_created": False,
              "no_french_threshold_selection": True, "balanced_sample_per_country": sample_per_country,
              "seed": seed, "raw_and_scored_reference_identity": ref_check,
              "sample_ordered_id_sha256": hashlib.sha256("\n".join(row[1] for row in sample).encode()).hexdigest(),
              "full_references": scored.nreference, "full_scored_pairs": scored.npairs,
              "by_country": result, "balanced_text_diagnostics": text,
              "france_vs_india_us": {"candidate_pair_score_ks": _ks(by_country["France"]["pair_scores"],
                                                                     by_country["India"]["pair_scores"] + by_country["US"]["pair_scores"]),
                                       "max_score_ks": _ks([float(top[row[0]]) for row in sample if row[4] == "France" and top[row[0]] >= 0],
                                                           [float(top[row[0]]) for row in sample if row[4] != "France" and top[row[0]] >= 0])},
              "score_semantics": scored.source, "scored_identity": scored.identity,
              "source_integrity_sha256": sha256(integrity),
              "limitations": "Distribution/cluster/format shift is label-free; neither French accuracy nor calibrated probabilities can be inferred from v1 raw blend scores."}
    output = _fresh_output(output)
    _write_json(output / "diagnostics.json", report)
    return report


def _scored_groups(scored: Scored, ref_codes: np.ndarray, pool_codes: np.ndarray, *, batch_size: int):
    """Yield *each* S1, including gaps with no candidate (the singleton convention)."""
    current = 0
    rights: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []

    def flush(reference: int):
        return (reference, np.concatenate(rights) if rights else np.empty(0, dtype=np.int64),
                np.concatenate(probabilities) if probabilities else np.empty(0, dtype=np.float64))

    for left, right, values, _ in _score_batches(scored, ref_codes, pool_codes, batch_size=batch_size):
        boundaries = np.r_[0, np.flatnonzero(left[1:] != left[:-1]) + 1, len(left)]
        for a, b in zip(boundaries[:-1], boundaries[1:]):
            reference = int(left[a])
            while current < reference:
                yield flush(current)
                current += 1
                rights.clear()
                probabilities.clear()
            rights.append(right[a:b])
            probabilities.append(values[a:b])
    while current < scored.nreference:
        yield flush(current)
        current += 1
        rights.clear()
        probabilities.clear()


PSEUDO_SCHEMA = pa.schema([("left_idx", pa.int32()), ("right_idx", pa.int32()),
                           ("p", pa.float32()), ("pseudo_label", pa.bool_()), ("weight", pa.float32())])
ENTITY_SCHEMA = pa.schema([("left_idx", pa.int32()), ("entity_id", pa.string()), ("country", pa.string()),
                           ("candidate_pairs", pa.int32()), ("high_claims", pa.int32()),
                           ("pseudo_positives", pa.int32()), ("eligible_negatives", pa.int32()),
                           ("pseudo_negatives", pa.int32()), ("conflicting_high_claims", pa.int32()),
                           ("maximum_p", pa.float32())])


def _append_rows(writer: pq.ParquetWriter, rows: list[tuple], schema: pa.Schema) -> None:
    writer.write_table(pa.Table.from_pylist([dict(zip(schema.names, row, strict=True)) for row in rows], schema=schema))


def select_pseudo(scored: Scored, test_dir: Path, integrity: Path, frozen: dict, output: Path,
                  *, max_negatives: int = MAX_NEGATIVES_PER_S1, batch_size: int = 100_000) -> dict:
    """Two full candidate passes: globally resolve conflicts, then retain ONLY retrieved edges."""
    if (scored.source != "v2_TUNE_calibrated" or scored.identity["frozen"] != frozen
            or not 1 <= max_negatives <= MAX_NEGATIVES_PER_S1):
        raise ValueError("Pseudo-labels require frozen v2 calibrated scores and the fixed <=10-negative S1 cap")
    inputs = _test_inputs(test_dir, integrity)
    if inputs != scored.identity["test_inputs"]:
        raise ValueError("Supplied test records changed after frozen v2 inference")
    sample, raw_counts, raw_sha = _raw_sample(Path(inputs["source1"]["path"]), scored.nreference, 1, SAMPLE_SEED)
    ref_codes, ref_check = _reference_codes(scored, sample, raw_sha)
    pool_codes = _pool_codes(scored.pool, scored.npool)
    # -1: unclaimed; -2: at least two S1s claim the same target at >=.98.
    owners = np.full(scored.npool, -1, dtype=np.int32)
    high_claims = 0
    for left, right, values, _ in _score_batches(scored, ref_codes, pool_codes, batch_size=batch_size):
        high = np.flatnonzero(values >= POSITIVE_MIN)
        high_claims += len(high)
        for index in high:
            target, reference = int(right[index]), int(left[index])
            old = owners[target]
            if old == -1:
                owners[target] = reference
            elif old != reference:
                owners[target] = -2
    conflicts = int(np.count_nonzero(owners == -2))
    output = _fresh_output(output)
    pseudo_path, entities_path = output / "pseudo_pairs.parquet", output / "entity_summary.parquet"
    pseudo_tmp, entities_tmp = output / "pseudo_pairs.parquet.tmp", output / "entity_summary.parquet.tmp"
    reference_ids = pd.read_parquet(scored.reference, columns=["entity_id"]).entity_id.tolist()
    if len(reference_ids) != scored.nreference:
        raise ValueError("Scored references changed after full ID-order verification")
    counts = Counter()
    countries = {country: Counter() for country in COUNTRIES}
    seen_targets = set()  # Only selected positives; the owner bitmap already enforces global uniqueness.
    with pq.ParquetWriter(pseudo_tmp, PSEUDO_SCHEMA, compression="zstd") as positive_writer, \
            pq.ParquetWriter(entities_tmp, ENTITY_SCHEMA, compression="zstd") as summary_writer:
        pseudo_rows: list[tuple] = []
        entity_rows: list[tuple] = []
        for reference, rights, probabilities in _scored_groups(scored, ref_codes, pool_codes, batch_size=batch_size):
            country = COUNTRIES[ref_codes[reference]]
            high = np.flatnonzero(probabilities >= POSITIVE_MIN)
            valid_pos = [int(i) for i in high if owners[int(rights[i])] == reference]
            ambiguous = len(high) - len(valid_pos)
            low = np.flatnonzero(probabilities <= NEGATIVE_MAX)
            eligible = [int(i) for i in low if owners[int(rights[i])] != -2]
            # Hardest of the specified low-probability negatives, with stable ID tie-breaking.
            chosen_neg = sorted(eligible, key=lambda i: (-float(probabilities[i]), int(rights[i])))[:max_negatives]
            selected = [(int(rights[i]), float(probabilities[i]), True) for i in valid_pos]
            selected += [(int(rights[i]), float(probabilities[i]), False) for i in chosen_neg]
            for target, p, label in sorted(selected):
                if label:
                    if target in seen_targets:
                        raise ValueError("Two pseudo-positive S1s survived the one-to-one filter")
                    seen_targets.add(target)
                pseudo_rows.append((reference, target, p, label, PSEUDO_WEIGHT))
            entity_rows.append((reference, reference_ids[reference], country, len(rights), len(high),
                                len(valid_pos), len(eligible), len(chosen_neg), ambiguous,
                                float(probabilities.max()) if len(rights) else None))
            counts["references"] += 1
            counts["candidates"] += len(rights)
            counts["high_raw"] += len(high)
            counts["pseudo_positive"] += len(valid_pos)
            counts["high_discarded_conflict"] += ambiguous
            counts["low_raw"] += len(low)
            counts["low_discarded_conflict"] += len(low) - len(eligible)
            counts["low_eligible"] += len(eligible)
            counts["pseudo_negative"] += len(chosen_neg)
            counts["references_without_candidates"] += not len(rights)
            counts["references_without_pseudo_positive"] += not valid_pos
            countries[country]["references"] += 1
            countries[country]["pseudo_positive"] += len(valid_pos)
            countries[country]["pseudo_negative"] += len(chosen_neg)
            countries[country]["without_candidates"] += not len(rights)
            countries[country]["high_discarded_conflict"] += ambiguous
            if len(pseudo_rows) >= 20_000:
                _append_rows(positive_writer, pseudo_rows, PSEUDO_SCHEMA)
                pseudo_rows.clear()
            if len(entity_rows) >= 10_000:
                _append_rows(summary_writer, entity_rows, ENTITY_SCHEMA)
                entity_rows.clear()
        if pseudo_rows:
            _append_rows(positive_writer, pseudo_rows, PSEUDO_SCHEMA)
        if entity_rows:
            _append_rows(summary_writer, entity_rows, ENTITY_SCHEMA)
    pseudo_tmp.replace(pseudo_path)
    entities_tmp.replace(entities_path)
    _check_count(counts["references"], scored.nreference, "pseudo audit Source 1 rows")
    _check_count(counts["candidates"], scored.npairs, "pseudo selection scored candidate pairs")
    _check_count(pq.ParquetFile(entities_path).metadata.num_rows, scored.nreference, "singleton-aware audit rows")
    _check_count(pq.ParquetFile(pseudo_path).metadata.num_rows,
                 counts["pseudo_positive"] + counts["pseudo_negative"], "selected pseudo candidate edges")
    if counts["high_raw"] - counts["pseudo_positive"] != counts["high_discarded_conflict"]:
        raise ValueError("Conflicting high-confidence claims were not all removed")
    for country in COUNTRIES:
        _check_count(countries[country]["references"], raw_counts[country], f"{country} test Source 1 coverage")
    report = {"schema": 1, "state": "complete", "population": "official_test_label_free",
              "labels_read": False, "positive_candidate_injection": False,
              "selected_only_from_retrieved_scored_pairs": True,
              "thresholds": {"positive_at_least": POSITIVE_MIN, "negative_at_most": NEGATIVE_MAX,
                             "pseudo_ce_loss_weight": PSEUDO_WEIGHT,
                             "max_negative_pairs_per_source1": max_negatives,
                             "ambiguous_high_targets": "discard every high claim and low claim on that target"},
              "conflicting_targets": conflicts, "counts": dict(counts),
              "by_country": {country: dict(countries[country]) for country in COUNTRIES},
              "raw_and_scored_reference_identity": ref_check, "frozen": frozen,
              "scored": scored.identity, "source_sha256": sha256(Path(__file__)),
              "outputs": {"pseudo_pairs": {"path": str(pseudo_path.resolve()),
                                            "sha256": sha256(pseudo_path),
                                            "rows": pq.ParquetFile(pseudo_path).metadata.num_rows},
                          "entity_summary": {"path": str(entities_path.resolve()),
                                             "sha256": sha256(entities_path), "rows": scored.nreference}},
              "fit_status": "blocked until separate frozen-v2/TUNE preflight and explicit GPU launch",
              "limitations": "These are unverified predictions, not French labels or test accuracy. Do not fit a CE without the TUNE gate."}
    _write_json(output / "selection.json", report)
    return report


def checked_selection(directory: Path, scored: Scored, frozen: dict) -> tuple[dict, pd.DataFrame]:
    """A completed selector output, tied to the exact frozen model and scored edges."""
    directory = Path(directory)
    manifest_path = directory / "selection.json"
    report = _json(manifest_path)
    if (report.get("state") != "complete" or report.get("labels_read") is not False
            or report.get("positive_candidate_injection") is not False
            or report.get("selected_only_from_retrieved_scored_pairs") is not True
            or report.get("frozen") != frozen or report.get("scored") != scored.identity
            or report.get("source_sha256") != sha256(Path(__file__))
            or report.get("counts", {}).get("references") != scored.nreference
            or report.get("counts", {}).get("candidates") != scored.npairs
            or report.get("thresholds") != {"positive_at_least": POSITIVE_MIN,
                                             "negative_at_most": NEGATIVE_MAX,
                                             "pseudo_ce_loss_weight": PSEUDO_WEIGHT,
                                             "max_negative_pairs_per_source1": MAX_NEGATIVES_PER_S1,
                                             "ambiguous_high_targets": "discard every high claim and low claim on that target"}):
        raise ValueError("P5 fitting needs complete, unchanged, full-test, frozen-v2 pseudo selection")
    outputs = report["outputs"]
    pairs, audit = (directory / "pseudo_pairs.parquet", directory / "entity_summary.parquet")
    for name, path, expected in (("pseudo_pairs", pairs, report["counts"]["pseudo_positive"]
                                  + report["counts"]["pseudo_negative"]),
                                 ("entity_summary", audit, scored.nreference)):
        if (Path(outputs[name]["path"]).resolve() != path.resolve() or outputs[name]["rows"] != expected
                or pq.ParquetFile(path).metadata.num_rows != expected):
            raise ValueError("Pseudo pair/audit row count or path differs from the completion marker")
        _verified_file(path, outputs[name]["sha256"])
    pairs_frame = pd.read_parquet(pairs)
    if (len(pairs_frame) == 0 or pairs_frame.duplicated(["left_idx", "right_idx"]).any()
            or pairs_frame.left_idx.min() < 0 or pairs_frame.left_idx.max() >= scored.nreference
            or pairs_frame.right_idx.min() < 0 or pairs_frame.right_idx.max() >= scored.npool
            or not np.isfinite(pairs_frame.p.to_numpy()).all()
            or not np.allclose(pairs_frame.weight.to_numpy(dtype=float), PSEUDO_WEIGHT, atol=1e-8)
            or not pairs_frame.pseudo_label.isin([True, False]).all()):
        raise ValueError("Selected pseudo edges are duplicated, unbounded or above the .3 CE weight cap")
    positive = pairs_frame.pseudo_label.to_numpy(dtype=bool)
    if (np.any(pairs_frame.p[positive] < POSITIVE_MIN) or np.any(pairs_frame.p[~positive] > NEGATIVE_MAX)
            or pairs_frame.loc[positive, "right_idx"].duplicated().any()
            or int(positive.sum()) != report["counts"]["pseudo_positive"]
            or int((~positive).sum()) != report["counts"]["pseudo_negative"]):
        raise ValueError("Pseudo confidence, one-to-one target assignment or label counts changed")
    entity_frame = pd.read_parquet(audit, columns=["left_idx", "entity_id", "candidate_pairs",
                                                  "pseudo_positives", "pseudo_negatives"])
    if (not np.array_equal(entity_frame.left_idx.to_numpy(), np.arange(scored.nreference))
            or entity_frame.entity_id.duplicated().any()
            or int(entity_frame.candidate_pairs.sum()) != scored.npairs
            or int(entity_frame.pseudo_positives.sum()) != int(positive.sum())
            or int(entity_frame.pseudo_negatives.sum()) != int((~positive).sum())):
        raise ValueError("Pseudo audit omitted/duplicated a Source 1 singleton or lost candidate coverage")
    keys = (pairs_frame.left_idx.to_numpy(dtype=np.uint64) << np.uint64(32)
            | pairs_frame.right_idx.to_numpy(dtype=np.uint64))
    if len(keys) > 1 and np.any(keys[1:] <= keys[:-1]):
        raise ValueError("Pseudo edges are duplicated or no longer ordered like retrieved candidates")
    reference_countries = pd.read_parquet(scored.reference, columns=["country"]).country
    if len(reference_countries) != scored.nreference or not reference_countries.isin(COUNTRIES).all():
        raise ValueError("Scored reference country blocks changed before fit")
    ref_codes = np.asarray([COUNTRY_CODE[country] for country in reference_countries], dtype=np.uint8)
    pool_codes = _pool_codes(scored.pool, scored.npool)
    cursor = 0
    for left, right, probabilities, _ in _score_batches(scored, ref_codes, pool_codes):
        candidate_keys = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
        end = int(np.searchsorted(keys, candidate_keys[-1], side="right"))
        wanted = keys[cursor:end]
        if len(wanted):
            positions = np.searchsorted(candidate_keys, wanted)
            if (np.any(positions >= len(candidate_keys))
                    or not np.array_equal(candidate_keys[positions], wanted)
                    or not np.allclose(pairs_frame.p.to_numpy()[cursor:end], probabilities[positions],
                                       rtol=0, atol=1e-7)):
                raise ValueError("Pseudo edge or calibrated probability was not in frozen test retrieval")
        cursor = end
    if cursor != len(keys):
        raise ValueError("Pseudo edge is absent from the completed frozen test candidate set")
    return report, pairs_frame


def assert_disjoint_populations(train_reference: pd.DataFrame, train_pool: pd.DataFrame,
                                tune_reference: pd.DataFrame, test_reference: pd.DataFrame,
                                test_pool: pd.DataFrame, pseudo_pairs: pd.DataFrame) -> dict:
    """No selected test ID may be interpreted as a TRAIN/TUNE record or target."""
    left = pseudo_pairs.left_idx.to_numpy(dtype=np.int64)
    right = pseudo_pairs.right_idx.to_numpy(dtype=np.int64)
    if (not len(left) or left.min() < 0 or right.min() < 0
            or left.max() >= len(test_reference) or right.max() >= len(test_pool)):
        raise ValueError("Pseudo indices cannot address another population or an absent test record")
    s1 = set(test_reference.entity_id.to_numpy()[np.unique(left)])
    targets = set(test_pool.entity_id.to_numpy()[np.unique(right)])
    if (len(s1) != len(np.unique(left)) or len(targets) != len(np.unique(right))
            or any(not entity.startswith("S1-") for entity in s1)
            or any(not entity.startswith(("S2-", "S3-")) for entity in targets)
            or s1 & targets or train_reference.entity_id.isin(s1 | targets).any()
            or train_pool.entity_id.isin(s1 | targets).any()
            or tune_reference.entity_id.isin(s1 | targets).any()
            or test_reference.entity_id.isin(set(train_reference.entity_id) | set(tune_reference.entity_id)).any()):
        raise ValueError("Cross-population test/TRAIN/TUNE Source 1 or target ID mixing is forbidden")
    country_ref = test_reference.country.to_numpy()[left]
    country_pool = test_pool.country.to_numpy()[right]
    if not np.array_equal(country_ref, country_pool):
        raise ValueError("Pseudo pair crossed a country block")
    return {"selected_test_source1_ids": len(s1), "selected_test_target_ids": len(targets),
            "train_source1_rows": len(train_reference), "train_target_rows": len(train_pool),
            "tune_source1_rows": len(tune_reference), "no_shared_population_ids": True,
            "same_country_pairs": len(pseudo_pairs)}


def mixed_batch_plan(positive: np.ndarray, negative: np.ndarray, pseudo: pd.DataFrame,
                     cycle, negative_channels: dict, rng: np.random.Generator,
                     *, effective_batch: int = 256, pseudo_fraction: float = .25) -> list[tuple]:
    """Balanced labels in two populations; real weight 1, test pseudo weight .3."""
    from .v2_ce_train import sample_hard_negatives

    if (effective_batch < 4 or effective_batch % 2 or not 0 < pseudo_fraction <= .3
            or not len(positive) or not len(negative)):
        raise ValueError("P5 needs an even, balanced CE batch and a capped pseudo presentation fraction")
    half = effective_batch // 2
    pseudo_per_class = int(half * pseudo_fraction)
    if pseudo_per_class < 1 or pseudo_per_class >= half:
        raise ValueError("Both real and pseudo positive/negative classes need at least one presentation")
    labelled = pseudo.pseudo_label.to_numpy(dtype=bool)
    pseudo_pos, pseudo_neg = np.flatnonzero(labelled), np.flatnonzero(~labelled)
    if not len(pseudo_pos) or not len(pseudo_neg):
        raise ValueError("Pseudo selection must contain both one-to-one positives and low-p candidate negatives")
    real_pos = positive[cycle.take(half - pseudo_per_class)]
    real_neg = negative[sample_hard_negatives(negative_channels, rng, half - pseudo_per_class)]
    picked_pos = pseudo.iloc[rng.choice(pseudo_pos, pseudo_per_class, replace=True)]
    picked_neg = pseudo.iloc[rng.choice(pseudo_neg, pseudo_per_class, replace=True)]
    plan = [("train", int(li), int(ri), 1., 1.) for li, ri in real_pos]
    plan += [("train", int(row["left"]), int(row["right"]), 0., 1.) for row in real_neg]
    plan += [("test", int(row.left_idx), int(row.right_idx), 1., PSEUDO_WEIGHT)
             for row in picked_pos.itertuples(index=False)]
    plan += [("test", int(row.left_idx), int(row.right_idx), 0., PSEUDO_WEIGHT)
             for row in picked_neg.itertuples(index=False)]
    return [plan[int(i)] for i in rng.permutation(len(plan))]


def weighted_binary_ce(logits, labels, weights, populations):
    """The actual CE adaptation objective; pseudo examples cannot outweigh .3."""
    import torch
    from torch.nn import functional as F

    if (logits.ndim != 1 or labels.shape != logits.shape or weights.shape != logits.shape
            or len(populations) != len(logits) or len(logits) < 2):
        raise ValueError("Weighted BCE requires an aligned, nonempty CE batch")
    if any(population not in ("train", "test") for population in populations):
        raise ValueError("Only separate TRAIN and test-pseudo CE populations are supported")
    expected = torch.as_tensor([1. if population == "train" else PSEUDO_WEIGHT
                                for population in populations], dtype=weights.dtype, device=weights.device)
    if not torch.allclose(weights, expected, atol=1e-6, rtol=0):
        raise ValueError("Only TRAIN=1 or test-pseudo<=.3 loss weights may be used")
    if (not torch.isfinite(logits).all() or not torch.isfinite(labels).all()
            or not torch.isfinite(weights).all() or not torch.all((labels == 0) | (labels == 1))):
        raise ValueError("CE batch has nonfinite logits/weights or nonbinary targets")
    return (F.binary_cross_entropy_with_logits(logits.float(), labels.float(), reduction="none")
            * weights.float()).sum() / weights.float().sum()


def fixed_tune_decision(decision: dict, tune, cross: np.ndarray) -> tuple[dict, pd.DataFrame, np.ndarray]:
    """Compare old/new CE with the *same* frozen v2 tree gate, blend and threshold."""
    from . import v2_ce_train as ce

    if (decision.get("kind") != "tree_ce_blend_threshold"
            or type(decision.get("tree_gate")) not in (int, float)
            or type(decision.get("ce_weight")) not in (int, float)
            or type(decision.get("threshold")) not in (int, float)
            or not 0 < decision["tree_gate"] < 1
            or not 0 < decision["ce_weight"] <= 1
            or not 0 < decision["threshold"] <= 1):
        raise ValueError("A controlled P5 comparison requires the frozen v2 tree+CE blend/threshold decision adapter")
    if (cross.shape != tune.base.shape or not np.isfinite(cross).all()
            or np.any((cross < 0) | (cross > 1))):
        raise ValueError("Adapted CE TUNE scores must align with every original full-pool TUNE candidate")
    gated = tune.base >= decision["tree_gate"]
    values = np.zeros(len(tune.base), dtype=np.float32)
    weight = decision["ce_weight"]
    values[gated] = ((1 - weight) * tune.base[gated] + weight * cross[gated]).astype(np.float32)
    frame = ce._entity_frame(tune, values, float(decision["threshold"]))
    return ce._entity_report(frame, float(decision["threshold"])), frame, values


def fit_preflight(args, scored: Scored, frozen: dict) -> tuple[dict, dict]:
    """CPU gate; the only labels opened are completed TRAIN and fixed TUNE."""
    from . import v2_ce_train as ce

    decision = _json(Path(frozen["path"])).get("decision", {})
    # P1's count-prior/assignment needs its *own* evaluation adapter. Never
    # compare a different decision procedure to the saved frozen-v2 baseline.
    if (decision.get("kind") != "tree_ce_blend_threshold"
            or any(type(decision.get(key)) not in (int, float)
                   for key in ("tree_gate", "ce_weight", "threshold"))
            or not 0 < decision["tree_gate"] < 1 or not 0 < decision["ce_weight"] <= 1
            or not 0 < decision["threshold"] <= 1):
        raise ValueError("Frozen v2 needs a supported fixed decision adapter before controlled P5 CE fitting")
    report, pseudo = checked_selection(args.selection, scored, frozen)
    ready, files = ce.validate_ready(args.train_ready)
    prepared = _json(args.prepared_dir / "prepared.json")
    if (prepared.get("identity", {}).get("p2_ready_sha256") != sha256(args.train_ready)
            or prepared["identity"].get("positive_candidate_injection") is not False
            or prepared["identity"].get("tune", {}).get("tune_id_sha256") != TUNE_ID_SHA256
            or sha256(args.prepared_dir / "positive_pairs.npy") != prepared["positive_sha256"]
            or sha256(args.prepared_dir / "negative_pairs.npy") != prepared["negative_sha256"]):
        raise ValueError("P3 TRAIN pools or fixed full-pool TUNE changed after v2 CE checkpoint selection")
    forbidden, sealed_sha = ce.sealed_ids(args.sealed_reservation)
    if sealed_sha != prepared["identity"]["sealed_reservation_sha256"]:
        raise ValueError("Fresh HOLDOUT-2 exclusion changed; P5 cannot fit on held-out entities")
    tune = ce.load_tune(args, forbidden, prepared)
    base_config = _json(Path(frozen["checkpoint"]) / "training_config.json")
    if (base_config.get("prepared_identity") != prepared["identity"]
            or base_config.get("tune_identity") != tune.identity
            or tune.identity["full_pool_records"] != ce.FULL_POOL_ROWS
            or decision["tree_gate"] != base_config["tree_gate"]):
        raise ValueError("Frozen CE checkpoint was not fitted/evaluated on these exact TRAIN/TUNE full-pool identities")
    train_ref = ce._records(files["reference"], reference=True)
    train_pool = ce._records(files["pool"], reference=False)
    test_ref = pd.read_parquet(scored.reference, columns=list(RECORD_COLUMNS))
    test_pool = pd.read_parquet(scored.pool, columns=list(RECORD_COLUMNS))
    if test_ref.entity_id.isin(forbidden).any():
        raise ValueError("Official-test Source 1 IDs overlap the sealed TRAIN/TUNE/holdout exclusion")
    disjoint = assert_disjoint_populations(train_ref, train_pool, tune.reference, test_ref, test_pool, pseudo)
    test_inputs = _test_inputs(args.test_dir, args.integrity)
    if test_inputs != scored.identity["test_inputs"]:
        raise ValueError("Raw supplied test records changed after pseudo selection")
    frozen_tune = _json(frozen["path"])["tune"]
    baseline = Path(frozen_tune["entities_path"])
    _verified_file(baseline, frozen_tune["entities_sha256"])
    if frozen_tune.get("baseline_decision") != decision:
        raise ValueError("Saved v2 baseline entities must use the SAME frozen decision as P5 TUNE evaluation")
    comparison = pd.read_csv(baseline, sep="\t")
    if (comparison.source1_entity_id.duplicated().any() or len(comparison) != 12_000
            or not np.array_equal(comparison.source1_entity_id.to_numpy(), tune.reference.entity_id.to_numpy())
            or not np.array_equal(comparison.country.to_numpy(), tune.reference.country.to_numpy())):
        raise ValueError("The frozen v2 baseline must contain every fixed TUNE entity in the same order")
    identity = {"state": "preflight_passed_no_training", "frozen_v2_sha256": frozen["sha256"],
                "p2_ready_sha256": sha256(args.train_ready),
                "p3_prepared_sha256": sha256(args.prepared_dir / "prepared.json"),
                "pseudo_selection_sha256": sha256(args.selection / "selection.json"),
                "tune_ids_sha256": tune.identity["tune_id_sha256"], "tune_pairs": len(tune.pairs),
                "full_competing_train_target_pool": ce.FULL_POOL_ROWS, "test_full_target_pool": EXPECTED_TARGETS,
                "train_and_test": disjoint, "tune_baseline_entities_sha256": sha256(baseline),
                "no_test_or_holdout_truth_read": True,
                "fixed_decision": decision,
                "policy": {"positive_min": POSITIVE_MIN, "negative_max": NEGATIVE_MAX,
                           "test_example_weight": PSEUDO_WEIGHT, "test_presentation_fraction": .25}}
    return identity, {"train_reference": train_ref, "train_pool": train_pool,
                      "test_reference": test_ref, "test_pool": test_pool,
                      "pseudo_pairs": pseudo, "prepared": prepared, "tune": tune, "baseline": comparison}


def fit(args, scored: Scored, frozen: dict) -> dict:
    """Explicit BF16 5–10k-step fine-tune from a frozen v2 checkpoint, then TUNE only."""
    if not args.launch_frozen_fit or not 5_000 <= args.steps <= 10_000 or args.device != "cuda":
        raise ValueError("P5 GPU fitting requires --launch-frozen-fit, 5–10k steps, a frozen v2 and free BF16 CUDA")
    import torch
    from . import v2_ce_train as ce
    from .hf_experiment import record_texts
    from .uncertainty import paired_bootstrap

    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.mem_get_info()[0] < 24 * 1024**3):
        raise ValueError("P5 adaptation needs free BF16 CUDA after the frozen-v2/TUNE launch gate")
    preflight, data = fit_preflight(args, scored, frozen)
    out = _fresh_output(args.output)
    config = {**preflight, "state": "configured", "source_sha256": sha256(Path(__file__)),
              "seed": args.seed, "steps": args.steps, "effective_batch": 256, "microbatch": 32,
              "learning_rate": 2e-5, "amp": "bf16", "optimizer": "AdamW", "weight_decay": .01,
              "model_id": "BAAI/bge-reranker-v2-m3", "selected_on": "TUNE_only"}
    _write_json(out / "training_config.json", config)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, tokenizer = ce._load_model(Path(frozen["checkpoint"]), "cuda")
    if _json(Path(frozen["checkpoint"]) / "training_config.json").get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if (not 500_000_000 <= total_parameters <= 650_000_000
            or total_parameters / 1e9 + frozen["inventory_params_b"] - .568 > 8):
        raise ValueError("Loaded CE architecture or final ensemble parameter count exceeds the licensed cap")
    positive = np.load(args.prepared_dir / "positive_pairs.npy", mmap_mode="r")
    negative = np.load(args.prepared_dir / "negative_pairs.npy", mmap_mode="r")
    if len(positive) != data["prepared"]["positive_pool"] or len(negative) != data["prepared"]["hard_negative_pool"]:
        raise ValueError("P3 TRAIN true/hard-negative pools changed before P5 adaptation")
    cycle = ce.PositiveCycle(len(positive), args.seed)
    channels = ce.hard_negative_indices(negative)
    rng = np.random.default_rng(args.seed + 1)
    left_text = {"train": record_texts(data["train_reference"]),
                 "test": record_texts(data["test_reference"])}
    right_text = {"train": record_texts(data["train_pool"]),
                  "test": record_texts(data["test_pool"])}
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=.01)
    losses = []
    for step in range(1, args.steps + 1):
        model.train()
        plan = mixed_batch_plan(positive, negative, data["pseudo_pairs"], cycle, channels, rng)
        optimizer.zero_grad(set_to_none=True)
        normalizer = sum(item[4] for item in plan)
        step_loss = 0.
        for first in range(0, len(plan), 32):
            chunk = plan[first:first + 32]
            inputs = tokenizer([left_text[population][left] for population, left, _, _, _ in chunk],
                               [right_text[population][right] for population, _, right, _, _ in chunk],
                               padding=True, truncation=True, max_length=256, pad_to_multiple_of=8,
                               return_tensors="pt").to("cuda")
            labels = torch.tensor([row[3] for row in chunk], device="cuda", dtype=torch.float32)
            weights = torch.tensor([row[4] for row in chunk], device="cuda", dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**inputs).logits
            if logits.shape != (len(chunk), 1):
                raise ValueError("P5 CE adapter must emit exactly one match logit per pair")
            loss = weighted_binary_ce(logits[:, 0].float(), labels, weights,
                                      [row[0] for row in chunk]) * (float(weights.sum()) / normalizer)
            loss.backward()
            step_loss += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 1 or step % 500 == 0 or step == args.steps:
            event = {"step": step, "weighted_bce": step_loss, "actual_ce_parameters": total_parameters}
            losses.append(event)
            print(json.dumps({"v2_test_adaptation": event}), flush=True)
    checkpoint_tmp, checkpoint = out / "ce_checkpoint.incomplete", out / "ce_checkpoint"
    checkpoint_tmp.mkdir()
    model.save_pretrained(checkpoint_tmp, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_tmp)
    _write_json(checkpoint_tmp / "training_config.json", config)
    checkpoint_files = {file.name: sha256(file) for file in checkpoint_tmp.iterdir() if file.is_file()}
    _write_json(checkpoint_tmp / "complete.json", {"state": "complete", "model_id": config["model_id"],
                                                   "base_v2_sha256": frozen["sha256"],
                                                   "files_sha256": checkpoint_files, "steps": args.steps})
    checkpoint_tmp.replace(checkpoint)
    tune = data["tune"]
    score_args = argparse.Namespace(tree_gate=preflight["fixed_decision"]["tree_gate"],
                                    max_length=256, inference_batch=128, device="cuda", amp="bf16",
                                    near_tie=.00025)
    cross, timing = ce._score_ce(model, tokenizer, tune, score_args)
    score, frame, selected_probabilities = fixed_tune_decision(preflight["fixed_decision"], tune, cross)
    baseline = data["baseline"]
    comparison = paired_bootstrap(frame.macro_f05, baseline.macro_f05,
                                  n_bootstrap=10_000, seed=args.seed)
    np.save(out / "tune_cross_probabilities.npy", cross)
    np.save(out / "tune_selected_probabilities.npy", selected_probabilities)
    frame.to_csv(out / "tune_entities.tsv", sep="\t", index=False)
    result = {"state": "complete", "scope": "full_pool_fixed_TUNE_12000_US_India_only",
              "selection_sha256": sha256(args.selection / "selection.json"),
              "source_checkpoint_complete_sha256": frozen["checkpoint_complete_sha256"],
              "checkpoint_complete_sha256": sha256(checkpoint / "complete.json"),
              "fixed_tune_evaluation": score, "fixed_decision": preflight["fixed_decision"],
              "fixed_decision_vs_frozen_v2_paired": comparison,
              "tune_inference": timing, "training_loss_trace": losses,
              "files_sha256": {name: sha256(out / name) for name in
                               ("training_config.json", "tune_cross_probabilities.npy",
                                "tune_selected_probabilities.npy", "tune_entities.tsv")},
              "test_labels_read": False, "french_accuracy_claimed": False,
              "decision": "TUNE comparison only; a later freeze/one-shot HOLDOUT-2 gate is required"}
    _write_json(out / "fit_result.json", result)
    return result


def generated_roundtrip(probe_marker: Path, test_dir: Path, integrity: Path,
                        *, sample_per_country: int = 256, seed: int = SAMPLE_SEED,
                        nreference: int = EXPECTED_REFERENCES) -> dict:
    """Format/digit safety of an already generated ByT5 test probe; no accuracy proxy."""
    inputs = _test_inputs(test_dir, integrity)
    report = _json(probe_marker)
    if (report.get("state") != "complete" or report.get("labels_read") is not False
            or report.get("positive_candidate_injection") is not False
            or report.get("test_input_sha256") != {name: item["sha256"] for name, item in inputs.items()}):
        raise ValueError("Generated canonicalization probes must be complete and test-label-free")
    checkpoint = Path(report["denoiser_checkpoint_complete_path"])
    _verified_file(checkpoint, report["denoiser_checkpoint_complete_sha256"])
    if _json(checkpoint).get("state") != "complete":
        raise ValueError("A complete, hashed TRAIN-only denoiser checkpoint is required")
    probes = _marker_file(probe_marker, report["files"]["probes"])
    sample, _, _ = _raw_sample(Path(inputs["source1"]["path"]), nreference, sample_per_country, seed)
    originals = {row[1]: row for row in sample}
    counts = {country: Counter() for country in COUNTRIES}
    seen = set()
    frame = pd.read_parquet(probes)
    required = {"entity_id", "generated", "input_truncated", "output_truncated"}
    if len(frame) != report["files"]["probes"]["rows"] or not required <= set(frame):
        raise ValueError("Canonicalization probe records changed after their completion marker")
    for row in frame.itertuples(index=False):
        if row.entity_id not in originals or row.entity_id in seen:
            raise ValueError("Generated probe has a duplicate or non-balanced-test Source 1 ID")
        seen.add(row.entity_id)
        reference = originals[row.entity_id]
        original = f"{reference[2]} | {reference[3]}"
        canonical, reason = safe_canonical_text(original, row.generated,
                                                 input_truncated=bool(row.input_truncated),
                                                 output_truncated=bool(row.output_truncated))
        country = reference[4]
        counts[country]["references"] += 1
        counts[country]["accepted"] += reason == "accepted"
        counts[country][f"reason/{reason}"] += 1
        counts[country]["digit_preserving"] += _DIGITS.findall(canonical.split(" | ", 1)[1]) == _DIGITS.findall(reference[3])
        counts[country]["name_or_address_changed"] += canonical != original
        counts[country]["format_roundtrip"] += canonical.count(" | ") == 1 and " | ".join(canonical.split(" | ")) == canonical
        counts[country]["normalized_idempotent"] += normalize_text(normalize_text(canonical)) == normalize_text(canonical)
    if seen != set(originals):
        raise ValueError("Generated probes omitted a balanced Source 1, including a no-candidate singleton")
    return {"scope": "generated canonicalization on balanced raw test S1s, no labels",
            "checkpoint_manifest_sha256": sha256(checkpoint), "probe_manifest_sha256": sha256(probe_marker),
            "by_country": {country: dict(counts[country]) for country in COUNTRIES},
            "limitation": "Format/digit/normalization round trips are not denoising accuracy or French match precision."}


def safe_canonical_text(original: str, generated: str, *, input_truncated: bool,
                        output_truncated: bool) -> tuple[str, str]:
    """Fail closed on a generated delimiter, empty name, or changed door/unit digits."""
    if input_truncated:
        return original, "input_truncated"
    if output_truncated:
        return original, "output_truncated"
    if original.count(" | ") != 1 or not isinstance(generated, str) or generated.count(" | ") != 1:
        return original, "invalid_delimiter"
    name, address = generated.split(" | ", 1)
    if not normalize_text(name):
        return original, "empty_name"
    if _DIGITS.findall(address) != _DIGITS.findall(original.split(" | ", 1)[1]):
        return original, "address_digits_changed"
    return generated, "accepted"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    def raw_inputs(command):
        command.add_argument("--test-dir", type=Path, required=True, help="Original supplied dataset/test directory")
        command.add_argument("--integrity", type=Path, required=True, help="Original docs/source_integrity.json")

    def sample(command):
        command.add_argument("--sample-per-country", type=int, default=256)
        command.add_argument("--seed", type=int, default=SAMPLE_SEED)

    v1 = commands.add_parser("screen-v1", help="Read-only validated v1 matching TSV; accepted-count/text diagnostics")
    raw_inputs(v1)
    sample(v1)
    v1.add_argument("--release-manifest", type=Path, required=True)
    v1.add_argument("--output", type=Path, required=True)

    scored_v1 = commands.add_parser("diagnose-v1-scores", help="Balanced v1 raw blend score shift, no pseudo-labels")
    raw_inputs(scored_v1)
    sample(scored_v1)
    scored_v1.add_argument("--release-manifest", type=Path, required=True)
    scored_v1.add_argument("--scored-dir", type=Path, required=True)
    scored_v1.add_argument("--output", type=Path, required=True)

    generated = commands.add_parser("diagnose-generated", help="Balanced ByT5 round-trip safety from an existing probe")
    raw_inputs(generated)
    sample(generated)
    generated.add_argument("--probe-marker", type=Path, required=True)
    generated.add_argument("--output", type=Path, required=True)

    def frozen_inputs(command):
        raw_inputs(command)
        command.add_argument("--frozen-v2", type=Path, required=True,
                             help="Frozen TUNE-selected v2 manifest, never a v1/medium-eval checkpoint")
        command.add_argument("--allowlist", type=Path, required=True, help="Checked MIT/Apache parameter inventory")
        command.add_argument("--scored-marker", type=Path, required=True,
                             help="Complete calibrated full-test score bundle, bound to frozen v2")
        command.add_argument("--output", type=Path, required=True)

    choose = commands.add_parser("select", help="P5.1 conflict-filtered candidate pseudo pairs, label-free")
    frozen_inputs(choose)
    inspect = commands.add_parser("diagnose-v2-scores", help="P5.2 balanced calibrated-score/text diagnostics")
    frozen_inputs(inspect)
    sample(inspect)

    training = commands.add_parser("fit", help="Preflight or explicit 5–10k-step BF16 P5 adaptation + TUNE evaluation")
    frozen_inputs(training)
    training.add_argument("--selection", type=Path, required=True)
    training.add_argument("--train-ready", type=Path, required=True)
    training.add_argument("--prepared-dir", type=Path, required=True)
    training.add_argument("--sealed-reservation", type=Path, required=True)
    training.add_argument("--tune-data", type=Path, required=True)
    training.add_argument("--tune-pairs", type=Path, required=True)
    training.add_argument("--tune-scores", type=Path, required=True)
    training.add_argument("--tune-score-pairs", type=Path, required=True)
    training.add_argument("--tune-tree-checkpoint", type=Path, required=True)
    training.add_argument("--launch-frozen-fit", action="store_true")
    training.add_argument("--steps", type=int, default=5_000)
    training.add_argument("--seed", type=int, default=SAMPLE_SEED)
    training.set_defaults(smoke=False, device="cuda", tune_base_dir=None)

    args = parser.parse_args(argv)
    if args.command == "screen-v1":
        report = screen_v1(args.test_dir, args.integrity, args.release_manifest, args.output,
                           sample_per_country=args.sample_per_country, seed=args.seed)
    elif args.command == "diagnose-v1-scores":
        inputs = _test_inputs(args.test_dir, args.integrity)
        scored = checked_v1_scores(args.scored_dir, inputs, args.release_manifest,
                                   nreference=EXPECTED_REFERENCES, full_targets=EXPECTED_TARGETS)
        report = diagnose_scores(scored, args.test_dir, args.integrity, args.output,
                                 sample_per_country=args.sample_per_country, seed=args.seed)
    elif args.command == "diagnose-generated":
        report = generated_roundtrip(args.probe_marker, args.test_dir, args.integrity,
                                     sample_per_country=args.sample_per_country, seed=args.seed)
        _write_json(_fresh_output(args.output) / "generated_roundtrip.json", report)
    else:
        frozen = checked_frozen(args.frozen_v2, allowlist=args.allowlist)
        inputs = _test_inputs(args.test_dir, args.integrity)
        scored = checked_scored_v2(args.scored_marker, frozen, inputs,
                                   nreference=EXPECTED_REFERENCES, full_targets=EXPECTED_TARGETS)
        if args.command == "select":
            report = select_pseudo(scored, args.test_dir, args.integrity, frozen, args.output)
        elif args.command == "diagnose-v2-scores":
            report = diagnose_scores(scored, args.test_dir, args.integrity, args.output,
                                     sample_per_country=args.sample_per_country, seed=args.seed)
        elif args.launch_frozen_fit:
            report = fit(args, scored, frozen)
        else:
            report, _ = fit_preflight(args, scored, frozen)
    print(json.dumps({"command": args.command, "scope": report.get("scope", report.get("state")),
                      "output": str(args.output.resolve()), "counts": report.get("counts"),
                      "references": report.get("full_references", report.get("full_reference_count"))},
                     sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
