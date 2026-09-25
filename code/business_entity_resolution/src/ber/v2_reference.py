"""Build the v2 label-free Source 1 population in original raw-row order.

The main population contains every future TRAIN/TUNE-eligible S1 and every
HOLDOUT-2 S1. The original medium TUNE S1s are a *separate* comparison file.
Only ID/split columns are projected from the old medium reference, and no
ground-truth file is opened. Complete markers are written after durable copies
have been checked; downstream jobs must require ``reference_manifest.json``.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .full_embeddings import atomic_json, file_digest


RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
POPULATION_COLUMNS = RECORD_COLUMNS + ["population", "raw_source1_row"]
MEDIUM_TUNE_COUNT = 12_000
MEDIUM_TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"


def id_sha256(ids: list[str] | set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def checked_ids(path: Path, count: int, file_sha: str, sorted_sha: str) -> set[str]:
    if file_digest(path) != file_sha:
        raise ValueError(f"Reservation ID file changed: {path}")
    data = path.read_bytes()
    ids = data.decode("utf-8").splitlines()
    if (len(ids) != count or len(set(ids)) != count or ids != sorted(ids)
            or data != ("\n".join(ids) + "\n").encode("utf-8")
            or any(not value.startswith("S1-") for value in ids)
            or id_sha256(ids) != sorted_sha):
        raise ValueError(f"Invalid, incomplete or reordered reservation ID file: {path}")
    return set(ids)


def input_identity(raw: Path, reservation_dir: Path, medium: Path,
                   tune_count: int, tune_sha: str) -> tuple[dict, set[str], set[str], list[str]]:
    reserved_manifest = reservation_dir / "reservation.json"
    reservation = json.loads(reserved_manifest.read_text(encoding="utf-8"))
    raw_info = reservation["inputs"]["raw_source1"]
    raw_stat = raw.stat()
    if raw_stat.st_size != raw_info["bytes"] or file_digest(raw) != raw_info["sha256"]:
        raise ValueError("Raw Source 1 differs from the sealed reservation input")
    old_medium = reservation["inputs"]["medium_all_splits"]
    if file_digest(medium) != old_medium["file_sha256"]:
        raise ValueError("Original medium reference differs from the reservation input")
    future = reservation["future_raw_train_tune"]
    holdout = reservation["holdout2"]
    blocked = checked_ids(reservation_dir / future["excluded_id_file"],
                          future["excluded_count"], future["excluded_file_sha256"],
                          future["excluded_sorted_id_sha256"])
    sealed = checked_ids(reservation_dir / holdout["id_file"], holdout["count"],
                         holdout["id_file_sha256"], holdout["sorted_id_sha256"])
    if not sealed <= blocked or len(blocked - sealed) != reservation["prior_exclusions"]["count"]:
        raise ValueError("The sealed holdout is not fully excluded from future fitting")
    # Project only ID and split; medium eval records and truth are never opened.
    frame = pd.read_parquet(medium, columns=["entity_id", "split"])
    ids = frame.entity_id.tolist()
    if (len(ids) != old_medium["ids"] or len(set(ids)) != len(ids)
            or id_sha256(ids) != old_medium["sorted_id_sha256"]):
        raise ValueError("Medium split IDs differ from the sealed reservation")
    tune = frame.loc[frame.split.eq("tune"), "entity_id"].tolist()
    if len(tune) != tune_count or id_sha256(tune) != tune_sha:
        raise ValueError("Original medium TUNE IDs differ from the controlled comparison")
    if not set(tune) <= blocked or set(tune) & sealed:
        raise ValueError("Original medium TUNE must be excluded from future fitting and HOLDOUT-2")
    identity = {
        "schema": 1,
        "raw_source1": {"path": str(raw.resolve()), "sha256": raw_info["sha256"],
                        "bytes": raw_info["bytes"], "rows": raw_info["rows"],
                        "mtime_ns": raw_stat.st_mtime_ns},
        "reservation": {"path": str(reserved_manifest.resolve()),
                        "sha256": file_digest(reserved_manifest),
                        "excluded_id_file_sha256": future["excluded_file_sha256"],
                        "holdout_id_file_sha256": holdout["id_file_sha256"]},
        "medium_reference": {"path": str(medium.resolve()), "sha256": old_medium["file_sha256"],
                             "tune_count": tune_count, "tune_sorted_id_sha256": tune_sha},
        "selection": "eligible_after_all_134k_exclusions + sealed_holdout2; medium_tune_separate",
        "row_order": "raw_train_source1_file_order",
        "labels_read": False,
        "positive_candidate_injection": False,
    }
    return identity, blocked, sealed, tune


def _copy_verified(src: Path, target: Path, digest: str) -> None:
    if src.resolve() == target.resolve():
        return
    if target.exists() and target.stat().st_size == src.stat().st_size and file_digest(target) == digest:
        return
    temporary = target.with_name(target.name + ".tmp")
    if temporary.exists():
        raise ValueError(f"Incomplete durable copy needs inspection: {temporary}")
    with src.open("rb") as source, temporary.open("xb") as dest:
        shutil.copyfileobj(source, dest, length=4 * 1024 * 1024)
        dest.flush()
        os.fsync(dest.fileno())
    if file_digest(temporary) != digest:
        raise ValueError(f"Durable copy failed SHA256 verification: {target}")
    temporary.replace(target)


def _parquet_writer(path: Path, table: pa.Table, writer: pq.ParquetWriter | None) -> pq.ParquetWriter:
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table, row_group_size=max(1, len(table)))
    return writer


def build(raw: Path, reservation_dir: Path, medium: Path, output: Path, durable: Path,
          *, chunk_size: int = 50_000, tune_count: int = MEDIUM_TUNE_COUNT,
          tune_sha: str = MEDIUM_TUNE_ID_SHA256) -> dict:
    if not 0 < chunk_size <= 50_000:
        raise ValueError("Chunk size must be between 1 and 50,000")
    output.mkdir(parents=True, exist_ok=True)
    durable.mkdir(parents=True, exist_ok=True)
    with (output / "reference.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity, blocked, sealed, medium_tune = input_identity(raw, reservation_dir, medium,
                                                                tune_count, tune_sha)
        manifest_path = output / "reference_manifest.json"
        if manifest_path.exists():
            report = json.loads(manifest_path.read_text(encoding="utf-8"))
            if report["identity"] != identity:
                raise ValueError("Existing reference population has a different input or exclusion identity")
            for name, digest in report["parquet_sha256"].items():
                if file_digest(output / name) != digest or file_digest(durable / name) != digest:
                    raise ValueError(f"Completed reference parquet differs from its recorded digest: {name}")
            return report
        if shutil.disk_usage(output).free < 4_500_000_000:
            raise ValueError("Less than 4.5 GB of scratch remains before population construction")
        path = output / "reference.parquet"
        temporary = output / "reference.parquet.tmp"
        if path.exists() or temporary.exists():
            raise ValueError("Uncommitted reference output exists; inspect it before rebuilding")
        writer = None
        seen_ids: set[str] = set()
        blocked_seen: set[str] = set()
        population_ids: dict[str, list[str]] = {"train_tune_eligible": [], "holdout2": []}
        country_counts: dict[str, Counter[str]] = {name: Counter() for name in population_ids}
        tune_records: dict[str, dict] = {}
        tune_set = set(medium_tune)
        raw_rows = 0
        order_digest = hashlib.sha256()
        try:
            for frame in pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False,
                                     usecols=RECORD_COLUMNS, chunksize=chunk_size):
                frame = frame.reset_index(drop=True)
                ids = frame.entity_id.tolist()
                if (len(set(ids)) != len(ids) or any(not value.startswith("S1-") for value in ids)
                        or seen_ids.intersection(ids) or not frame.country.isin(["India", "US"]).all()):
                    raise ValueError("Duplicate/malformed Source 1 ID or unexpected country")
                seen_ids.update(ids)
                blocked_seen.update(set(ids) & blocked)
                mask = ~frame.entity_id.isin(blocked) | frame.entity_id.isin(sealed)
                selected = frame.loc[mask, RECORD_COLUMNS].copy()
                raw_offsets = raw_rows + np.flatnonzero(mask.to_numpy())
                selected["population"] = np.where(selected.entity_id.isin(sealed),
                                                   "holdout2", "train_tune_eligible")
                selected["raw_source1_row"] = raw_offsets.astype(np.int32)
                for group, group_frame in selected.groupby("population", sort=False):
                    population_ids[group].extend(group_frame.entity_id.tolist())
                    country_counts[group].update(group_frame.country.value_counts().to_dict())
                for entity_id in selected.entity_id:
                    order_digest.update(entity_id.encode("utf-8") + b"\n")
                matches = frame.loc[frame.entity_id.isin(tune_set), RECORD_COLUMNS]
                for idx, record in matches.iterrows():
                    tune_records[record.entity_id] = {**record.to_dict(), "population": "medium_tune",
                                                     "raw_source1_row": int(raw_rows + idx)}
                if len(selected):
                    writer = _parquet_writer(temporary, pa.Table.from_pandas(
                        selected[POPULATION_COLUMNS], preserve_index=False), writer)
                raw_rows += len(frame)
        finally:
            if writer is not None:
                writer.close()
        expected_rows = identity["raw_source1"]["rows"]
        reservation = json.loads((reservation_dir / "reservation.json").read_text(encoding="utf-8"))
        expected_eligible = reservation["future_raw_train_tune"]["eligible_count"]
        if (raw_rows != expected_rows or len(seen_ids) != expected_rows or blocked_seen != blocked
                or len(population_ids["train_tune_eligible"]) != expected_eligible
                or set(population_ids["holdout2"]) != sealed or len(tune_records) != tune_count):
            raise ValueError("Population coverage, exclusions, or original medium TUNE records disagree")
        if raw.stat().st_mtime_ns != identity["raw_source1"]["mtime_ns"]:
            raise ValueError("Raw source changed during population construction")
        if pq.ParquetFile(temporary).metadata.num_rows != expected_eligible + len(sealed):
            raise ValueError("Reference parquet row count changed during construction")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary.replace(path)
        tune_path = output / "medium_tune_reference.parquet"
        if tune_path.exists():
            raise ValueError("An uncommitted medium TUNE artifact already exists")
        pd.DataFrame([tune_records[entity_id] for entity_id in medium_tune],
                     columns=POPULATION_COLUMNS).to_parquet(tune_path, index=False)
        files = {name: file_digest(output / name) for name in
                 ("reference.parquet", "medium_tune_reference.parquet")}
        for name, digest in files.items():
            _copy_verified(output / name, durable / name, digest)
        report = {
            "identity": identity,
            "parquet_sha256": files,
            "population": {"rows": expected_eligible + len(sealed),
                           "row_order_id_sha256_with_terminal_lf": order_digest.hexdigest(),
                           "groups": {group: {"count": len(ids), "sorted_id_sha256": id_sha256(ids),
                                              "country_counts": dict(sorted(country_counts[group].items()))}
                                      for group, ids in population_ids.items()}},
            "medium_tune": {"rows": len(medium_tune), "sorted_id_sha256": id_sha256(medium_tune),
                            "row_order": "original_medium_reference_tune_order",
                            "comparisons_only": True},
            "labels_read": False,
            "holdout_truth_read": False,
            "medium_eval_truth_read": False,
            "complete": True,
        }
        atomic_json(durable / "reference_manifest.json", report)
        atomic_json(manifest_path, report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--reservation-dir", type=Path, required=True)
    parser.add_argument("--medium-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durable-output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    args = parser.parse_args()
    report = build(args.raw, args.reservation_dir, args.medium_reference,
                   args.output, args.durable_output, chunk_size=args.chunk_size)
    print(json.dumps({"output": str(args.output), "durable": str(args.durable_output),
                      "population": report["population"], "medium_tune": report["medium_tune"],
                      "labels_read": report["labels_read"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
