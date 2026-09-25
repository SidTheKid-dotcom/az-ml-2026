"""Seal HOLDOUT-2 using Source 1 IDs/countries; never open match labels.

The combined exclusion file is the input guard for any subsequent raw-source
TRAIN/TUNE assignment. It includes quick, all medium splits, the older sealed
holdout, and HOLDOUT-2. Existing development splits remain separate artifacts.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import heapq
import json
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class ReservationSpec:
    seed: int = 20260929
    raw_rows: int = 2_206_821
    raw_sha256: str = "591af0e1dfeb65cab71ea6ee8cb69df00f92d6ba6fa79e05746c938775d14973"
    raw_bytes: int = 210_069_713
    quick_count: int = 12_000
    quick_sha256: str = "8980d7ae3c65c38da6c9b9fddb7dfe23eec0c1659099d707af86a9f959f9464e"
    medium_count: int = 60_000
    medium_sha256: str = "451f7e94bc76080b0be4de8dfd1dd3d3ff452d66aa899f77eb29520281f80042"
    medium_splits: tuple[tuple[str, int], ...] = (("train", 35_999), ("tune", 12_000), ("eval", 12_001))
    older_count: int = 12_000
    older_sha256: str = "ec057370b8dc8b3dbef24e277a6920f905a08481ca2b35a5245dcdf9d22f5187"
    holdout_count: int = 50_000


DEFAULT_SPEC = ReservationSpec()
OUTPUT_FILES = frozenset({"reserved_ids.txt", "train_tune_excluded_ids.txt", "reservation.json"})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def id_sha256(ids: list[str] | set[str]) -> str:
    """SHA256 of lexically sorted IDs joined by LF, with no terminal LF."""
    return hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()


def _id_file_bytes(sorted_ids: list[str]) -> bytes:
    return ("\n".join(sorted_ids) + "\n").encode("utf-8")


def _validate_ids(ids: list[str], expected_count: int, expected_sha256: str, label: str) -> set[str]:
    if (len(ids) != expected_count or len(set(ids)) != len(ids)
            or any(not isinstance(entity_id, str) or not entity_id.startswith("S1-")
                   or "\n" in entity_id or "\r" in entity_id for entity_id in ids)
            or id_sha256(ids) != expected_sha256):
        raise ValueError(f"{label}: incomplete, duplicate, malformed, or unexpected Source 1 IDs")
    return set(ids)


class _ReverseLex(str):
    """Make a max-heap on (hash rank, ID) using Python's min-heap."""

    def __lt__(self, other: str) -> bool:
        return str.__gt__(self, other)


def _rank(seed: int, entity_id: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(f"{seed}\0{entity_id}".encode("utf-8"), digest_size=16).digest(), "big"
    )


def _compute(raw: Path, quick: Path, medium: Path, older_ids: Path,
             spec: ReservationSpec) -> tuple[dict, list[str], list[str]]:
    raw, quick, medium, older_ids = map(Path, (raw, quick, medium, older_ids))
    raw_sha = file_sha256(raw)
    if raw.stat().st_size != spec.raw_bytes or raw_sha != spec.raw_sha256:
        raise ValueError("Raw train_source1.tsv differs from the verified supplied source")

    # Parquet projection: do not load names, addresses, pool records, or truth.
    quick_ids = pd.read_parquet(quick, columns=["entity_id"])["entity_id"].tolist()
    medium_frame = pd.read_parquet(medium, columns=["entity_id", "split"])
    medium_ids = medium_frame["entity_id"].tolist()
    quick_sha, medium_sha = file_sha256(quick), file_sha256(medium)
    old_bytes = older_ids.read_bytes()
    old_list = old_bytes.decode("utf-8").splitlines()
    if old_bytes != _id_file_bytes(sorted(old_list)):
        raise ValueError("Older sealed holdout ID file is not a sorted, newline-terminated ID list")
    old_sha = hashlib.sha256(old_bytes).hexdigest()
    older_manifest_path = older_ids.parent / "reservation.json"
    older_manifest = json.loads(older_manifest_path.read_text(encoding="utf-8"))
    older_manifest_sha = file_sha256(older_manifest_path)

    quick_set = _validate_ids(quick_ids, spec.quick_count, spec.quick_sha256, "quick")
    medium_set = _validate_ids(medium_ids, spec.medium_count, spec.medium_sha256, "medium")
    old_set = _validate_ids(old_list, spec.older_count, spec.older_sha256, "older sealed holdout")
    splits = dict(Counter(medium_frame["split"].tolist()))
    if splits != dict(spec.medium_splits):
        raise ValueError("Medium exclusion does not include the expected TRAIN/TUNE/EVAL splits")
    excluded = quick_set | medium_set | old_set
    if len(excluded) != spec.quick_count + spec.medium_count + spec.older_count:
        raise ValueError("Quick, medium, and older sealed holdout IDs must be pairwise disjoint")
    if (older_manifest.get("source_sha256") != spec.raw_sha256
            or older_manifest.get("excluded_quick_id_sha256") != spec.quick_sha256
            or older_manifest.get("excluded_medium_id_sha256") != spec.medium_sha256
            or older_manifest.get("exclusion_reference_sha256") != {"quick": quick_sha, "medium": medium_sha}
            or older_manifest.get("reserved_ids") != spec.older_count
            or older_manifest.get("reserved_id_sha256") != spec.older_sha256
            or older_manifest.get("id_file_sha256") != old_sha
            or older_manifest.get("labels_read") is not False):
        raise ValueError("Older holdout IDs do not match their sealed reservation manifest")

    seen_ids: set[str] = set()
    seen_excluded: set[str] = set()
    source_countries: Counter[str] = Counter()
    eligible_countries: Counter[str] = Counter()
    # Keep the largest selected (rank, ID) at the root, replacing it with each
    # smaller eligible item. This does not depend on input row order.
    heap: list[tuple[int, _ReverseLex, str]] = []
    for chunk in pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False,
                             usecols=["entity_id", "country"], chunksize=100_000):
        for entity_id, country in chunk[["entity_id", "country"]].itertuples(index=False, name=None):
            if (not entity_id.startswith("S1-") or entity_id in seen_ids
                    or country not in ("India", "US")):
                raise ValueError("Raw Source 1 has a malformed/duplicate ID or unexpected country")
            seen_ids.add(entity_id)
            source_countries[country] += 1
            if entity_id in excluded:
                seen_excluded.add(entity_id)
                continue
            eligible_countries[country] += 1
            rank = _rank(spec.seed, entity_id)
            item = (-rank, _ReverseLex(entity_id), country)
            if len(heap) < spec.holdout_count:
                heapq.heappush(heap, item)
            elif (rank, entity_id) < (-heap[0][0], str(heap[0][1])):
                heapq.heapreplace(heap, item)
    if len(seen_ids) != spec.raw_rows or seen_excluded != excluded or len(heap) != spec.holdout_count:
        raise ValueError("Raw Source 1 coverage, exclusion membership, or holdout size differs")

    selected = sorted(str(item[1]) for item in heap)
    if len(set(selected)) != spec.holdout_count or set(selected) & excluded:
        raise ValueError("HOLDOUT-2 is not distinct from every development/older holdout ID")
    future_excluded = sorted(excluded | set(selected))
    if len(future_excluded) != len(excluded) + spec.holdout_count:
        raise ValueError("Future TRAIN/TUNE exclusion does not include all holdout IDs")
    reserved_bytes = _id_file_bytes(selected)
    future_bytes = _id_file_bytes(future_excluded)

    report = {
        "schema": 1,
        "name": "HOLDOUT-2",
        "seed": spec.seed,
        "method": f"Globally select the {spec.holdout_count:,} smallest (BLAKE2b-128(seed\\0entity_id), entity_id) ranks among eligible raw Source 1 IDs; row-order invariant; country is used only for accounting.",
        "inputs": {
            "raw_source1": {"path": str(raw.resolve()), "sha256": raw_sha,
                            "bytes": raw.stat().st_size, "rows": len(seen_ids),
                            "columns_read": ["entity_id", "country"],
                            "country_counts": dict(sorted(source_countries.items()))},
            "quick": {"path": str(quick.resolve()), "file_sha256": quick_sha,
                      "ids": len(quick_set), "sorted_id_sha256": id_sha256(quick_set),
                      "columns_read": ["entity_id"]},
            "medium_all_splits": {"path": str(medium.resolve()), "file_sha256": medium_sha,
                                  "ids": len(medium_set), "sorted_id_sha256": id_sha256(medium_set),
                                  "split_counts": dict(sorted(splits.items())),
                                  "columns_read": ["entity_id", "split"]},
            "older_sealed_holdout": {"path": str(older_ids.resolve()), "file_sha256": old_sha,
                                      "manifest_path": str(older_manifest_path.resolve()),
                                      "manifest_sha256": older_manifest_sha,
                                      "ids": len(old_set), "sorted_id_sha256": id_sha256(old_set)},
        },
        "prior_exclusions": {"count": len(excluded), "sorted_id_sha256": id_sha256(excluded),
                             "all_present_in_raw": True, "pairwise_disjoint": True},
        "eligible_after_prior_exclusions": {"count": sum(eligible_countries.values()),
                                            "country_counts": dict(sorted(eligible_countries.items()))},
        "holdout2": {"count": len(selected),
                     "country_counts": dict(sorted(Counter(item[2] for item in heap).items())),
                     "sorted_id_sha256": id_sha256(selected), "id_file": "reserved_ids.txt",
                     "id_file_sha256": hashlib.sha256(reserved_bytes).hexdigest()},
        "future_raw_train_tune": {
            "excluded_id_file": "train_tune_excluded_ids.txt", "excluded_count": len(future_excluded),
            "excluded_sorted_id_sha256": id_sha256(future_excluded),
            "excluded_file_sha256": hashlib.sha256(future_bytes).hexdigest(),
            "eligible_count": spec.raw_rows - len(future_excluded),
            "rule": "Subtract this entire file before creating any new TRAIN or TUNE raw-Source-1 split; the older 12k and HOLDOUT-2 are both included.",
        },
        "verification": {"source_matches_verified_original": True,
                         "exclusion_id_counts_and_digests_match": True,
                         "older_reservation_manifest_matches": True,
                         "all_exclusions_present_in_raw": True,
                         "selected_and_prior_exclusions_disjoint": True,
                         "older_and_new_holdouts_excluded_from_future_train_tune": True,
                         "labels_read": False},
    }
    return report, selected, future_excluded


def reserve(raw: Path, quick: Path, medium: Path, older_ids: Path, output: Path,
            spec: ReservationSpec = DEFAULT_SPEC) -> dict:
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Reservation already exists; refusing to overwrite: {output}")
    report, selected, future_excluded = _compute(raw, quick, medium, older_ids, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir()  # Refuse a concurrent reservation at the same destination.
    contents = {
        "reserved_ids.txt": _id_file_bytes(selected),
        "train_tune_excluded_ids.txt": _id_file_bytes(future_excluded),
        "reservation.json": (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    for name, content in contents.items():
        path = output / name
        with path.open("xb") as stream:
            stream.write(content)
        path.chmod(0o444)
    verify(output, spec=spec)
    return report


def verify(output: Path, *, raw: Path | None = None, quick: Path | None = None,
           medium: Path | None = None, older_ids: Path | None = None,
           spec: ReservationSpec = DEFAULT_SPEC) -> dict:
    output = Path(output)
    if {path.name for path in output.iterdir()} != OUTPUT_FILES:
        raise ValueError("Unexpected or missing HOLDOUT-2 artifact files")
    stored = json.loads((output / "reservation.json").read_text(encoding="utf-8"))
    paths = stored["inputs"]
    report, selected, future = _compute(
        Path(raw or paths["raw_source1"]["path"]), Path(quick or paths["quick"]["path"]),
        Path(medium or paths["medium_all_splits"]["path"]),
        Path(older_ids or paths["older_sealed_holdout"]["path"]), spec,
    )
    # A relocated byte-identical source is valid, but its new workstation/pod
    # path must not rewrite the historical selection snapshot.
    for label, supplied in (("raw_source1", raw), ("quick", quick),
                            ("medium_all_splits", medium), ("older_sealed_holdout", older_ids)):
        if supplied is not None:
            report["inputs"][label]["path"] = paths[label]["path"]
            if label == "older_sealed_holdout":
                report["inputs"][label]["manifest_path"] = paths[label]["manifest_path"]
    if report != stored:
        raise ValueError("HOLDOUT-2 manifest differs from independently recomputed source IDs")
    for name, ids in (("reserved_ids.txt", selected), ("train_tune_excluded_ids.txt", future)):
        path = output / name
        if path.read_bytes() != _id_file_bytes(ids) or path.stat().st_mode & 0o222:
            raise ValueError(f"HOLDOUT-2 ID file differs or is writable: {name}")
    if (output / "reservation.json").stat().st_mode & 0o222:
        raise ValueError("HOLDOUT-2 manifest is writable")
    return stored


def check_train_tune(output: Path, train_ids: Path, tune_ids: Path) -> dict:
    """ID-only gate before using any newly sampled raw-source TRAIN/TUNE split."""
    output = Path(output)
    manifest = json.loads((output / "reservation.json").read_text(encoding="utf-8"))
    rule = manifest["future_raw_train_tune"]
    forbidden_path = output / rule["excluded_id_file"]
    forbidden_bytes = forbidden_path.read_bytes()
    forbidden = forbidden_bytes.decode("utf-8").splitlines()
    if (len(forbidden) != rule["excluded_count"] or len(set(forbidden)) != len(forbidden)
            or forbidden_bytes != _id_file_bytes(sorted(forbidden))
            or id_sha256(forbidden) != rule["excluded_sorted_id_sha256"]
            or hashlib.sha256(forbidden_bytes).hexdigest() != rule["excluded_file_sha256"]):
        raise ValueError("Future TRAIN/TUNE exclusion file is incomplete or modified")
    blocked = set(forbidden)
    populations = {}
    for label, path in (("train", train_ids), ("tune", tune_ids)):
        ids = Path(path).read_text(encoding="utf-8").splitlines()
        if len(ids) != len(set(ids)) or any(not entity_id.startswith("S1-") for entity_id in ids):
            raise ValueError(f"{label} has duplicate or malformed IDs")
        if set(ids) & blocked:
            raise ValueError(f"{label} includes reserved/development Source 1 IDs")
        populations[label] = set(ids)
    if populations["train"] & populations["tune"]:
        raise ValueError("TRAIN and TUNE Source 1 IDs overlap")
    return {"train": len(populations["train"]), "tune": len(populations["tune"]),
            "excluded": len(blocked), "disjoint": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reserve_parser = commands.add_parser("reserve", help="Create an immutable, ID-only reservation")
    for name in ("raw", "quick", "medium", "older-ids"):
        reserve_parser.add_argument(f"--{name}", type=Path, required=True)
    reserve_parser.add_argument("--output", type=Path, required=True)
    verify_parser = commands.add_parser("verify", help="Recompute the reservation from the same inputs")
    verify_parser.add_argument("--output", type=Path, required=True)
    for name in ("raw", "quick", "medium", "older-ids"):
        verify_parser.add_argument(f"--{name}", type=Path)
    check_parser = commands.add_parser("check-train-tune", help="Reject later raw-source split leakage")
    check_parser.add_argument("--output", type=Path, required=True)
    check_parser.add_argument("--train-ids", type=Path, required=True)
    check_parser.add_argument("--tune-ids", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "reserve":
        result = reserve(args.raw, args.quick, args.medium, args.older_ids, args.output)
    elif args.command == "verify":
        result = verify(args.output, raw=args.raw, quick=args.quick,
                        medium=args.medium, older_ids=args.older_ids)
    else:
        result = check_train_tune(args.output, args.train_ids, args.tune_ids)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
