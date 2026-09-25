"""Reserve unused original TRAIN Source 1 IDs without opening any truth labels."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from pathlib import Path
import time

import pandas as pd

from .cpu_decision import file_sha256
from .experiment import json_write


SEED = 20260928
QUICK_SHA256 = "8980d7ae3c65c38da6c9b9fddb7dfe23eec0c1659099d707af86a9f959f9464e"
MEDIUM_SHA256 = "451f7e94bc76080b0be4de8dfd1dd3d3ff452d66aa899f77eb29520281f80042"
COUNTS = {"India": 4787, "US": 7213}


def id_digest(ids):
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def run(args):
    started = time.monotonic()
    raw, quick, medium, output = map(Path, (args.raw, args.quick, args.medium, args.output))
    quick_ids = pd.read_parquet(quick, columns=["entity_id"]).entity_id.tolist()
    medium_frame = pd.read_parquet(medium, columns=["entity_id", "split", "country"])
    medium_ids = medium_frame.entity_id.tolist()
    if (len(quick_ids) != 12000 or id_digest(quick_ids) != QUICK_SHA256
            or len(medium_ids) != 60000 or id_digest(medium_ids) != MEDIUM_SHA256
            or medium_frame.split.value_counts().to_dict() != {"train": 35999, "tune": 12000, "eval": 12001}
            or len(set(quick_ids) | set(medium_ids)) != 72000):
        raise ValueError("The quick and medium exclusion populations are incomplete or overlapping")
    excluded = set(quick_ids) | set(medium_ids)
    heaps = {country: [] for country in COUNTS}
    eligible = {country: 0 for country in COUNTS}
    total = 0
    for chunk in pd.read_csv(raw, sep="\t", dtype=str, keep_default_na=False,
                             usecols=["entity_id", "country"], chunksize=100000):
        total += len(chunk)
        for entity_id, country in zip(chunk.entity_id, chunk.country):
            if country not in heaps:
                raise ValueError(f"Unexpected raw Source 1 country: {country!r}")
            if entity_id in excluded:
                continue
            eligible[country] += 1
            # Stable order-independent hash sample: smallest 64-bit values per country.
            rank = int.from_bytes(hashlib.blake2b(f"{SEED}\0{entity_id}".encode(), digest_size=8).digest(), "big")
            heap = heaps[country]
            if len(heap) < COUNTS[country]:
                heapq.heappush(heap, (-rank, entity_id))
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, (-rank, entity_id))
    if total != 2206821 or any(len(heaps[key]) != size for key, size in COUNTS.items()):
        raise ValueError("Raw Source 1 population/counts differ from the completed training release")
    selected = sorted(entity_id for heap in heaps.values() for _, entity_id in heap)
    if len(selected) != 12000 or len(set(selected)) != len(selected) or set(selected) & excluded:
        raise ValueError("Reservation is not 12,000 unique, disjoint businesses")
    id_file = output / "reserved_ids.txt"
    if id_file.exists() and id_file.read_text().splitlines() != selected:
        raise ValueError("Existing holdout reservation differs; choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    id_file.write_text("\n".join(selected) + "\n", encoding="utf-8")
    report = {"schema": 1, "seed": SEED, "method": "smallest blake2b-64 rank of seed\\0entity_id within country",
              "source": "original train_source1.tsv, entity_id and country only",
              "source_rows": total, "source_sha256": file_sha256(raw),
              "excluded_quick": len(quick_ids), "excluded_quick_id_sha256": QUICK_SHA256,
              "excluded_medium_all_splits": len(medium_ids), "excluded_medium_id_sha256": MEDIUM_SHA256,
              "exclusion_reference_sha256": {"quick": file_sha256(quick), "medium": file_sha256(medium)},
              "eligible_after_exclusion": eligible, "reserved_country_counts": COUNTS,
              "reserved_ids": len(selected), "reserved_id_sha256": id_digest(selected),
              "id_file_sha256": file_sha256(id_file), "labels_read": False,
              "v1_holdout_truth_or_errors_read": False,
              "usage": "Keep sealed until next procedure and thresholds are frozen; evaluate against all training targets.",
              "runtime_seconds": time.monotonic() - started}
    json_write(output / "reservation.json", report)
    print(json.dumps(report, allow_nan=False), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", required=True, help="Original supplied train_source1.tsv")
    parser.add_argument("--quick", required=True, help="Saved quick reference.parquet")
    parser.add_argument("--medium", required=True, help="Saved 60,000-S1 medium reference.parquet")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
