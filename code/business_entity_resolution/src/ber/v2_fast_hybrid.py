"""Build an LF-only, full-coverage leaderboard diagnostic from two fixed TSVs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from itertools import zip_longest
from pathlib import Path


def run(source: Path, assignment: Path, decision: Path, output: Path) -> dict:
    if output.exists() or output.with_suffix(".tmp").exists():
        raise FileExistsError(output)
    counts = {"France": 0, "US": 0, "India": 0}
    changed = {key: 0 for key in counts}
    temporary = output.with_suffix(".tmp")
    with source.open(encoding="utf-8", newline="") as raw, assignment.open(encoding="utf-8", newline="") as a, decision.open(encoding="utf-8", newline="") as d, temporary.open("x", encoding="utf-8", newline="\n") as dest:
        s1 = csv.reader(raw, delimiter="\t")
        if next(s1) != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError("Unexpected source header")
        header = "source1_entity_id\tmatched_entity_ids\n"
        if next(a) != header or next(d) != header:
            raise ValueError("The two inputs must be LF-only matching TSVs")
        dest.write(header)
        for record, left, right in zip_longest(s1, a, d):
            if record is None or left is None or right is None or len(record) != 4:
                raise ValueError("Missing or extra S1 row")
            entity, country = record[0], record[3]
            if country not in counts or "\r" in left or "\r" in right or not left.endswith("\n") or not right.endswith("\n"):
                raise ValueError("Unknown country or non-LF input")
            if left.split("\t", 1)[0] != entity or right.split("\t", 1)[0] != entity:
                raise ValueError("Reference IDs or order differ")
            counts[country] += 1
            changed[country] += int(left != right)
            dest.write(left if country == "France" else right)
        dest.flush()
        os.fsync(dest.fileno())
    if sum(counts.values()) != 1_732_544:
        raise ValueError("Incomplete official test coverage")
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    report = {"matching_results": str(output.resolve()), "sha256": digest,
              "bytes": output.stat().st_size, "source1_rows": sum(counts.values()),
              "country_rows": counts, "variant_disagreements": changed,
              "France": "assignment_only", "India_US": "P1_count_rule_and_assignment",
              "labels_used": False, "line_endings": "LF"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "assignment", "decision", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    print(json.dumps(run(**vars(parser.parse_args()))))


if __name__ == "__main__":
    main()
