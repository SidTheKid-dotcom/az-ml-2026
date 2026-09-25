"""Strict format gate around the unmodified official validator; computes no metric."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from itertools import zip_longest
from pathlib import Path


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _output_row(line, label):
    fields = line.rstrip("\r\n").split("\t")
    if len(fields) != 2 or not fields[0] or fields[0] != fields[0].strip():
        raise ValueError(f"{label}: malformed or empty source1 row")
    ids = fields[1].split(",") if fields[1] else []
    if any(not value.startswith(("S2-", "S3-")) or value != value.strip() for value in ids):
        raise ValueError(f"{label}: empty, malformed, or wrong-source target ID")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{label}: duplicate target ID")
    return fields[0], set(ids)


def stream_check(matching, candidate, source1, expected_rows, counts):
    with Path(source1).open(encoding="utf-8") as source, Path(matching).open(encoding="utf-8") as matches, Path(candidate).open(encoding="utf-8") as candidates:
        if source.readline().split("\t", 1)[0] != "entity_id":
            raise ValueError("Original test_source1.tsv has an unexpected header")
        for handle, header in ((matches, "source1_entity_id\tmatched_entity_ids"), (candidates, "source1_entity_id\tcandidate_entity_ids")):
            if handle.readline().rstrip("\r\n") != header:
                raise ValueError(f"Expected exact header {header!r}")
        seen = set()
        for number, lines in enumerate(zip_longest(source, matches, candidates), start=2):
            if any(line is None for line in lines):
                raise ValueError(f"Row count mismatch at line {number}")
            entity = lines[0].split("\t", 1)[0].rstrip("\r\n")
            if not entity.startswith("S1-") or entity in seen or entity != entity.strip():
                raise ValueError(f"Invalid/duplicate original Source 1 ID at line {number}")
            seen.add(entity)
            mid, matched = _output_row(lines[1], f"matching line {number}")
            cid, considered = _output_row(lines[2], f"candidate line {number}")
            if mid != entity or cid != entity:
                raise ValueError(f"Missing, unknown, duplicate, or out-of-order Source 1 row at line {number}")
            if not matched <= considered:
                raise ValueError(f"Matches are not a subset of candidates at line {number}")
            counts["rows_checked"] += 1
            counts["matched_links"] += len(matched)
            counts["candidate_links"] += len(considered)
            counts["empty_match_rows"] += int(not matched)
        if counts["rows_checked"] != expected_rows:
            raise ValueError(f"Expected {expected_rows} Source 1 rows, found {counts['rows_checked']}")


def validate_release(matching, candidate, test_dir, report, *, validator="data/raw/utils/validate_submission.py", expected_rows=1_732_544):
    matching, candidate, test_dir, report, validator = map(Path, (matching, candidate, test_dir, report, validator))
    report.parent.mkdir(parents=True, exist_ok=True)
    log_path = report.with_suffix(".validator.log")
    errors, counts = [], dict(rows_checked=0, matched_links=0, candidate_links=0, empty_match_rows=0)
    command = [sys.executable, str(validator), "--matching", str(matching), "--candidate", str(candidate), "--test-dir", str(test_dir), "--check-ids"]
    with log_path.open("w", encoding="utf-8") as log:
        official = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    stdout = log_path.read_text(encoding="utf-8", errors="replace")
    warnings = [line for line in stdout.splitlines() if line.lstrip().startswith("WARNING:")]
    official_pass = official.returncode == 0 and not warnings and any(line.startswith("PASS") for line in stdout.splitlines())
    if not official_pass:
        errors.append("Official validator failed, warned, or did not confirm PASS")
    stream_complete = False
    try:
        stream_check(matching, candidate, test_dir / "test_source1.tsv", expected_rows, counts)
        stream_complete = True
    except (OSError, UnicodeError, ValueError) as error:
        errors.append(str(error))
    hashes = {}
    for path in (matching, candidate, *(test_dir / f"test_source{i}.tsv" for i in (1, 2, 3)), validator):
        try:
            hashes[str(path.resolve())] = file_sha256(path)
        except OSError as error:
            hashes[str(path.resolve())] = None
            errors.append(str(error))
    result = {"passed": not errors, "validation_kind": "format_and_integrity_only", "metric_score_computed": False,
              "expected_source1_rows": expected_rows, "counts": counts, "stream_complete": stream_complete,
              "file_sha256": hashes, "errors": errors,
              "original_validator": {"path": str(validator.resolve()), "sha256": hashes.get(str(validator.resolve())),
                  "exit_code": official.returncode, "warnings": warnings, "passed_without_warnings": official_pass,
                  "stdout_path": str(log_path.resolve()), "command": command}}
    report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matching", "candidate", "test-dir"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--validator", default="data/raw/utils/validate_submission.py")
    parser.add_argument("--report", default="artifacts/release_validation.json")
    parser.add_argument("--expected-rows", type=int, default=1_732_544)
    result = validate_release(**vars(parser.parse_args()))
    print(json.dumps({"passed": result["passed"], "counts": result["counts"], "errors": result["errors"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
