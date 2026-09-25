"""Exploratory assignment-only fallback on *original* v1 thresholded scores.

Never calibrates, refits, changes retrieval/thresholds, or reads official-test labels.
The original v1 matching TSV is the sole source of accepted edges. Every accepted
edge must agree with the aligned float32 score and positional candidate pair.
Conflicts keep highest original score, then lowest S1 position, then pair row.
Output is a separate, non-release-ready TSV, with unchanged bytes for every
uncontested Source 1 row (including empty rows and CRLF line endings).
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .uncertainty import paired_bootstrap
from .validate_release import validate_release


V1_MATCH_SHA256 = "5df9ddbf42137d4e403f0072fe85169aaabc0167a0b431f675f926754371e1ed"
V1_CANDIDATE_SHA256 = "fc0ec7a43888208b25fef4deecea352a98dcce75797ea9288c0f7fb9e6ebca34"
TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
N_REFERENCES = 1_732_544
N_PAIRS = 86_627_200
THRESHOLD = .775
CLAIM_DTYPE = np.dtype([("left", "<i4"), ("right", "<i4"),
                        ("pair_row", "<i8"), ("score", "<f4")])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> dict:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)}


def winners(claims: np.ndarray) -> tuple[np.ndarray, dict]:
    """P1's delta=0 global ordering, only on the original accepted v1 links."""
    keep = np.zeros(len(claims), dtype=bool)
    if not len(claims):
        return keep, {"conflicting_targets": 0, "removed_claims": 0, "tied_winning_scores": 0}
    order = np.lexsort((claims["pair_row"], claims["left"], -claims["score"], claims["right"]))
    right = claims["right"][order]
    first = np.r_[True, right[1:] != right[:-1]]
    keep[order[first]] = True
    starts = np.flatnonzero(first)
    lengths = np.diff(np.r_[starts, len(claims)])
    conflicts = starts[lengths > 1]
    ties = sum(claims["score"][order[pos]] == claims["score"][order[pos + 1]]
               for pos in conflicts)
    return keep, {"conflicting_targets": int(len(conflicts)),
                  "removed_claims": int(len(claims) - len(starts)),
                  "tied_winning_scores": int(ties)}


def parse_row(line: bytes, *, header: bool = False) -> tuple[bytes, list[bytes]]:
    if not line or not line.endswith(b"\n"):
        raise ValueError("Truncated or unterminated v1 matching row")
    fields = line.rstrip(b"\r\n").split(b"\t")
    if len(fields) != 2 or not fields[0]:
        raise ValueError("Malformed original v1 matching row")
    if header:
        if fields != [b"source1_entity_id", b"matched_entity_ids"]:
            raise ValueError("Unexpected original v1 matching header")
        return fields[0], []
    ids = fields[1].split(b",") if fields[1] else []
    if len(ids) != len(set(ids)) or any(not x.startswith((b"S2-", b"S3-")) for x in ids):
        raise ValueError("Malformed or duplicate v1 accepted ID")
    return fields[0], ids


def pair_batches(path: Path, nref: int, npool: int, *, batch_size: int = 200_000):
    """Project positional pairs; never split an S1 across yielded batches."""
    file = pq.ParquetFile(path)
    last_left = last_right = -1
    pending_l = pending_r = np.empty(0, dtype=np.int32)
    offset = seen = 0
    for batch in file.iter_batches(batch_size=batch_size, columns=["left_idx", "right_idx"], use_threads=False):
        left, right = (batch.column(i).to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
                       for i in (0, 1))
        if not len(left):
            continue
        if (left.min() < 0 or left.max() >= nref or right.min() < 0 or right.max() >= npool
                or (int(left[0]), int(right[0])) <= (last_left, last_right)
                or np.any(left[1:] < left[:-1])
                or np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1]))):
            raise ValueError("Candidate pairs are duplicated, unsorted, or out of bounds")
        last_left, last_right = int(left[-1]), int(right[-1])
        seen += len(left)
        if len(pending_l):
            left, right = np.r_[pending_l, left], np.r_[pending_r, right]
        stop = int(np.searchsorted(left, left[-1], side="left"))
        if stop:
            yield offset, left[:stop], right[:stop]
            offset += stop
        pending_l, pending_r = left[stop:], right[stop:]
    if len(pending_l):
        yield offset, pending_l, pending_r
        offset += len(pending_l)
    if seen != file.metadata.num_rows or offset != seen:
        raise ValueError("Positional pair batches lost or duplicated rows")


def checked_inputs(v1: Path, data: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    ready = json.loads((v1 / "ready.json").read_text())
    report = json.loads((v1 / "report.json").read_text())
    complete = json.loads((v1 / "complete.json").read_text())
    original, candidate = v1 / "matching_results.tsv", v1 / "candidate_pairs.tsv"
    pairs = v1 / "scored_pairs.parquet"
    if (ready.get("ready_for_submission") is not True or complete.get("state") != "complete"
            or complete.get("report") != report or report.get("labels_accessed") is not False
            or report.get("benchmark_only") is not False or report.get("threshold") != THRESHOLD
            or report.get("reference_count") != N_REFERENCES or report.get("candidate_pairs") != N_PAIRS
            or ready.get("source1_rows") != N_REFERENCES
            or ready["file_sha256"].get("matching_results.tsv") != V1_MATCH_SHA256
            or ready["file_sha256"].get("candidate_pairs.tsv") != V1_CANDIDATE_SHA256
            or not pairs.samefile(data / "pairs.parquet")
            or pq.ParquetFile(pairs).metadata.num_rows != N_PAIRS):
        raise ValueError("Expected the complete original v1 full-test scores and release")
    identities = {name: fingerprint(path) for name, path in {
        "original_matching": original, "original_candidates": candidate,
        "scored_pairs": pairs, "blended_scores": v1 / "probabilities.npy",
        "reference": data / "reference.parquet", "pool": data / "pool.parquet",
        "ready": v1 / "ready.json", "complete": v1 / "complete.json"}.items()}
    if (identities["original_matching"]["sha256"] != V1_MATCH_SHA256
            or identities["original_candidates"]["sha256"] != V1_CANDIDATE_SHA256):
        raise ValueError("Original v1 TSV hash mismatch")
    for label in ("reference", "pool", "pairs"):
        identity = report["input_identities"][label]
        path = data / f"{label}.parquet"
        if path.stat().st_size != identity["size_bytes"] or path.stat().st_mtime_ns != identity["mtime_ns"]:
            raise ValueError(f"Saved v1 {label} identity changed")
    refs = pd.read_parquet(data / "reference.parquet", columns=["entity_id", "country"])
    pool = pd.read_parquet(data / "pool.parquet", columns=["entity_id"])
    if (len(refs) != N_REFERENCES or refs.entity_id.duplicated().any()
            or len(pool) != report["pool_count"] or pool.entity_id.duplicated().any()):
        raise ValueError("Original v1 reference/compact target population changed")
    scores = np.load(v1 / "probabilities.npy", mmap_mode="r", allow_pickle=False)
    if scores.dtype != np.float32 or scores.shape != (N_PAIRS,):
        raise ValueError("Saved v1 blend scores are incomplete or not float32")
    return identities, refs, pool.entity_id.to_numpy(), scores


def extract_claims(original: Path, pairs: Path, scores: np.ndarray, refs: pd.DataFrame,
                   pool_ids: np.ndarray, dest: Path, *, expected_pairs: int) -> int:
    """Verify ALL threshold decisions against v1, including zero-candidate S1s."""
    ref_ids = refs.entity_id.to_numpy()
    row = pair_count = claim_count = 0
    with original.open("rb") as source, dest.open("xb") as sink:
        parse_row(source.readline(), header=True)

        def check_reference(index: int, right: np.ndarray, pair_offset: int):
            nonlocal row, claim_count
            entity, original_ids = parse_row(source.readline())
            if entity != ref_ids[index].encode():
                raise ValueError(f"Original v1 Source 1 order differs at row {index}")
            chosen = np.flatnonzero(scores[pair_offset:pair_offset + len(right)] >= THRESHOLD)
            accepted_ids = [str(pool_ids[right[j]]).encode() for j in chosen]
            if sorted(accepted_ids) != sorted(original_ids):
                raise ValueError(f"Saved score, candidate, and original v1 accepted links disagree at row {index}")
            if len(chosen):
                claims = np.empty(len(chosen), dtype=CLAIM_DTYPE)
                claims["left"] = index
                claims["right"] = right[chosen]
                claims["pair_row"] = pair_offset + chosen
                claims["score"] = scores[pair_offset + chosen]
                claims.tofile(sink)
                claim_count += len(chosen)
            row += 1

        for offset, left, right in pair_batches(pairs, len(refs), len(pool_ids)):
            raw = np.asarray(scores[offset:offset + len(left)])
            if len(raw) != len(left) or not np.isfinite(raw).all() or np.any((raw < 0) | (raw > 1)):
                raise ValueError("Aligned saved v1 scores are invalid")
            boundaries = np.r_[0, np.flatnonzero(left[1:] != left[:-1]) + 1, len(left)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                index = int(left[start])
                while row < index:
                    check_reference(row, np.empty(0, dtype=np.int32), offset + start)
                if row != index:
                    raise ValueError("Candidate pair order revisited a Source 1")
                check_reference(index, right[start:end], offset + start)
            pair_count = offset + len(left)
        while row < len(refs):
            check_reference(row, np.empty(0, dtype=np.int32), pair_count)
        sink.flush()
        if source.readline() or pair_count != expected_pairs or claim_count * CLAIM_DTYPE.itemsize != dest.stat().st_size:
            raise ValueError("Full original v1 test population or score/pair alignment is incomplete")
    return claim_count


def rewrite(original: Path, destination: Path, refs: pd.DataFrame,
            losers: dict[int, set[bytes]]) -> dict:
    """Retain original rows byte-for-byte unless removing specifically a lost ID."""
    changed = removed = before = after = empty_before = empty_after = 0
    with original.open("rb") as source, destination.open("xb") as sink:
        header = source.readline()
        parse_row(header, header=True)
        sink.write(header)
        for index, entity_id in enumerate(refs.entity_id):
            raw = source.readline()
            entity, ids = parse_row(raw)
            if entity != entity_id.encode():
                raise ValueError("Original v1 Source 1 ID order changed before output")
            before += len(ids)
            empty_before += not ids
            lost = losers.get(index)
            if lost:
                remaining = [key for key in ids if key not in lost]
                if len(ids) - len(remaining) != len(lost):
                    raise ValueError("A lost claim was absent from its original v1 TSV row")
                newline = b"\r\n" if raw.endswith(b"\r\n") else b"\n"
                sink.write(entity + b"\t" + b",".join(remaining) + newline)
                changed += 1
                removed += len(lost)
                after += len(remaining)
                empty_after += not remaining
            else:
                sink.write(raw)
                after += len(ids)
                empty_after += not ids
        if source.readline():
            raise ValueError("Unexpected extra v1 matching row")
    return {"rows": len(refs), "changed_rows": changed, "removed_links": removed,
            "original_links": before, "assignment_only_links": after,
            "original_empty_rows": empty_before, "assignment_only_empty_rows": empty_after}


def _score_tune(left: np.ndarray, selected: np.ndarray, labels: np.ndarray,
                actual: np.ndarray) -> tuple[dict, np.ndarray]:
    predicted = np.bincount(left[selected], minlength=len(actual))
    tp = np.bincount(left[selected & labels], minlength=len(actual))
    values = np.divide(5. * tp, 4 * predicted + actual, out=np.zeros(len(actual)),
                       where=(4 * predicted + actual) > 0)
    values[(actual == 0) & (predicted == 0)] = 1
    return {"macro_f05": float(values.mean()), "predicted_links": int(predicted.sum()),
            "true_positive_links": int(tp.sum()), "false_positive_links": int((predicted - tp).sum()),
            "micro_precision": float(tp.sum() / predicted.sum()),
            "micro_recall": float(tp.sum() / actual.sum()),
            "singleton_accuracy": float(np.mean(predicted[actual == 0] == 0))}, values


def tune(pairs_path: Path, references_path: Path, p1_results: Path, destination: Path) -> dict:
    """Re-score the saved 12k medium TUNE, never an opened holdout or test labels."""
    if destination.exists():
        raise FileExistsError(destination)
    refs = pd.read_parquet(references_path, columns=["source1_entity_id", "original_left_idx", "true_count"])
    pairs = pd.read_parquet(pairs_path, columns=["original_pair_row", "left_idx", "right_idx",
                                                 "source1_entity_id", "original_blend_probability",
                                                 "is_truth_for_scoring_only", "baseline_accept"])
    if (len(refs) != 12_000 or len(pairs) != 600_000 or refs.source1_entity_id.duplicated().any()
            or hashlib.sha256("\n".join(sorted(refs.source1_entity_id)).encode()).hexdigest() != TUNE_ID_SHA256
            or pairs.original_pair_row.duplicated().any()
            or not pairs.original_pair_row.is_monotonic_increasing):
        raise ValueError("Saved 12k TUNE / 600k aligned pair identities changed")
    positions = pd.Index(refs.original_left_idx).get_indexer(pairs.left_idx)
    if (np.any(positions < 0) or not np.array_equal(refs.source1_entity_id.to_numpy()[positions],
                                                    pairs.source1_entity_id.to_numpy())):
        raise ValueError("TUNE candidates lost reference order/identity")
    raw = pairs.original_blend_probability.to_numpy(dtype=np.float32)
    baseline = raw >= THRESHOLD
    if not np.array_equal(baseline, pairs.baseline_accept.to_numpy()):
        raise ValueError("Frozen v1 TUNE threshold does not replay")
    claimed = np.flatnonzero(baseline)
    claims = np.empty(len(claimed), dtype=CLAIM_DTYPE)
    claims["left"] = positions[claimed]
    claims["right"] = pairs.right_idx.to_numpy(dtype=np.int32)[claimed]
    claims["pair_row"] = pairs.original_pair_row.to_numpy(dtype=np.int64)[claimed]
    claims["score"] = raw[claimed]
    chosen, conflicts = winners(claims)
    assigned = baseline.copy()
    assigned[claimed[~chosen]] = False
    labels = pairs.is_truth_for_scoring_only.to_numpy(dtype=bool)
    actual = refs.true_count.to_numpy(dtype=np.int32)
    old, per_old = _score_tune(positions, baseline, labels, actual)
    new, per_new = _score_tune(positions, assigned, labels, actual)
    saved = json.loads(p1_results.read_text())
    reported = next(x for x in saved["assignment"] if x["variant"] == "assignment_delta_0.00")
    if (abs(old["macro_f05"] - saved["baseline"]["macro_f05"]) > 1e-12
            or abs(new["macro_f05"] - reported["score"]["macro_f05"]) > 1e-12
            or conflicts["conflicting_targets"] != reported["detail"]["conflicting_targets"]
            or int(np.count_nonzero(labels[claimed[~chosen]])) != 0):
        raise ValueError("Independent pure assignment does not reproduce saved P1 TUNE result")
    comparison = {"scope": "reused_12k_medium_TUNE_full_training_target_pool",
                  "baseline": old, "assignment_only": new, "conflicts": conflicts,
                  "removed_true_links": int(np.count_nonzero(labels[claimed[~chosen]])),
                  "paired_entity_bootstrap_10k": paired_bootstrap(per_new, per_old, n_bootstrap=10_000, seed=20260925),
                  "inputs": {"tune_pairs": fingerprint(pairs_path), "tune_references": fingerprint(references_path),
                             "p1_results": fingerprint(p1_results)},
                  "old_sealed12k_used": False, "independent_validation": False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n")
    return comparison


def test(v1: Path, data: Path, test_dir: Path, validator: Path, output: Path) -> dict:
    """Full-test label-free fallback in an exclusive directory; never a release marker."""
    if output.exists() or output == v1 or output == data or v1 in output.parents or data in output.parents:
        raise FileExistsError("Assignment-only output must be fresh and outside original v1 artifacts")
    started = time.monotonic()
    identities, refs, pool_ids, scores = checked_inputs(v1, data)
    output.mkdir(parents=True, exist_ok=False)
    claims_path = output / "original_accepted_claims.bin"
    nclaims = extract_claims(v1 / "matching_results.tsv", v1 / "scored_pairs.parquet",
                             scores, refs, pool_ids, claims_path, expected_pairs=N_PAIRS)
    if nclaims != 5_753_111:
        raise ValueError("Original v1 accepted link count differs from its frozen report")
    claims = np.memmap(claims_path, mode="r", dtype=CLAIM_DTYPE, shape=(nclaims,))
    keep, conflicts = winners(claims)
    losers: dict[int, set[bytes]] = defaultdict(set)
    countries = Counter()
    for claim in claims[np.flatnonzero(~keep)]:
        left, right = int(claim["left"]), int(claim["right"])
        target = str(pool_ids[right]).encode()
        if target in losers[left]:
            raise ValueError("Duplicate original accepted target for one S1")
        losers[left].add(target)
        countries[str(refs.country.iat[left])] += 1
    del claims
    del keep
    changed = rewrite(v1 / "matching_results.tsv", output / "matching_results.tsv", refs, losers)
    if (changed["removed_links"] != conflicts["removed_claims"]
            or changed["original_links"] != nclaims
            or changed["changed_rows"] != len(losers)):
        raise ValueError("Assignment changed any non-conflicting original v1 pair")
    checked = validate_release(output / "matching_results.tsv", v1 / "candidate_pairs.tsv",
                               test_dir, output / "validation.json", validator=validator,
                               expected_rows=N_REFERENCES)
    if (not checked["passed"] or checked["counts"]["matched_links"] != changed["assignment_only_links"]
            or checked["counts"]["candidate_links"] != N_PAIRS):
        raise ValueError(f"Official --check-ids or strict full coverage/subset failed: {checked['errors']}")
    if (sha256(v1 / "matching_results.tsv") != V1_MATCH_SHA256
            or sha256(v1 / "probabilities.npy") != identities["blended_scores"]["sha256"]):
        raise ValueError("Immutable original v1 inputs changed during fallback construction")
    result = {"status": "exploratory_fallback_only", "ready_for_submission": False,
              "upload_performed": False, "test_labels_used": False, "old_sealed12k_used": False,
              "rule": "original v1 >=0.775 accept set; contested target keeps highest original float32 blend, then lowest S1 index/pair row",
              "original_v1": identities, "output": fingerprint(output / "matching_results.tsv"),
              "claims": fingerprint(claims_path), "source_sha256": sha256(Path(__file__)),
              "conflicts": conflicts, "removed_claims_by_country": dict(countries),
              "diff": changed, "validation_report": fingerprint(output / "validation.json"),
              "official_check_ids_and_strict_full_coverage_subset_passed": checked["passed"],
              "elapsed_seconds": time.monotonic() - started,
              "limits": "No test score or French accuracy. Reused TUNE is not independent confirmation."}
    (output / "assignment_only.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    tune_cmd = commands.add_parser("tune")
    for name in ("pairs", "references", "p1-results", "output"):
        tune_cmd.add_argument(f"--{name}", type=Path, required=True)
    test_cmd = commands.add_parser("test")
    for name in ("v1", "data", "test-dir", "validator", "output"):
        test_cmd.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "tune":
        result = tune(args.pairs, args.references, args.p1_results, args.output)
        print(json.dumps({"scope": result["scope"], "baseline": result["baseline"]["macro_f05"],
                          "assignment_only": result["assignment_only"]["macro_f05"],
                          "conflicts": result["conflicts"],
                          "ci": result["paired_entity_bootstrap_10k"]["confidence_intervals"]["left_minus_right"]}), flush=True)
    else:
        result = test(args.v1, args.data, args.test_dir, args.validator, args.output)
        print(json.dumps({"output": result["output"], "conflicts": result["conflicts"],
                          "diff": result["diff"], "validated": result["official_check_ids_and_strict_full_coverage_subset_passed"],
                          "seconds": result["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
