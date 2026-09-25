"""P4.1: CPU-only, full-target-pool cluster-mate retrieval screen.

Build text keys in raw S2-then-S3 row order; optional FAISS IndexLSH codes are
computed over the already saved E5 pool vectors. Clusters are *one-hop* groups,
not transitive connected components. Only the saved dense top-50 hits seed new
candidate edges. TRAIN truth assesses purity; TUNE truth is opened only after
the complete, label-free candidate parquet and its manifest are committed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import time

import numpy as np
import pandas as pd

from .full_embeddings import atomic_json, file_digest
from .normalization import normalize_text


EXACT, TEXT, VECTOR = 2, 4, 8
TUNE_COUNT = 12_000
TUNE_SHA = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
RAW_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
DIGITS = re.compile(r"\d+")
DEVANAGARI = re.compile(r"[\u0900-\u097f]")
TEXT_THRESHOLDS = (0.75, 0.85, 0.90, 0.95)
VECTOR_THRESHOLDS = ((0.60, 0.75), (0.70, 0.85), (0.80, 0.90))


def id_sha(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def input_sha(path: Path, expected: str) -> None:
    if file_digest(path) != expected:
        raise ValueError(f"Raw or saved input SHA256 mismatch: {path}")


def digits(text: str) -> str:
    return " ".join(DIGITS.findall(text))


def hash64(text: str) -> int:
    return int.from_bytes(hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest(), "little")


def exact_key(country: str, name: str, address: str) -> np.void:
    data = (country + "\x1f" + name + "\x1f" + address).encode("utf-8")
    return np.void(hashlib.blake2b(data, digest_size=16).digest())


def blocking_keys(country: str, name: str, address: str) -> tuple[int, int]:
    numbers = digits(address)
    normalized_name = normalize_text(name)
    if not numbers or not normalized_name:
        return 0, 0
    return (hash64(country + "\x1f" + normalized_name + "\x1f" + numbers),
            hash64(country + "\x1f" + numbers))


def raw_identity(source_paths: tuple[Path, Path], embedding_complete: Path,
                 integrity: Path) -> dict:
    complete = json.loads(embedding_complete.read_text())
    if (complete.get("state") != "complete" or not complete.get("full_dataset_complete")
            or not complete.get("verification", {}).get("all_finite")):
        raise ValueError("Saved full-pool embeddings lack a valid completion marker")
    identity = complete["identity"]
    original = {Path(f["path"]).name: f for f in json.loads(integrity.read_text())["files"]}
    info = []
    for source, saved in zip(source_paths, identity["inputs_in_order"], strict=True):
        if source.name != Path(saved["path"]).name or source.stat().st_size != saved["size_bytes"]:
            raise ValueError("Raw pool file order/size differs from saved embedding identity")
        original_item = original[source.name]
        if original_item["bytes"] != saved["size_bytes"]:
            raise ValueError("Original ZIP identity differs from saved embedding identity")
        input_sha(source, original_item["sha256"])
        info.append({"path": str(source.resolve()), "sha256": original_item["sha256"],
                     "bytes": saved["size_bytes"], "rows": saved["expected_rows"]})
    if sum(i["rows"] for i in info) != identity["target_rows"]:
        raise ValueError("Pool row count differs from embedding identity")
    return {"sources": info, "pool_rows": identity["target_rows"],
            "embedding_checkpoint_sha256": identity["checkpoint"]["sha256"],
            "embedding_dimension": identity["dimension"]}


def build_text_index(source_paths: tuple[Path, Path], identity: dict, output: Path,
                     chunk_size: int = 50_000) -> dict:
    if not 0 < chunk_size <= 50_000:
        raise ValueError("Text index chunks must be within 1..50,000")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "index_complete.json"
    paths = {name: output / f"{name}.npy" for name in ("exact", "text", "digits")}
    if marker.exists():
        report = json.loads(marker.read_text())
        if report["identity"] != identity:
            raise ValueError("Text index belongs to different raw/vector inputs")
        for name, dtype in (("exact", "V16"), ("text", "uint64"), ("digits", "uint64")):
            array = np.load(paths[name], mmap_mode="r")
            if array.shape != (identity["pool_rows"],) or array.dtype != np.dtype(dtype):
                raise ValueError(f"Incomplete or incompatible text index: {name}")
        return report
    if any(path.exists() for path in paths.values()):
        raise ValueError("Uncommitted text index arrays exist; inspect before retrying")
    started = time.monotonic()
    arrays = {"exact": np.lib.format.open_memmap(paths["exact"], mode="w+", dtype="V16",
                                                  shape=(identity["pool_rows"],)),
              "text": np.lib.format.open_memmap(paths["text"], mode="w+", dtype=np.uint64,
                                                 shape=(identity["pool_rows"],)),
              "digits": np.lib.format.open_memmap(paths["digits"], mode="w+", dtype=np.uint64,
                                                   shape=(identity["pool_rows"],))}
    offset = 0
    for path, metadata in zip(source_paths, identity["sources"], strict=True):
        count = 0
        for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=RAW_COLUMNS, chunksize=chunk_size):
            for target_id, name, address, country in frame.itertuples(index=False, name=None):
                if not target_id.startswith(("S2-", "S3-")):
                    raise ValueError("Unexpected raw target ID")
                idx = offset + count
                arrays["exact"][idx] = exact_key(country, name, address)
                arrays["text"][idx], arrays["digits"][idx] = blocking_keys(country, name, address)
                count += 1
            if count % 500_000 < len(frame):
                print(f"text_index {offset + count:,}/{identity['pool_rows']:,}", flush=True)
        if count != metadata["rows"]:
            raise ValueError(f"Raw target row count changed while indexing {path}")
        offset += count
    if offset != identity["pool_rows"]:
        raise ValueError("Text index did not cover the full target pool")
    for array in arrays.values():
        array.flush()
    del arrays
    report = {"identity": identity, "index_method": "raw_exact_128 + normalized_name_digit_64 + country_digit_64",
              "rows": offset, "elapsed_seconds": time.monotonic() - started,
              "labels_read": False, "target_sampling": False}
    atomic_json(marker, report)
    return report


def build_vector_codes(embeddings: Path, complete_path: Path, preservation_path: Path,
                       output: Path, *, threads: int = 2, chunk_size: int = 25_000) -> dict:
    """FAISS random-hyperplane LSH; encode every full-pool vector on CPU in chunks."""
    if not 1 <= threads <= 3 or not 0 < chunk_size <= 50_000:
        raise ValueError("Vector LSH requires <=3 CPU threads and <=50k-row chunks")
    import faiss  # Optional; this command runs on the pod's existing CPU FAISS runtime.

    complete = json.loads(complete_path.read_text())
    preservation = json.loads(preservation_path.read_text())
    identity = complete["identity"]
    if (complete.get("state") != "complete" or not complete.get("full_dataset_complete")
            or not preservation.get("copy_verified") or embeddings.stat().st_size != preservation["bytes"]):
        raise ValueError("Incomplete/unverified full-pool vector input")
    vectors = np.load(embeddings, mmap_mode="r")
    if (vectors.shape != (identity["target_rows"], identity["dimension"])
            or vectors.dtype != np.float16 or identity["dimension"] != 384):
        raise ValueError("Saved E5 pool vector shape/dtype/identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    marker = output / "vector_code_complete.json"
    codes_path, transform_path = output / "vector_codes.npy", output / "faiss_lsh.index"
    config = {"pool_rows": len(vectors), "dimension": 384, "bits": 24,
              "bands": [0, 12], "faiss_version": faiss.__version__,
              "pool_embedding_sha256": preservation["sha256"],
              "checkpoint_sha256": identity["checkpoint"]["sha256"]}
    if marker.exists():
        report = json.loads(marker.read_text())
        if (report["config"] != config or np.load(codes_path, mmap_mode="r").shape != (len(vectors),)
                or file_digest(transform_path) != report["transform_sha256"]):
            raise ValueError("Existing vector codes have a different E5/FAISS identity")
        return report
    if codes_path.exists() or transform_path.exists():
        raise ValueError("Uncommitted vector LSH arrays exist; inspect before retrying")
    started = time.monotonic()
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexLSH(384, 24, True)
    faiss.write_index(index, str(transform_path))
    coded = np.lib.format.open_memmap(codes_path, mode="w+", dtype=np.uint32,
                                      shape=(len(vectors),))
    for start in range(0, len(vectors), chunk_size):
        end = min(start + chunk_size, len(vectors))
        data = np.asarray(vectors[start:end], dtype=np.float32)
        bits = index.sa_encode(data)
        coded[start:end] = (bits[:, 0].astype(np.uint32) | (bits[:, 1].astype(np.uint32) << 8)
                            | (bits[:, 2].astype(np.uint32) << 16))
        if end % 500_000 < chunk_size or end == len(vectors):
            coded.flush()
            print(f"faiss_lsh {end:,}/{len(vectors):,}", flush=True)
    coded.flush()
    del coded
    report = {"config": config, "transform_sha256": file_digest(transform_path),
              "codes_sha256": file_digest(codes_path), "threads": threads,
              "chunk_size": chunk_size, "elapsed_seconds": time.monotonic() - started,
              "labels_read": False, "target_sampling": False}
    atomic_json(marker, report)
    return report


def vector_band_keys(digit_keys: np.ndarray, codes: np.ndarray, band: int) -> np.ndarray:
    key = np.empty(len(codes), dtype=[("digit", "<u8"), ("band", "<u2")])
    key["digit"] = digit_keys
    key["band"] = ((codes >> band) & 4095).astype(np.uint16)
    return key


def propose_mates(keys: np.ndarray, seeds: np.ndarray, flag: int, max_group: int,
                  proposals: dict[tuple[int, int], int], max_proposals: int,
                  *, tune_seeds: set[int] | None = None,
                  candidate_edges: dict[tuple[int, int], int] | None = None) -> dict:
    """Look up complete full-pool hash buckets, with explicit size/memory caps."""
    started = time.monotonic()
    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]
    wanted = keys[seeds]
    starts = np.searchsorted(sorted_keys, wanted, side="left")
    stops = np.searchsorted(sorted_keys, wanted, side="right")
    visited = accepted = oversized = 0
    for seed, key, start, stop in zip(seeds, wanted, starts, stops, strict=True):
        size = int(stop - start)
        if ((flag == VECTOR and key["digit"] == 0) or
                (flag == TEXT and key == 0) or size <= 1):
            continue
        visited += 1
        if size > max_group:
            oversized += 1
            continue
        for mate in order[start:stop]:
            a, b = sorted((int(seed), int(mate)))
            if a == b:
                continue
            proposals[a, b] = proposals.get((a, b), 0) | flag
            if tune_seeds is not None and int(seed) in tune_seeds:
                if candidate_edges is None:
                    raise ValueError("TUNE-seeded candidate edge set is required")
                candidate_edges[a, b] = candidate_edges.get((a, b), 0) | flag
            accepted += 1
        if len(proposals) > max_proposals:
            raise ValueError("Near-twin bucket proposals exceeded the bounded pilot limit")
    return {"flag": flag, "seeds_with_mates": visited, "seeds_oversized": oversized,
            "directed_proposals": accepted, "unique_edges_total": len(proposals),
            "elapsed_seconds": time.monotonic() - started}


@dataclass(frozen=True)
class Target:
    entity_id: str
    name: str
    address: str
    country: str


def extract_raw_rows(source_paths: tuple[Path, Path], identity: dict, needed: np.ndarray,
                     chunk_size: int = 50_000) -> dict[int, Target]:
    """One bounded pass; resolve original IDs/text only for proposed cluster rows."""
    needed = np.unique(np.asarray(needed, dtype=np.int64))
    if len(needed) and (needed[0] < 0 or needed[-1] >= identity["pool_rows"]):
        raise ValueError("Proposed raw target row is outside the saved pool")
    records: dict[int, Target] = {}
    offset = 0
    for path, metadata in zip(source_paths, identity["sources"], strict=True):
        count = 0
        for frame in pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                                 usecols=RAW_COLUMNS, chunksize=chunk_size):
            start = offset + count
            lo, hi = np.searchsorted(needed, [start, start + len(frame)])
            for row in needed[lo:hi]:
                entity_id, name, address, country = frame.iloc[int(row - start)]
                records[int(row)] = Target(entity_id, name, address, country)
            count += len(frame)
        if count != metadata["rows"]:
            raise ValueError("Raw target row count changed while resolving IDs")
        offset += count
    if len(records) != len(needed):
        raise ValueError("Failed to resolve every original raw target ID")
    return records


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


def trigrams(text: str) -> frozenset[str]:
    return frozenset((" " + text + " ")[i:i + 3] for i in range(len(text)))


def edge_features(proposals: dict[tuple[int, int], int], records: dict[int, Target]) -> dict:
    """Recheck hash collisions and measure pairwise text evidence before any labels."""
    features = {}
    text_cache: dict[int, tuple[str, str, frozenset[str], frozenset[str]]] = {}

    def canonical(raw: int):
        if raw not in text_cache:
            row = records[raw]
            name, address = normalize_text(row.name), normalize_text(row.address)
            text_cache[raw] = (name, digits(row.address), frozenset(address.split()), trigrams(name))
        return text_cache[raw]

    for (a, b), flags in proposals.items():
        x, y = records[a], records[b]
        if x.country != y.country or x.entity_id == y.entity_id:
            continue
        exact = bool(flags & EXACT and x.name == y.name and x.address == y.address)
        nx, dx, ax, cx = canonical(a)
        ny, dy, ay, cy = canonical(b)
        same_digits = bool(dx and dx == dy)
        addr_j = jaccard(ax, ay)
        name_j = jaccard(cx, cy)
        text = bool(flags & TEXT and same_digits and nx and nx == ny)
        vector = bool(flags & VECTOR and same_digits and nx and ny)
        if exact or text or vector:
            features[a, b] = {"flags": (EXACT if exact else 0) | (TEXT if text else 0)
                              | (VECTOR if vector else 0), "address_j": addr_j,
                              "name_j": name_j}
    return features


def train_truth_rows(ground_truth: Path, train_ids: set[str], compact_pool: pd.DataFrame,
                     expected_sha: str) -> dict[int, str]:
    """Stream ground truth selecting TRAIN IDs only; never parse TUNE/eval values."""
    input_sha(ground_truth, expected_sha)
    owner_by_id = {}
    seen = set()
    with ground_truth.open(encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("Unexpected original ground-truth header")
        for line in stream:
            entity_id, separator, targets = line.rstrip("\r\n").partition("\t")
            if separator and entity_id in train_ids:
                if entity_id in seen:
                    raise ValueError("Duplicate TRAIN truth reference")
                seen.add(entity_id)
                for target in targets.split(","):
                    if target:
                        if target in owner_by_id:
                            raise ValueError("One target has multiple TRAIN owners")
                        owner_by_id[target] = entity_id
    if seen != train_ids:
        raise ValueError("TRAIN truth does not cover every fixed TRAIN S1")
    row_for_id = dict(zip(compact_pool.entity_id, compact_pool.raw_row_idx, strict=True))
    if set(owner_by_id) - row_for_id.keys():
        raise ValueError("Saved compact pool omitted a TRAIN true target record")
    return {int(row_for_id[entity_id]): owner for entity_id, owner in owner_by_id.items()}


def train_purity(edges: dict, train_owners: dict[int, str], min_purity: float = 0.99) -> tuple[dict, dict]:
    """Anchor-known precision: a mate must be true for the *same* TRAIN S1."""
    if not 0 < min_purity <= 1:
        raise ValueError("Purity threshold must be between 0 and 1")
    policies = {"exact": (EXACT, None),
                **{f"text_{threshold:.2f}": (TEXT, threshold) for threshold in TEXT_THRESHOLDS},
                **{f"vector_{name:.2f}_{addr:.2f}": (VECTOR, (name, addr))
                   for name, addr in VECTOR_THRESHOLDS}}
    counters = {label: [0, 0, 0] for label in policies}
    for (a, b), features in edges.items():
        for anchor, mate in ((a, b), (b, a)):
            owner = train_owners.get(anchor)
            if owner is None:
                continue
            same_owner = bool(train_owners.get(mate) == owner)
            cross_owner = bool(mate in train_owners and not same_owner)
            for label, (flag, threshold) in policies.items():
                if not features["flags"] & flag:
                    continue
                if flag == TEXT and features["address_j"] < threshold:
                    continue
                if flag == VECTOR and (features["name_j"] < threshold[0]
                                       or features["address_j"] < threshold[1]):
                    continue
                result = counters[label]
                result[0] += 1
                result[1] += int(same_owner)
                result[2] += int(cross_owner)
    report = {label: {"anchor_true_mate_edges": count, "same_train_s1": good,
                      "other_or_unlabeled_for_s1": count - good,
                      "different_known_train_s1": cross,
                      "anchor_positive_precision": good / count if count else None}
              for label, (count, good, cross) in counters.items()}

    def choose(prefix: str, variants):
        for label in variants:
            result = report[label]
            if result["anchor_true_mate_edges"] >= 50 and result["anchor_positive_precision"] >= min_purity:
                return label
        return None

    text_choice = choose("text", [f"text_{x:.2f}" for x in TEXT_THRESHOLDS])
    vector_choice = choose("vector", [f"vector_{n:.2f}_{a:.2f}" for n, a in VECTOR_THRESHOLDS])
    selected = {"exact": True, "text": text_choice, "vector": vector_choice,
                "minimum_train_anchor_precision": min_purity,
                "minimum_anchor_edges_for_near_policy": 50}
    return report, selected


def selected_edges(edges: dict, selected: dict,
                   allowed_edges: dict[tuple[int, int], int] | None = None) -> dict[int, list[tuple[int, int]]]:
    adjacent: dict[int, list[tuple[int, int]]] = defaultdict(list)
    text_threshold = float(selected["text"].split("_")[1]) if selected["text"] else None
    vector_threshold = (tuple(map(float, selected["vector"].split("_")[1:]))
                        if selected["vector"] else None)
    for (a, b), features in edges.items():
        tune_flags = (allowed_edges.get((a, b), 0) if allowed_edges is not None
                      else EXACT | TEXT | VECTOR)
        if not tune_flags:
            continue
        flags = features["flags"] & tune_flags & EXACT
        if (text_threshold is not None and features["flags"] & tune_flags & TEXT
                and features["address_j"] >= text_threshold):
            flags |= TEXT
        if (vector_threshold is not None and features["flags"] & tune_flags & VECTOR
                and features["name_j"] >= vector_threshold[0]
                and features["address_j"] >= vector_threshold[1]):
            flags |= VECTOR
        if flags:
            adjacent[a].append((b, flags))
            adjacent[b].append((a, flags))
    return adjacent


def expand_tune(raw_pairs: pd.DataFrame, reference: pd.DataFrame, pool: pd.DataFrame,
                adjacent: dict[int, list[tuple[int, int]]], records: dict[int, Target],
                *, max_extra_per_reference: int = 100) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    tune_positions = np.flatnonzero(reference.split.to_numpy() == "tune")
    tune = reference.iloc[tune_positions].reset_index(drop=True)
    if len(tune) != TUNE_COUNT or id_sha(tune.entity_id) != TUNE_SHA:
        raise ValueError("Expected exactly the saved, fixed 12k medium TUNE S1s")
    old_to_new = np.full(len(reference), -1, dtype=np.int32)
    old_to_new[tune_positions] = np.arange(len(tune), dtype=np.int32)
    old_left = raw_pairs.left_idx.to_numpy(dtype=np.int64)
    if not np.all((old_left >= 0) & (old_left < len(reference))):
        raise ValueError("Saved dense reference index is out of bounds")
    keep = old_to_new[old_left] >= 0
    base = raw_pairs.loc[keep].copy().reset_index(drop=True)
    left = old_to_new[old_left[keep]]
    right = base.right_raw_idx.to_numpy(dtype=np.int64)
    ranks = base.dense_rank.to_numpy(dtype=np.int16)
    if len(base) != 50 * TUNE_COUNT or np.bincount(left, minlength=TUNE_COUNT).tolist() != [50] * TUNE_COUNT:
        raise ValueError("Saved baseline differs from 50 dense full-pool hits per TUNE S1")
    if len(set(zip(left, right, strict=True))) != len(base):
        raise ValueError("Saved dense pairs contain duplicate reference/target edges")
    pool_rows = pool.raw_row_idx.to_numpy(dtype=np.int64)
    if len(np.unique(pool_rows)) != len(pool_rows) or not np.all(pool_rows[:-1] <= pool_rows[1:]):
        raise ValueError("Saved compact-pool raw indices must be unique and sorted")
    indexes = np.searchsorted(pool_rows, right)
    if not np.array_equal(pool_rows[indexes], right):
        raise ValueError("Saved dense target is missing from compact pool")
    ids = pool.entity_id.to_numpy()[indexes]
    original = [set() for _ in range(TUNE_COUNT)]
    for ref, row in zip(left, right, strict=True):
        original[int(ref)].add(int(row))
    extra: dict[tuple[int, int], tuple[int, int, int]] = {}
    extras_by_ref = np.zeros(TUNE_COUNT, dtype=np.int32)
    capped = 0
    for ref, seed, rank in zip(left, right, ranks, strict=True):
        ref, seed, rank = int(ref), int(seed), int(rank)
        for mate, flags in sorted(adjacent.get(seed, ())):
            if mate in original[ref]:
                continue
            key = (ref, mate)
            if key in extra:
                old_flags, old_rank, old_seed = extra[key]
                extra[key] = (old_flags | flags, min(old_rank, rank), old_seed)
                continue
            if extras_by_ref[ref] >= max_extra_per_reference:
                capped += 1
                continue
            extra[key] = (flags, rank, seed)
            extras_by_ref[ref] += 1
    if any(mate not in records for _, mate in extra):
        raise ValueError("Unresolved original target ID for a new cluster candidate")
    for (ref, mate) in extra:
        if records[mate].country != tune.country.iloc[ref]:
            raise ValueError("Country-conflicting cluster candidate")
    base_frame = pd.DataFrame({"reference_idx": left, "reference_id": tune.entity_id.to_numpy()[left],
                               "target_raw_idx": right, "target_id": ids, "source_flags": np.uint8(1),
                               "seed_raw_idx": right, "seed_dense_rank": ranks})
    additions = pd.DataFrame(
        [(ref, tune.entity_id.iloc[ref], mate, records[mate].entity_id, flags, seed, rank)
         for (ref, mate), (flags, rank, seed) in sorted(extra.items())],
        columns=base_frame.columns,
    )
    pairs = pd.concat((base_frame, additions), ignore_index=True)
    if (pairs.duplicated(["reference_id", "target_raw_idx"]).any()
            or pairs.duplicated(["reference_id", "target_id"]).any()
            or pairs.reference_idx.nunique() != len(tune)):
        raise ValueError("Cluster expansion lost S1 coverage or duplicated a candidate pair/ID")
    return tune, pairs, {"dense_pairs": len(base), "new_pairs": len(extra), "cap": max_extra_per_reference,
                         "capped_proposals": capped, "references_with_extras": int((extras_by_ref > 0).sum()),
                         "max_extras_per_reference": int(extras_by_ref.max())}


def oracle_scores(true_counts: np.ndarray, found: np.ndarray) -> np.ndarray:
    if np.any(found > true_counts) or np.any(true_counts < 0):
        raise ValueError("Duplicate/injected positives or invalid truth counts")
    scores = np.divide(5.0 * found, 4.0 * found + true_counts,
                       out=np.zeros(len(true_counts), dtype=np.float64), where=true_counts > 0)
    scores[true_counts == 0] = 1.0
    return scores


def paired_bootstrap(baseline: np.ndarray, challenger: np.ndarray, *, seed: int = 20260925,
                     replicates: int = 10_000, block: int = 200) -> dict:
    if baseline.shape != challenger.shape or baseline.ndim != 1 or not len(baseline):
        raise ValueError("Paired bootstrap requires aligned, complete S1 score vectors")
    rng = np.random.default_rng(seed)
    diff = challenger - baseline
    samples = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, block):
        end = min(start + block, replicates)
        indices = rng.integers(len(diff), size=(end - start, len(diff)), dtype=np.int32)
        samples[start:end] = diff[indices].mean(axis=1)
    return {"delta": float(diff.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
            "replicates": replicates, "seed": seed, "reference_count": len(diff),
            "improved": int((diff > 0).sum()), "worsened": int((diff < 0).sum()),
            "tied": int((diff == 0).sum())}


def script_label(name: str, address: str) -> str:
    combined = name + " " + address
    if DEVANAGARI.search(combined):
        return "Devanagari"
    if not combined.isascii():
        return "other_non_ASCII"
    return "ASCII"


def score_candidates(pairs: pd.DataFrame, tune: pd.DataFrame, pool: pd.DataFrame,
                     truth_path: Path, saved_baseline: dict, *, seed: int = 20260925) -> dict:
    """Called only after label-free candidates are durable on local disk."""
    truth_all = json.loads(truth_path.read_text())
    ids = tune.entity_id.tolist()
    if len(ids) != TUNE_COUNT or id_sha(ids) != TUNE_SHA or set(ids) - truth_all.keys():
        raise ValueError("Scoring must use every original fixed medium TUNE business")
    truth = {ref: set(truth_all[ref]) for ref in ids}
    true_counts = np.fromiter((len(truth[ref]) for ref in ids), dtype=np.int32)
    pool_ids = set(pool.entity_id)
    if set.union(set(), *truth.values()) - pool_ids:
        raise ValueError("Missed true target record absent from compact scoring pool")
    left = pairs.reference_idx.to_numpy(dtype=np.int32)
    flags = pairs.source_flags.to_numpy(dtype=np.uint8)
    positives = np.fromiter((target in truth[ids[int(ref)]]
                             for ref, target in zip(left, pairs.target_id, strict=True)),
                            dtype=bool, count=len(pairs))
    variants = {"baseline_dense50": flags == 1,
                "exact": (flags == 1) | ((flags & EXACT) > 0),
                "exact_text": (flags == 1) | ((flags & (EXACT | TEXT)) > 0),
                "exact_text_vector": np.ones(len(pairs), dtype=bool)}
    scripts = np.asarray([script_label(name, address) for name, address
                          in zip(tune.business_name, tune.business_address, strict=True)])
    country = tune.country.to_numpy()
    reports, scores = {}, {}
    for label, mask in variants.items():
        found = np.bincount(left[mask & positives], minlength=len(tune))
        per_ref = oracle_scores(true_counts, found)
        scores[label] = per_ref
        extra = mask & (flags != 1)
        report = {"reference_count": len(tune), "candidate_pairs": int(mask.sum()),
                  "new_candidate_pairs": int(extra.sum()), "new_true_links": int(positives[extra].sum()),
                  "new_negative_candidates": int((extra & ~positives).sum()),
                  "macro_f05": float(per_ref.mean()),
                  "link_recall": float(found.sum() / true_counts.sum()),
                  "complete_match_set_recall": float((found[true_counts > 0] == true_counts[true_counts > 0]).mean()),
                  "oracle_precision": 1.0, "oracle_false_links": 0,
                  "oracle_singleton_accuracy": 1.0, "true_links": int(true_counts.sum()),
                  "found_links": int(found.sum()),
                  "singleton_new_candidates": int((extra & (true_counts[left] == 0)).sum()),
                  "groups": {}}
        for grouping, values in (("country", country), ("script", scripts)):
            for value in sorted(set(values)):
                selected = values == value
                selected_pairs = selected[left]
                extra_group = extra & selected_pairs
                report["groups"][f"{grouping}/{value}"] = {
                    "references": int(selected.sum()), "oracle_macro_f05": float(per_ref[selected].mean()),
                    "link_recall": (float(found[selected].sum() / true_counts[selected].sum())
                                    if true_counts[selected].sum() else None),
                    "new_candidates": int(extra_group.sum()),
                    "new_true_links": int(positives[extra_group].sum()),
                    "new_negative_candidates": int((extra_group & ~positives).sum()),
                    "singleton_new_candidates": int((extra_group & (true_counts[left] == 0)).sum())}
        reports[label] = report
    if (abs(reports["baseline_dense50"]["macro_f05"] - saved_baseline["oracle_macro_f05"]) > 1e-12
            or abs(reports["baseline_dense50"]["link_recall"] - saved_baseline["link_recall"]) > 1e-12
            or reports["baseline_dense50"]["candidate_pairs"] != 50 * TUNE_COUNT):
        raise ValueError("Same-S1/full-pool oracle baseline does not reproduce the saved result")
    for label in ("exact", "exact_text", "exact_text_vector"):
        reports[label]["paired_vs_dense"] = paired_bootstrap(scores["baseline_dense50"], scores[label],
                                                               seed=seed)
    # Risk if a matcher blindly propagates a confidently *true* dense hit's cluster mates.
    original = set(zip(left[flags == 1], pairs.target_raw_idx[flags == 1], strict=True))
    original_id = dict(zip(zip(left[flags == 1], pairs.target_raw_idx[flags == 1], strict=True),
                           pairs.target_id[flags == 1], strict=True))
    added = np.flatnonzero(flags != 1)
    anchored = np.fromiter((original_id[(int(left[i]), int(pairs.seed_raw_idx.iloc[i]))]
                            in truth[ids[int(left[i])]]
                            if (int(left[i]), int(pairs.seed_raw_idx.iloc[i])) in original else False
                            for i in added), dtype=bool, count=len(added))
    reports["fp_risk_proxy"] = {
        "meaning": "new negatives attached to a truly matching dense anchor; not predicted false links",
        "extra_from_true_anchors": int(anchored.sum()),
        "negatives_from_true_anchors": int((anchored & ~positives[added]).sum()),
        "singleton_with_extras": int(np.unique(left[added][true_counts[left[added]] == 0]).size)}
    return reports


def run_screen(args) -> dict:
    start = time.monotonic()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cluster.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / "screen_complete.json").exists():
            raise ValueError("Cluster screen already complete; inspect saved report instead of rerunning")
        source_paths = (args.source2, args.source3)
        identity = raw_identity(source_paths, args.embedding_complete, args.source_integrity)
        baseline_complete = json.loads((args.dense_dir / "complete.json").read_text())
        if (baseline_complete.get("state") != "complete" or
                baseline_complete["config"]["pool_embedding_identity"]["checkpoint"]["sha256"]
                != identity["embedding_checkpoint_sha256"] or
                baseline_complete["metadata"]["mining_full_pool_records"] != identity["pool_rows"] or
                baseline_complete["metadata"]["candidate_pairs"] != 2_399_950):
            raise ValueError("Baseline dense full-pool identity/completion mismatch")
        input_files = ("complete.json", "raw_pairs.parquet", "reference.parquet",
                       "pool.parquet", "truth.json")
        inputs = {name: file_digest(args.dense_dir / name) for name in input_files}
        index_report = build_text_index(source_paths, identity, out / "text_index", args.chunk_size)
        reference = pd.read_parquet(args.dense_dir / "reference.parquet")
        pool = pd.read_parquet(args.dense_dir / "pool.parquet", columns=["entity_id", "raw_row_idx"])
        pairs = pd.read_parquet(args.dense_dir / "raw_pairs.parquet")
        if (len(reference) != 47_999 or reference.entity_id.duplicated().any() or
                len(pairs) != 2_399_950 or len(pool) != baseline_complete["metadata"]["pool_records"]):
            raise ValueError("Saved dense reference/pair/target identities changed")
        if (id_sha(reference.entity_id) != baseline_complete["metadata"]["reference_id_sha256"] or
                set(reference.split) != {"train", "tune"}):
            raise ValueError("Saved reference set contains another split or changed IDs")
        train_ids = set(reference.loc[reference.split.eq("train"), "entity_id"])
        if len(train_ids) != 35_999:
            raise ValueError("TRAIN purity must use only the 35,999 medium TRAIN S1s")
        truth_sha = next(x["sha256"] for x in json.loads(args.source_integrity.read_text())["files"]
                         if Path(x["path"]).name == args.ground_truth.name)
        train_owners = train_truth_rows(args.ground_truth, train_ids, pool, truth_sha)
        selected_raw = pairs.loc[reference.split.to_numpy()[pairs.left_idx.to_numpy()] == "tune",
                                 "right_raw_idx"].to_numpy(dtype=np.int64)
        tune_seeds = set(map(int, selected_raw))
        seeds = np.unique(np.r_[selected_raw, np.fromiter(train_owners, dtype=np.int64)])
        candidate_edges: dict[tuple[int, int], int] = {}
        index_dir = out / "text_index"
        proposals: dict[tuple[int, int], int] = {}
        searches = []
        searches.append(propose_mates(np.load(index_dir / "exact.npy", mmap_mode="r"), seeds,
                                      EXACT, args.max_exact_group, proposals, args.max_proposals,
                                      tune_seeds=tune_seeds, candidate_edges=candidate_edges))
        searches.append(propose_mates(np.load(index_dir / "text.npy", mmap_mode="r"), seeds,
                                      TEXT, args.max_near_group, proposals, args.max_proposals,
                                      tune_seeds=tune_seeds, candidate_edges=candidate_edges))
        vector_report = None
        if args.vector_dir:
            vector_report = json.loads((args.vector_dir / "vector_code_complete.json").read_text())
            if (vector_report["config"]["pool_rows"] != identity["pool_rows"] or
                    vector_report["config"]["checkpoint_sha256"] != identity["embedding_checkpoint_sha256"] or
                    file_digest(args.vector_dir / "vector_codes.npy") != vector_report["codes_sha256"] or
                    file_digest(args.vector_dir / "faiss_lsh.index") != vector_report["transform_sha256"]):
                raise ValueError("FAISS-LSH codes do not match complete full-pool E5 vectors")
            codes = np.load(args.vector_dir / "vector_codes.npy", mmap_mode="r")
            if codes.shape != (identity["pool_rows"],) or codes.dtype != np.uint32:
                raise ValueError("FAISS-LSH codes have an incomplete shape/dtype")
            digit_keys = np.load(index_dir / "digits.npy", mmap_mode="r")
            for band in (0, 12):
                keys = vector_band_keys(digit_keys, codes, band)
                searches.append(propose_mates(keys, seeds, VECTOR, args.max_near_group,
                                              proposals, args.max_proposals,
                                              tune_seeds=tune_seeds, candidate_edges=candidate_edges))
                del keys
        needed = np.fromiter((row for edge in proposals for row in edge), dtype=np.int64,
                             count=2 * len(proposals))
        records = extract_raw_rows(source_paths, identity, needed, args.chunk_size)
        edges = edge_features(proposals, records)
        purity, selected = train_purity(edges, train_owners, args.min_train_purity)
        adjacency = selected_edges(edges, selected, allowed_edges=candidate_edges)
        tune, expanded, expansion_report = expand_tune(pairs, reference, pool, adjacency, records,
                                                       max_extra_per_reference=args.max_extra_per_reference)
        tune_path, pair_path = out / "tune_references.parquet", out / "tune_candidates.parquet"
        if tune_path.exists() or pair_path.exists():
            raise ValueError("Uncommitted candidate file exists; inspect before retrying")
        tune.to_parquet(tune_path, index=False)
        expanded.to_parquet(pair_path, index=False)
        config = {"identity": identity, "saved_dense_sha256": inputs,
                  "full_pool_target_competition": identity["pool_rows"],
                  "baseline_reference_sha256": baseline_complete["metadata"]["reference_id_sha256"],
                  "tune_id_sha256": id_sha(tune.entity_id), "train_id_sha256": id_sha(train_ids),
                  "seeds": len(seeds), "tune_top_hit_seeds": len(tune_seeds),
                  "top_hits": 50, "max_exact_group": args.max_exact_group,
                  "max_near_group": args.max_near_group, "max_proposals": args.max_proposals,
                  "max_extra_per_reference": args.max_extra_per_reference,
                  "train_only_purity_policy": selected, "vector_lsh": vector_report,
                  "bootstrap_seed": args.bootstrap_seed, "bootstrap_replicates": 10_000,
                  "code_sha256": file_digest(Path(__file__)), "positive_candidate_injection": False,
                  "candidate_edges_seeded_only_by_saved_tune_dense_hits": True}
        pretruth = {"config": config, "text_index": index_report, "searches": searches,
                    "proposed_unique_undirected_edges": len(proposals),
                    "verified_text_edges": len(edges), "candidate_edges_from_tune_hits": len(candidate_edges),
                    "train_purity": purity,
                    "train_only_truth_used_for_purity": True, "tune_truth_opened": False,
                    "candidate_file_sha256": file_digest(pair_path),
                    "reference_file_sha256": file_digest(tune_path),
                    "candidate_pair_count": len(expanded), "expansion": expansion_report,
                    "elapsed_before_tune_truth_seconds": time.monotonic() - start}
        atomic_json(out / "candidate_manifest.json", pretruth)
        # The only TUNE-truth read is below, after complete, original-ID candidates exist.
        scored = score_candidates(expanded, tune, pool, args.dense_dir / "truth.json",
                                  baseline_complete["metadata"]["retrieval"]["tune"],
                                  seed=args.bootstrap_seed)
        gate = scored["exact_text_vector"]["paired_vs_dense"]["delta"] >= 0.003
        report = {**pretruth, "tune_truth_opened_after_candidates": True,
                  "retrieval_only_gate": {"required_delta_macro_f05": 0.003, "passed": gate,
                                          "ce_cluster_mean_or_assignment_allowed": gate},
                  "tune_retrieval": scored, "total_elapsed_seconds": time.monotonic() - start,
                  "peak_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2),
                  "scope": "fixed_medium_TUNE_12000_vs_complete_training_pool_10320219"}
        atomic_json(out / "screen_complete.json", report)
        return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    code = sub.add_parser("vector-codes", help="CPU FAISS-LSH all saved E5 pool rows")
    code.add_argument("--embeddings", type=Path, required=True)
    code.add_argument("--embedding-complete", type=Path, required=True)
    code.add_argument("--preservation", type=Path, required=True)
    code.add_argument("--output", type=Path, required=True)
    code.add_argument("--threads", type=int, default=2)
    code.add_argument("--chunk-size", type=int, default=25_000)
    screen = sub.add_parser("screen", help="Full-pool text/optional FAISS-LSH cluster screen")
    for flag in ("source2", "source3", "ground-truth", "source-integrity", "embedding-complete",
                 "dense-dir", "output"):
        screen.add_argument("--" + flag, type=Path, required=True)
    screen.add_argument("--vector-dir", type=Path)
    screen.add_argument("--chunk-size", type=int, default=50_000)
    screen.add_argument("--max-exact-group", type=int, default=24)
    screen.add_argument("--max-near-group", type=int, default=12)
    screen.add_argument("--max-proposals", type=int, default=5_000_000)
    screen.add_argument("--max-extra-per-reference", type=int, default=100)
    screen.add_argument("--min-train-purity", type=float, default=0.99)
    screen.add_argument("--bootstrap-seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.command == "vector-codes":
        report = build_vector_codes(args.embeddings, args.embedding_complete, args.preservation,
                                    args.output, threads=args.threads, chunk_size=args.chunk_size)
        print(json.dumps(report, sort_keys=True), flush=True)
    else:
        if min(args.max_exact_group, args.max_near_group, args.max_proposals,
               args.max_extra_per_reference) <= 0:
            parser.error("Group, proposal, and per-reference limits must be positive")
        result = run_screen(args)
        print(json.dumps({"gate": result["retrieval_only_gate"],
                          "tune": result["tune_retrieval"]["exact_text_vector"],
                          "runtime_seconds": result["total_elapsed_seconds"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
