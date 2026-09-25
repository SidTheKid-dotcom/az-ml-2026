"""P3.1: bounded-memory CatBoost on complete, full-pool candidate groups.

The screen uses saved medium dense candidates. A full run requires *completed*
P2 union artifacts for both the new TRAIN and the fixed 12,000-S1 medium TUNE.
Only TRAIN pairs fit the tree; TUNE labels choose the decision threshold. Neither
mode accepts an eval/holdout split or injects unretrieved truth edges.

Example (from the v2 repository root, using the existing CPU environment)::

    PYTHONPATH=code/business_entity_resolution/src /path/to/.venv/bin/python \
      -m ber.v2_tree_train --mode screen \
      --train-data artifacts/v2_tree_screen/inputs \
      --tune-data artifacts/v2_tree_screen/inputs \
      --reference-checkpoint artifacts/v2_tree_screen/inputs/best.joblib \
      --output artifacts/v2_tree_screen/dense_compat --iterations 80 --threads 6

For P2, supply its atomic --train-ready marker and the completed medium-TUNE
union as --tune-data/--tune-pairs; keep --mode full. Pairs must be
ordered by left_idx and contain every candidate of each selected S1. The source
Parquets are never modified. The quantized numeric training Pool is saved for
replay; CatBoost's used_ram_limit is advisory, so a conservative pair cap and
available-memory checks are applied before quantization and fitting. Use
--mode memory-probe with saved medium TRAIN features to establish a real no-swap
resource envelope before choosing local versus pod CPU full-scale training.
The provisional co-resident pod cap is 10M pairs, subject to live memory gates;
no local 60M-pair fit has been measured.

After the P2.5 *full* ready marker and the scored 12k TUNE union are complete,
the pod CPU command (output on the v2 durable volume, not the 30 GB overlay) is::

    python -m ber.v2_tree_train --mode full --execution-site pod-cpu \
      --train-ready /workspace/az-ml-v2/train_union/full/ready.json \
      --tune-data /tmp/az-ml-v2/lexical/medium_tune_union_features \
      --tune-pairs /tmp/az-ml-v2/lexical/medium_tune_union_features/candidate_pairs_scored.parquet \
      --output /workspace/az-ml-v2/tree_screen/full_p2 \
      --threads 3 --max-train-pairs 10000000 --ram-budget-gb 22 \
      --max-process-rss-gb 22 --min-available-gb 12

Check the actual ready-marker location, disk space, cgroup headroom and active
GPU CPU load immediately before queueing. A 10M-pair fit is only a guarded
proposal until a real P2 full-pair resource measurement passes.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import gc
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import resource
import subprocess
import shutil
import threading
import time

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import psutil

from .features import BASE_FEATURE_NAMES, CONTEXT_FEATURE_NAMES
from .numeric_features import NAMES as NUMERIC_NAMES
from .predict import feature_batch
from .predict_streaming import iter_reference_batches
from .uncertainty import paired_bootstrap


FULL_POOL_ROWS = 10_320_219
MEDIUM_TUNE_COUNT = 12_000
MEDIUM_TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
V1_TUNE_F05 = 0.9689921356332671
RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
OPTIONAL_FEATURES = ("phonetic_score", "phonetic_only", "dense_score", "dense_only",
                     "lexical_scores_missing", "lexical_retrieval_score")
SHARED_UNION_COLUMNS = {"left_idx", "right_idx", "dense_score", "retrieval_score",
                        "retrieval_rank", "dense_hit", "phonetic_key_hit"}
REPO_ROOT = Path(__file__).resolve().parents[4]
GIB = 1024 ** 3
POD_COHOST_MAX_PAIRS = 10_000_000  # unmeasured ceiling; not a claim of feasibility


def _cgroup_int(path: str) -> int | None:
    try:
        value = Path(path).read_text().strip()
        return int(value) if value != "max" else None
    except (OSError, ValueError):
        return None


def effective_available_bytes(host_available: int, limit: int | None,
                              used: int | None, inactive_file: int) -> int:
    """Conservatively count only inactive file cache as reclaimable in a cgroup."""
    if limit is None or used is None or limit >= 1 << 60:
        return host_available
    return max(0, min(host_available, limit - used + inactive_file))


def resource_snapshot() -> dict:
    """Host RAM plus cgroup v1/v2 quota, avoiding pod host-free illusions."""
    host = psutil.virtual_memory()
    v1 = Path("/sys/fs/cgroup/memory/memory.stat")
    v2 = Path("/sys/fs/cgroup/memory.stat")
    if v1.is_file():
        stat = v1
        limit = _cgroup_int("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        used = _cgroup_int("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        inactive_key = "total_inactive_file"
        quota = _cgroup_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _cgroup_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if quota is not None and quota < 0:
            quota = None
    elif v2.is_file():
        stat = v2
        limit = _cgroup_int("/sys/fs/cgroup/memory.max")
        used = _cgroup_int("/sys/fs/cgroup/memory.current")
        inactive_key = "inactive_file"
        cpu_max = Path("/sys/fs/cgroup/cpu.max")
        if cpu_max.is_file():
            cpu_values = cpu_max.read_text().split()
            quota = int(cpu_values[0]) if cpu_values[0] != "max" else None
            period = int(cpu_values[1])
        else:
            quota = period = None
    else:
        stat, limit, used, inactive_key, quota, period = None, None, None, "", None, None
    inactive = 0
    if stat is not None:
        for line in stat.read_text().splitlines():
            name, value = line.split()
            if name == inactive_key:
                inactive = int(value)
                break
    swap = psutil.swap_memory()
    return {"host_total_bytes": host.total, "host_available_bytes": host.available,
            "cgroup_limit_bytes": limit if limit is not None and limit < 1 << 60 else None,
            "cgroup_used_bytes": used if limit is not None and limit < 1 << 60 else None,
            "cgroup_inactive_file_bytes": inactive,
            "effective_available_bytes": effective_available_bytes(host.available, limit, used, inactive),
            "cgroup_cpu_quota": quota / period if quota is not None and period else None,
            "swap_used_bytes": swap.used, "swap_in_bytes": swap.sin, "swap_out_bytes": swap.sout}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def log(message: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {message}", flush=True)


def check_memory(min_available_gb: float, stage: str) -> None:
    available = resource_snapshot()["effective_available_bytes"]
    if available < min_available_gb * GIB:
        raise MemoryError(f"{stage}: only {available / GIB:.2f} GiB available (host/cgroup minimum); "
                          f"required floor is {min_available_gb:.2f} GiB")


class MemoryMonitor:
    """Sample native-fit RSS, physical headroom and swap activity."""

    def __init__(self):
        self._stop = threading.Event()
        self.peak_rss = 0
        self.peak_process_swap = 0
        self.lowest_available = 2 ** 63 - 1
        self.initial = resource_snapshot()
        self.final = self.initial
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while True:
            self.peak_rss = max(self.peak_rss, process.memory_info().rss)
            self.peak_process_swap = max(self.peak_process_swap, getattr(process.memory_full_info(), "swap", 0))
            self.final = resource_snapshot()
            self.lowest_available = min(self.lowest_available, self.final["effective_available_bytes"])
            if self._stop.wait(0.5):
                break

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join()

    def report(self) -> dict:
        return {"peak_process_rss_gib": self.peak_rss / GIB,
                "lowest_effective_available_gib": self.lowest_available / GIB,
                "peak_process_swap_bytes": self.peak_process_swap,
                "system_swap_out_delta_bytes": max(0, self.final["swap_out_bytes"] - self.initial["swap_out_bytes"]),
                "system_swap_used_delta_bytes": self.final["swap_used_bytes"] - self.initial["swap_used_bytes"],
                "cgroup_limit_bytes": self.initial["cgroup_limit_bytes"]}


def feature_columns(pair_columns) -> list[str]:
    """The exact tabular_sweep/predict.feature_batch order, including numeric."""
    pair_columns = set(pair_columns)
    if {"left_idx", "right_idx"} - pair_columns:
        raise ValueError("Candidate pairs require left_idx and right_idx")
    return (BASE_FEATURE_NAMES + CONTEXT_FEATURE_NAMES
            + [name for name in OPTIONAL_FEATURES if name in pair_columns] + NUMERIC_NAMES)


def union_feature_columns(pair_columns) -> list[str]:
    """P2.5 and medium TUNE share dense/key evidence, not legacy lexical TF-IDF."""
    if SHARED_UNION_COLUMNS - set(pair_columns):
        raise ValueError(f"Scored P2 union lacks shared feature provenance: "
                         f"{sorted(SHARED_UNION_COLUMNS - set(pair_columns))}")
    return feature_columns(set(pair_columns) | set(OPTIONAL_FEATURES))


def check_reference_feature_order(columns: list[str], checkpoint: Path | None) -> dict | None:
    if checkpoint is None:
        return None
    digest = sha256_file(checkpoint)
    bundle = joblib.load(checkpoint)
    prior = list(bundle["feature_columns"])
    if len(prior) != len(set(prior)) or not prior:
        raise ValueError("Reference checkpoint has duplicate/empty feature columns")
    # A union adds retrieval-channel columns before the numeric suffix. The
    # projection onto the earlier feature vocabulary must still be identical.
    if [name for name in columns if name in prior] != prior:
        raise ValueError("Original tree feature order/provenance differs from the new feature order")
    if set(columns) == set(prior) and columns != prior:
        raise ValueError("Features match the original tree names but not their order")
    return {"path": str(checkpoint.resolve()), "sha256": digest,
            "depth": bundle.get("record", {}).get("depth"),
            "negative_weight": bundle.get("record", {}).get("negative_weight"),
            "ordered_feature_columns": prior, "extra_union_columns": [x for x in columns if x not in prior]}


def _pairs_path(directory: Path, override: str | None) -> Path:
    if override:
        return Path(override).resolve()
    for name in ("candidate_pairs.parquet", "pairs.parquet"):
        if (directory / name).is_file():
            return (directory / name).resolve()
    raise FileNotFoundError(f"No completed candidate_pairs.parquet or pairs.parquet in {directory}")


def _source_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "sha256": sha256_file(path)}


def _full_pool_count(metadata: dict) -> int | None:
    for key in ("mining_full_pool_records", "full_pool_records", "full_competing_target_pool"):
        if key in metadata:
            return int(metadata[key])
    return None


def _ipc_records(source: Path, destination: Path, columns: tuple[str, ...]) -> pa.Table:
    """Stream Parquet to mmap-able Arrow IPC; never materialize all pool text."""
    expected = {"source": _source_identity(source), "columns": list(columns)}
    marker = destination.with_suffix(".json")
    if destination.exists() or marker.exists():
        if not destination.is_file() or not marker.is_file() or json.loads(marker.read_text()) != expected:
            raise ValueError(f"Record cache is incomplete or belongs to different source: {destination}")
    else:
        parquet = pq.ParquetFile(source)
        if set(columns) - set(parquet.schema_arrow.names):
            raise ValueError(f"Missing record fields in {source}: {set(columns)-set(parquet.schema_arrow.names)}")
        temporary = destination.with_name(destination.name + ".tmp")
        schema = pa.schema([(name, pa.string()) for name in columns])
        with pa.OSFile(str(temporary), "wb") as sink, ipc.new_file(sink, schema) as writer:
            for batch in parquet.iter_batches(batch_size=50_000, columns=list(columns), use_threads=False):
                arrays = [pc.fill_null(pc.cast(batch.column(i), pa.string()), "")
                          for i in range(len(columns))]
                writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
        temporary.replace(destination)
        write_json(marker, expected)
    # The table retains references to the Arrow memory map after the reader dies.
    return ipc.open_file(pa.memory_map(str(destination), "r")).read_all()


@dataclass
class Dataset:
    directory: Path
    pairs_path: Path
    metadata: dict
    inputs: dict
    reference: pa.Table
    pool: pa.Table
    ref_ids: list[str]
    country: np.ndarray
    splits: np.ndarray
    truth: dict[str, list[str]]
    features: list[str]
    truth_pair_keys: np.ndarray | None = None  # TRAIN-only P2 ready: all real links, even retrieval misses
    truth_counts: np.ndarray | None = None
    pair_policy: str = "as_saved"


def _pair_keys(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if (len(left) != len(right) or np.any(left < 0) or np.any(right < 0)
            or np.any(left >= 2 ** 32) or np.any(right >= 2 ** 32)):
        raise ValueError("Pair indices must fit exact unsigned 32-bit positional keys")
    return (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)


def _tune_union_files(directory: Path, pairs_path: Path, complete: dict) -> dict:
    """Accept P2's TUNE union only after its separate dense-context pass commits."""
    if (complete.get("state") != "complete" or complete.get("references") != MEDIUM_TUNE_COUNT
            or complete.get("full_target_pool_rows") != FULL_POOL_ROWS
            or complete.get("reference_ids_sha256") != MEDIUM_TUNE_ID_SHA256
            or complete.get("positive_candidate_injection") is not False
            or complete.get("truth_used_in_retrieval") is not False):
        raise ValueError("TUNE union completion marker lacks the fixed, label-free full-pool identity")
    original = directory / "candidate_pairs.parquet"
    raw = directory / "candidate_pairs_raw.parquet"
    if (not original.is_file() or not raw.is_file() or
            sha256_file(raw) != complete["raw_pairs"]["sha256"] or
            pq.ParquetFile(original).metadata.num_rows != complete["raw_pairs"]["rows"]):
        raise ValueError("The completed TUNE union lost its original mapped candidates")
    context = pairs_path.with_suffix(".context.json")
    if not context.is_file():
        raise ValueError("P2 TUNE union requires a completed score-context Parquet/manifest")
    scored = json.loads(context.read_text())
    if (scored.get("state") != "complete" or
            scored.get("positive_candidate_injection") is not False or
            scored["inputs"]["pairs"]["sha256"] != sha256_file(original) or
            scored["pair_file"]["sha256"] != sha256_file(pairs_path) or
            scored["pair_file"].get("rows", complete["raw_pairs"]["rows"]) != complete["raw_pairs"]["rows"]):
        raise ValueError("Scored TUNE pair manifest is incomplete or belongs to another union")
    if pq.ParquetFile(pairs_path).metadata.num_rows != complete["raw_pairs"]["rows"]:
        raise ValueError("Scored TUNE union omitted or added candidate edges")
    return {"candidate_pairs": pq.ParquetFile(pairs_path).metadata.num_rows,
            "references": MEDIUM_TUNE_COUNT,
            "pool_records": pq.ParquetFile(directory / "pool.parquet").metadata.num_rows,
            "full_pool_records": FULL_POOL_ROWS, "positive_candidate_injection": False,
            "dataset_kind": "full_pool_medium_tune_union",
            "context_manifest": _source_identity(context)}


def load_dataset(directory: str | Path, pairs_override: str | None, cache: Path,
                 *, require_union: bool, min_available_gb: float) -> Dataset:
    directory = Path(directory).resolve()
    pairs_path = _pairs_path(directory, pairs_override)
    files = {name: directory / name for name in ("reference.parquet", "pool.parquet", "truth.json",
                                                "complete.json")}
    metadata_path = directory / "metadata.json"
    for path in (*files.values(), pairs_path):
        if not path.is_file():
            raise FileNotFoundError(f"P2/dense artifact is not complete: {path}")
    complete = json.loads(files["complete.json"].read_text())
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        files["metadata.json"] = metadata_path
        if complete.get("state") != "complete" or complete.get("metadata") != metadata:
            raise ValueError("Candidate union needs a matching COMPLETE marker and metadata")
        if complete.get("config", {}).get("labels_used_for_mining", False) is not False:
            raise ValueError("TUNE candidate retrieval used labels")
    else:
        if not require_union:
            raise FileNotFoundError(metadata_path)
        metadata = _tune_union_files(directory, pairs_path, complete)
    if _full_pool_count(metadata) != FULL_POOL_ROWS or metadata.get("positive_candidate_injection") is not False:
        raise ValueError("Candidates must be mined against all 10,320,219 supplied targets without truth-edge injection")
    if require_union and "union" not in str(metadata.get("dataset_kind", "")).lower():
        raise ValueError("Full P3.1 training requires the completed P2 candidate union, not dense-only candidates")
    parquet = pq.ParquetFile(pairs_path)
    columns = (union_feature_columns(parquet.schema_arrow.names) if require_union
               else feature_columns(parquet.schema_arrow.names))
    if "right_idx" not in parquet.schema_arrow.names or parquet.metadata.num_rows != metadata["candidate_pairs"]:
        raise ValueError("Candidate row count/schema differs from the completion metadata")
    cache.mkdir(parents=True, exist_ok=False)
    ref = _ipc_records(files["reference.parquet"], cache / "reference.arrow",
                       (*RECORD_COLUMNS, "split"))
    pool = _ipc_records(files["pool.parquet"], cache / "pool.arrow", RECORD_COLUMNS)
    ref_ids = ref.column("entity_id").to_pylist()
    splits = np.asarray(ref.column("split").to_pylist(), dtype=object)
    country = np.asarray(ref.column("country").to_pylist(), dtype=object)
    if (len(ref_ids) != len(set(ref_ids)) or not all(ref_ids)
            or pc.count_distinct(pool.column("entity_id")).as_py() != len(pool)
            or pc.any(pc.equal(pool.column("entity_id"), "")).as_py()):
        raise ValueError("Reference/target entity IDs must be nonempty and unique")
    if set(splits) - {"train", "tune"} or not set(splits):
        raise ValueError("Eval/HOLDOUT-2 references or unknown splits are forbidden in a tree fit")
    if len(ref) != metadata["references"] or len(pool) != metadata["pool_records"]:
        raise ValueError("Reference or compact target count differs from metadata")
    if (metadata.get("reference_id_sha256", ids_digest(ref_ids)) != ids_digest(ref_ids)):
        raise ValueError("Reference IDs differ from the completed retrieval metadata")
    check_memory(min_available_gb, "Loading memory-mapped candidate records")
    # Do not open truth until the completion marker and split exclusion pass.
    truth = json.loads(files["truth.json"].read_text())
    if set(truth) != set(ref_ids) or any(not isinstance(v, list) for v in truth.values()):
        raise ValueError("Truth must cover exactly the TRAIN/TUNE references, including all singletons")
    check_memory(min_available_gb, "Reading selected TRAIN/TUNE truth")
    inputs = {name: _source_identity(path) for name, path in {**files, "pairs": pairs_path}.items()}
    return Dataset(directory, pairs_path, metadata, inputs, ref, pool, ref_ids, country, splits, truth,
                   columns, pair_policy="dense_key_shared" if require_union else "as_saved")


def load_p2_ready(path: str | Path, cache: Path, min_available_gb: float) -> Dataset:
    """Validate P2's atomic TRAIN-only handoff without needing a giant truth JSON."""
    path = Path(path).resolve()
    ready = json.loads(path.read_text())
    required = {"state": "complete", "split": "train", "dense_mining_complete": True,
                "train_truth_join_complete": True, "positive_candidate_injection": False,
                "full_pool_records": FULL_POOL_ROWS}
    if any(type(ready.get(key)) is not type(value) or ready[key] != value for key, value in required.items()):
        raise ValueError("P2 full TRAIN requires an atomic, truth-joined, full-pool union ready marker")
    if ready.get("truth_used_in_retrieval", False) is not False:
        raise ValueError("P2 retrieval used truth before fixing candidate edges")
    named = {}
    for name in ("reference", "pool", "candidate_pairs", "train_truth_pairs"):
        item = ready.get("files", {}).get(name)
        if not isinstance(item, dict) or not all(k in item for k in ("path", "sha256", "rows")):
            raise ValueError(f"P2 ready is missing the {name} file identity")
        candidate = Path(item["path"])
        candidate = candidate if candidate.is_absolute() else path.parent / candidate
        if (not candidate.is_file() or sha256_file(candidate) != item["sha256"]
                or pq.ParquetFile(candidate).metadata.num_rows != item["rows"]):
            raise ValueError(f"P2 {name} file differs from the committed hash/row count")
        named[name] = candidate.resolve()
    pair_schema = pq.ParquetFile(named["candidate_pairs"]).schema_arrow.names
    columns = union_feature_columns(pair_schema)
    cache.mkdir(parents=True, exist_ok=False)
    ref = _ipc_records(named["reference"], cache / "reference.arrow", (*RECORD_COLUMNS, "split"))
    pool = _ipc_records(named["pool"], cache / "pool.arrow", RECORD_COLUMNS)
    ref_ids = ref.column("entity_id").to_pylist()
    splits = np.asarray(ref.column("split").to_pylist(), dtype=object)
    country = np.asarray(ref.column("country").to_pylist(), dtype=object)
    if (len(ref_ids) != len(set(ref_ids)) or not all(ref_ids) or set(splits) != {"train"}
            or ids_digest(ref_ids) != ready.get("train_reference_id_sha256")
            or pc.count_distinct(pool.column("entity_id")).as_py() != len(pool)):
        raise ValueError("P2 TRAIN records/IDs disagree with the sealed ready marker")
    check_memory(min_available_gb, "Loading P2 records")
    # Compact uint64 positional keys represent *every* true TRAIN link; truth
    # targets absent from candidates remain in the denominator, not as edges.
    truth = pq.read_table(named["train_truth_pairs"], columns=["left_idx", "right_idx"]).to_pandas()
    left = truth.left_idx.to_numpy(dtype=np.int64)
    right = truth.right_idx.to_numpy(dtype=np.int64)
    if (len(left) == 0 or left.min() < 0 or left.max() >= len(ref) or
            right.min() < 0 or right.max() >= len(pool)):
        raise ValueError("P2 TRAIN truth positions are out of bounds")
    keys = np.sort(_pair_keys(left, right))
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError("P2 TRAIN truth has duplicate pair IDs")
    counts = np.bincount(left, minlength=len(ref)).astype(np.int32)
    del truth, left, right
    check_memory(min_available_gb, "Indexing complete TRAIN truth for accounting")
    metadata = {"dataset_kind": "p2_full_union", "references": len(ref), "pool_records": len(pool),
                "candidate_pairs": ready["files"]["candidate_pairs"]["rows"],
                "full_pool_records": FULL_POOL_ROWS, "positive_candidate_injection": False,
                "true_links": int(counts.sum())}
    inputs = {name: _source_identity(item) for name, item in {"ready": path, **named}.items()}
    return Dataset(path.parent, named["candidate_pairs"], metadata, inputs, ref, pool, ref_ids,
                   country, splits, {}, columns, truth_pair_keys=keys, truth_counts=counts,
                   pair_policy="dense_key_shared")


def _select_references(data: Dataset, split: str, limit: int | None, seed: int) -> np.ndarray:
    positions = np.flatnonzero(data.splits == split)
    if not len(positions):
        raise ValueError(f"No {split} references in {data.directory}")
    if limit is not None and len(positions) > limit:
        # Stable across file row ordering, while keeping every candidate for an S1.
        ordered = np.asarray(sorted(positions, key=lambda i: data.ref_ids[i]), dtype=np.int64)
        positions = np.sort(np.random.default_rng(seed).choice(ordered, size=limit, replace=False))
    selected = np.zeros(len(data.ref_ids), dtype=bool)
    selected[positions] = True
    return selected


def validate_populations(train: Dataset, tune: Dataset, train_mask: np.ndarray,
                         tune_mask: np.ndarray, mode: str, excluded_file: Path) -> dict:
    train_ids = [train.ref_ids[i] for i in np.flatnonzero(train_mask)]
    tune_ids = [tune.ref_ids[i] for i in np.flatnonzero(tune_mask)]
    if len(tune_ids) != MEDIUM_TUNE_COUNT or ids_digest(tune_ids) != MEDIUM_TUNE_ID_SHA256:
        raise ValueError("TUNE must be the identical saved 12,000-S1 medium split")
    if set(train_ids) & set(tune_ids):
        raise ValueError("TRAIN and TUNE contain the same business ID")
    excluded = None
    if mode in {"full", "p2-screen"}:
        reservation = json.loads((excluded_file.parent / "reservation.json").read_text())
        if reservation.get("verification", {}).get("labels_read") is not False:
            raise ValueError("HOLDOUT-2 reservation must be label-free")
        expected_count = reservation["future_raw_train_tune"]["eligible_count"]
        if len(train.ref_ids) != expected_count or (mode == "full" and len(train_ids) != expected_count):
            raise ValueError(f"Full-scale P2 TRAIN needs all {expected_count:,} eligible S1s")
        expected = reservation["future_raw_train_tune"]["excluded_file_sha256"]
        if sha256_file(excluded_file) != expected:
            raise ValueError("HOLDOUT-2/earlier split exclusion file changed")
        excluded = set(excluded_file.read_text().splitlines())
        held = reservation["holdout2"]
        reserved_path = excluded_file.parent / held["id_file"]
        reserved = set(reserved_path.read_text().splitlines())
        if (len(reserved) != held["count"] or held["count"] != 50_000
                or sha256_file(reserved_path) != held["id_file_sha256"]
                or not reserved <= excluded):
            raise ValueError("HOLDOUT-2 IDs are not wholly covered by the sealed TRAIN exclusion")
        if set(train_ids) & excluded:
            raise ValueError("P2 TRAIN includes prior split or reserved HOLDOUT-2 IDs")
    return {"train_references": len(train_ids), "train_ids_sha256": ids_digest(train_ids),
            "tune_references": len(tune_ids), "tune_ids_sha256": ids_digest(tune_ids),
            "train_tune_overlap": 0,
            "train_exclusion_file": {"path": str(excluded_file.resolve()), "sha256": sha256_file(excluded_file),
                                     "excluded_ids": len(excluded)} if excluded is not None else None}


def _selected_batches(data: Dataset, selected: np.ndarray, args):
    for batch in iter_reference_batches(data.pairs_path, len(data.ref_ids), len(data.pool),
                                        args.reference_batch_size, args.pair_read_batch_size,
                                        args.max_pairs_per_batch):
        if batch.pairs.empty:
            continue
        keep = selected[batch.pairs.left_idx.to_numpy(dtype=np.int64)]
        if np.any(keep):
            yield batch, batch.pairs.loc[keep].reset_index(drop=True)


def pair_labels(data: Dataset, pairs: pd.DataFrame, *,
                include_target_ids: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Join only retrieved edges to their S1 truth; missed links remain misses."""
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    targets = None
    if data.truth_pair_keys is not None:
        query = _pair_keys(left, right)
        positions = np.searchsorted(data.truth_pair_keys, query)
        labels = np.zeros(len(query), dtype=np.bool_)
        valid = positions < len(data.truth_pair_keys)
        labels[valid] = data.truth_pair_keys[positions[valid]] == query[valid]
    else:
        targets = data.pool.column("entity_id").take(pa.array(right)).to_pylist()
        truth_sets = {i: set(data.truth[data.ref_ids[i]]) for i in np.unique(left)}
        labels = np.fromiter((target in truth_sets[i] for i, target in zip(left, targets)),
                             dtype=np.bool_, count=len(pairs))
    if "label" in pairs and not np.array_equal(labels, pairs.label.to_numpy(dtype=bool)):
        raise ValueError("P2 candidate labels disagree with independent TRAIN truth-pair accounting")
    if targets is None and include_target_ids:
        targets = data.pool.column("entity_id").take(pa.array(right)).to_pylist()
    return labels, np.asarray(targets, dtype=object) if targets is not None else None


def prepare_exact_features(data: Dataset, start: int, end: int,
                           pairs: pd.DataFrame) -> pd.DataFrame:
    """Remap one complete S1 interval to compact records, preserving context."""
    if pairs.empty or pairs.left_idx.min() < start or pairs.left_idx.max() >= end:
        raise ValueError("Feature preparation requires complete, nonempty reference groups")
    reference = data.reference.slice(start, end - start).select(RECORD_COLUMNS).to_pandas().fillna("")
    rights, inverse = np.unique(pairs.right_idx.to_numpy(dtype=np.int64), return_inverse=True)
    pool = data.pool.take(pa.array(rights)).to_pandas().fillna("")
    local = pairs.copy()
    local["left_idx"] = local.left_idx.to_numpy(dtype=np.int64) - start
    local["right_idx"] = inverse
    if data.pair_policy == "dense_key_shared":
        # P2.5 has no legacy char-TF-IDF scores; medium TUNE does. Normalize
        # both sides to the same label-free dense/key representation before
        # recomputing name/address score gaps and candidate counts.
        if SHARED_UNION_COLUMNS - set(local):
            raise ValueError("P2 union lost dense/key hit flags or scored context")
        if not np.array_equal(local.retrieval_score.to_numpy(dtype=np.float32),
                              local.dense_score.to_numpy(dtype=np.float32)):
            raise ValueError("P2 TRAIN/TUNE retrieval_score must be the full-union dense cosine")
        local["name_score"] = np.float32(0)
        local["address_score"] = np.float32(0)
        local["phonetic_score"] = np.float32(0)
        local["lexical_retrieval_score"] = np.float32(0)
        local["lexical_scores_missing"] = True
        local["phonetic_only"] = local.phonetic_key_hit.to_numpy(dtype=bool)
        local["dense_only"] = local.dense_hit.to_numpy(dtype=np.float32)
    result = feature_batch(reference, pool, local, data.features)
    if list(result.columns) != data.features or result.shape != (len(pairs), len(data.features)):
        raise ValueError("Exact feature order or candidate row alignment changed")
    if not np.isfinite(result.to_numpy(dtype=np.float32)).all():
        raise ValueError("Missing/nonfinite P2 retrieval channels or numeric/string features")
    return result


def count_training_pairs(data: Dataset, selected: np.ndarray, args) -> tuple[int, int, int]:
    total = positives = negative_references = 0
    for batch, pairs in _selected_batches(data, selected, args):
        labels, _ = pair_labels(data, pairs, include_target_ids=False)
        total += len(labels)
        positives += int(labels.sum())
        negative_references += len(np.unique(pairs.left_idx.to_numpy(dtype=np.int64)[~labels]))
        if batch.end % 100_000 < args.reference_batch_size:
            log(f"Counted {batch.end:,}/{len(data.ref_ids):,} S1s; {total:,} TRAIN candidate pairs")
        check_memory(args.min_available_gb, "Counting original candidate pairs")
    if not total or not positives or total == positives:
        raise ValueError("TRAIN retrieved candidates need both positive and hard-negative pairs")
    return total, positives, negative_references


def pair_budget(positive_count: int, pair_count: int, features: int, args,
                mandatory_negatives: int = 0) -> dict:
    resources = resource_snapshot()
    available = resources["effective_available_bytes"]
    usable = min(int(args.ram_budget_gb * GIB), max(0, available - int(args.min_available_gb * GIB)))
    # Quantized rows cost ~one byte/feature; allow four times the original
    # float32 matrix plus per-row overhead for CatBoost's CPU training buffers.
    bytes_per_pair = 4 * features * 4 + 64
    memory_cap = usable // bytes_per_pair
    mode = getattr(args, "mode", "screen")
    site = getattr(args, "execution_site", "local")
    verified_cap = None
    verification = None
    if mode in {"full", "p2-screen"}:
        if site == "local":
            report = Path(args.resource_report)
            verified_cap = no_swap_verified_pairs(json.loads(report.read_text()))
            verification = {"path": str(report.resolve()), "sha256": sha256_file(report),
                            "maximum_measured_pairs_without_child_swap": verified_cap}
            if not verified_cap:
                raise MemoryError("No real TRAIN CatBoost pair count has passed the local no-swap probe")
        else:
            if (resources["cgroup_limit_bytes"] is None or
                    not 45 * GIB <= resources["cgroup_limit_bytes"] <= 55 * GIB or
                    resources["cgroup_cpu_quota"] is None or resources["cgroup_cpu_quota"] > 8 or
                    args.threads > 3 or args.min_available_gb < 12 or
                    args.ram_budget_gb > 22 or args.max_process_rss_gb > 22):
                raise ValueError("Pod co-residency needs its measured ~50 GiB / <=8 CPU cgroup, "
                                 "<=3 CatBoost threads, <=22 GiB fit/RSS budget and >=12 GiB spare")
            verified_cap = min(POD_COHOST_MAX_PAIRS, 4_000_000) if mode == "p2-screen" else POD_COHOST_MAX_PAIRS
            verification = {"pod_cgroup_only": True, "not_yet_measured_at_full_scale": True,
                            "conservative_requested_pair_ceiling": POD_COHOST_MAX_PAIRS}
    cap = min(pair_count, args.max_train_pairs, memory_cap,
              verified_cap if verified_cap is not None else pair_count)
    if cap < positive_count + mandatory_negatives:
        raise MemoryError(f"{positive_count:,} retrieved TRAIN positives and "
                          f"{mandatory_negatives:,} hardest-per-S1 negatives exceed the "
                          f"{cap:,}-pair measured/profile cap; do not silently drop positives. "
                          "Use a bounded pod CPU fit if the workstation has not passed a real no-swap probe.")
    negatives = pair_count - positive_count
    quota = cap - positive_count - mandatory_negatives
    remaining_negatives = negatives - mandatory_negatives
    if remaining_negatives < 0:
        raise ValueError("A reference cannot require more mandatory negatives than real negatives")
    if quota >= remaining_negatives:
        probability = 1.0
    else:
        # Leave six standard deviations of headroom: selected rows stay <= cap
        # without a row-order-dependent 'first N' truncation.
        probability = max(0.0, (quota - 6 * np.sqrt(quota) - 128) / remaining_negatives)
    return {"input_train_pairs": pair_count, "retrieved_positive_pairs": positive_count,
            "mandatory_negative_s1s": mandatory_negatives,
            "execution_site": site, "resource_snapshot": resources,
            "verified_pair_cap": verification, "available_before_training_gib": available / GIB,
            "configured_ram_budget_gib": args.ram_budget_gb,
            "min_available_gib": args.min_available_gb, "bytes_per_pair_conservative": bytes_per_pair,
            "memory_pair_cap": int(memory_cap), "max_requested_pairs": args.max_train_pairs,
            "actual_pair_cap": int(cap), "negative_keep_probability": float(probability),
            "sampling": "all retrieved positives + hardest negative per S1 + seeded stable positional-pair-hash extra negatives"}


def no_swap_verified_pairs(report: dict) -> int:
    """Only completed, TRAIN-only numeric CatBoost measurements justify a local cap."""
    if (report.get("no_tune_or_holdout_model_score") is not True or
            "TRAIN-only medium dense full-target-pool" not in report.get("scope", "")):
        raise ValueError("Resource report is not a real TRAIN-only full-pool memory probe")
    passed = [int(item["train_pairs"]) for item in report.get("measurements", [])
              if item.get("status") == "pass" and item.get("exitcode") == 0
              and item.get("peak_child_swap_bytes") == 0
              and item.get("system_swap_out_delta_bytes", 1) == 0
              and {stage.get("stage") for stage in item.get("stages", [])} == {"quantized", "fitted"}]
    return max(passed, default=0)


def mandatory_negative_rows(pairs: pd.DataFrame, labels: np.ndarray) -> np.ndarray:
    """Retain each S1's hardest real negative, including every true singleton."""
    chosen = np.zeros(len(pairs), dtype=bool)
    rows = np.flatnonzero(~labels)
    if not len(rows):
        return chosen
    left = pairs.left_idx.to_numpy(dtype=np.int64)[rows]
    right = pairs.right_idx.to_numpy(dtype=np.int64)[rows]
    priority = (pairs.hard_negative_priority.to_numpy(dtype=np.int64)[rows]
                if "hard_negative_priority" in pairs else np.zeros(len(rows), dtype=np.int64))
    score = (pairs.dense_score.to_numpy(dtype=np.float32)[rows]
             if "dense_score" in pairs else np.zeros(len(rows), dtype=np.float32))
    ordering = np.lexsort((right, -score, priority, left))
    ordered_left = left[ordering]
    first = np.r_[True, ordered_left[1:] != ordered_left[:-1]]
    chosen[rows[ordering[first]]] = True
    return chosen


def sampled_negatives(pairs: pd.DataFrame, labels: np.ndarray, probability: float,
                      seed: int, mandatory: np.ndarray | None = None) -> np.ndarray:
    """Deterministic per pair regardless of Parquet row-group/chunk boundaries."""
    if probability >= 1:
        return np.ones(len(labels), dtype=bool)
    if mandatory is None:
        mandatory = mandatory_negative_rows(pairs, labels)
    if mandatory.shape != labels.shape or np.any(mandatory & labels):
        raise ValueError("Mandatory negative selection is not pair-aligned")
    fingerprint = pd.util.hash_pandas_object(pairs[["left_idx", "right_idx"]], index=False).to_numpy(np.uint64).copy()
    with np.errstate(over="ignore"):
        fingerprint ^= np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15)
        fingerprint ^= fingerprint >> np.uint64(30)
        fingerprint *= np.uint64(0xBF58476D1CE4E5B9)
        fingerprint ^= fingerprint >> np.uint64(27)
        fingerprint *= np.uint64(0x94D049BB133111EB)
        fingerprint ^= fingerprint >> np.uint64(31)
    cutoff = np.uint64(int(probability * (2 ** 64 - 1)))
    return labels | mandatory | (fingerprint <= cutoff)


def _pair_table(data: Dataset, pairs: pd.DataFrame, labels: np.ndarray,
                targets: np.ndarray, reference_local: np.ndarray) -> pd.DataFrame:
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    return pd.DataFrame({"left_idx": left, "right_idx": pairs.right_idx.to_numpy(dtype=np.int64),
                         "reference_local_idx": reference_local[left],
                         "source1_entity_id": [data.ref_ids[i] for i in left],
                         "candidate_entity_id": targets, "is_match": labels})


def _write_rows(writer: pq.ParquetWriter | None, frame: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
    elif writer.schema != table.schema:
        raise ValueError("A chunk changed its feature or pair schema")
    writer.write_table(table)
    return writer


def prepare_training(data: Dataset, selected: np.ndarray, budget: dict, output: Path, args) -> dict:
    positions = np.flatnonzero(selected)
    local_index = np.full(len(data.ref_ids), -1, dtype=np.int32)
    local_index[positions] = np.arange(len(positions), dtype=np.int32)
    feature_path = output / "train_features.tsv"
    description = output / "train_features.cd"
    description.write_text("0\tLabel\n" + "".join(
        f"{i}\tNum\t{name}\n" for i, name in enumerate(data.features, 1)), encoding="utf-8")
    pair_path = output / "train_selected_pairs.parquet"
    writer = None
    count = positives = mandatory_refs = 0
    prep_started = time.monotonic()
    last_log = -1
    try:
        with feature_path.open("w", encoding="utf-8") as stream:
            for batch, pairs in _selected_batches(data, selected, args):
                labels, target_ids = pair_labels(data, pairs, include_target_ids=False)
                # Candidate-group features must be computed BEFORE any negative
                # subsampling; score gaps and candidate count use the full group.
                X = prepare_exact_features(data, batch.start, batch.end, pairs)
                mandatory = mandatory_negative_rows(pairs, labels)
                keep = sampled_negatives(pairs, labels, budget["negative_keep_probability"], args.seed, mandatory)
                mandatory_refs += int(mandatory.sum())
                if np.any(keep):
                    matrix = X.to_numpy(dtype=np.float32)[keep]
                    y = labels[keep].astype(np.int8)
                    np.savetxt(stream, np.column_stack((y, matrix)), fmt="%.9g", delimiter="\t")
                    if target_ids is None:
                        right = pairs.right_idx.to_numpy(dtype=np.int64)[keep]
                        target_ids = np.asarray(data.pool.column("entity_id").take(pa.array(right)).to_pylist(),
                                                dtype=object)
                    else:
                        target_ids = target_ids[keep]
                    writer = _write_rows(writer, _pair_table(data, pairs.loc[keep], labels[keep],
                        target_ids, local_index), pair_path)
                    count += int(keep.sum())
                    positives += int(y.sum())
                if args.mode == "p2-screen" and batch.end // 25_000 > last_log:
                    last_log = batch.end // 25_000
                    log(f"Prepared {count:,} TRAIN rows / {batch.end:,} source S1s in "
                        f"{time.monotonic()-prep_started:.1f}s")
                elif batch.end % 100_000 < args.reference_batch_size:
                    log(f"Prepared TRAIN {batch.end:,}/{len(data.ref_ids):,} S1s; {count:,} retained pairs")
                check_memory(args.min_available_gb, "Preparing complete TRAIN feature groups")
                if (args.mode == "p2-screen" and time.monotonic() - prep_started > args.max_feature_prep_seconds):
                    raise TimeoutError("Bounded TRAIN feature preparation exceeded the 45-minute deadline gate")
    finally:
        if writer is not None:
            writer.close()
    if (writer is None or count > budget["actual_pair_cap"]
            or positives != budget["retrieved_positive_pairs"]
            or mandatory_refs != budget["mandatory_negative_s1s"]):
        raise ValueError("Training sample exceeded its RAM cap or dropped a positive candidate")
    all_truth_links = (int(data.truth_counts[selected].sum()) if data.truth_counts is not None
                       else sum(len(data.truth[data.ref_ids[i]]) for i in positions))
    return {"selected_pairs": count, "selected_positives": positives,
            "selected_negatives": count - positives, "mandatory_negative_s1s": mandatory_refs,
            "selected_pair_ids": _source_identity(pair_path),
            "unretrieved_train_true_links": all_truth_links - budget["retrieved_positive_pairs"],
            "feature_text_bytes": feature_path.stat().st_size, "feature_text_sha256": sha256_file(feature_path),
            "catboost_column_description_sha256": sha256_file(description)}


def prepare_tuning(data: Dataset, selected: np.ndarray, output: Path, args) -> dict:
    positions = np.flatnonzero(selected)
    local_index = np.full(len(data.ref_ids), -1, dtype=np.int32)
    local_index[positions] = np.arange(len(positions), dtype=np.int32)
    feature_path, pair_path = output / "tune_features.parquet", output / "tune_pairs.parquet"
    feature_writer = pair_writer = None
    total = found = 0
    try:
        for batch, pairs in _selected_batches(data, selected, args):
            labels, targets = pair_labels(data, pairs)
            X = prepare_exact_features(data, batch.start, batch.end, pairs)
            feature_writer = _write_rows(feature_writer, X, feature_path)
            pair_writer = _write_rows(pair_writer, _pair_table(data, pairs, labels, targets, local_index), pair_path)
            total += len(pairs)
            found += int(labels.sum())
            check_memory(args.min_available_gb, "Preparing all TUNE candidate groups")
    finally:
        if feature_writer is not None:
            feature_writer.close()
        if pair_writer is not None:
            pair_writer.close()
    if pair_writer is None or total != pq.ParquetFile(feature_path).metadata.num_rows:
        raise ValueError("TUNE pairs and exact features are missing or misaligned")
    true_count = np.fromiter((len(data.truth[data.ref_ids[i]]) for i in positions), dtype=np.int32)
    true_links = int(true_count.sum())
    if found > true_links:
        raise ValueError("Candidate labels outnumber TUNE true links")
    return {"candidate_pairs": total, "retrieved_true_links": found,
            "true_links": true_links, "link_recall": found / true_links if true_links else 0.0,
            "pairs": _source_identity(pair_path), "features": _source_identity(feature_path)}


def train_tree(data: Dataset, output: Path, args) -> tuple[object, dict]:
    """Quantize numeric TSV while loading, then fit without a TUNE eval_set."""
    from catboost import CatBoostClassifier
    from catboost.utils import quantize

    started = time.monotonic()
    device = getattr(args, "device", "CPU").upper()
    pool = quantize(data_path=str(output / "train_features.tsv"),
                    column_description=str(output / "train_features.cd"),
                    delimiter="\t", has_header=False, thread_count=args.threads,
                    random_seed=args.seed, border_count=128 if device == "GPU" else 254,
                    used_ram_limit=f"{args.ram_budget_gb:g}gb")
    if list(pool.get_feature_names()) != data.features:
        raise ValueError("CatBoost quantization changed the ordered feature names")
    quantize_seconds = time.monotonic() - started
    quantized_path = output / "train_pool.quantized"
    pool.save(str(quantized_path))
    check_memory(args.min_available_gb, "Quantizing numeric TRAIN features")
    model = CatBoostClassifier(iterations=args.iterations, depth=args.depth, learning_rate=0.08,
                               loss_function="Logloss", eval_metric="Logloss", random_seed=args.seed,
                               task_type=device, thread_count=args.threads, allow_writing_files=False,
                               l2_leaf_reg=5.0, class_weights=[args.negative_weight, 1.0],
                               **({"gpu_ram_part": args.gpu_ram_part, "devices": "0"} if device == "GPU"
                                  else {"used_ram_limit": f"{args.ram_budget_gb:g}gb"}))
    started = time.monotonic()
    model.fit(pool, verbose=False)  # strictly TRAIN, fixed iterations; no TUNE early stopping
    fit_seconds = time.monotonic() - started
    if list(model.feature_names_) != data.features:
        raise ValueError("Fitted CatBoost feature names do not match feature preparation")
    model_path = output / "tree.cbm"
    model.save_model(str(model_path))
    del pool
    check_memory(args.min_available_gb, "Fitting the TRAIN-only CatBoost tree")
    return model, {"quantization_seconds": quantize_seconds, "fit_seconds": fit_seconds,
                   "quantized_pool": _source_identity(quantized_path),
                   "native_tree": _source_identity(model_path), "tree_count": model.tree_count_}


def _fit_tree_child(output: str, columns: list[str], settings: dict, messages) -> None:
    """Train in a fresh process so co-resident GPU work retains a hard RAM floor."""
    try:
        from types import SimpleNamespace

        _, report = train_tree(SimpleNamespace(features=columns), Path(output), SimpleNamespace(**settings))
        messages.put({"stage": "complete", "report": report,
                      "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    except Exception as error:
        messages.put({"stage": "error", "error": f"{type(error).__name__}: {error}"})


def supervised_full_fit(data: Dataset, output: Path, args) -> tuple[object, dict]:
    """Terminate only our CPU fit if cgroup headroom/child RSS becomes unsafe."""
    from catboost import CatBoostClassifier

    before = resource_snapshot()
    settings = {"threads": args.threads, "iterations": args.iterations, "depth": args.depth,
                "seed": args.seed, "negative_weight": args.negative_weight,
                "ram_budget_gb": args.ram_budget_gb, "min_available_gb": args.min_available_gb,
                "device": getattr(args, "device", "CPU"),
                "gpu_ram_part": getattr(args, "gpu_ram_part", .75)}
    context = mp.get_context("spawn")
    messages = context.Queue()
    process = context.Process(target=_fit_tree_child,
                              args=(str(output), data.features, settings, messages))
    process.start()
    peak_rss = peak_swap = peak_gpu_mib = 0
    least_available = before["effective_available_bytes"]
    stopped = None
    while process.is_alive():
        sample = resource_snapshot()
        least_available = min(least_available, sample["effective_available_bytes"])
        try:
            child = psutil.Process(process.pid)
            peak_rss = max(peak_rss, child.memory_info().rss)
            peak_swap = max(peak_swap, getattr(child.memory_full_info(), "swap", 0))
        except psutil.NoSuchProcess:
            pass
        if getattr(args, "device", "CPU") == "GPU":
            try:
                usage = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
                for line in usage.stdout.splitlines():
                    fields = line.split(",")
                    if len(fields) == 2 and int(fields[0].strip()) == process.pid:
                        peak_gpu_mib = max(peak_gpu_mib, int(fields[1].strip()))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
        if least_available < args.min_available_gb * GIB:
            stopped = "physical/cgroup available RAM crossed the protected floor"
        elif peak_rss > args.max_process_rss_gb * GIB:
            stopped = "CatBoost RSS exceeded the measured process allowance"
        elif peak_swap:
            stopped = "CatBoost process entered swap"
        elif peak_gpu_mib > 40 * 1024:
            stopped = "CatBoost exceeded the 40 GiB GPU memory limit"
        elif (args.execution_site == "local" and
              sample["swap_out_bytes"] - before["swap_out_bytes"] > args.max_swap_out_mb * 1024 ** 2):
            stopped = "workstation swap-out exceeded the guarded allowance"
        if stopped:
            process.terminate()
            break
        process.join(timeout=.2)
    process.join()
    try:
        message = messages.get(timeout=2)
    except Empty:
        message = {"stage": "error", "error": "Native fit produced no completion report"}
    messages.close()
    peak_rss = max(peak_rss, int(message.get("maxrss_kib", 0)) * 1024)
    after = resource_snapshot()
    if stopped is None and peak_rss > args.max_process_rss_gb * GIB:
        stopped = "CatBoost reported a peak RSS above the process allowance"
    if (stopped is None and args.execution_site == "local" and
            after["swap_out_bytes"] - before["swap_out_bytes"] > args.max_swap_out_mb * 1024 ** 2):
        stopped = "workstation swap-out exceeded the guarded allowance"
    state = {"exitcode": process.exitcode, "stopped_reason": stopped, "child_message": message,
             "peak_child_rss_gib": peak_rss / GIB, "peak_child_swap_bytes": peak_swap,
             "peak_gpu_memory_mib": peak_gpu_mib,
             "lowest_effective_available_gib": least_available / GIB,
             "system_swap_out_delta_bytes": max(0, after["swap_out_bytes"] - before["swap_out_bytes"]),
             "cgroup_limit_bytes": before["cgroup_limit_bytes"], "cgroup_cpu_quota": before["cgroup_cpu_quota"]}
    write_json(output / "fit_resource.json", state)
    if process.exitcode != 0 or stopped or message["stage"] != "complete":
        raise MemoryError(f"Bounded CatBoost fit did not complete: {stopped or message.get('error')}")
    native = output / "tree.cbm"
    if sha256_file(native) != message["report"]["native_tree"]["sha256"]:
        raise ValueError("Bounded CPU fit's native checkpoint changed before loading")
    model = CatBoostClassifier()
    model.load_model(str(native))
    if list(model.feature_names_) != data.features or model.tree_count_ != message["report"]["tree_count"]:
        raise ValueError("Bounded CPU fit returned another model or feature order")
    return model, {**message["report"], "bounded_cpu_resource": state}


def predict_tuning(model, columns: list[str], output: Path, args) -> np.ndarray:
    path = output / "tune_features.parquet"
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.names != columns:
        raise ValueError("TUNE feature order differs from trained CatBoost")
    length = parquet.metadata.num_rows
    temporary = output / "tune_probabilities.npy.tmp"
    scores = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(length,))
    offset = 0
    for batch in parquet.iter_batches(batch_size=50_000, use_threads=False):
        frame = batch.to_pandas()
        values = np.asarray(model.predict_proba(frame, thread_count=args.threads))[:, 1]
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError("CatBoost returned nonfinite or misaligned TUNE probabilities")
        scores[offset:offset + len(frame)] = values.astype(np.float32)
        offset += len(frame)
        check_memory(args.min_available_gb, "Scoring fixed TUNE candidates")
    if offset != length:
        raise ValueError("Not every TUNE pair was scored")
    scores.flush()
    temporary.replace(output / "tune_probabilities.npy")
    return scores


def entity_scores(left: np.ndarray, labels: np.ndarray, chosen: np.ndarray,
                  true_count: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(true_count)
    predicted = np.bincount(left[chosen], minlength=n)
    tp = np.bincount(left[chosen & labels], minlength=n)
    denominator = 4 * predicted + true_count
    f05 = np.divide(5.0 * tp, denominator, out=np.zeros(n), where=denominator > 0)
    f1 = np.divide(2.0 * tp, predicted + true_count,
                   out=np.zeros(n), where=(predicted + true_count) > 0)
    singletons = (true_count == 0) & (predicted == 0)
    f05[singletons] = f1[singletons] = 1.0
    return f05, f1, predicted, tp


def score_tune(data: Dataset, selected: np.ndarray, output: Path,
               scores: np.ndarray) -> tuple[dict, np.ndarray]:
    pairs = pd.read_parquet(output / "tune_pairs.parquet")
    positions = np.flatnonzero(selected)
    ref_ids = [data.ref_ids[i] for i in positions]
    countries = data.country[positions]
    actual = np.asarray([len(data.truth[entity]) for entity in ref_ids], dtype=np.int32)
    left = pairs.reference_local_idx.to_numpy(dtype=np.int64)
    labels = pairs.is_match.to_numpy(dtype=bool)
    if (len(pairs) != len(scores) or np.any(np.diff(left) < 0) or (len(left) and
        (left.min() < 0 or left.max() >= len(ref_ids))) or
        not np.array_equal(np.asarray(ref_ids, dtype=object)[left], pairs.source1_entity_id.to_numpy())):
        raise ValueError("TUNE pair IDs, reference order and probabilities are not aligned")
    positives = np.bincount(left[labels], minlength=len(ref_ids))
    if np.any(positives > actual):
        raise ValueError("TUNE label count exceeds truth; duplicate candidate edge?")
    maximum = float(scores.max()) if len(scores) else 1.0
    thresholds = np.unique(np.r_[0.0, np.linspace(0.05, 0.9, 35), .925, .95, .975,
                                 .99, .995, np.nextafter(maximum, float("inf"))])
    best = (-1.0, 0.5)
    trace = []
    for threshold in thresholds:
        values, _, _, _ = entity_scores(left, labels, np.asarray(scores, dtype=np.float64) >= threshold, actual)
        macro = float(values.mean())
        trace.append({"threshold": float(threshold), "macro_f05": macro})
        if (macro, float(threshold)) > best:
            best = (macro, float(threshold))
    threshold = best[1]
    f05, f1, predicted, tp = entity_scores(left, labels, np.asarray(scores, dtype=np.float64) >= threshold, actual)

    def describe(mask):
        n_pred, n_true, n_tp = int(predicted[mask].sum()), int(actual[mask].sum()), int(tp[mask].sum())
        singleton = mask & (actual == 0)
        return {"reference_count": int(mask.sum()), "macro_f05": float(f05[mask].mean()),
                "macro_f1": float(f1[mask].mean()), "micro_precision": n_tp / n_pred if n_pred else 0.0,
                "micro_recall": n_tp / n_true if n_true else 0.0,
                "predicted_links": n_pred, "true_links": n_true, "true_positive_links": n_tp,
                "false_positive_links": n_pred - n_tp, "missed_true_links": n_true - n_tp,
                "singleton_accuracy": float((predicted[singleton] == 0).mean()) if singleton.any() else None,
                "singleton_false_merges": int((predicted[singleton] > 0).sum())}

    retrieval = {"pairs": len(pairs), "pairs_per_reference": len(pairs) / len(ref_ids),
                 "link_recall": int(positives.sum()) / int(actual.sum()),
                 "complete_match_set_recall": float((positives[actual > 0] == actual[actual > 0]).mean()),
                 "oracle_macro_f05": float(entity_scores(left, labels, labels, actual)[0].mean())}
    pd.DataFrame({"source1_entity_id": ref_ids, "country": countries, "true_count": actual,
                  "predicted_count": predicted, "true_positive_count": tp, "f05": f05,
                  "f1": f1}).to_parquet(output / "tune_entities.parquet", index=False)
    write_json(output / "thresholds.json", {"selected": threshold, "tune_trace": trace})
    report = describe(np.ones(len(ref_ids), dtype=bool))
    report["threshold"] = threshold
    report["by_country"] = {key: describe(countries == key) for key in sorted(set(countries))}
    report["by_target_name_script"] = target_script_report(data, ref_ids, pairs,
                                                             np.asarray(scores, dtype=np.float64) >= threshold)
    return {"tune": report, "retrieval_ceiling": retrieval,
            "tune_pair_probabilities": _source_identity(output / "tune_probabilities.npy"),
            "tune_entities": _source_identity(output / "tune_entities.parquet")}, f05


def target_script_report(data: Dataset, ref_ids: list[str], pairs: pd.DataFrame,
                         accepted: np.ndarray) -> list[dict]:
    """All TUNE truth targets, including misses, versus retrieved/accepted links."""
    from .retrieval_ablation import _script

    truth_targets = {target for ref_id in ref_ids for target in data.truth[ref_id]}
    locations = pc.is_in(data.pool.column("entity_id"), value_set=pa.array(sorted(truth_targets)))
    rows = np.flatnonzero(locations.to_numpy(zero_copy_only=False))
    records = data.pool.take(pa.array(rows)).select(RECORD_COLUMNS).to_pandas()
    if len(records) != len(truth_targets):
        raise ValueError("Missing TUNE truth records prevent honest script/miss accounting")
    scripts = {target: _script(name) for target, name in zip(records.entity_id, records.business_name)}
    countries = {target: country for target, country in zip(records.entity_id, records.country)}
    counts: dict[tuple[str, str], dict] = {}

    def group(target: str) -> dict:
        key = (countries[target], scripts[target])
        return counts.setdefault(key, {"country": key[0], "target_name_script": key[1],
                                       "true_links": 0, "retrieved_true_links": 0,
                                       "accepted_true_links": 0, "false_positive_links": 0})

    for ref in ref_ids:
        for target in data.truth[ref]:
            group(target)["true_links"] += 1
    labels = pairs.is_match.to_numpy(dtype=bool)
    for target, keep in zip(pairs.loc[labels, "candidate_entity_id"], accepted[labels]):
        record = group(target)
        record["retrieved_true_links"] += 1
        record["accepted_true_links"] += int(keep)
    false_rows = pairs.loc[accepted & ~labels, "right_idx"].to_numpy(dtype=np.int64)
    false_records = data.pool.take(pa.array(false_rows)).select(RECORD_COLUMNS).to_pandas()
    for country, name in zip(false_records.country, false_records.business_name):
        key = (country, _script(name))
        counts.setdefault(key, {"country": key[0], "target_name_script": key[1],
                                "true_links": 0, "retrieved_true_links": 0,
                                "accepted_true_links": 0, "false_positive_links": 0})["false_positive_links"] += 1
    for entry in counts.values():
        entry["retrieval_link_recall"] = (entry["retrieved_true_links"] / entry["true_links"]
                                          if entry["true_links"] else None)
        entry["accepted_link_recall"] = (entry["accepted_true_links"] / entry["true_links"]
                                         if entry["true_links"] else None)
    return [counts[key] for key in sorted(counts)]


def compare_v1(data: Dataset, selected: np.ndarray, f05: np.ndarray,
               path: Path, seed: int) -> dict:
    """Full-run-only paired comparison on *exactly* the same medium TUNE IDs."""
    saved = pd.read_parquet(path)
    if (saved.source1_entity_id.duplicated().any() or len(saved) != MEDIUM_TUNE_COUNT
            or ids_digest(saved.source1_entity_id) != MEDIUM_TUNE_ID_SHA256
            or abs(float(saved.baseline_f05.mean()) - V1_TUNE_F05) > 1e-12):
        raise ValueError("Saved v1 CE baseline differs from fixed 12k medium TUNE")
    positions = np.flatnonzero(selected)
    ids = [data.ref_ids[i] for i in positions]
    prior = saved.set_index("source1_entity_id").loc[ids]
    actual = np.asarray([len(data.truth[entity]) for entity in ids])
    if not np.array_equal(prior.true_count.to_numpy(), actual) or not np.array_equal(
            prior.country.to_numpy(), data.country[positions]):
        raise ValueError("Same-ID v1 comparison has different truth counts or countries")
    baseline = prior.baseline_f05.to_numpy(dtype=np.float64)
    return {"v1_saved_entities": _source_identity(path), "v1_medium_tune_macro_f05": float(baseline.mean()),
            "tree_minus_v1": paired_bootstrap(f05, baseline, n_bootstrap=10_000, seed=seed),
            "improved_entities": int((f05 > baseline).sum()),
            "worsened_entities": int((f05 < baseline).sum()),
            "ties": int((f05 == baseline).sum()),
            "scope": "same 12,000 S1 IDs; v1 is dense/tree/BGE, v2 is a tree on P2 union candidates"}


def compare_p1(data: Dataset, selected: np.ndarray, f05: np.ndarray,
               directory: Path, seed: int) -> dict:
    """Pair against the saved 0.970099 no-source rule + pure assignment by S1."""
    refs_path = directory / "tune_references.parquet"
    pairs_path = directory / "tune_pairs.parquet"
    accept_path = directory / "no_source_rule_plus_assignment_accept.npy"
    refs = pd.read_parquet(refs_path)
    pairs = pd.read_parquet(pairs_path, columns=["left_idx", "is_truth_for_scoring_only"])
    decisions = np.load(accept_path, mmap_mode="r")
    if (len(refs) != MEDIUM_TUNE_COUNT or ids_digest(refs.source1_entity_id) != MEDIUM_TUNE_ID_SHA256
            or len(pairs) != len(decisions) or refs.source1_entity_id.duplicated().any()):
        raise ValueError("P1 paired comparison is not the complete saved medium TUNE set")
    remap = np.full(int(max(refs.original_left_idx.max(), pairs.left_idx.max())) + 1, -1, dtype=np.int32)
    remap[refs.original_left_idx.to_numpy(dtype=np.int64)] = np.arange(len(refs), dtype=np.int32)
    left = remap[pairs.left_idx.to_numpy(dtype=np.int64)]
    if np.any(left < 0):
        raise ValueError("P1 saved pair references do not belong to its complete TUNE split")
    labels = pairs.is_truth_for_scoring_only.to_numpy(dtype=bool)
    p1, _, _, _ = entity_scores(left, labels, np.asarray(decisions, dtype=bool),
                                refs.true_count.to_numpy(dtype=np.int32))
    if not np.isclose(p1.mean(), .9700994007431162, rtol=0, atol=1e-12):
        raise ValueError("P1 decision baseline changed from its recorded 0.970099 score")
    ids = [data.ref_ids[i] for i in np.flatnonzero(selected)]
    aligned = pd.Series(p1, index=refs.source1_entity_id).loc[ids].to_numpy()
    if not np.array_equal(refs.set_index("source1_entity_id").loc[ids].true_count.to_numpy(),
                          np.asarray([len(data.truth[entity]) for entity in ids])):
        raise ValueError("P1 and P3.1 true match counts differ for the same TUNE IDs")
    return {"p1_macro_f05": float(aligned.mean()), "tree_minus_p1": paired_bootstrap(
        f05, aligned, n_bootstrap=10_000, seed=seed),
        "inputs": {name: _source_identity(path) for name, path in
                   (("references", refs_path), ("pairs", pairs_path), ("decisions", accept_path))}}


def verify_immutable_inputs(*datasets: Dataset) -> None:
    seen = set()
    for data in datasets:
        for entry in data.inputs.values():
            path = Path(entry["path"])
            if path in seen:
                continue
            seen.add(path)
            if path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                raise ValueError(f"A completed candidate/label input changed during training: {path}")


def stage_medium_memory_probe(directory: Path, output: Path, counts: list[int],
                              min_available_gb: float, max_swap_out_mb: float = 32.) -> dict:
    """Reuse saved exact medium features; fit only real TRAIN rows, never TUNE."""
    directory = Path(directory)
    paths = {name: directory / name for name in ("reference.parquet", "pool.parquet", "pairs.parquet",
                                                "truth.json", "metadata.json", "complete.json",
                                                "base_features.parquet", "numeric_features.parquet")}
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("Memory probe requires saved medium full-pool candidates and cached tree features")
    metadata = json.loads(paths["metadata.json"].read_text())
    complete = json.loads(paths["complete.json"].read_text())
    if (complete.get("state") != "complete" or complete.get("metadata") != metadata
            or metadata.get("mining_full_pool_records") != FULL_POOL_ROWS
            or metadata.get("positive_candidate_injection") is not False):
        raise ValueError("Probe input is not committed, label-free full-pool medium retrieval")
    reference = pd.read_parquet(paths["reference.parquet"], columns=["entity_id", "split"])
    if (reference.entity_id.duplicated().any() or len(reference) != 47_999
            or reference.split.eq("train").sum() != 35_999
            or reference.split.eq("tune").sum() != MEDIUM_TUNE_COUNT
            or ids_digest(reference.loc[reference.split.eq("tune"), "entity_id"]) != MEDIUM_TUNE_ID_SHA256):
        raise ValueError("Memory probe must use the saved disjoint medium TRAIN/TUNE IDs")
    allowed = reference.split.eq("train").to_numpy()
    ids = reference.entity_id.to_numpy()
    pool_ids = pq.read_table(paths["pool.parquet"], columns=["entity_id"]).column("entity_id")
    truth = json.loads(paths["truth.json"].read_text())
    if set(truth) != set(ids):
        raise ValueError("The saved medium truth must cover every fixed TRAIN/TUNE S1")
    truth = {key: set(truth[key]) for key in ids[allowed]}
    readers = [pq.ParquetFile(paths[name]) for name in
               ("pairs.parquet", "base_features.parquet", "numeric_features.parquet")]
    source_rows = readers[0].metadata.num_rows
    groups = [[reader.metadata.row_group(i).num_rows for i in range(reader.metadata.num_row_groups)]
              for reader in readers]
    if (groups[0] != groups[1] or groups[0] != groups[2] or source_rows != metadata["candidate_pairs"]
            or readers[1].schema_arrow.names + readers[2].schema_arrow.names
            != feature_columns(readers[0].schema_arrow.names)):
        raise ValueError("The saved medium feature rows/columns are not aligned with candidate pairs")
    expected_train = metadata["retrieval"]["train"]["pairs"]
    if counts != sorted(set(counts)) or not counts or counts[0] < 1 or counts[-1] > expected_train:
        raise ValueError(f"Probe sizes must be unique, sorted, and <= {expected_train:,} real TRAIN pairs")
    cd = output / "train_features.cd"
    columns = feature_columns(readers[0].schema_arrow.names)
    cd.write_text("0\tLabel\n" + "".join(f"{i}\tNum\t{name}\n" for i, name in enumerate(columns, 1)),
                  encoding="utf-8")
    files = {count: output / f"train_{count}.tsv" for count in counts}
    true_counts = {count: 0 for count in counts}
    covered = {count: set() for count in counts}
    written = 0
    baseline = resource_snapshot()
    own_process = psutil.Process()
    peak_own_rss = peak_own_swap = 0
    lowest_available = baseline["effective_available_bytes"]
    with ExitStack() as stack:
        streams = {count: stack.enter_context(path.open("w", encoding="utf-8")) for count, path in files.items()}
        batches = [reader.iter_batches(batch_size=50_000, use_threads=False) for reader in readers]
        for pair_batch, base_batch, numeric_batch in zip(*batches, strict=True):
            if len(pair_batch) != len(base_batch) or len(pair_batch) != len(numeric_batch):
                raise ValueError("Cached feature and candidate batch boundaries differ")
            left = pair_batch.column(pair_batch.schema.get_field_index("left_idx")).to_numpy(zero_copy_only=False)
            right = pair_batch.column(pair_batch.schema.get_field_index("right_idx")).to_numpy(zero_copy_only=False)
            if len(left) and (left.min() < 0 or left.max() >= len(ids) or right.min() < 0 or right.max() >= len(pool_ids)):
                raise ValueError("Cached TRAIN pairs contain invalid positional IDs")
            keep = allowed[left]
            if not keep.any():
                continue
            left, right = left[keep], right[keep]
            reference_ids = ids[left]
            target_ids = pool_ids.take(pa.array(right)).to_pylist()
            labels = np.fromiter((target in truth[ref] for ref, target in zip(reference_ids, target_ids)),
                                 dtype=np.int8, count=len(left))
            base = base_batch.to_pandas().to_numpy(dtype=np.float32)[keep]
            numeric = numeric_batch.to_pandas().to_numpy(dtype=np.float32)[keep]
            matrix = np.column_stack((labels, base, numeric))
            if not np.isfinite(matrix).all():
                raise ValueError("Cached TRAIN features contain nonfinite values")
            for count, stream in streams.items():
                length = max(0, min(len(matrix), count - written))
                if length:
                    np.savetxt(stream, matrix[:length], fmt="%.9g", delimiter="\t")
                    true_counts[count] += int(labels[:length].sum())
                    covered[count].update(left[:length].tolist())
            written += len(matrix)
            check_memory(min_available_gb, "Staging real medium TRAIN numeric rows")
            snapshot = resource_snapshot()
            lowest_available = min(lowest_available, snapshot["effective_available_bytes"])
            peak_own_rss = max(peak_own_rss, own_process.memory_info().rss)
            peak_own_swap = max(peak_own_swap, getattr(own_process.memory_full_info(), "swap", 0))
            swap_out_delta = max(0, snapshot["swap_out_bytes"] - baseline["swap_out_bytes"])
            if written % 250_000 < len(matrix):
                write_json(output / "stage_progress.json", {"real_train_pairs_written": written,
                    "peak_staging_rss_gib": peak_own_rss / GIB, "peak_staging_swap_bytes": peak_own_swap,
                    "lowest_effective_available_gib": lowest_available / GIB,
                    "system_swap_out_delta_bytes": swap_out_delta})
            if peak_own_swap or swap_out_delta > max_swap_out_mb * 1024 ** 2:
                raise MemoryError("TRAIN feature staging swapped this process or exceeded the system swap-out allowance")
            if written >= counts[-1]:
                break
    if written < counts[-1] or counts[-1] == expected_train and written != expected_train:
        raise ValueError("Probe failed to stage the requested real TRAIN rows")
    inputs = {name: _source_identity(path) for name, path in paths.items()}
    return {"scope": "TRAIN-only medium dense full-target-pool memory probe; no TUNE fitting or model score",
            "feature_columns": columns, "reference_id_sha256": ids_digest(ids[allowed]),
            "input_candidates": source_rows, "all_train_candidate_pairs": expected_train,
            "input_identities": inputs, "column_description_sha256": sha256_file(cd),
            "staging_resources": {"peak_process_rss_gib": peak_own_rss / GIB,
                                  "peak_process_swap_bytes": peak_own_swap,
                                  "lowest_effective_available_gib": lowest_available / GIB,
                                  "system_swap_out_delta_bytes": swap_out_delta},
            "staged": {str(count): {"file": _source_identity(files[count]),
                                     "train_positive_rows": true_counts[count],
                                     "distinct_train_s1s": len(covered[count])} for count in counts}}


def _memory_probe_child(tsv: str, cd: str, destination: str, threads: int,
                        iterations: int, ram_gb: float, messages) -> None:
    """Isolated CatBoost native memory; parent can stop it before swapping."""
    try:
        from catboost import CatBoostClassifier
        from catboost.utils import quantize

        started = time.monotonic()
        pool = quantize(data_path=tsv, column_description=cd, delimiter="\t", has_header=False,
                        thread_count=threads, random_seed=20260925, border_count=254,
                        used_ram_limit=f"{ram_gb:g}gb")
        quant_seconds = time.monotonic() - started
        pool.save(destination + ".quantized")
        messages.put({"stage": "quantized", "seconds": quant_seconds,
                      "quantized_bytes": Path(destination + ".quantized").stat().st_size})
        model = CatBoostClassifier(iterations=iterations, depth=8, learning_rate=.08,
                                   loss_function="Logloss", eval_metric="Logloss", random_seed=20260925,
                                   task_type="CPU", thread_count=threads, allow_writing_files=False,
                                   l2_leaf_reg=5., class_weights=[1., 1.],
                                   used_ram_limit=f"{ram_gb:g}gb")
        started = time.monotonic()
        model.fit(pool, verbose=False)
        model.save_model(destination + ".cbm")
        messages.put({"stage": "fitted", "seconds": time.monotonic() - started,
                      "maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      "major_faults": resource.getrusage(resource.RUSAGE_SELF).ru_majflt})
    except Exception as error:
        messages.put({"stage": "error", "error": f"{type(error).__name__}: {error}"})


def _supervise_memory_probe(count: int, stage: dict, output: Path, args) -> dict:
    before = resource_snapshot()
    if before["effective_available_bytes"] < (args.min_available_gb + 1) * GIB:
        raise MemoryError("Not enough uncommitted physical RAM to start a native CPU memory probe")
    context = mp.get_context("spawn")
    messages = context.Queue()
    dest = output / f"train_{count}"
    process = context.Process(target=_memory_probe_child, args=(stage["staged"][str(count)]["file"]["path"],
        str(output / "train_features.cd"), str(dest), args.threads, args.iterations,
        args.ram_budget_gb, messages))
    started = time.monotonic()
    process.start()
    maximum_rss = maximum_swap = 0
    minimum_available = before["effective_available_bytes"]
    stopped_reason = None
    while process.is_alive():
        sample = resource_snapshot()
        minimum_available = min(minimum_available, sample["effective_available_bytes"])
        try:
            child = psutil.Process(process.pid)
            maximum_rss = max(maximum_rss, child.memory_info().rss)
            maximum_swap = max(maximum_swap, getattr(child.memory_full_info(), "swap", 0))
        except psutil.NoSuchProcess:
            pass
        if minimum_available < args.min_available_gb * GIB:
            stopped_reason = "physical/cgroup available RAM fell below the protected floor"
        elif maximum_rss > args.max_process_rss_gb * GIB:
            stopped_reason = "child RSS exceeded the bounded workstation budget"
        elif maximum_swap > 0 or sample["swap_out_bytes"] - before["swap_out_bytes"] > args.max_swap_out_mb * 1024 ** 2:
            stopped_reason = "child swapped or system swap-out exceeded the guarded allowance"
        if stopped_reason:
            process.terminate()
            break
        process.join(timeout=.2)
    process.join()
    seen = []
    while len(seen) < 2:
        try:
            seen.append(messages.get(timeout=.5))
        except Empty:
            break
    messages.close()
    maximum_rss = max(maximum_rss, *(int(item.get("maxrss_kib", 0)) * 1024 for item in seen))
    after = resource_snapshot()
    if stopped_reason is None and maximum_rss > args.max_process_rss_gb * GIB:
        stopped_reason = "child highwater RSS exceeded the bounded workstation budget"
    if (stopped_reason is None and
            after["swap_out_bytes"] - before["swap_out_bytes"] > args.max_swap_out_mb * 1024 ** 2):
        stopped_reason = "system swap-out exceeded the guarded allowance"
    status = "pass" if process.exitcode == 0 and not stopped_reason and [m["stage"] for m in seen] == ["quantized", "fitted"] else "unsafe_or_failed"
    return {"train_pairs": count, "status": status, "exitcode": process.exitcode,
            "stopped_reason": stopped_reason, "stages": seen,
            "peak_child_rss_gib": maximum_rss / GIB, "peak_child_swap_bytes": maximum_swap,
            "guarded_system_swap_out_allowance_mb": args.max_swap_out_mb,
            "minimum_effective_available_gib": minimum_available / GIB,
            "system_swap_out_delta_bytes": max(0, after["swap_out_bytes"] - before["swap_out_bytes"]),
            "system_swap_used_delta_bytes": after["swap_used_bytes"] - before["swap_used_bytes"],
            "quantized_pool_bytes": (dest.with_suffix(".quantized").stat().st_size
                                     if dest.with_suffix(".quantized").is_file() else None),
            "elapsed_seconds": time.monotonic() - started, "resource_before": before,
            "resource_after": after}


def memory_probe(args) -> dict:
    """Bounded real TRAIN-only resource experiment; reports no F0.5."""
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a memory probe: {output}")
    if args.threads < 1 or args.threads > 8:
        raise ValueError("The local probe must use at most eight CPU cores")
    counts = args.probe_pairs
    if args.iterations < 1 or args.ram_budget_gb <= 0 or args.max_process_rss_gb <= 0:
        raise ValueError("Probe training and memory limits must be positive")
    check_memory(args.min_available_gb + 1, "Before staging the TRAIN-only memory probe")
    output.mkdir(parents=True, exist_ok=True)
    stage = stage_medium_memory_probe(Path(args.train_data), output, counts, args.min_available_gb,
                                      args.max_stage_swap_out_mb)
    write_json(output / "staged.json", stage)
    gc.collect()
    report = {"scope": stage["scope"], "seed": 20260925, "threads": args.threads,
              "iterations": args.iterations, "depth": 8, "negative_weight": 1.,
              "max_child_rss_gib": args.max_process_rss_gb,
              "guarded_system_swap_out_allowance_mb": args.max_swap_out_mb,
              "protected_available_gib": args.min_available_gb,
              "staged_sha256": sha256_file(output / "staged.json"), "measurements": [],
              "no_tune_or_holdout_model_score": True}
    for count in counts:
        result = _supervise_memory_probe(count, stage, output, args)
        report["measurements"].append(result)
        write_json(output / "resource_results.json", report)
        log(f"Real TRAIN memory probe {count:,}: {result['status']}, "
            f"peak child RSS {result['peak_child_rss_gib']:.2f} GiB, "
            f"swap-out delta {result['system_swap_out_delta_bytes']} bytes")
        if result["status"] != "pass":
            break
    return report


def run(args) -> dict:
    started = time.monotonic()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing tree experiment: {output}")
    if args.threads < 1 or args.threads > min(8, len(os.sched_getaffinity(0))):
        raise ValueError("Use 1–8 available workstation CPU cores")
    if min(args.iterations, args.max_train_pairs, args.reference_batch_size,
           args.pair_read_batch_size, args.max_pairs_per_batch) < 1:
        raise ValueError("Iterations, pair cap and batch sizes must be positive")
    if args.ram_budget_gb <= 0 or args.min_available_gb <= 0 or args.negative_weight <= 0:
        raise ValueError("RAM budgets and negative weight must be positive")
    if args.mode == "screen" and args.screen_train_references < 1:
        raise ValueError("A screen must specify positive screen-train-references")
    if args.mode == "screen" and (args.train_ready or not args.train_data):
        raise ValueError("Screen mode needs --train-data and does not consume P2's full TRAIN ready marker")
    if args.mode in {"full", "p2-screen"} and (not args.train_ready or args.train_data or args.train_pairs):
        raise ValueError("Full mode needs only --train-ready for the sealed P2 TRAIN candidate union")
    if args.mode == "full" and args.screen_train_references != 1200:
        raise ValueError("Full-scale training must not screen a subset of TRAIN S1s")
    if args.mode == "p2-screen" and (args.execution_site != "pod-gpu" or args.device != "GPU"
                                    or not 50_000 <= args.screen_train_references <= 200_000
                                    or args.max_train_pairs > 4_000_000 or args.iterations > 300
                                    or args.gpu_ram_part > .85 or args.threads > 3):
        raise ValueError("Fast P2 screen requires GPU pod, 50k–200k TRAIN S1s, <=4M pairs/300 trees, <=3 CPU threads and <40GB VRAM")
    if args.mode == "p2-screen" and shutil.disk_usage("/tmp").free < 4 * GIB:
        raise RuntimeError("Fast P2 screen requires at least 4 GiB free pod scratch")
    if args.mode in {"full", "p2-screen"}:
        ready = json.loads(Path(args.train_ready).read_text())
        if (ready.get("mode") != "full" or ready.get("screen_only") is not False
                or ready.get("files", {}).get("reference", {}).get("rows") != 2_072_821):
            raise ValueError("P3.1 full fit requires the 2,072,821-S1 P2.5 FULL ready marker, not a 10k screen")
    if args.mode in {"full", "p2-screen"} and args.execution_site == "local":
        measured = no_swap_verified_pairs(json.loads(Path(args.resource_report).read_text()))
        all_true_links = ready.get("files", {}).get("train_truth_pairs", {}).get("rows")
        if not isinstance(all_true_links, int) or measured < .75 * all_true_links:
            raise MemoryError(f"Local no-swap probe verified only {measured:,} pairs against "
                              f"{all_true_links!r} P2 TRAIN true links; wait for a larger real "
                              "no-swap probe or choose --execution-site pod-cpu")
    check_memory(args.min_available_gb, "Starting P3.1")
    output.mkdir(parents=True, exist_ok=True)
    train = (load_p2_ready(args.train_ready, output / "train_records", args.min_available_gb)
             if args.mode in {"full", "p2-screen"} else
             load_dataset(args.train_data, args.train_pairs, output / "train_records", require_union=False,
                          min_available_gb=args.min_available_gb))
    if (args.mode == "screen" and Path(args.tune_data).resolve() == train.directory and
            _pairs_path(Path(args.tune_data), args.tune_pairs) == train.pairs_path):
        tune = train
    else:
        tune = load_dataset(args.tune_data, args.tune_pairs, output / "tune_records",
                            require_union=args.mode in {"full", "p2-screen"}, min_available_gb=args.min_available_gb)
    if train.features != tune.features:
        raise ValueError("TRAIN and TUNE pair feature schemas/order differ; P2 must materialize the same scored union")
    prior = check_reference_feature_order(train.features,
                                          Path(args.reference_checkpoint) if args.reference_checkpoint else None)
    train_mask = _select_references(train, "train", args.screen_train_references if args.mode in {"screen", "p2-screen"} else None,
                                    args.seed)
    tune_mask = _select_references(tune, "tune", None, args.seed)
    population = validate_populations(train, tune, train_mask, tune_mask, args.mode, Path(args.excluded_ids_file))
    from catboost import __version__ as catboost_version
    config = {"schema": 1, "mode": args.mode, "hypothesis": "More retrieved hard negatives and TRAIN businesses improve a numeric CatBoost tree",
              "changed_variable": "TRAIN scale on fixed completed candidates (P2 union required for full fit)",
              "train_population": population, "seed": args.seed, "threads": args.threads,
              "execution_site": args.execution_site,
              "iterations": args.iterations, "depth": args.depth, "negative_weight": args.negative_weight,
              "learning_rate": 0.08, "l2_leaf_reg": 5.0, "eval_set": None, "device": args.device,
              "gpu_ram_part": args.gpu_ram_part if args.device == "GPU" else None,
              "feature_columns": train.features, "reference_checkpoint": prior,
              "pair_feature_policy": train.pair_policy,
              "union_feature_semantics": ("legacy lexical/phonetic scores zeroed on TRAIN and TUNE; "
                                          "dense_only=dense_hit, phonetic_only=phonetic_key_hit; "
                                          "retrieval_score=dense_score over complete union") if args.mode in {"full", "p2-screen"} else None,
              "full_competing_target_pool": FULL_POOL_ROWS, "positive_candidate_injection": False,
              "input_train": train.inputs, "input_tune": tune.inputs,
              "input_train_metadata": train.metadata, "input_tune_metadata": tune.metadata,
              "feature_code_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                      for name in ("features.py", "numeric_features.py", "predict.py", "v2_tree_train.py")},
              "versions": {"catboost": catboost_version, "numpy": np.__version__,
                           "pandas": pd.__version__, "pyarrow": pa.__version__}}
    pair_count, positive_count, mandatory_refs = count_training_pairs(train, train_mask, args)
    budget = pair_budget(positive_count, pair_count, len(train.features), args, mandatory_refs)
    config["sampling_and_memory_budget"] = budget
    write_json(output / "experiment_config.json", config)
    log(f"{args.mode}: TRAIN {population['train_references']:,} S1s, {pair_count:,} mined pairs; "
        f"TUNE {population['tune_references']:,} S1s; safe pair cap {budget['actual_pair_cap']:,}")
    with MemoryMonitor() as memory:
        train_preparation = prepare_training(train, train_mask, budget, output, args)
        tune_preparation = prepare_tuning(tune, tune_mask, output, args)
        if (tune_preparation["candidate_pairs"] != tune.metadata["retrieval"]["tune"]["pairs"]
                if "retrieval" in tune.metadata and "tune" in tune.metadata["retrieval"] else False):
            raise ValueError("TUNE pair count no longer matches its saved full-pool retrieval")
        model, training = (supervised_full_fit(train, output, args) if args.mode in {"full", "p2-screen"}
                           else train_tree(train, output, args))
        probabilities = predict_tuning(model, train.features, output, args)
        evaluation, f05 = score_tune(tune, tune_mask, output, probabilities)
        bundle = {"model": model, "threshold": evaluation["tune"]["threshold"],
                  "feature_columns": train.features, "numeric_features": True,
                  "record": {"mode": args.mode, "depth": args.depth, "negative_weight": args.negative_weight,
                             "training_pairs": train_preparation["selected_pairs"],
                             "tune": evaluation["tune"]}}
        joblib.dump(bundle, output / "tree_bundle.joblib")
    comparison = None
    p1_comparison = None
    if args.mode in {"full", "p2-screen"}:
        comparison = compare_v1(tune, tune_mask, f05, Path(args.v1_tune_entities), args.seed)
        if args.p1_decision_dir:
            p1_comparison = compare_p1(tune, tune_mask, f05, Path(args.p1_decision_dir), args.seed)
    verify_immutable_inputs(train, tune)
    results = {"scope": ("COMPATIBILITY SCREEN: sampled medium TRAIN; dense-only full 10,320,219-target "
                         "retrieval and fixed 12,000 medium TUNE; no P2 union/full-scale claim"
                         if args.mode == "screen" else
                         "BOUNDED P2 TRAIN-ENTITY SCREEN: full-pool union candidates, sampled TRAIN S1s; fixed 12,000 medium TUNE; NOT a full TRAIN fit"
                         if args.mode == "p2-screen" else
                         "FULL P3.1: P2 union TRAIN and fixed 12,000 medium TUNE against the full target pool"),
               "train": train_preparation, "tune_preparation": tune_preparation,
               "evaluation": evaluation, "v1_paired_tune": comparison, "p1_paired_tune": p1_comparison,
               "training": training, "memory": memory.report(),
               "tree_bundle": _source_identity(output / "tree_bundle.joblib"),
               "experiment_config_sha256": sha256_file(output / "experiment_config.json"),
               "elapsed_seconds": time.monotonic() - started,
               "holdout_labels_used": False, "france_accuracy_measured": False,
               "limitations": ["TUNE was reused for threshold selection; a paired tuning CI does not establish held-out gain.",
                               "Retrieval misses remain false negatives; truth never enters candidate generation."]}
    write_json(output / "results.json", results)
    write_json(output / "complete.json", {"state": "complete", "results_sha256": sha256_file(output / "results.json"),
                                          "tree_checkpoint_sha256": results["tree_bundle"]["sha256"]})
    log(f"Completed {args.mode}: TUNE macro F0.5 {evaluation['tune']['macro_f05']:.6f}, "
        f"precision {evaluation['tune']['micro_precision']:.6f}, "
        f"recall {evaluation['tune']['micro_recall']:.6f}; {results['elapsed_seconds']:.1f}s")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("screen", "p2-screen", "full", "memory-probe"), required=True)
    parser.add_argument("--train-data", help="Completed saved medium dense directory; screen only")
    parser.add_argument("--train-ready", help="P2 atomic full TRAIN union ready.json; full mode only")
    parser.add_argument("--train-pairs")
    parser.add_argument("--tune-data")
    parser.add_argument("--tune-pairs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reference-checkpoint", help="Read-only v1 checkpoint feature-order comparison")
    parser.add_argument("--screen-train-references", type=int, default=1200)
    parser.add_argument("--excluded-ids-file", default=str(REPO_ROOT / "artifacts/v2_holdout2/train_tune_excluded_ids.txt"))
    parser.add_argument("--v1-tune-entities", default=str(REPO_ROOT / "artifacts/v2_p1_decision/tune_references.parquet"))
    parser.add_argument("--p1-decision-dir", help="Saved exact same-entity P1 0.970099 decision inputs")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--iterations", type=int, default=650)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--execution-site", choices=("local", "pod-cpu", "pod-gpu"), default="local",
                        help="Full local mode requires a measured no-swap pair cap; pod-cpu requires cgroup/reserve guards")
    parser.add_argument("--resource-report", default=str(REPO_ROOT / "artifacts/v2_tree_screen/memory_probe_medium_bounded/resource_results.json"),
                        help="TRAIN-only measured no-swap resource probe required for a local full fit")
    parser.add_argument("--max-train-pairs", type=int, default=10_000_000,
                        help="Provisional upper bound; pod co-residency never exceeds 10M without a new probe")
    parser.add_argument("--ram-budget-gb", type=float, default=6.0)
    parser.add_argument("--min-available-gb", type=float, default=1.5)
    parser.add_argument("--reference-batch-size", type=int, default=1000)
    parser.add_argument("--pair-read-batch-size", type=int, default=100_000)
    parser.add_argument("--max-pairs-per-batch", type=int, default=250_000)
    parser.add_argument("--probe-pairs", type=int, nargs="+", default=[250_000, 750_000, 1_500_000, 1_799_950],
                        help="TRAIN-only cached-medium pair prefixes for memory-probe mode")
    parser.add_argument("--max-process-rss-gb", type=float, default=4.5,
                        help="Terminate the isolated memory probe at this RSS budget")
    parser.add_argument("--max-swap-out-mb", type=float, default=8.,
                        help="Abort probe if this much system swap-out occurs; child swap always aborts")
    parser.add_argument("--max-stage-swap-out-mb", type=float, default=32.,
                        help="Bound unrelated system swap-out while writing cached TRAIN TSVs")
    parser.add_argument("--device", choices=("CPU", "GPU"), default="CPU")
    parser.add_argument("--gpu-ram-part", type=float, default=.75)
    parser.add_argument("--max-feature-prep-seconds", type=int, default=2700)
    args = parser.parse_args()
    if args.mode == "memory-probe":
        if not args.train_data or args.tune_data or args.train_ready:
            parser.error("Memory probe needs only saved medium --train-data; it never opens TUNE for scoring")
        memory_probe(args)
    else:
        if not args.tune_data:
            parser.error("Screen/full matching requires --tune-data")
        run(args)


if __name__ == "__main__":
    main()
