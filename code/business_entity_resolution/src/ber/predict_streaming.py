"""Frozen tree inference with ordered Parquet input and bounded CPU workers.

Candidate input must be ordered by left_idx; right_idx may be unordered within
one business. Complete S1 groups are validated, sorted, and sent to workers with
only their referenced records. No truth, fitting, or threshold selection occurs.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
from dataclasses import dataclass
import gc
import json
import multiprocessing as mp
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .full_embeddings import atomic_json, checkpoint_identity, file_digest
from .models import predict_proba
from .predict import feature_batch
from .prepare import log


RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
_WORKER_BUNDLE = None
_WORKER_THREAD_LIMIT = None


@dataclass
class CandidateBatch:
    start: int
    end: int
    pairs: pd.DataFrame


def load_records(data_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = Path(data_path)
    reference = pd.read_parquet(data / "reference.parquet", columns=RECORD_COLUMNS).fillna("")
    pool = pd.read_parquet(data / "pool.parquet", columns=RECORD_COLUMNS).fillna("")
    for label, frame in [("reference", reference), ("pool", pool)]:
        if frame.entity_id.eq("").any() or frame.entity_id.duplicated().any():
            raise ValueError(f"{label} entity IDs must be nonempty and unique")
    return reference, pool


def _candidate_schema(parquet: pq.ParquetFile) -> None:
    schema = parquet.schema_arrow
    for name in ("left_idx", "right_idx"):
        if name not in schema.names or not pa.types.is_integer(schema.field(name).type):
            raise ValueError(f"Candidate Parquet requires an integer {name} column")


def iter_reference_batches(path: str | Path, reference_count: int, pool_count: int,
                           reference_batch_size: int = 1000, pair_read_batch_size: int = 100000,
                           max_pairs_per_batch: int = 200000,
                           stop_reference: int | None = None, require_sorted_right: bool = False):
    """Yield complete, sorted groups, including empty reference intervals.

    Validation spans Arrow row-batch boundaries. Global left ordering is required;
    right ordering is established within each output batch. The bounded prefix
    option exists only for explicitly labeled development benchmarks.
    """
    if min(reference_batch_size, pair_read_batch_size, max_pairs_per_batch) <= 0:
        raise ValueError("Candidate batch limits must be positive")
    stop_reference = reference_count if stop_reference is None else min(stop_reference, reference_count)
    if stop_reference < 0:
        raise ValueError("Reference prefix cannot be negative")
    parquet = pq.ParquetFile(path)
    _candidate_schema(parquet)
    empty = pa.Table.from_batches([], schema=parquet.schema_arrow).to_pandas()
    iterator = iter(parquet.iter_batches(batch_size=pair_read_batch_size, use_threads=False))
    previous_left = -1

    def next_frame():
        nonlocal previous_left
        try:
            table = next(iterator)
        except StopIteration:
            return None
        frame = table.to_pandas()
        if frame.empty:
            return next_frame()
        if frame[["left_idx", "right_idx"]].isna().any().any():
            raise ValueError("Candidate indices cannot be null")
        left = frame.left_idx.to_numpy(dtype=np.int64)
        right = frame.right_idx.to_numpy(dtype=np.int64)
        if left.min() < 0 or left.max() >= reference_count or right.min() < 0 or right.max() >= pool_count:
            raise ValueError("Candidate indices are outside reference/pool records")
        if left[0] < previous_left or np.any(left[1:] < left[:-1]):
            raise ValueError("Candidate Parquet must be globally ordered by left_idx")
        previous_left = int(left[-1])
        return frame

    current, offset = next_frame(), 0
    for start in range(0, stop_reference, reference_batch_size):
        end = min(start + reference_batch_size, stop_reference)
        pieces, pair_count = [], 0
        while current is not None:
            left = current.left_idx.to_numpy(dtype=np.int64)
            stop = int(np.searchsorted(left, end, side="left"))
            if stop > offset:
                pieces.append(current.iloc[offset:stop])
                pair_count += stop - offset
                if pair_count > max_pairs_per_batch:
                    raise ValueError("A complete reference batch exceeds --max-pairs-per-batch; "
                                     "reduce --reference-batch-size or raise the explicit pair limit")
            offset = stop
            if offset == len(current):
                current, offset = next_frame(), 0
            else:
                break
        if pieces:
            pairs = pd.concat(pieces, ignore_index=True) if len(pieces) > 1 else pieces[0].reset_index(drop=True)
            if pairs.duplicated(["left_idx", "right_idx"]).any():
                raise ValueError("Duplicate candidate pair, possibly spanning Parquet row groups")
            left = pairs.left_idx.to_numpy(dtype=np.int64)
            right = pairs.right_idx.to_numpy(dtype=np.int64)
            if np.any((left[1:] == left[:-1]) & (right[1:] < right[:-1])):
                if require_sorted_right:
                    raise ValueError("Candidate right_idx is not ordered within its S1 group; aligned scores cannot be reordered")
                pairs = pairs.sort_values(["left_idx", "right_idx"], kind="stable").reset_index(drop=True)
        else:
            pairs = empty.copy()
        yield CandidateBatch(start, end, pairs)
    if stop_reference == reference_count and current is not None:
        raise ValueError("Candidate input has unconsumed rows after the final reference")


def compact_batch(reference: pd.DataFrame, pool: pd.DataFrame,
                  batch: CandidateBatch) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Only within-S1 relative features exist, so this reindexing is exact."""
    left = reference.iloc[batch.start:batch.end][RECORD_COLUMNS].reset_index(drop=True)
    rows, inverse = np.unique(batch.pairs.right_idx.to_numpy(dtype=np.int64), return_inverse=True)
    right = pool.iloc[rows][RECORD_COLUMNS].reset_index(drop=True)
    pairs = batch.pairs.copy()
    pairs["left_idx"] = pairs.left_idx.to_numpy(dtype=np.int64) - batch.start
    pairs["right_idx"] = inverse
    return left, right, pairs


def initialize_worker(checkpoint: str, expected_hash: str) -> None:
    global _WORKER_BUNDLE, _WORKER_THREAD_LIMIT
    from threadpoolctl import threadpool_limits

    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    _WORKER_THREAD_LIMIT = threadpool_limits(limits=1)
    path = Path(checkpoint)
    if file_digest(path) != expected_hash:
        raise ValueError("Selected matcher checkpoint changed before worker initialization")
    _WORKER_BUNDLE = joblib.load(path)
    if file_digest(path) != expected_hash:
        raise ValueError("Selected matcher checkpoint changed while loading a worker")


def score_compact_batch(reference: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame) -> tuple[np.ndarray, dict]:
    if _WORKER_BUNDLE is None:
        raise RuntimeError("Frozen inference worker was not initialized")
    started = time.perf_counter()
    if pairs.empty:
        return np.empty(0, dtype=np.float32), {"feature_seconds": 0., "predict_seconds": 0.}
    features = feature_batch(reference, pool, pairs, _WORKER_BUNDLE["feature_columns"])
    feature_seconds = time.perf_counter() - started
    started = time.perf_counter()
    model = _WORKER_BUNDLE["model"]
    if type(model).__module__.startswith("catboost"):
        prediction = np.asarray(model.predict_proba(features, thread_count=1))
        if prediction.shape != (len(pairs), 2):
            raise ValueError("Frozen matcher returned an invalid probability shape")
        values = prediction[:, 1].astype(np.float32)
    else:
        values = predict_proba(model, features)
    if not np.isfinite(values).all():
        raise ValueError("Frozen matcher returned non-finite probabilities")
    anchor = float(_WORKER_BUNDLE.get("anchor_threshold", 0))
    if anchor > 0:
        _, inverse = np.unique(pairs.left_idx.to_numpy(), return_inverse=True)
        maxima = np.full(inverse.max() + 1, -1., dtype=np.float32)
        np.maximum.at(maxima, inverse, values)
        values = np.where(maxima[inverse] >= anchor, values, np.float32(-1))
    return values, {"feature_seconds": feature_seconds, "predict_seconds": time.perf_counter() - started}


def _prefix_pair_count(path: Path, limit: int, total_references: int) -> int:
    """A cheap index-column pass for explicitly requested benchmark prefixes."""
    parquet = pq.ParquetFile(path)
    if limit == total_references:
        return parquet.metadata.num_rows
    _candidate_schema(parquet)
    count, previous = 0, -1
    for batch in parquet.iter_batches(columns=["left_idx"], batch_size=250000, use_threads=False):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        if len(values) and (values[0] < previous or np.any(values[1:] < values[:-1])):
            raise ValueError("Benchmark candidate input is not ordered by left_idx")
        position = int(np.searchsorted(values, limit))
        count += position
        if position < len(values):
            break
        if len(values):
            previous = values[-1]
    return count


def _fsync(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _file_identity(path: Path) -> dict:
    stat = path.stat()
    return {"path": str(path.resolve()), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def accepted_mask(values: np.ndarray, threshold: float) -> np.ndarray:
    # NumPy's float32 array/scalar comparison can round a float64 threshold down.
    # Preserve the exact frozen decision rule used by ber.metrics/ber.predict.
    return np.asarray(values, dtype=np.float64) >= threshold


def _prepare_pipeline(args, checkpoint_hash: str, output: Path) -> dict | None:
    path_value = getattr(args, "pipeline", None)
    if not path_value:
        return None
    path = Path(path_value).resolve()
    config, digest = json.loads(path.read_text()), file_digest(path)
    for name in ("alpha", "gate"):
        if not np.isfinite(config[name]) or not 0 <= config[name] <= 1:
            raise ValueError("Frozen pipeline alpha and gate must lie in [0,1]")
    if not np.isfinite(config["threshold"]) or config["threshold"] <= 0:
        raise ValueError("Frozen blend threshold must be finite and positive")
    if any(type(config[name]) is not int or config[name] <= 0 for name in ("max_length", "inference_batch_size")):
        raise ValueError("Frozen CE sequence and inference batch sizes must be positive integers")
    if type(config.get("skip_provably_rejected_pairs", False)) is not bool:
        raise ValueError("Frozen skip_provably_rejected_pairs must be a boolean")
    if config.get("reference_batch_size", args.reference_batch_size) != args.reference_batch_size:
        raise ValueError("Reference batch size differs from the frozen pipeline's CE grouping")
    if "length_bucketing" in config:
        if type(config["length_bucketing"]) is not bool:
            raise ValueError("Frozen length_bucketing must be a boolean")
        multiple = config.get("pad_to_multiple_of", 8)
        if multiple is not None and (type(multiple) is not int or multiple <= 0):
            raise ValueError("Frozen pad_to_multiple_of must be a positive integer or null")
    if config["tree_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Frozen pipeline identifies a different tree checkpoint")
    ce = checkpoint_identity(Path(config["cross_encoder_checkpoint"]))
    if ce["sha256"] != config["cross_encoder_sha256"]:
        raise ValueError("Frozen pipeline CE checkpoint digest mismatch")
    runtime = {"device": getattr(args, "ce_device", "cuda"), "threads": getattr(args, "ce_threads", 2),
               "interop_threads": 1, "reference_batch_size": args.reference_batch_size,
               "inference_batch_size": config["inference_batch_size"], "max_length": config["max_length"],
               "gpu_memory_fraction": getattr(args, "ce_gpu_memory_fraction", None),
               "inference_backend": "batched" if "length_bucketing" in config else "legacy",
               "skip_provably_rejected_pairs": config.get("skip_provably_rejected_pairs", False),
               "length_bucketing": config.get("length_bucketing"),
               "pad_to_multiple_of": config.get("pad_to_multiple_of", 8) if "length_bucketing" in config else None}
    shutil.copyfile(path, output / "frozen_pipeline.json")
    if file_digest(output / "frozen_pipeline.json") != digest:
        raise ValueError("Frozen pipeline changed while being copied")
    value = {"pipeline": config, "pipeline_path": str(path), "pipeline_sha256": digest,
             "cross_encoder_identity": ce, "runtime": runtime, "labels_accessed": False}
    atomic_json(output / "frozen_pipeline_manifest.json", value)
    return value


def _load_ce_scorer(pipeline: dict):
    from .cross_encoder_experiment import load_reranker, score_reranker
    import torch

    runtime, config = pipeline["runtime"], pipeline["pipeline"]
    torch.set_num_threads(runtime["threads"])
    torch.set_num_interop_threads(1)
    if runtime["device"].startswith("cuda") and runtime["gpu_memory_fraction"] is not None:
        torch.cuda.set_per_process_memory_fraction(runtime["gpu_memory_fraction"])
    model, tokenizer = load_reranker(config["cross_encoder_checkpoint"], runtime["device"])
    options = SimpleNamespace(device=runtime["device"], max_length=config["max_length"],
                              inference_batch_size=config["inference_batch_size"],
                              length_bucketing=config.get("length_bucketing", False),
                              pad_to_multiple_of=config.get("pad_to_multiple_of", 8))
    scorer = score_reranker
    if "length_bucketing" in config:
        from .rerank_inference import score_reranker_batched

        scorer = score_reranker_batched
    return lambda reference, pool, pairs: scorer(model, tokenizer, reference, pool, pairs,
                                                 np.ones(len(pairs), dtype=bool), options)


def run(args, *, ce_scorer=None) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    import fcntl

    with (output / "prediction.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another predictor owns this output directory") from error
        return _run_locked(args, output, ce_scorer=ce_scorer)


def _run_locked(args, output: Path, *, ce_scorer=None) -> dict:
    started = time.perf_counter()
    data = Path(args.data)
    if (output / "complete.json").exists():
        raise ValueError("Output already contains completed predictions; choose a new directory")
    checkpoint, pairs_path = Path(args.checkpoint).resolve(), Path(args.pairs).resolve()
    inputs = {label: _file_identity(path) for label, path in
              [("reference", data / "reference.parquet"), ("pool", data / "pool.parquet"), ("pairs", pairs_path)]}
    checkpoint_hash = file_digest(checkpoint)
    bundle = joblib.load(checkpoint)
    threshold, anchor_threshold = float(bundle["threshold"]), float(bundle.get("anchor_threshold", 0))
    if not np.isfinite(threshold) or not 0 <= threshold <= np.nextafter(1., np.inf):
        raise ValueError("Frozen threshold must be in [0,1] or the nextafter(1,+inf) reject-all value")
    if not np.isfinite(anchor_threshold) or not 0 <= anchor_threshold <= 1:
        raise ValueError("Frozen anchor threshold must be finite and between zero and one")
    feature_columns = list(bundle["feature_columns"])
    if not feature_columns or not all(isinstance(value, str) for value in feature_columns):
        raise ValueError("Frozen matcher must include its ordered feature columns")
    del bundle
    gc.collect()
    pipeline = _prepare_pipeline(args, checkpoint_hash, output)
    tree_threshold = threshold
    if pipeline is not None:
        threshold = float(pipeline["pipeline"]["threshold"])
    reference, pool = load_records(data)
    if args.expected_references is not None and len(reference) != args.expected_references:
        raise ValueError("Reference count differs from --expected-references")
    benchmark = args.benchmark_references is not None
    reference_count = min(args.benchmark_references, len(reference)) if benchmark else len(reference)
    pair_count = _prefix_pair_count(pairs_path, reference_count, len(reference))
    reuse_pairs = bool(getattr(args, "reuse_sorted_pairs", False))
    save_cross = bool(getattr(args, "save_cross_probabilities", False))
    if reuse_pairs and reference_count != len(reference):
        raise ValueError("A benchmark prefix cannot hardlink the full candidate file")
    config = {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
              "threshold": threshold, "anchor_threshold": anchor_threshold,
              "feature_columns": feature_columns, "reference_count": reference_count,
              "full_reference_count": len(reference), "pool_count": len(pool), "candidate_pairs": pair_count,
              "data": str(data.resolve()), "pairs": str(pairs_path), "workers": args.workers,
              "input_identities": inputs,
              "max_pending_batches": args.workers * 2, "benchmark_only": benchmark,
              "reference_batch_size": args.reference_batch_size, "pair_read_batch_size": args.pair_read_batch_size,
              "max_pairs_per_batch": args.max_pairs_per_batch,
              "pair_order": ["left_idx", "right_idx"], "probability_order": "scored_pairs.parquet row order",
              "threshold_changed": False, "labels_accessed": False, "scorer": "frozen_tree"}
    config["pair_artifact_mode"] = "hardlink" if reuse_pairs else "streamed_rewrite"
    if pipeline is not None:
        config.update(scorer="frozen_tree_ce", tree_threshold=tree_threshold,
                      pipeline_sha256=pipeline["pipeline_sha256"], pipeline=pipeline["pipeline"],
                      cross_encoder_sha256=pipeline["cross_encoder_identity"]["sha256"], ce_runtime=pipeline["runtime"],
                      cross_probabilities_saved=save_cross)
    atomic_json(output / "prediction_config.json", config)
    probabilities_path = output / "probabilities.npy.tmp"
    probabilities = np.lib.format.open_memmap(probabilities_path, mode="w+", dtype=np.float32, shape=(pair_count,))
    base_probabilities = cross_probabilities = None
    if pipeline is not None:
        base_probabilities = np.lib.format.open_memmap(output / "base_probabilities.npy.tmp", mode="w+", dtype=np.float32, shape=(pair_count,))
        if save_cross:
            cross_probabilities = np.lib.format.open_memmap(output / "cross_probabilities.npy.tmp", mode="w+", dtype=np.float32, shape=(pair_count,))
    pool_ids, reference_ids = pool.entity_id.to_numpy(), reference.entity_id.to_numpy()
    pairs_temporary = output / "scored_pairs.parquet.tmp"
    match_temporary, candidate_temporary = output / "matching_results.tsv.tmp", output / "candidate_pairs.tsv.tmp"
    batches = iter_reference_batches(pairs_path, len(reference), len(pool), args.reference_batch_size,
                                     args.pair_read_batch_size, args.max_pairs_per_batch, reference_count,
                                     require_sorted_right=reuse_pairs)
    pending, executor, writer = deque(), None, None
    read_prepare_seconds = worker_feature_seconds = worker_predict_seconds = 0.
    row_offset = references_written = accepted_pairs = batches_written = 0
    reranked_pairs, ce_seconds = 0, 0.
    pipeline_started = time.perf_counter()
    try:
        if args.workers == 1:
            initialize_worker(str(checkpoint), checkpoint_hash)
        else:
            executor = ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                           initializer=initialize_worker, initargs=(str(checkpoint), checkpoint_hash))
        if pipeline is not None:
            # Imported only in the parent, avoiding the module cycle and keeping
            # CUDA/model state out of the spawned CPU feature workers.
            from .rerank_frozen import rerank_batch

            if ce_scorer is None:
                ce_scorer = _load_ce_scorer(pipeline)
        with match_temporary.open("w", newline="") as matches_stream, candidate_temporary.open("w", newline="") as candidates_stream:
            matches, candidates = csv.writer(matches_stream, delimiter="\t"), csv.writer(candidates_stream, delimiter="\t")
            matches.writerow(["source1_entity_id", "matched_entity_ids"])
            candidates.writerow(["source1_entity_id", "candidate_entity_ids"])
            exhausted = False
            while pending or not exhausted:
                while not exhausted and len(pending) < (args.workers * 2 if executor is not None else 1):
                    preparation_started = time.perf_counter()
                    try:
                        batch = next(batches)
                    except StopIteration:
                        exhausted = True
                        break
                    compact = compact_batch(reference, pool, batch)
                    read_prepare_seconds += time.perf_counter() - preparation_started
                    if executor is None:
                        result = score_compact_batch(*compact)
                    else:
                        result = executor.submit(score_compact_batch, *compact)
                    pending.append((batch, result))
                if not pending:
                    break
                batch, result = pending.popleft()
                values, timing = result.result() if executor is not None else result
                if values.shape != (len(batch.pairs),) or not np.isfinite(values).all():
                    raise ValueError("Worker probability output lost pair alignment")
                worker_feature_seconds += timing["feature_seconds"]
                worker_predict_seconds += timing["predict_seconds"]
                end_offset = row_offset + len(values)
                if end_offset > pair_count:
                    raise ValueError("Scored more candidate rows than the input Parquet records")
                if pipeline is not None:
                    base_probabilities[row_offset:end_offset] = values
                    ce_started = time.perf_counter()
                    values, cross, gated = rerank_batch(reference, pool, batch.pairs, values, pipeline["pipeline"], ce_scorer)
                    ce_seconds += time.perf_counter() - ce_started
                    if cross_probabilities is not None:
                        cross_probabilities[row_offset:end_offset] = cross
                    reranked_pairs += int(gated.sum())
                probabilities[row_offset:end_offset] = values
                if not reuse_pairs:
                    table = pa.Table.from_pandas(batch.pairs, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(pairs_temporary, table.schema, compression="zstd")
                    writer.write_table(table, row_group_size=max(1, len(batch.pairs)))
                left = batch.pairs.left_idx.to_numpy(dtype=np.int64)
                right = batch.pairs.right_idx.to_numpy(dtype=np.int64)
                boundaries = np.searchsorted(left, np.arange(batch.start, batch.end + 1, dtype=np.int64))
                for index, entity in enumerate(range(batch.start, batch.end)):
                    first, last = boundaries[index:index + 2]
                    ids = pool_ids[right[first:last]]
                    accepted = accepted_mask(values[first:last], threshold)
                    matches.writerow([reference_ids[entity], ",".join(sorted(ids[accepted]))])
                    candidates.writerow([reference_ids[entity], ",".join(sorted(ids))])
                    references_written += 1
                    accepted_pairs += int(accepted.sum())
                row_offset, batches_written = end_offset, batches_written + 1
                if batches_written == 1 or batches_written % 10 == 0:
                    elapsed = time.perf_counter() - pipeline_started
                    log(f"Streaming frozen prediction: {batch.end:,}/{reference_count:,} S1; "
                        f"{row_offset:,} pairs; pipeline={elapsed:.1f}s")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        if writer is not None:
            writer.close()
    if row_offset != pair_count or references_written != reference_count:
        raise ValueError("Final reference/candidate counts do not match the selected input")
    if reuse_pairs:
        # Do not silently allocate another multi-GB copy if the caller requested
        # zero-copy reuse. Keep outputs on the candidate filesystem for this mode.
        os.link(pairs_path, pairs_temporary)
    elif writer is None:
        empty = pa.Table.from_batches([], schema=pq.ParquetFile(pairs_path).schema_arrow)
        pq.write_table(empty, pairs_temporary, compression="zstd")
    probabilities.flush()
    if pipeline is not None:
        base_probabilities.flush()
        if cross_probabilities is not None:
            cross_probabilities.flush()
        if file_digest(Path(pipeline["pipeline_path"])) != pipeline["pipeline_sha256"]:
            raise ValueError("Frozen pipeline changed during streaming inference")
        if file_digest(output / "frozen_pipeline.json") != pipeline["pipeline_sha256"]:
            raise ValueError("Saved pipeline identity changed during streaming inference")
        if checkpoint_identity(Path(pipeline["pipeline"]["cross_encoder_checkpoint"]))["sha256"] != pipeline["cross_encoder_identity"]["sha256"]:
            raise ValueError("Frozen CE checkpoint changed during streaming inference")
    if file_digest(checkpoint) != checkpoint_hash:
        raise ValueError("Frozen checkpoint changed during prediction")
    if any(_file_identity(Path(identity["path"])) != identity for identity in inputs.values()):
        raise ValueError("Reference, pool, or candidate input changed during prediction")
    completed_files = [(probabilities_path, output / "probabilities.npy"),
                       (pairs_temporary, output / "scored_pairs.parquet"),
                       (match_temporary, output / "matching_results.tsv"),
                       (candidate_temporary, output / "candidate_pairs.tsv")]
    if pipeline is not None:
        arrays = ["base_probabilities"] + (["cross_probabilities"] if save_cross else [])
        completed_files.extend((output / f"{name}.npy.tmp", output / f"{name}.npy") for name in arrays)
    for temporary, final in completed_files:
        _fsync(temporary)
        temporary.replace(final)
    elapsed = time.perf_counter() - started
    report = {**config, "reference_count": references_written, "candidate_pairs": row_offset,
              "accepted_pairs": accepted_pairs, "elapsed_seconds": elapsed,
              "pipeline_seconds": time.perf_counter() - pipeline_started,
              "pairs_per_second_including_load": row_offset / max(elapsed, 1e-6),
              "read_prepare_seconds": read_prepare_seconds,
              "worker_feature_seconds_sum": worker_feature_seconds,
              "worker_predict_seconds_sum": worker_predict_seconds,
              "all_selected_references_written": True, "ready_for_submission": False,
              "validation_required": "Run the official validator and verify the selected full test population."}
    if pipeline is not None:
        report.update(reranked_pairs=reranked_pairs, ce_seconds=ce_seconds,
                      base_probability_order="scored_pairs.parquet row order")
    atomic_json(output / "report.json", report)
    atomic_json(output / "complete.json", {"state": "complete", "report": report})
    log(json.dumps({key: report[key] for key in ("scorer", "reference_count", "candidate_pairs", "accepted_pairs",
                                               "workers", "benchmark_only", "elapsed_seconds",
                                               "pairs_per_second_including_load")}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--reference-batch-size", type=int, default=1000)
    parser.add_argument("--pair-read-batch-size", type=int, default=100000)
    parser.add_argument("--max-pairs-per-batch", type=int, default=200000)
    parser.add_argument("--expected-references", type=int)
    parser.add_argument("--benchmark-references", type=int,
                        help="Explicit development-only S1 prefix; output is not a full submission")
    parser.add_argument("--pipeline", help="Optional frozen CE blend JSON; overlaps parent GPU reranking with CPU workers")
    parser.add_argument("--ce-device", default="cuda")
    parser.add_argument("--ce-threads", type=int, default=2)
    parser.add_argument("--ce-gpu-memory-fraction", type=float, help="Optional bounded CUDA allocator fraction for a shared GPU")
    parser.add_argument("--reuse-sorted-pairs", action="store_true",
                        help="Require already sorted pairs and hardlink the immutable full input instead of rewriting it")
    parser.add_argument("--save-cross-probabilities", action="store_true",
                        help="Optional diagnostic CE array; normally only base and final probabilities are stored")
    args = parser.parse_args()
    if min(args.workers, args.reference_batch_size, args.pair_read_batch_size, args.max_pairs_per_batch, args.ce_threads) <= 0:
        parser.error("Worker and batch limits must be positive")
    if args.benchmark_references is not None and args.benchmark_references <= 0:
        parser.error("--benchmark-references must be positive")
    if args.expected_references is not None and args.expected_references < 0:
        parser.error("--expected-references must be nonnegative")
    if args.ce_gpu_memory_fraction is not None and not 0 < args.ce_gpu_memory_fraction <= 1:
        parser.error("--ce-gpu-memory-fraction must lie in (0,1]")
    if args.save_cross_probabilities and not args.pipeline:
        parser.error("--save-cross-probabilities requires --pipeline")
    run(args)


if __name__ == "__main__":
    main()
