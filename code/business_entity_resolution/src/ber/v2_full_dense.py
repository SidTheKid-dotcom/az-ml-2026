"""Checkpointed, label-free E5 query encoding and exact full-pool dense top-50.

``screen`` searches the first 1,000 *original medium TUNE* references against
all 10,320,219 targets, comparing retrieval to the saved v1 TUNE candidates
without reading any truth. ``full`` mines every future-fit-eligible S1 and
every HOLDOUT-2 S1; holdout candidates are never scored here. The frozen pool
vectors and country cache are opened read-only. Query vectors, top-K snapshots,
complete pairs, and their manifests are mirrored under the v2 durable directory.

The search is pool-outer: each 50,000-row pool tile is read once from the slow
workspace store; 512-query/50,000-target score tiles use float32 dot products
from the saved float16 pool, with country blocking and no truth injection.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .full_embeddings import Reporter, atomic_json, checkpoint_identity, embedding_texts, file_digest
from .normalization import normalize_text
from .v2_reference import POPULATION_COLUMNS, RECORD_COLUMNS, id_sha256


def _flush(path: Path, array=None) -> None:
    if array is not None:
        array.flush()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _copy_verified(src: Path, dest: Path, sha256: str) -> None:
    if src.resolve() == dest.resolve():
        return
    if dest.exists() and file_digest(dest) == sha256:
        return
    tmp = dest.with_name(dest.name + ".copying")
    if tmp.exists():
        # The prior attempt did not publish this file; it is owned by this job.
        tmp.unlink()
    with src.open("rb") as source, tmp.open("xb") as output:
        shutil.copyfileobj(source, output, length=4 * 1024 * 1024)
        output.flush()
        os.fsync(output.fileno())
    if file_digest(tmp) != sha256:
        raise ValueError(f"Durable copy SHA256 mismatch: {dest}")
    tmp.replace(dest)


def _manifest(reporter: Reporter, value: dict, name: str) -> None:
    # Durable marker first: a local marker must never claim a checkpoint that
    # has not yet reached the durable project directory.
    for directory in reversed(reporter.directories):
        atomic_json(directory / name, value)


def _sha_array(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def _storage_room(path: Path, required: int) -> None:
    free = shutil.disk_usage(path).free
    if free < required + 4_000_000_000:
        raise ValueError(f"Scratch would fall below 4 GB: free={free:,}, needed={required:,}")


def _input_checks(args) -> dict:
    ref_manifest = json.loads((args.reference_root / "reference_manifest.json").read_text())
    if not ref_manifest.get("complete") or ref_manifest.get("labels_read") is not False:
        raise ValueError("Reference population must be complete and label-free")
    name = "medium_tune_reference.parquet" if args.mode == "screen" else "reference.parquet"
    path = args.reference_root / name
    if file_digest(path) != ref_manifest["parquet_sha256"][name]:
        raise ValueError("Reference parquet differs from its sealed manifest")
    if args.mode == "screen":
        source = pd.read_parquet(path, columns=POPULATION_COLUMNS)
        reference = source.iloc[:args.screen_rows].reset_index(drop=True)
        if len(reference) != args.screen_rows or not reference.population.eq("medium_tune").all():
            raise ValueError("Screen must contain exactly the requested original medium TUNE rows")
        screen_file = args.output / "reference.parquet"
        if screen_file.exists():
            saved = pd.read_parquet(screen_file, columns=["entity_id"])
            if saved.entity_id.tolist() != reference.entity_id.tolist():
                raise ValueError("Existing screen belongs to different TUNE reference IDs")
        else:
            temporary = screen_file.with_name(screen_file.name + ".tmp")
            if temporary.exists():
                raise ValueError("Incomplete screen reference file needs inspection")
            reference.to_parquet(temporary, index=False)
            temporary.replace(screen_file)
        digest = file_digest(screen_file)
        _copy_verified(screen_file, args.durable_output / "reference.parquet", digest)
        reference_sha = digest
        n = len(reference)
        selected_sha = id_sha256(reference.entity_id.tolist())
    else:
        reference_sha = ref_manifest["parquet_sha256"]["reference.parquet"]
        n = ref_manifest["population"]["rows"]
        selected_sha = ref_manifest["population"]["row_order_id_sha256_with_terminal_lf"]
        if pq.ParquetFile(path).metadata.num_rows != n:
            raise ValueError("Main reference file lost raw Source 1 records")
    pool_complete = json.loads(args.pool_complete.read_text())
    pool_identity = pool_complete["identity"]
    preservation = json.loads(args.pool_preservation.read_text())
    if (pool_complete.get("state") != "complete" or not pool_complete.get("full_dataset_complete")
            or not pool_complete.get("verification", {}).get("all_finite")
            or not preservation.get("copy_verified")
            or Path(preservation["durable"]).resolve() != args.pool_embeddings.resolve()
            or args.pool_embeddings.stat().st_size != preservation["bytes"]):
        raise ValueError("Full, preserved pool embedding identity is incomplete or relocated unexpectedly")
    pool = np.load(args.pool_embeddings, mmap_mode="r")
    if (pool.shape != (pool_identity["target_rows"], pool_identity["dimension"])
            or pool.dtype != np.float16 or pool_identity["checkpoint"]["sha256"]
            != checkpoint_identity(args.model)["sha256"]):
        raise ValueError("Frozen E5 checkpoint or saved full-pool embeddings differ")
    for item in pool_identity["inputs_in_order"]:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("A raw S2/S3 pool file changed since the frozen pool was encoded")
    country_manifest = json.loads(args.country_cache.with_suffix(".json").read_text())
    if (country_manifest["config"]["inputs_in_order"] != pool_identity["inputs_in_order"]
            or country_manifest["config"]["rows"] != pool_identity["target_rows"]
            or country_manifest["config"]["normalization"] != "ber.normalize_text_v1"):
        raise ValueError("Existing country cache does not align with frozen pool vectors")
    country_codes = np.load(args.country_cache, mmap_mode="r")
    if country_codes.shape != (len(pool),) or country_codes.dtype != np.int32:
        raise ValueError("Existing pool country cache shape/dtype is incorrect")
    if len(set(country_manifest["labels"])) != len(country_manifest["labels"]):
        raise ValueError("Duplicate normalized pool country keys")
    return {
        "schema": 1, "scope": args.mode, "references": n,
        "reference_sha256": reference_sha, "reference_id_sha256": selected_sha,
        "reference_source_manifest_sha256": file_digest(args.reference_root / "reference_manifest.json"),
        "reference_source1_sha256": ref_manifest["identity"]["raw_source1"]["sha256"],
        "pool_embeddings": {"path": str(args.pool_embeddings.resolve()), "bytes": preservation["bytes"],
                            "sha256": preservation["sha256"], "rows": len(pool),
                            "dimension": pool_identity["dimension"], "dtype": "float16"},
        "pool_checkpoint_sha256": pool_identity["checkpoint"]["sha256"],
        "model_path": str(args.model.resolve()), "model_sha256": pool_identity["checkpoint"]["sha256"],
        "country_cache_path": str(args.country_cache.resolve()),
        "country_cache_sha256": file_digest(args.country_cache),
        "country_labels": country_manifest["labels"],
        "prefix": "query: ", "text_format": "business_name | business_address",
        "max_length": pool_identity["max_length"], "model_execution_dtype": "bfloat16",
        "query_storage_dtype": "float32", "query_normalization": "float32_l2",
        "score_precision": "float32_dot_from_float16_pool", "top_k": args.top_k,
        "query_tile": args.query_tile, "pool_tile": args.pool_tile,
        "query_chunk_size": args.query_chunk_size, "encoding_batch_size_requested": args.batch_size,
        "positive_candidate_injection": False, "labels_read": False,
        "holdout_truth_read": False, "medium_eval_truth_read": False,
    }


def _reference_batches(path: Path, batch_size: int):
    reader = pq.ParquetFile(path)
    for record_batch in reader.iter_batches(batch_size=batch_size, columns=RECORD_COLUMNS):
        yield record_batch.to_pandas()


def _model(identity: dict, args):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Full v2 E5 query encoding requires CUDA BF16 support")
    model = SentenceTransformer(str(args.model), device="cuda", local_files_only=True,
                                trust_remote_code=False)
    model.max_seq_length = identity["max_length"]
    model.bfloat16()
    model.eval()
    if model.get_sentence_embedding_dimension() != identity["pool_embeddings"]["dimension"]:
        raise ValueError("E5 query dimension does not match the frozen target pool")
    return model


def _encode_chunk(model, frame: pd.DataFrame, args, batch: int) -> tuple[np.ndarray, int]:
    import torch

    while True:
        try:
            values = model.encode(embedding_texts(frame, "query: "), batch_size=batch,
                                  show_progress_bar=False, convert_to_numpy=True,
                                  normalize_embeddings=True)
            break
        except torch.cuda.OutOfMemoryError:
            if batch <= 128:
                raise
            gc.collect()
            torch.cuda.empty_cache()
            batch = max(128, batch // 2)
            print(f"CUDA OOM in E5 encoding; retrying the same chunk at batch={batch}", flush=True)
    values = np.asarray(values, dtype=np.float32)
    if values.shape != (len(frame), args.dimension) or not np.isfinite(values).all():
        raise ValueError("E5 query encoder returned non-finite or wrongly shaped embeddings")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("E5 query encoder returned a zero vector")
    return values / norms, batch


def _progress(reporter: Reporter, name: str, identity: dict) -> dict | None:
    local, durable = [directory / name for directory in reporter.directories]
    existing = durable if durable.exists() else local
    if not existing.exists():
        return None
    value = json.loads(existing.read_text())
    if value.get("identity") != identity:
        raise ValueError(f"Resume identity mismatch: {name}")
    if local.exists() and json.loads(local.read_text()).get("identity") != identity:
        raise ValueError(f"Local resume identity mismatch: {name}")
    return value


def _open_query_arrays(reporter: Reporter, progress: dict, n: int, dimension: int):
    path, durable_path = (directory / "query_embeddings.npy" for directory in reporter.directories)
    shape = (n, dimension)
    completed = progress["completed_rows"]
    if durable_path.exists():
        durable = np.load(durable_path, mmap_mode="r+")
        if durable.shape != shape or durable.dtype != np.float32:
            raise ValueError("Durable query checkpoint has wrong shape/dtype")
    elif completed:
        raise ValueError("Query progress has no durable embedding checkpoint")
    else:
        durable = np.lib.format.open_memmap(durable_path, mode="w+", dtype=np.float32, shape=shape)
    if path.exists():
        local = np.load(path, mmap_mode="r+")
        if local.shape != shape or local.dtype != np.float32:
            raise ValueError("Local query checkpoint has wrong shape/dtype")
    else:
        _storage_room(path.parent, n * dimension * 4)
        local = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape)
    for item in progress["chunks"]:
        start, stop = item["start"], item["stop"]
        recorded = item["sha256"]
        durable_values = np.asarray(durable[start:stop])
        if _sha_array(durable_values) != recorded:
            raise ValueError(f"Damaged durable query checkpoint at rows {start}:{stop}")
        if _sha_array(np.asarray(local[start:stop])) != recorded:
            local[start:stop] = durable_values
    if completed:
        _flush(path, local)
    return path, local, durable_path, durable


def encode_queries(args, identity: dict, reporter: Reporter) -> dict:
    complete = _progress(reporter, "query_complete.json", identity)
    if complete is not None:
        for directory in reporter.directories:
            path = directory / "query_embeddings.npy"
            if (not path.exists() or path.stat().st_size != complete["embedding_bytes"]
                    or file_digest(path) != complete["embedding_sha256"]):
                raise ValueError("Completed query vectors differ from their recorded checksum")
        return complete
    progress = _progress(reporter, "query_progress.json", identity)
    n, dimension = identity["references"], identity["pool_embeddings"]["dimension"]
    if progress is None:
        progress = {"identity": identity, "state": "encoding", "completed_rows": 0,
                    "chunks": [], "actual_batch_size": args.batch_size,
                    "started_unix": time.time()}
        _manifest(reporter, progress, "query_progress.json")
    completed = progress["completed_rows"]
    cursor = 0
    for item in progress["chunks"]:
        if item["start"] != cursor or item["stop"] <= cursor:
            raise ValueError("Non-contiguous or missing committed query chunks")
        cursor = item["stop"]
    if completed != cursor:
        raise ValueError("Query progress does not match the last committed chunk")
    path, local, durable_path, durable = _open_query_arrays(reporter, progress, n, dimension)
    model = _model(identity, args) if completed < n else None
    reporter.write(f"E5 bf16 encoding: {completed:,}/{n:,}, batch={progress['actual_batch_size']}")
    visited = 0
    try:
        for frame in _reference_batches(args.output / "reference.parquet", args.query_chunk_size):
            start, stop = visited, visited + len(frame)
            visited = stop
            if stop <= completed:
                continue
            if start < completed:
                frame = frame.iloc[completed - start:].reset_index(drop=True)
                start = completed
            if start != completed or stop > n:
                raise ValueError("Query chunk did not resume on a contiguous reference row")
            tic = time.time()
            values, actual_batch = _encode_chunk(model, frame, args, progress["actual_batch_size"])
            sha = _sha_array(values)
            local[start:stop] = values
            durable[start:stop] = values
            _flush(path, local)
            _flush(durable_path, durable)
            completed = stop
            progress["completed_rows"] = completed
            progress["chunks"].append({"start": start, "stop": stop, "sha256": sha,
                                       "actual_batch_size": actual_batch})
            progress["actual_batch_size"] = actual_batch
            progress["last_entity_id"] = str(frame.entity_id.iloc[-1])
            progress["last_chunk_seconds"] = round(time.time() - tic, 3)
            progress["last_update_unix"] = time.time()
            _manifest(reporter, progress, "query_progress.json")
            reporter.write(f"E5 bf16 rows={completed:,}/{n:,}, batch={actual_batch}, "
                           f"chunk_seconds={progress['last_chunk_seconds']:.1f}")
            del values
    finally:
        del model
        gc.collect()
        import torch
        torch.cuda.empty_cache()
    if visited != n or completed != n:
        raise ValueError("Reference parquet row count differs from the sealed population")
    min_norm, max_norm = float("inf"), 0.
    for start in range(0, n, args.query_chunk_size):
        batch = np.asarray(local[start:start + args.query_chunk_size], dtype=np.float32)
        if not np.isfinite(batch).all():
            raise ValueError("Non-finite committed E5 query vector")
        norms = np.linalg.norm(batch, axis=1)
        if np.any(np.abs(norms - 1.) > .01):
            raise ValueError("A committed query vector is not normalized")
        min_norm, max_norm = min(min_norm, float(norms.min())), max(max_norm, float(norms.max()))
    _flush(path, local)
    _flush(durable_path, durable)
    del local, durable
    sha = file_digest(path)
    if file_digest(durable_path) != sha:
        raise ValueError("Complete scratch and durable query vectors disagree")
    report = {"identity": identity, "state": "complete", "completed_rows": n,
              "embedding_sha256": sha, "embedding_bytes": path.stat().st_size,
              "chunks": len(progress["chunks"]), "actual_batch_size": progress["actual_batch_size"],
              "min_l2_norm": min_norm, "max_l2_norm": max_norm,
              "elapsed_seconds": time.time() - progress["started_unix"]}
    _manifest(reporter, report, "query_complete.json")
    reporter.write(f"COMPLETE E5 bf16 queries={n:,}, sha256={sha}")
    return report


def _query_country_codes(path: Path, n: int, labels: list[str], chunk_size: int) -> tuple[np.ndarray, dict]:
    codes = np.full(n, -1, dtype=np.int8)
    lookup = {name: code for code, name in enumerate(labels)}
    counts = Counter()
    offset = 0
    for record in pq.ParquetFile(path).iter_batches(batch_size=chunk_size, columns=["country"]):
        frame = record.to_pandas()
        normalized = frame.country.map({c: normalize_text(c) for c in frame.country.unique()})
        values = normalized.map(lookup).fillna(-1).to_numpy(dtype=np.int8)
        codes[offset:offset + len(values)] = values
        counts.update(frame.country.value_counts().to_dict())
        offset += len(values)
    if offset != n:
        raise ValueError("Query country codes lost or duplicated reference rows")
    return codes, dict(sorted(counts.items()))


def _mining_config(identity: dict, query_complete: dict, args) -> dict:
    return {"schema": 1, "input_identity": identity,
            "query_embedding_sha256": query_complete["embedding_sha256"],
            "country_blocked": True, "backend": "torch", "device": "cuda",
            "scores": "float32 dot product of float32 normalized bf16 queries and frozen fp16 pool",
            "query_tile": args.query_tile, "pool_tile": args.pool_tile,
            "top_k": args.top_k, "checkpoint_tiles": args.checkpoint_tiles,
            "labels_read": False, "positive_candidate_injection": False}


def _checkpoint_names(tiles: int, final: bool) -> tuple[str, str]:
    if final:
        return "topk_raw_rows.npy", "topk_scores.npy"
    return f"topk_raw_rows_{tiles:04d}.npy", f"topk_scores_{tiles:04d}.npy"


def _write_snapshot(reporter: Reporter, scores, rows, n: int, k: int,
                    tiles: int, final: bool, block_rows: int = 20_000) -> dict:
    names = _checkpoint_names(tiles, final)
    _storage_room(reporter.directories[0], n * k * 8)
    output = reporter.directories[0]
    paths = [output / name for name in names]
    # If interrupted between writing this snapshot and advancing the manifest,
    # the same locked job must recompute the incomplete, uncommitted files.
    for path in paths:
        if path.exists():
            path.unlink()
    local_rows = np.lib.format.open_memmap(paths[0], mode="w+", dtype=np.int32, shape=(n, k))
    local_scores = np.lib.format.open_memmap(paths[1], mode="w+", dtype=np.float32, shape=(n, k))
    for start in range(0, n, block_rows):
        stop = min(start + block_rows, n)
        local_rows[start:stop] = rows[start:stop].cpu().numpy()
        local_scores[start:stop] = scores[start:stop].cpu().numpy()
    for path, values in zip(paths, (local_rows, local_scores)):
        _flush(path, values)
    del local_rows, local_scores
    files = {name: {"bytes": path.stat().st_size, "sha256": file_digest(path)}
             for name, path in zip(names, paths)}
    for name, details in files.items():
        _copy_verified(output / name, reporter.directories[-1] / name, details["sha256"])
    return {"tiles": tiles, "files": files, "final": final}


def _restore_snapshot(reporter: Reporter, snapshot: dict, scores, rows, n: int, k: int) -> None:
    for name, details in snapshot["files"].items():
        durable = reporter.directories[-1] / name
        if durable.stat().st_size != details["bytes"] or file_digest(durable) != details["sha256"]:
            raise ValueError(f"Damaged durable top-K checkpoint: {durable}")
        local = reporter.directories[0] / name
        if not local.exists() or local.stat().st_size != details["bytes"] or file_digest(local) != details["sha256"]:
            _copy_verified(durable, local, details["sha256"])
    raw_name = next(name for name in snapshot["files"] if "raw_rows" in name)
    score_name = next(name for name in snapshot["files"] if "scores" in name)
    raw = np.load(reporter.directories[0] / raw_name, mmap_mode="r")
    values = np.load(reporter.directories[0] / score_name, mmap_mode="r")
    if raw.shape != (n, k) or raw.dtype != np.int32 or values.shape != (n, k) or values.dtype != np.float32:
        raise ValueError("Top-K checkpoint is not aligned with this reference population")
    for start in range(0, n, 20_000):
        stop = min(n, start + 20_000)
        rows[start:stop] = np.asarray(raw[start:stop], dtype=np.int32)
        scores[start:stop] = np.asarray(values[start:stop], dtype=np.float32)


def _retire_snapshot(reporter: Reporter, previous: dict | None, current: dict) -> None:
    if previous is None or previous["files"] == current["files"]:
        return
    for directory in reporter.directories:
        for name, details in previous["files"].items():
            path = directory / name
            if file_digest(path) != details["sha256"]:
                raise ValueError("Refusing to remove a modified previous top-K checkpoint")
            path.unlink()


def _resume_config(reporter: Reporter, config: dict) -> dict:
    existing = _progress(reporter, "mining_progress.json", config)
    if existing is not None:
        return existing
    for name in ("topk_raw_rows.npy", "topk_scores.npy"):
        if (reporter.directories[0] / name).exists() or (reporter.directories[-1] / name).exists():
            raise ValueError("Uncommitted top-K result exists without a mining manifest")
    value = {"identity": config, "state": "mining", "pool_rows_scanned": 0,
             "pool_tiles_scanned": 0, "checkpoint": None,
             "comparisons": 0, "started_unix": time.time()}
    _manifest(reporter, value, "mining_progress.json")
    return value


def _prepare_gpu(args, n: int, k: int, query_path: Path, query_codes: np.ndarray,
                 pool_labels: list[str]):
    import torch

    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")  # Preserve the original float32 score semantics.
    gpu_queries = torch.empty((n, args.dimension), dtype=torch.float32, device="cuda")
    query_memmap = np.load(query_path, mmap_mode="r")
    for start in range(0, n, 40_000):
        stop = min(start + 40_000, n)
        gpu_queries[start:stop] = torch.as_tensor(np.asarray(query_memmap[start:stop]), device="cuda")
    best_scores = torch.full((n, k), -float("inf"), dtype=torch.float32, device="cuda")
    best_rows = torch.full((n, k), -1, dtype=torch.int32, device="cuda")
    groups = {code: torch.as_tensor(np.flatnonzero(query_codes == code), dtype=torch.long, device="cuda")
              for code in range(len(pool_labels)) if np.any(query_codes == code)}
    return gpu_queries, best_scores, best_rows, groups


def _update_topk(queries, best_scores, best_rows, qrows, pool_vectors,
                 raw_rows, query_tile: int, k: int) -> None:
    import torch

    if not len(qrows) or not len(raw_rows):
        return
    block_k = min(k, len(raw_rows))
    raw_tensor = torch.as_tensor(raw_rows, dtype=torch.int32, device="cuda")
    with torch.inference_mode():
        for start in range(0, len(qrows), query_tile):
            index = qrows[start:start + query_tile]
            scores = queries.index_select(0, index) @ pool_vectors.T
            values, positions = torch.topk(scores, block_k, dim=1)
            candidates = torch.cat((best_scores.index_select(0, index), values), dim=1)
            candidate_rows = torch.cat((best_rows.index_select(0, index), raw_tensor[positions]), dim=1)
            selected, order = torch.topk(candidates, k, dim=1)
            best_scores.index_copy_(0, index, selected)
            best_rows.index_copy_(0, index, torch.gather(candidate_rows, 1, order))


def mine_topk(args, identity: dict, query_complete: dict, reporter: Reporter) -> dict:
    import torch

    config = _mining_config(identity, query_complete, args)
    completed = _progress(reporter, "topk_complete.json", config)
    if completed is not None:
        for name, details in completed["checkpoint"]["files"].items():
            for directory in reporter.directories:
                path = directory / name
                if path.stat().st_size != details["bytes"] or file_digest(path) != details["sha256"]:
                    raise ValueError("Completed top-K results differ from their recorded checksums")
        return completed
    progress = _resume_config(reporter, config)
    n, k = identity["references"], args.top_k
    pool = np.load(args.pool_embeddings, mmap_mode="r")
    codes = np.load(args.country_cache, mmap_mode="r")
    query_codes, country_counts = _query_country_codes(args.output / "reference.parquet", n,
                                                        identity["country_labels"], args.query_chunk_size)
    queries, best_scores, best_rows, groups = _prepare_gpu(
        args, n, k, args.output / "query_embeddings.npy", query_codes, identity["country_labels"])
    previous = progress["checkpoint"]
    if previous is not None:
        _restore_snapshot(reporter, previous, best_scores, best_rows, n, k)
    next_pool = progress["pool_rows_scanned"]
    if (next_pool < 0 or next_pool > len(pool) or next_pool % args.pool_tile != 0 and next_pool != len(pool)
            or progress["pool_tiles_scanned"] != (next_pool + args.pool_tile - 1) // args.pool_tile):
        raise ValueError("Mining checkpoint has an invalid pool tile boundary")
    if next_pool and previous is None:
        raise ValueError("Mining progress is missing its top-K checkpoint")
    reporter.write(f"Mining exact top{k}, {n:,} references x {len(pool):,} full targets; "
                   f"resume pool row {next_pool:,}; tile {args.query_tile}x{args.pool_tile}")
    for start in range(next_pool, len(pool), args.pool_tile):
        stop = min(start + args.pool_tile, len(pool))
        # Sequential access matters: the 7.93-GB frozen mmap lives on slower
        # object-backed storage. Never index its rows at random in the hot loop.
        tile = torch.as_tensor(np.asarray(pool[start:stop], dtype=np.float32), device="cuda")
        tile_codes = np.asarray(codes[start:stop])
        for country in sorted(set(tile_codes) & set(groups)):
            local_rows = np.flatnonzero(tile_codes == country)
            passages = tile.index_select(0, torch.as_tensor(local_rows, dtype=torch.long, device="cuda"))
            _update_topk(queries, best_scores, best_rows, groups[country], passages,
                         (start + local_rows).astype(np.int32), args.query_tile, k)
            progress["comparisons"] += int(len(groups[country]) * len(local_rows))
        del tile
        tiles = progress["pool_tiles_scanned"] + 1
        final = stop == len(pool)
        if tiles % args.checkpoint_tiles == 0 or final:
            snapshot = _write_snapshot(reporter, best_scores, best_rows, n, k, tiles, final)
            old = previous
            progress.update(pool_rows_scanned=stop, pool_tiles_scanned=tiles,
                            checkpoint=snapshot, last_update_unix=time.time())
            _manifest(reporter, progress, "mining_progress.json")
            _retire_snapshot(reporter, old, snapshot)
            previous = snapshot
            reporter.write(f"Exact top{k} pool rows={stop:,}/{len(pool):,}; "
                           f"tile={tiles}; comparisons={progress['comparisons']:,}; "
                           f"elapsed={time.time()-progress['started_unix']:.1f}s")
        else:
            progress["pool_tiles_scanned"] = tiles
    if progress["pool_rows_scanned"] != len(pool) or not progress["checkpoint"]["final"]:
        raise ValueError("Full pool was not checkpointed through the last target row")
    del queries, best_scores, best_rows, groups
    gc.collect()
    torch.cuda.empty_cache()
    raw = np.load(args.output / "topk_raw_rows.npy", mmap_mode="r")
    values = np.load(args.output / "topk_scores.npy", mmap_mode="r")
    valid_count = 0
    without = 0
    for start in range(0, n, args.query_chunk_size):
        stop = min(start + args.query_chunk_size, n)
        r, s = np.asarray(raw[start:stop]), np.asarray(values[start:stop])
        valid = r >= 0
        if (np.any(r[valid] >= len(pool)) or not np.isfinite(s[valid]).all()
                or np.any(valid & (codes[np.maximum(r, 0)] != query_codes[start:stop, None]))):
            raise ValueError("Invalid, non-finite, or cross-country dense candidate")
        if np.any((r[:, 1:] == r[:, :-1]) & valid[:, 1:]):
            raise ValueError("Duplicate raw target within a query's top-K")
        if np.any(np.diff(s, axis=1)[valid[:, 1:]] > 1e-5):
            raise ValueError("Top-K scores were not stored in descending rank order")
        valid_count += int(valid.sum())
        without += int((~valid.any(axis=1)).sum())
    report = {"identity": config, "state": "complete", "checkpoint": progress["checkpoint"],
              "references": n, "references_without_candidates": without,
              "candidate_pairs": valid_count, "pool_rows_scanned": len(pool),
              "pool_tiles_scanned": progress["pool_tiles_scanned"],
              "country_counts": country_counts, "dot_products": progress["comparisons"],
              "elapsed_seconds": time.time() - progress["started_unix"],
              "positive_candidate_injection": False, "labels_read": False}
    _manifest(reporter, report, "topk_complete.json")
    reporter.write(f"COMPLETE exact full-pool top{k}, pairs={valid_count:,}, no-candidate refs={without:,}")
    return report


def write_pairs(args, identity: dict, topk: dict, reporter: Reporter) -> dict:
    output, durable = reporter.directories
    config = {"schema": 1, "scope": args.mode, "identity": identity,
              "topk_snapshot": topk["checkpoint"], "pair_order": "left_idx_then_dense_rank",
              "right_index": "global raw S2-then-S3 row number", "labels_read": False,
              "positive_candidate_injection": False}
    existing = _progress(reporter, "dense_complete.json", config)
    if existing is not None:
        for directory in reporter.directories:
            path = directory / "candidate_pairs.parquet"
            if file_digest(path) != existing["candidate_pairs_sha256"]:
                raise ValueError("Completed candidate pair artifact differs from its digest")
        return existing
    rows = np.load(output / "topk_raw_rows.npy", mmap_mode="r")
    scores = np.load(output / "topk_scores.npy", mmap_mode="r")
    n, k = rows.shape
    path = output / "candidate_pairs.parquet"
    temp = output / "candidate_pairs.parquet.tmp"
    if path.exists():
        # A completed parquet may have survived interruption before the final
        # marker/durable copy; its metadata is checked below and then reused.
        if pq.ParquetFile(path).metadata.num_rows != topk["candidate_pairs"]:
            raise ValueError("Uncommitted candidate parquet row count is inconsistent")
    else:
        if temp.exists():
            # This exact job owns the incomplete ParquetWriter output; rerender
            # deterministically from the committed top-K checkpoint.
            if _progress(reporter, "pairs_progress.json", config) is None:
                raise ValueError("Uncommitted pair temporary file has no matching identity")
            temp.unlink()
        _storage_room(output, 1_000_000_000)
        _manifest(reporter, {"identity": config, "state": "writing", "completed_rows": 0},
                  "pairs_progress.json")
        writer, count = None, 0
        try:
            for start in range(0, n, args.pair_reference_batch_size):
                stop = min(n, start + args.pair_reference_batch_size)
                raw = np.asarray(rows[start:stop])
                dense = np.asarray(scores[start:stop])
                valid = raw >= 0
                left = np.broadcast_to(np.arange(start, stop, dtype=np.int32)[:, None], raw.shape)[valid]
                rank = np.broadcast_to(np.arange(1, k + 1, dtype=np.int32)[None, :], raw.shape)[valid]
                frame = pd.DataFrame({"left_idx": left, "right_raw_idx": raw[valid],
                                      "dense_score": np.clip(dense[valid], -1, 1).astype(np.float32),
                                      "dense_rank": rank})
                if frame.duplicated(["left_idx", "right_raw_idx"]).any():
                    raise ValueError("Repeated candidate for one reference")
                table = pa.Table.from_pandas(frame, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(temp, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=max(1, len(table)))
                count += len(frame)
                if stop == n or stop % (args.pair_reference_batch_size * 20) == 0:
                    reporter.write(f"Pair artifact references={stop:,}/{n:,}; pairs={count:,}")
        finally:
            if writer is not None:
                writer.close()
        if count != topk["candidate_pairs"] or pq.ParquetFile(temp).metadata.num_rows != count:
            raise ValueError("Candidate parquet lost or duplicated a scored raw-row pair")
        _flush(temp)
        temp.replace(path)
    sha = file_digest(path)
    _copy_verified(path, durable / path.name, sha)
    report = {"identity": config, "state": "complete", "references": n,
              "candidate_pairs": topk["candidate_pairs"], "candidate_pairs_sha256": sha,
              "candidate_pairs_bytes": path.stat().st_size,
              "reference_file": str(output / "reference.parquet"),
              "query_embeddings_file": str(output / "query_embeddings.npy"),
              "target_pool": identity["pool_embeddings"],
              "heldout_population_mined_but_not_scored": args.mode == "full",
              "labels_read": False, "positive_candidate_injection": False}
    _manifest(reporter, report, "dense_complete.json")
    reporter.write(f"COMPLETE label-free dense candidate pairs={report['candidate_pairs']:,}; sha256={sha}")
    return report


def compare_saved_tune(args, identity: dict, topk: dict, reporter: Reporter) -> dict:
    """ID-aligned retrieval-only comparison; do not open medium truth or eval."""
    if args.mode != "screen" or args.baseline_medium is None:
        return {}
    medium_source = pd.read_parquet(args.medium_reference, columns=["entity_id", "split"])
    selected = medium_source.loc[medium_source.split.isin(["train", "tune"])].reset_index(drop=True)
    screen = pd.read_parquet(args.output / "reference.parquet", columns=["entity_id"])
    lookup = pd.Index(selected.entity_id).get_indexer(screen.entity_id)
    if np.any(lookup < 0) or not medium_source.iloc[medium_source.index[medium_source.split.eq("tune")]]["entity_id"].isin(
            screen.entity_id).sum() == len(screen):
        raise ValueError("Screen references are not original medium TUNE businesses")
    baseline_manifest = json.loads((args.baseline_medium / "query_manifest.json").read_text())
    if baseline_manifest["checkpoint_sha256"] != identity["model_sha256"]:
        raise ValueError("Saved medium baseline used a different retrieval encoder")
    old_queries = np.load(args.baseline_medium / "query_embeddings.npy", mmap_mode="r")
    if old_queries.shape != (len(selected), args.dimension):
        raise ValueError("Medium baseline query vectors do not align with original TRAIN+TUNE rows")
    new_queries = np.load(args.output / "query_embeddings.npy", mmap_mode="r")
    cosine = np.einsum("ij,ij->i", np.asarray(new_queries), np.asarray(old_queries[lookup]))
    previous = pd.read_parquet(args.baseline_medium / "raw_pairs.parquet",
                               columns=["left_idx", "right_raw_idx", "dense_rank"])
    previous = previous.loc[previous.left_idx.isin(lookup)]
    grouped = {key: part.sort_values("dense_rank").right_raw_idx.to_numpy()
               for key, part in previous.groupby("left_idx", sort=False)}
    rows = np.load(args.output / "topk_raw_rows.npy", mmap_mode="r")
    matched = 0
    identical = 0
    top1 = 0
    for i, idx in enumerate(lookup):
        old = grouped.get(idx)
        if old is None or len(old) != args.top_k:
            raise ValueError("Saved full-pool baseline lost a medium TUNE query")
        actual = np.asarray(rows[i])
        matched += len(set(actual) & set(old))
        identical += bool(set(actual) == set(old))
        top1 += bool(actual[0] == old[0])
    report = {"scope": "first_1000_original_medium_tune_full_10_320_219_target_pool",
              "references": len(screen), "pairs_per_reference": args.top_k,
              "saved_baseline_path": str(args.baseline_medium.resolve()),
              "saved_baseline_pairs_sha256": file_digest(args.baseline_medium / "raw_pairs.parquet"),
              "fraction_candidate_overlap": matched / (len(screen) * args.top_k),
              "references_with_identical_candidate_sets": identical,
              "references_with_identical_top1": top1,
              "mean_saved_query_cosine": float(cosine.mean()),
              "min_saved_query_cosine": float(cosine.min()),
              "baseline_execution_dtype": "float16", "new_execution_dtype": "bfloat16",
              "labels_read": False, "truth_injection": False,
              "note": "Candidate identity comparison only; not macro F0.5 or a holdout score."}
    _manifest(reporter, report, "screen_comparison.json")
    reporter.write(f"SCREEN comparison: {json.dumps(report, sort_keys=True)}")
    return report


def run(args) -> dict:
    args.output.mkdir(parents=True, exist_ok=True)
    args.durable_output.mkdir(parents=True, exist_ok=True)
    reporter = Reporter(args.output, args.durable_output)
    with (args.output / "dense.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = _input_checks(args)
        args.dimension = identity["pool_embeddings"]["dimension"]
        config_path = args.output / "dense_config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != identity:
            raise ValueError("Dense output belongs to another population, model, or retrieval config")
        _manifest(reporter, identity, "dense_config.json")
        query = encode_queries(args, identity, reporter)
        if args.stage == "encode":
            return query
        topk = mine_topk(args, identity, query, reporter)
        pairs = write_pairs(args, identity, topk, reporter)
        if args.mode == "screen":
            compare_saved_tune(args, identity, topk, reporter)
        return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["screen", "full"], required=True)
    parser.add_argument("--stage", choices=["encode", "all"], default="all")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--medium-reference", type=Path, help="Original medium reference; screen only")
    parser.add_argument("--baseline-medium", type=Path, help="Read-only saved medium dense retrieval; screen only")
    parser.add_argument("--screen-rows", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--durable-output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pool-embeddings", type=Path, required=True)
    parser.add_argument("--pool-complete", type=Path, required=True)
    parser.add_argument("--pool-preservation", type=Path, required=True)
    parser.add_argument("--country-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--query-chunk-size", type=int, default=20_000)
    parser.add_argument("--query-tile", type=int, default=512)
    parser.add_argument("--pool-tile", type=int, default=50_000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--checkpoint-tiles", type=int, default=20)
    parser.add_argument("--pair-reference-batch-size", type=int, default=5_000)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if (min(args.screen_rows, args.batch_size, args.query_chunk_size, args.query_tile,
            args.pool_tile, args.top_k, args.checkpoint_tiles, args.pair_reference_batch_size,
            args.threads) <= 0 or args.query_tile > 512 or args.pool_tile > 50_000
            or args.query_chunk_size > 50_000):
        parser.error("All sizes must be positive; tiles <=512x50k and chunks <=50k")
    if args.mode == "screen" and (not args.medium_reference or not args.baseline_medium):
        parser.error("A screen requires original --medium-reference and --baseline-medium")
    if args.mode == "full" and args.output.resolve() != args.reference_root.resolve():
        parser.error("Full v2 dense retrieval must share the sealed main reference directory")
    if args.durable_output.resolve() == args.output.resolve():
        parser.error("Scratch and durable outputs must be distinct")
    result = run(args)
    print(json.dumps({"status": result["state"], "mode": args.mode,
                      "references": result.get("references", result.get("completed_rows")),
                      "candidate_pairs": result.get("candidate_pairs"),
                      "output": str(args.output.resolve()),
                      "durable": str(args.durable_output.resolve())}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
