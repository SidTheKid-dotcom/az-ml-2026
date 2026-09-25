"""Bounded-memory, unlabeled full-pool candidate mining for official inference.

Called by ``ber.full_dense_candidates --query-input test_source1.tsv``. Query
text, embeddings, top-K results, pool compaction, and candidate writing are all
streamed or memory mapped. This module never opens a truth file or computes a
label-based metric. Every raw S1 record is retained, even with no country match.
"""
from __future__ import annotations

from collections import Counter
import fcntl
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .full_embeddings import Reporter, atomic_json, checkpoint_identity, embedding_texts
from .normalization import normalize_text


RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _input_identity(path: Path, expected_rows: int) -> dict:
    path = path.resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "expected_rows": expected_rows}


def _assert_input(item: dict) -> None:
    if _input_identity(Path(item["path"]), item["expected_rows"]) != item:
        raise ValueError("An input changed during unlabeled candidate mining")


def _flush(path: Path, array=None) -> None:
    if array is not None:
        array.flush()
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _write_frame(writer, frame: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table, row_group_size=max(1, len(frame)))
    return writer


def _check_unique_hashes(path: Path, expected_rows: int) -> None:
    """Duplicates cannot be missed; an exceedingly rare hash collision aborts."""
    values = np.load(path, mmap_mode="r")
    if values.shape != (expected_rows,):
        raise ValueError("ID hash verification has the wrong row count")
    if len(values) > 1:
        ordered = np.sort(values)
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("Repeated entity ID or hash collision; inspect raw IDs before proceeding")


def encode_raw_queries(args, identity: dict, query_input: dict, reporter: Reporter) -> dict:
    """Encode S1 in input order without materializing its full text table."""
    output = Path(args.output)
    metadata_path = output / "query_complete.json"
    config = {"input": query_input, "checkpoint_sha256": identity["checkpoint"]["sha256"],
              "prefix": "query: ", "text_format": "business_name | business_address",
              "max_length": identity["max_length"], "dimension": identity["dimension"],
              "storage_dtype": "float32", "row_order": "raw_query_input"}
    if metadata_path.exists():
        value = json.loads(metadata_path.read_text())
        if value["config"] != config:
            raise ValueError("Unlabeled query cache belongs to a different input or checkpoint")
        for name, shape, dtype in [("query_embeddings.npy", (query_input["expected_rows"], identity["dimension"]), np.float32),
                                   ("query_country_codes.npy", (query_input["expected_rows"],), np.int32)]:
            array = np.load(output / name, mmap_mode="r")
            if array.shape != shape or array.dtype != dtype:
                raise ValueError("Completed query artifact has an invalid shape or dtype")
        if pq.ParquetFile(output / "reference.parquet").metadata.num_rows != query_input["expected_rows"]:
            raise ValueError("Completed query artifact lost reference rows")
        return value

    from sentence_transformers import SentenceTransformer
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(args.model), device=args.device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = identity["max_length"]
    if args.device.startswith("cuda"):
        model.half()
    model.eval()
    n = query_input["expected_rows"]
    vectors_path, country_path = output / "query_embeddings.npy", output / "query_country_codes.npy"
    hash_path = output / "query_id_hashes.npy"
    vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=np.float32, shape=(n, identity["dimension"]))
    countries = np.lib.format.open_memmap(country_path, mode="w+", dtype=np.int32, shape=(n,))
    hashes = np.lib.format.open_memmap(hash_path, mode="w+", dtype=np.uint64, shape=(n,))
    labels, country_counts = {}, Counter()
    temporary = output / "reference.parquet.tmp"
    writer, rows = None, 0
    started = time.time()
    try:
        for frame in pd.read_csv(query_input["path"], sep="\t", dtype=str, keep_default_na=False,
                                 usecols=RECORD_COLUMNS, chunksize=args.query_stream_size):
            stop = rows + len(frame)
            if stop > n:
                raise ValueError("Query input has more rows than --expected-query-rows")
            if frame.entity_id.eq("").any():
                raise ValueError("Empty query entity ID")
            values = model.encode(embedding_texts(frame, "query: "), batch_size=args.batch_size,
                                  show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            values = np.asarray(values, dtype=np.float32)
            if values.shape != (len(frame), identity["dimension"]) or not np.isfinite(values).all():
                raise ValueError("Query encoder returned invalid embeddings")
            norms = np.linalg.norm(values, axis=1, keepdims=True)
            if np.any(norms <= 1e-8):
                raise ValueError("Query encoder returned a zero vector")
            vectors[rows:stop] = values / norms
            normalized = frame.country.map({value: normalize_text(value) for value in frame.country.unique()})
            for country in normalized.unique():
                if country not in labels:
                    labels[country] = len(labels)
            countries[rows:stop] = normalized.map(labels).to_numpy(dtype=np.int32)
            country_counts.update(frame.country.value_counts().to_dict())
            hashes[rows:stop] = pd.util.hash_array(frame.entity_id.to_numpy(dtype=object))
            writer = _write_frame(writer, frame[RECORD_COLUMNS], temporary)
            rows = stop
            reporter.manifest({"state": "encoding_queries", "completed_rows": rows, "expected_rows": n,
                               "elapsed_seconds": time.time() - started}, "query_progress.json")
            reporter.write(f"Unlabeled query encoding: {rows:,}/{n:,}; elapsed={time.time()-started:.1f}s")
    finally:
        if writer is not None:
            writer.close()
    if rows != n:
        raise ValueError(f"Query row count {rows:,} differs from expected {n:,}")
    _assert_input(query_input)
    for path, array in [(vectors_path, vectors), (country_path, countries), (hash_path, hashes)]:
        _flush(path, array)
    _check_unique_hashes(hash_path, n)
    _flush(temporary)
    temporary.replace(output / "reference.parquet")
    value = {"config": config, "reference_count": rows, "country_codes": labels,
             "country_counts": dict(country_counts), "all_finite": True,
             "id_uniqueness_verified": True, "elapsed_seconds": time.time() - started}
    reporter.manifest(value, "query_complete.json")
    del model, vectors, countries, hashes
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value


def mine_unlabeled_topk(args, identity: dict, queries: dict, reporter: Reporter) -> dict:
    """Pool-outer search; retain GPU top-K, then copy results to disk in shards."""
    from .full_dense_candidates import load_country_codes

    output = Path(args.output)
    n, k = queries["reference_count"], args.top_k
    rows_path, scores_path = output / "topk_raw_rows.npy", output / "topk_scores.npy"
    marker = output / "topk_complete.json"
    if marker.exists():
        value = json.loads(marker.read_text())
        for path, dtype in [(rows_path, np.int64), (scores_path, np.float32)]:
            array = np.load(path, mmap_mode="r")
            if array.shape != (n, k) or array.dtype != dtype:
                raise ValueError("Completed top-K artifact has invalid shape/dtype")
        return value
    pool_vectors = np.load(args.embeddings, mmap_mode="r")
    query_vectors = np.load(output / "query_embeddings.npy", mmap_mode="r")
    country_codes = np.load(output / "query_country_codes.npy", mmap_mode="r")
    by_country = {country: np.flatnonzero(country_codes == code)
                  for country, code in queries["country_codes"].items()}
    backend = "torch" if args.backend == "auto" else args.backend
    if backend == "torch":
        import torch

        torch.set_num_threads(args.threads)
        torch.set_float32_matmul_precision("highest")
        query_tensor = torch.as_tensor(np.asarray(query_vectors), dtype=torch.float32, device=args.device)
        best_scores = torch.full((n, k), -float("inf"), dtype=torch.float32, device=args.device)
        best_rows = torch.full((n, k), -1, dtype=torch.long, device=args.device)
    else:
        # CPU support is intended for tiny deterministic correctness checks.
        if n * len(pool_vectors) > 100_000_000:
            raise ValueError("Large unlabeled exact retrieval requires --backend torch")
        query_tensor = np.asarray(query_vectors, dtype=np.float32)
        best_scores = np.full((n, k), -np.inf, dtype=np.float32)
        best_rows = np.full((n, k), -1, dtype=np.int64)
    global_start, chunks, comparisons = 0, 0, 0
    started = time.time()
    pool_country_codes, pool_country_labels = load_country_codes(Path(args.embeddings), identity,
                                                               args.pool_batch_size, reporter)
    for item in identity["inputs_in_order"]:
        rows_seen = 0
        for source_start in range(0, item["expected_rows"], args.pool_batch_size):
            count = min(args.pool_batch_size, item["expected_rows"] - source_start)
            keys = pool_country_labels[pool_country_codes[global_start:global_start + count]]
            rows_seen += count
            if rows_seen > item["expected_rows"]:
                raise ValueError("Pool input has more rows than its embedding manifest")
            for country in sorted(set(keys) & set(by_country)):
                qrows = by_country[country]
                raw_rows = global_start + np.flatnonzero(keys == country)
                block_k = min(k, len(raw_rows))
                comparisons += len(qrows) * len(raw_rows)
                if backend == "torch":
                    passage = torch.as_tensor(np.array(pool_vectors[raw_rows], copy=True), dtype=torch.float32,
                                              device=args.device)
                    raw_tensor = torch.as_tensor(raw_rows, dtype=torch.long, device=args.device)
                    with torch.inference_mode():
                        for start in range(0, len(qrows), args.query_batch_size):
                            index = torch.as_tensor(qrows[start:start + args.query_batch_size],
                                                    dtype=torch.long, device=args.device)
                            scores = query_tensor[index] @ passage.T
                            block_scores, positions = torch.topk(scores, block_k, dim=1)
                            candidates = torch.cat((best_scores[index], block_scores), dim=1)
                            candidate_rows = torch.cat((best_rows[index], raw_tensor[positions]), dim=1)
                            values, order = torch.topk(candidates, k, dim=1)
                            best_scores.index_copy_(0, index, values)
                            best_rows.index_copy_(0, index, torch.gather(candidate_rows, 1, order))
                            del scores
                    del passage, raw_tensor
                else:
                    passage = np.asarray(pool_vectors[raw_rows], dtype=np.float32)
                    for start in range(0, len(qrows), args.query_batch_size):
                        index = qrows[start:start + args.query_batch_size]
                        scores = query_tensor[index] @ passage.T
                        positions = np.argsort(-scores, axis=1, kind="stable")[:, :block_k]
                        candidates = np.concatenate((best_scores[index], np.take_along_axis(scores, positions, axis=1)), axis=1)
                        candidate_rows = np.concatenate((best_rows[index], raw_rows[positions]), axis=1)
                        order = np.argsort(-candidates, axis=1, kind="stable")[:, :k]
                        best_scores[index] = np.take_along_axis(candidates, order, axis=1)
                        best_rows[index] = np.take_along_axis(candidate_rows, order, axis=1)
            global_start += count
            chunks += 1
            if chunks == 1 or chunks % 10 == 0:
                if backend == "torch" and args.device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.time() - started
                reporter.manifest({"state": "mining", "pool_rows_scanned": global_start,
                                   "pool_rows_total": identity["target_rows"], "references": n,
                                   "dot_products": comparisons, "elapsed_seconds": elapsed,
                                   "dot_products_per_second": comparisons / max(elapsed, 1e-6)}, "mining_progress.json")
                reporter.write(f"Unlabeled mining: pool_rows={global_start:,}/{identity['target_rows']:,}; "
                               f"dot_products={comparisons:,}; elapsed={elapsed:.1f}s")
        if rows_seen != item["expected_rows"]:
            raise ValueError("Pool row count changed during search")
        _assert_input(item)
    if global_start != len(pool_vectors):
        raise ValueError("Pool scan and embedding row counts differ")
    raw_rows = np.lib.format.open_memmap(rows_path, mode="w+", dtype=np.int64, shape=(n, k))
    scores = np.lib.format.open_memmap(scores_path, mode="w+", dtype=np.float32, shape=(n, k))
    candidates, with_candidates = 0, 0
    for start in range(0, n, args.pair_reference_batch_size):
        stop = min(n, start + args.pair_reference_batch_size)
        r, s = best_rows[start:stop], best_scores[start:stop]
        if backend == "torch":
            r, s = r.cpu().numpy(), s.cpu().numpy()
        valid = r >= 0
        if not np.isfinite(s[valid]).all():
            raise ValueError("Non-finite valid candidate score")
        raw_rows[start:stop], scores[start:stop] = r, np.clip(s, -1, 1)
        candidates += int(valid.sum())
        with_candidates += int(valid.any(axis=1).sum())
    _flush(rows_path, raw_rows)
    _flush(scores_path, scores)
    value = {"state": "complete", "references": n, "references_with_candidates": with_candidates,
             "references_without_candidates": n - with_candidates, "candidate_pairs": candidates,
             "pool_rows_scanned": global_start, "top_k": k, "dot_products": comparisons,
             "elapsed_seconds": time.time() - started, "score_precision": "float32_dot_from_float16_pool"}
    reporter.manifest(value, "topk_complete.json")
    del best_rows, best_scores, query_tensor, raw_rows, scores
    gc.collect()
    if backend == "torch" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return value


def compact_unlabeled_pool(args, identity: dict, reporter: Reporter) -> dict:
    """Use a 10MB selected-row bitmap and 80MB row map for a ten-million pool."""
    output = Path(args.output)
    marker = output / "pool_complete.json"
    if marker.exists():
        return json.loads(marker.read_text())
    topk = np.load(output / "topk_raw_rows.npy", mmap_mode="r")
    count = identity["target_rows"]
    selected_path, mapping_path = output / "selected_raw_rows.npy", output / "raw_to_pool_row.npy"
    selected = np.lib.format.open_memmap(selected_path, mode="w+", dtype=np.bool_, shape=(count,))
    selected[:] = False
    for start in range(0, len(topk), args.pair_reference_batch_size):
        rows = np.asarray(topk[start:start + args.pair_reference_batch_size]).ravel()
        valid = rows >= 0
        if valid.any() and rows[valid].max() >= count:
            raise ValueError("Candidate raw row index exceeds the pool")
        selected[rows[valid]] = True
    expected = int(selected.sum())
    mapping = np.lib.format.open_memmap(mapping_path, mode="w+", dtype=np.int64, shape=(count,))
    mapping[:] = -1
    hash_path = output / "pool_id_hashes.npy"
    hashes = np.lib.format.open_memmap(hash_path, mode="w+", dtype=np.uint64, shape=(expected,))
    temporary = output / "pool.parquet.tmp"
    writer, raw_start, kept = None, 0, 0
    country_counts = Counter()
    try:
        for item in identity["inputs_in_order"]:
            source_rows = 0
            for frame in pd.read_csv(item["path"], sep="\t", dtype=str, keep_default_na=False,
                                     usecols=RECORD_COLUMNS, chunksize=args.pool_batch_size):
                raw_stop = raw_start + len(frame)
                retain = np.asarray(selected[raw_start:raw_stop])
                piece = frame.loc[retain, RECORD_COLUMNS].copy()
                piece["raw_row_idx"] = raw_start + np.flatnonzero(retain)
                if len(piece):
                    if piece.entity_id.eq("").any():
                        raise ValueError("Empty retained pool entity ID")
                    stop = kept + len(piece)
                    mapping[piece.raw_row_idx.to_numpy(dtype=np.int64)] = np.arange(kept, stop, dtype=np.int64)
                    hashes[kept:stop] = pd.util.hash_array(piece.entity_id.to_numpy(dtype=object))
                    country_counts.update(piece.country.value_counts().to_dict())
                    kept = stop
                # An empty first table still establishes a valid Parquet schema.
                writer = _write_frame(writer, piece, temporary)
                raw_start, source_rows = raw_stop, source_rows + len(frame)
            if source_rows != item["expected_rows"]:
                raise ValueError("Pool row count changed during compaction")
            _assert_input(item)
    finally:
        if writer is not None:
            writer.close()
    if kept != expected or raw_start != count:
        raise ValueError("Pool compaction retained an inconsistent row count")
    for path, array in [(selected_path, selected), (mapping_path, mapping), (hash_path, hashes)]:
        _flush(path, array)
    _check_unique_hashes(hash_path, kept)
    _flush(temporary)
    temporary.replace(output / "pool.parquet")
    value = {"pool_records": kept, "raw_pool_records": count, "country_counts": dict(country_counts),
             "row_order": "raw_source_order", "id_uniqueness_verified": True,
             "truth_targets_retained": False}
    reporter.manifest(value, "pool_complete.json")
    reporter.write(f"Unlabeled pool compaction: retained {kept:,}/{count:,} candidate records")
    return value


def write_unlabeled_pairs(args, topk_report: dict, reporter: Reporter) -> int:
    """Write complete reference groups, already ordered by left_idx/right_idx."""
    output = Path(args.output)
    raw_rows = np.load(output / "topk_raw_rows.npy", mmap_mode="r")
    scores = np.load(output / "topk_scores.npy", mmap_mode="r")
    mapping = np.load(output / "raw_to_pool_row.npy", mmap_mode="r")
    temporary = output / "pairs.parquet.tmp"
    n, k = raw_rows.shape
    if max(n, len(mapping), k) >= np.iinfo(np.int32).max:
        raise ValueError("Unlabeled pair artifacts require indices smaller than int32 limits")
    writer, written = None, 0
    try:
        for start in range(0, n, args.pair_reference_batch_size):
            stop = min(n, start + args.pair_reference_batch_size)
            raw = np.asarray(raw_rows[start:stop])
            valid = raw >= 0
            right = np.full(raw.shape, np.iinfo(np.int64).max, dtype=np.int64)
            right[valid] = mapping[raw[valid]]
            if np.any(right[valid] < 0):
                raise ValueError("Compaction omitted a retrieved candidate")
            order = np.argsort(right, axis=1, kind="stable")
            right = np.take_along_axis(right, order, axis=1)
            valid = np.take_along_axis(valid, order, axis=1)
            raw = np.take_along_axis(raw, order, axis=1)
            dense = np.take_along_axis(scores[start:stop], order, axis=1)
            rank = order + 1
            left = np.broadcast_to(np.arange(start, stop, dtype=np.int32)[:, None], raw.shape)[valid]
            values = dense[valid].astype(np.float32)
            frame = pd.DataFrame({"left_idx": left, "right_idx": right[valid].astype(np.int32),
                                  "right_raw_idx": raw[valid].astype(np.int32), "dense_score": values,
                                  "dense_rank": rank[valid].astype(np.int32), "name_score": np.float32(0),
                                  "address_score": np.float32(0), "retrieval_score": values,
                                  "retrieval_rank": rank[valid].astype(np.int32), "dense_only": np.float32(1)})
            if frame.duplicated(["left_idx", "right_idx"]).any():
                raise ValueError("Duplicate candidate within an unlabeled reference group")
            writer = _write_frame(writer, frame, temporary)
            written += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if written != topk_report["candidate_pairs"]:
        raise ValueError("Candidate count changed during pair artifact writing")
    _flush(temporary)
    temporary.replace(output / "pairs.parquet")
    reporter.write(f"Unlabeled candidate artifact: {written:,} pairs; ordered by left_idx/right_idx")
    return written


def run_unlabeled(args) -> dict:
    from .full_dense_candidates import read_embedding_manifest

    if args.data:
        raise ValueError("Unlabeled --query-input mode must not receive --data or a truth directory")
    if not args.expected_query_rows or args.expected_query_rows <= 0:
        raise ValueError("Unlabeled mode requires a positive --expected-query-rows")
    started = time.time()
    output = Path(args.output)
    reporter = Reporter(output, Path(args.durable_output) if args.durable_output else None)
    lock = (output / "mining.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another miner owns this output directory") from error
    identity = read_embedding_manifest(Path(args.embeddings))
    checkpoint = checkpoint_identity(Path(args.model))
    if checkpoint["sha256"] != identity["checkpoint"]["sha256"]:
        raise ValueError("Query checkpoint differs from the full-pool embedding checkpoint")
    query_input = _input_identity(Path(args.query_input), args.expected_query_rows)
    config = {"schema": 1, "mode": "unlabeled_inference", "query_input": query_input,
              "pool_embedding_identity": identity, "top_k": args.top_k, "backend": args.backend,
              "device": args.device, "score_precision": "float32_dot_from_float16_pool",
              "query_batch_size": args.query_batch_size, "pool_batch_size": args.pool_batch_size,
              "labels_accessed": False, "positive_candidate_injection": False}
    config_path = output / "mining_manifest.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to a different unlabeled inference configuration")
    atomic_json(config_path, config)
    if (output / "complete.json").exists():
        complete = json.loads((output / "complete.json").read_text())
        if complete.get("config") != config:
            raise ValueError("Completion marker belongs to a different configuration")
        return complete["metadata"]
    queries = encode_raw_queries(args, identity, query_input, reporter)
    topk = mine_unlabeled_topk(args, identity, queries, reporter)
    pool = compact_unlabeled_pool(args, identity, reporter)
    pairs = write_unlabeled_pairs(args, topk, reporter)
    _assert_input(query_input)
    metadata = {"mode": "unlabeled_inference", "references": queries["reference_count"],
                "pool_records": pool["pool_records"], "mining_full_pool_records": identity["target_rows"],
                "candidate_pairs": pairs, "reference_country_counts": queries["country_counts"],
                "pool_country_counts": pool["country_counts"],
                "references_with_candidates": topk["references_with_candidates"],
                "references_without_candidates": topk["references_without_candidates"],
                "all_input_references_retained": True, "labels_accessed": False,
                "positive_candidate_injection": False, "pair_order": ["left_idx", "right_idx"],
                "rank_columns_preserved": ["dense_rank", "retrieval_rank"],
                "pairs_unique": True, "top_k": args.top_k,
                "elapsed_seconds": time.time() - started, "mining": topk,
                "feature_semantics": {"name_score": "unavailable; zero", "address_score": "unavailable; zero",
                                      "retrieval_score": "dense cosine", "dense_only": "always one"},
                "memory_plan": {"full_query_text_loaded": False, "full_pair_dataframe_loaded": False,
                                "query_embedding_bytes": args.expected_query_rows * identity["dimension"] * 4,
                                "persistent_gpu_topk_bytes": args.expected_query_rows * args.top_k * 12,
                                "raw_row_mapping_bytes": identity["target_rows"] * 9,
                                "pair_reference_batch_size": args.pair_reference_batch_size},
                "limitations": ["Exact retrieval runtime scales with country-constrained query/pool products.",
                                "Prediction should consume ordered Parquet groups without globally copying or sorting pairs."]}
    for name, expected in [("reference.parquet", queries["reference_count"]),
                            ("pool.parquet", pool["pool_records"]), ("pairs.parquet", pairs)]:
        if pq.ParquetFile(output / name).metadata.num_rows != expected:
            raise ValueError(f"Final artifact row count mismatch: {name}")
    reporter.manifest(metadata, "metadata.json")
    reporter.manifest({"state": "complete", "config": config, "metadata": metadata}, "complete.json")
    reporter.write(f"UNLABELED COMPLETE: {queries['reference_count']:,} references, {pairs:,} candidate pairs")
    return metadata
