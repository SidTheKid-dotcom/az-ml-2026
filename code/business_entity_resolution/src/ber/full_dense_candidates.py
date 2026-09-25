"""Mine exact country-blocked neighbors from streamed full-pool embeddings.

Candidate mining finishes and raw-row pairs are saved before truth is read.
Ground-truth records missing from top-K are then retained in the compact pool for
honest recall measurement; they are never inserted into the candidate pairs.
Labeled mode supports explicit train/tune reference splits. Final eval retrieval
requires a deliberate flag and a hash-verified, frozen matcher bundle. Unlabeled
inference uses --query-input and a separate bounded-memory implementation with no
truth access, preserving every raw source1 query, including unseen countries.
"""
from __future__ import annotations

import argparse
import fcntl
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from .full_embeddings import Reporter, atomic_json, checkpoint_identity, embedding_texts, file_digest
from .hf_experiment import frame_digest
from .metrics import candidate_report
from .normalization import normalize_text


def _country_keys(series: pd.Series) -> np.ndarray:
    mapping = {value: normalize_text(value) for value in series.unique()}
    return series.map(mapping).fillna("").to_numpy()


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = Path(str(path) + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def _atomic_npy(values: np.ndarray, path: Path) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def frozen_evaluation_metadata(args, requested_splits: tuple[str, ...]) -> dict | None:
    """Fail closed on eval unless an already selected matcher is identified."""
    final_evaluation = bool(getattr(args, "final_evaluation", False))
    bundle_path = getattr(args, "frozen_model_bundle", None)
    expected_hash = getattr(args, "frozen_model_sha256", None)
    if not final_evaluation:
        if bundle_path or expected_hash:
            raise ValueError("Frozen-model flags require explicit --final-evaluation")
        if not requested_splits or any(split not in {"train", "tune"} for split in requested_splits):
            raise ValueError("Standard retrieval supports train/tune only; eval requires explicit frozen final evaluation")
        return None
    if requested_splits != ("eval",):
        raise ValueError("--final-evaluation permits exactly --splits eval, never training/tuning splits")
    if not bundle_path or not expected_hash:
        raise ValueError("Final evaluation requires --frozen-model-bundle and --frozen-model-sha256")
    expected_hash = str(expected_hash).lower()
    if len(expected_hash) != 64 or any(value not in "0123456789abcdef" for value in expected_hash):
        raise ValueError("--frozen-model-sha256 must contain exactly 64 hexadecimal characters")
    path = Path(bundle_path).resolve()
    actual_hash = file_digest(path)
    if actual_hash != expected_hash:
        raise ValueError("Frozen matcher SHA256 differs from the explicitly selected hash")
    import joblib

    bundle = joblib.load(path)
    if file_digest(path) != actual_hash:
        raise ValueError("Frozen matcher changed while its selection was being verified")
    if not isinstance(bundle, dict) or "model" not in bundle or "threshold" not in bundle:
        raise ValueError("Frozen matcher must be a saved ber.predict bundle with model and threshold")
    threshold = float(bundle["threshold"])
    anchor_threshold = float(bundle.get("anchor_threshold", 0))
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Frozen matcher threshold must be finite and between zero and one")
    if not np.isfinite(anchor_threshold) or not 0 <= anchor_threshold <= 1:
        raise ValueError("Frozen matcher anchor threshold must be finite and between zero and one")
    columns = bundle.get("feature_columns")
    if columns is None or not len(columns) or any(not isinstance(value, str) for value in columns):
        raise ValueError("Frozen matcher must include its ordered feature_columns")
    value = {"bundle_path": str(path), "bundle_sha256": actual_hash, "bundle_size_bytes": path.stat().st_size,
             "threshold": threshold, "anchor_threshold": anchor_threshold,
             "feature_columns": list(columns), "threshold_selection_allowed": False,
             "scope": "sealed_final_eval_only"}
    del bundle
    return value


def read_embedding_manifest(path: Path) -> dict:
    marker = path.parent / "complete.json"
    if not marker.exists():
        raise ValueError("Full-pool embeddings are not complete; wait for their completion marker")
    complete = json.loads(marker.read_text())
    if complete.get("state") != "complete" or not complete.get("full_dataset_complete"):
        raise ValueError("Mining requires a completed full-pool embedding artifact, not a smoke prefix")
    identity = complete["identity"]
    if not complete.get("verification", {}).get("all_finite"):
        raise ValueError("Embedding artifact does not record finite-vector verification")
    for item in identity["inputs_in_order"]:
        stat = Path(item["path"]).stat()
        if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError("A raw input changed since full-pool encoding")
    array = np.load(path, mmap_mode="r")
    if array.shape != (identity["target_rows"], identity["dimension"]) or array.dtype != np.float16:
        raise ValueError("Full-pool embedding shape/dtype differs from its completion manifest")
    return identity


def load_country_codes(embeddings_path: Path, identity: dict, chunk_size: int,
                       reporter: Reporter) -> tuple[np.ndarray, np.ndarray]:
    """Cache only normalized country codes, verified against raw-file identity."""
    directory = embeddings_path.parent
    path, manifest_path = directory / "country_codes.npy", directory / "country_codes.json"
    config = {"schema": 1, "inputs_in_order": identity["inputs_in_order"],
              "rows": identity["target_rows"], "normalization": "ber.normalize_text_v1"}
    lock = (directory / "country_codes.lock").open("a")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if manifest_path.exists():
            cached = json.loads(manifest_path.read_text())
            if cached["config"] != config:
                raise ValueError("Country-code cache does not match raw-file order/identity")
            codes = np.load(path, mmap_mode="r")
            if codes.shape != (config["rows"],) or codes.dtype != np.int32:
                raise ValueError("Country-code cache has an invalid shape/dtype")
            return codes, np.asarray(cached["labels"], dtype=object)
        temporary = Path(str(path) + ".tmp")
        codes = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.int32, shape=(config["rows"],))
        labels, start = {}, 0
        started = time.time()
        for item in identity["inputs_in_order"]:
            rows = 0
            for frame in pd.read_csv(item["path"], sep="\t", dtype=str, keep_default_na=False,
                                     usecols=["country"], chunksize=chunk_size):
                countries = _country_keys(frame.country)
                for country in set(countries):
                    if country not in labels:
                        labels[country] = len(labels)
                stop = start + len(frame)
                if stop > config["rows"] or rows + len(frame) > item["expected_rows"]:
                    raise ValueError("Raw row count grew while building country cache")
                codes[start:stop] = pd.Series(countries).map(labels).to_numpy(dtype=np.int32)
                start, rows = stop, rows + len(frame)
            if rows != item["expected_rows"]:
                raise ValueError("Raw row count differs from country-cache identity")
            stat = Path(item["path"]).stat()
            if stat.st_size != item["size_bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
                raise ValueError("Raw input changed while building country cache")
        if start != config["rows"]:
            raise ValueError("Country cache lost raw pool rows")
        codes.flush()
        import os

        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        del codes
        temporary.replace(path)
        cached = {"config": config, "labels": list(labels), "elapsed_seconds": time.time() - started}
        atomic_json(manifest_path, cached)
        reporter.write(f"Cached normalized country codes for {start:,} pool rows in {time.time()-started:.2f}s")
        return np.load(path, mmap_mode="r"), np.asarray(cached["labels"], dtype=object)


def encode_queries(reference: pd.DataFrame, args, identity: dict, reporter: Reporter) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    import torch

    path = Path(args.output) / "query_embeddings.npy"
    metadata = {"reference_digest": frame_digest(reference, ["entity_id", "business_name", "business_address", "country"]),
                "checkpoint_sha256": identity["checkpoint"]["sha256"],
                "max_length": identity["max_length"], "prefix": "query: ", "dimension": identity["dimension"]}
    config_path = Path(args.output) / "query_manifest.json"
    if config_path.exists() and path.exists():
        if json.loads(config_path.read_text()) != metadata:
            raise ValueError("Query embedding cache does not match selected references/checkpoint")
        return np.load(path, mmap_mode="r")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(args.model), device=args.device, local_files_only=True, trust_remote_code=False)
    model.max_seq_length = identity["max_length"]
    if args.device.startswith("cuda"):
        model.half()
    model.eval()
    reporter.write(f"Encoding {len(reference):,} selected query references")
    vectors = model.encode(embedding_texts(reference, "query: "), batch_size=args.batch_size,
                           show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.shape != (len(reference), identity["dimension"]) or not np.isfinite(vectors).all():
        raise ValueError("Invalid query embedding output")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Query encoder returned a zero vector")
    vectors /= norms
    _atomic_npy(vectors, path)
    atomic_json(config_path, metadata)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return vectors


def mine_pairs(reference: pd.DataFrame, query_vectors: np.ndarray, embeddings_path: Path,
               identity: dict, args, reporter: Reporter) -> pd.DataFrame:
    """Pool-outer exact top-K; only bounded score tiles are ever allocated."""
    pool_vectors = np.load(embeddings_path, mmap_mode="r")
    query_countries = _country_keys(reference.country)
    by_country = {country: np.flatnonzero(query_countries == country) for country in set(query_countries)}
    n, k = len(reference), args.top_k
    backend = "torch" if args.backend == "auto" else args.backend
    if backend == "torch":
        import torch

        torch.set_num_threads(args.threads)
        # Scores stay float32 even though the saved pool is fp16: rounding the
        # final dot product to fp16 can tie close business-name variants.
        dtype = torch.float32
        torch.set_float32_matmul_precision("highest")
        queries = torch.as_tensor(np.asarray(query_vectors), dtype=dtype, device=args.device)
        best_scores = torch.full((n, k), -float("inf"), dtype=torch.float32, device=args.device)
        best_rows = torch.full((n, k), -1, dtype=torch.long, device=args.device)
    else:
        queries = np.asarray(query_vectors, dtype=np.float32)
        best_scores = np.full((n, k), -np.inf, dtype=np.float32)
        best_rows = np.full((n, k), -1, dtype=np.int64)
    global_start, chunks = 0, 0
    started = time.time()
    country_codes, country_labels = load_country_codes(embeddings_path, identity, args.pool_batch_size, reporter)
    for input_index, item in enumerate(identity["inputs_in_order"]):
        rows_seen = 0
        for source_start in range(0, item["expected_rows"], args.pool_batch_size):
            count = min(args.pool_batch_size, item["expected_rows"] - source_start)
            countries = country_labels[country_codes[global_start:global_start + count]]
            rows_seen += count
            if rows_seen > item["expected_rows"]:
                raise ValueError("Raw input has more rows than the embedding manifest records")
            for country in sorted(set(countries) & set(by_country)):
                query_rows = by_country[country]
                pool_local = np.flatnonzero(countries == country)
                raw_rows = global_start + pool_local
                block_k = min(k, len(pool_local))
                if backend == "torch":
                    passage = torch.as_tensor(np.array(pool_vectors[raw_rows], copy=True), dtype=dtype, device=args.device)
                    raw_rows_tensor = torch.as_tensor(raw_rows, dtype=torch.long, device=args.device)
                    with torch.inference_mode():
                        for query_start in range(0, len(query_rows), args.query_batch_size):
                            index = torch.as_tensor(query_rows[query_start:query_start + args.query_batch_size],
                                                     dtype=torch.long, device=args.device)
                            scores = queries[index] @ passage.T
                            block_scores, block_positions = torch.topk(scores, block_k, dim=1)
                            candidates = torch.cat((best_scores[index], block_scores.float()), dim=1)
                            positions = torch.cat((best_rows[index], raw_rows_tensor[block_positions]), dim=1)
                            values, order = torch.topk(candidates, k, dim=1)
                            best_scores.index_copy_(0, index, values)
                            best_rows.index_copy_(0, index, torch.gather(positions, 1, order))
                            del scores
                    del passage, raw_rows_tensor
                else:
                    passage = np.asarray(pool_vectors[raw_rows], dtype=np.float32)
                    for query_start in range(0, len(query_rows), args.query_batch_size):
                        index = query_rows[query_start:query_start + args.query_batch_size]
                        scores = queries[index] @ passage.T
                        local_order = np.argsort(-scores, axis=1, kind="stable")[:, :block_k]
                        candidates = np.concatenate((best_scores[index], np.take_along_axis(scores, local_order, axis=1)), axis=1)
                        positions = np.concatenate((best_rows[index], raw_rows[local_order]), axis=1)
                        order = np.argsort(-candidates, axis=1, kind="stable")[:, :k]
                        best_scores[index] = np.take_along_axis(candidates, order, axis=1)
                        best_rows[index] = np.take_along_axis(positions, order, axis=1)
            global_start += count
            chunks += 1
            if chunks == 1 or chunks % 10 == 0:
                progress = {"state": "mining", "pool_rows_scanned": global_start,
                            "pool_rows_total": identity["target_rows"], "references": n,
                            "top_k": k, "elapsed_seconds": time.time() - started}
                reporter.manifest(progress, "mining_progress.json")
                reporter.write(f"Mining pool_rows={global_start:,}/{identity['target_rows']:,}; "
                               f"queries={n:,}; elapsed={time.time()-started:.1f}s")
        if rows_seen != item["expected_rows"]:
            raise ValueError(f"Source {input_index + 1} count changed: {rows_seen}")
    if global_start != len(pool_vectors):
        raise ValueError("Raw source row count and pool embedding row count disagree")
    if backend == "torch":
        best_scores, best_rows = best_scores.cpu().numpy(), best_rows.cpu().numpy()
    valid = best_rows >= 0
    left = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], (n, k))[valid]
    rank = np.broadcast_to(np.arange(1, k + 1, dtype=np.int32)[None, :], (n, k))[valid]
    return pd.DataFrame({"left_idx": left, "right_raw_idx": best_rows[valid],
                         "dense_score": np.clip(best_scores[valid], -1, 1).astype(np.float32),
                         "dense_rank": rank})


def compact_pool(raw_pairs: pd.DataFrame, reference: pd.DataFrame, truth: dict,
                 identity: dict, reporter: Reporter) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep candidate records and all truth targets without changing candidates."""
    requested = np.unique(raw_pairs.right_raw_idx.to_numpy(dtype=np.int64))
    truth_targets = {target for values in truth.values() for target in values}
    # Reusing Arrow's value set avoids converting a large Python set for every
    # 50K-row chunk. The selected rows are identical to Series.isin(truth_targets).
    truth_values = pa.array(list(truth_targets), type=pa.string())
    pieces, global_start = [], 0
    for item in identity["inputs_in_order"]:
        rows_seen = 0
        for frame in pd.read_csv(item["path"], sep="\t", dtype=str, keep_default_na=False,
                                 usecols=["entity_id", "business_name", "business_address", "country"], chunksize=50000):
            stop = global_start + len(frame)
            low, high = np.searchsorted(requested, [global_start, stop])
            retain = pc.is_in(pa.array(frame.entity_id), value_set=truth_values).to_numpy(zero_copy_only=False).copy()
            retain[requested[low:high] - global_start] = True
            if retain.any():
                piece = frame.loc[retain].copy()
                piece["raw_row_idx"] = global_start + np.flatnonzero(retain)
                pieces.append(piece)
            global_start = stop
            rows_seen += len(frame)
        if rows_seen != item["expected_rows"]:
            raise ValueError("Source row counts changed during compact-pool construction")
    if not pieces:
        raise ValueError("No candidate or ground-truth records were retained")
    pool = pd.concat(pieces, ignore_index=True)
    if pool.entity_id.duplicated().any():
        raise ValueError("Raw source pool contains duplicate entity IDs")
    missing = truth_targets - set(pool.entity_id)
    if missing:
        raise ValueError(f"{len(missing)} provided truth targets are absent from the full raw pool")
    raw_rows = pool.raw_row_idx.to_numpy(dtype=np.int64)
    right = np.searchsorted(raw_rows, raw_pairs.right_raw_idx.to_numpy(dtype=np.int64))
    if not np.array_equal(raw_rows[right], raw_pairs.right_raw_idx.to_numpy(dtype=np.int64)):
        raise ValueError("Compact-pool candidate mapping lost raw rows")
    pairs = raw_pairs.copy()
    pairs["right_idx"] = right
    pairs["name_score"] = np.float32(0)
    pairs["address_score"] = np.float32(0)
    pairs["retrieval_score"] = pairs.dense_score.astype(np.float32)
    pairs["retrieval_rank"] = pairs.dense_rank.astype(np.int32)
    pairs["dense_only"] = np.float32(1)
    reporter.write(f"Compacted {len(pool):,} pool records for {len(pairs):,} mined pairs; "
                   "unretrieved truth targets retained only as pool records")
    return pool, pairs


def run(args):
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
    requested_splits = tuple(value.strip() for value in args.splits.split(","))
    frozen = frozen_evaluation_metadata(args, requested_splits)
    source_reference = pd.read_parquet(Path(args.data) / "reference.parquet").fillna("")
    reference = source_reference.loc[source_reference.split.isin(requested_splits)].reset_index(drop=True)
    if reference.empty:
        raise ValueError("No reference rows match the requested splits")
    config = {"schema": 1, "checkpoint_sha256": checkpoint["sha256"],
              "pool_embedding_identity": identity, "reference_digest": frame_digest(reference,
              ["entity_id", "business_name", "business_address", "country", "split"]),
              "splits": list(requested_splits), "top_k": args.top_k,
              "backend": args.backend, "score_precision": "float32_dot_from_float16_pool",
              "device": args.device, "query_batch_size": args.query_batch_size,
              "pool_batch_size": args.pool_batch_size, "threads": args.threads,
              "labels_used_for_mining": False, "positive_candidate_injection": False}
    if frozen is not None:
        config["final_evaluation"] = True
        config["frozen_matcher"] = frozen
    config_path = output / "mining_manifest.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to another mining configuration; choose a different directory")
    atomic_json(config_path, config)
    if frozen is not None:
        # Freeze before queries are encoded or raw candidates are mined. On a
        # resumed run, never replace the original frozen selection or timestamp.
        freeze_path = output / "frozen_evaluation.json"
        if freeze_path.exists():
            freeze = json.loads(freeze_path.read_text())
            if freeze["config"] != config:
                raise ValueError("Existing final-evaluation freeze metadata cannot be changed")
        else:
            reporter.manifest({"frozen_at_unix": time.time(), "config": config,
                               "retrieval_started": False, "labels_accessed": False}, "frozen_evaluation.json")
        reporter.write("Final-evaluation matcher/hash/threshold are frozen; retrieval begins without labels")
    raw_path = output / "raw_pairs.parquet"
    if raw_path.exists():
        reporter.write("Reusing completed raw-row candidate mining")
        raw_pairs = pd.read_parquet(raw_path)
    else:
        query_vectors = encode_queries(reference, args, identity, reporter)
        raw_pairs = mine_pairs(reference, query_vectors, Path(args.embeddings), identity, args, reporter)
        _atomic_parquet(raw_pairs, raw_path)
    reporter.write("Raw-row candidates are saved; loading selected reference truth for diagnostics and compaction")
    # This is intentionally the first truth access anywhere in the mining path.
    all_truth = json.loads((Path(args.data) / "truth.json").read_text())
    selected_ids = set(reference.entity_id)
    truth = {key: set(values) for key, values in all_truth.items() if key in selected_ids}
    if set(truth) != selected_ids:
        raise ValueError("Selected references are missing from the provided truth file")
    pool, pairs = compact_pool(raw_pairs, reference, truth, identity, reporter)
    _atomic_parquet(reference, output / "reference.parquet")
    _atomic_parquet(pool, output / "pool.parquet")
    _atomic_parquet(pairs, output / "pairs.parquet")
    atomic_json(output / "truth.json", {key: sorted(values) for key, values in truth.items()})
    retrieval = {split: candidate_report(reference, pool, pairs, truth, reference.split.eq(split))
                 for split in requested_splits}
    metadata = {"references": len(reference), "pool_records": len(pool),
                "mining_full_pool_records": identity["target_rows"], "true_links": sum(map(len, truth.values())),
                "reference_id_sha256": hashlib.sha256("\n".join(sorted(selected_ids)).encode()).hexdigest(),
                "split_counts": {f"{split}/{country}": int(count) for (split, country), count in
                                 reference.groupby(["split", "country"]).size().items()},
                "candidate_pairs": len(pairs), "retrieval": retrieval,
                "positive_candidate_injection": False, "truth_targets_retained_after_mining": True,
                "feature_semantics": {"name_score": "unavailable; zero", "address_score": "unavailable; zero",
                                      "retrieval_score": "dense cosine", "dense_only": "always one"},
                "elapsed_seconds": time.time() - started,
                "limitations": ["No French reference labels are available.",
                                "Candidate retrieval uses float32 dot products over the full fp16 provided training pool."]}
    if frozen is not None:
        metadata["final_evaluation"] = True
        metadata["frozen_matcher"] = frozen
    reporter.manifest(metadata, "metadata.json")
    reporter.manifest({"state": "complete", "config": config, "metadata": metadata}, "complete.json")
    reporter.write(f"COMPLETE retrieval={json.dumps(retrieval)}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="Labeled reference/truth directory, e.g. data/quick or data/medium")
    parser.add_argument("--splits", default="train,tune", help="Train/tune normally; eval only with --final-evaluation")
    parser.add_argument("--final-evaluation", action="store_true",
                        help="Deliberately open only eval references after freezing a selected matcher")
    parser.add_argument("--frozen-model-bundle", help="Selected ber.predict joblib bundle; required for final eval")
    parser.add_argument("--frozen-model-sha256", help="Explicitly selected bundle SHA256; required for final eval")
    parser.add_argument("--query-input", help="Unlabeled raw source1 TSV; mutually exclusive with --data")
    parser.add_argument("--expected-query-rows", type=int, help="Required exact S1 row count for unlabeled inference")
    parser.add_argument("--query-stream-size", type=int, default=50000)
    parser.add_argument("--pair-reference-batch-size", type=int, default=2000)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--durable-output")
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=512)
    parser.add_argument("--pool-batch-size", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["auto", "torch", "numpy"], default="auto")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    if min(args.top_k, args.batch_size, args.query_batch_size, args.pool_batch_size, args.threads,
           args.query_stream_size, args.pair_reference_batch_size) <= 0:
        parser.error("K, batch sizes and thread count must all be positive")
    if args.pool_batch_size > 50000:
        parser.error("Pool chunks are bounded at 50,000 rows")
    if args.query_stream_size > 50000:
        parser.error("Query text chunks are bounded at 50,000 rows")
    if bool(args.data) == bool(args.query_input):
        parser.error("Choose exactly one of --data (labeled train/tune) and --query-input (unlabeled)")
    if args.query_input:
        if args.final_evaluation or args.frozen_model_bundle or args.frozen_model_sha256:
            parser.error("Unlabeled inference does not accept final-evaluation or frozen-matcher flags")
        if not args.expected_query_rows or args.expected_query_rows <= 0:
            parser.error("Unlabeled --query-input requires a positive --expected-query-rows")
        from .full_dense_inference import run_unlabeled

        run_unlabeled(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
