"""Resume-safe streaming embeddings for the supplied full record pool.

Rows retain the exact input-file order. Only names and addresses enter the model;
no labels are read. Each completed chunk is flushed before either progress
manifest advances. The durable directory receives compact metadata/logs only.
"""
from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd


def atomic_json(path: Path, value: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        try:
            os.fsync(descriptor)
        except OSError as error:
            if error.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
    finally:
        os.close(descriptor)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_identity(path: Path) -> dict:
    if not path.is_dir():
        raise ValueError("--model must be a complete existing local checkpoint directory")
    files = []
    for candidate in sorted(path.rglob("*")):
        if candidate.is_file() and "__pycache__" not in candidate.parts:
            files.append({"path": str(candidate.relative_to(path)), "size": candidate.stat().st_size,
                          "sha256": file_digest(candidate)})
    if not files or not any(row["path"].endswith((".safetensors", ".bin")) for row in files):
        raise ValueError("Checkpoint contains no model weight file")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "files": files}


class Reporter:
    def __init__(self, output: Path, durable: Path | None):
        self.directories = [output]
        if durable is not None and durable.resolve() != output.resolve():
            self.directories.append(durable)
        for directory in self.directories:
            directory.mkdir(parents=True, exist_ok=True)

    def write(self, message: str) -> None:
        line = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + " " + message
        print(line, flush=True)
        for directory in self.directories:
            with (directory / "progress.log").open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
                stream.flush()
                os.fsync(stream.fileno())

    def manifest(self, value: dict, name: str = "progress.json") -> None:
        for directory in self.directories:
            atomic_json(directory / name, value)


def embedding_texts(frame: pd.DataFrame, prefix: str) -> list[str]:
    return (prefix + frame.business_name.fillna("").astype(str) + " | "
            + frame.business_address.fillna("").astype(str)).tolist()


def load_encoder(args):
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = SentenceTransformer(str(args.model), device=args.device,
                                local_files_only=True, trust_remote_code=False)
    model.max_seq_length = args.max_length
    if args.device.startswith("cuda"):
        model.half()
    model.eval()
    dimension = model.get_sentence_embedding_dimension()
    if dimension != args.dimension:
        raise ValueError(f"Expected {args.dimension}-dimensional encoder, got {dimension}")
    return model


def encode_chunk(model, texts: list[str], args) -> np.ndarray:
    result = model.encode(texts, batch_size=args.batch_size, show_progress_bar=False,
                          convert_to_numpy=True, normalize_embeddings=True)
    vectors = np.asarray(result, dtype=np.float32)
    if vectors.shape != (len(texts), args.dimension) or not np.isfinite(vectors).all():
        raise ValueError("Encoder output has invalid shape or non-finite values")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 1e-8):
        raise ValueError("Encoder produced a zero vector")
    # Normalize in float32 before storage even if the transformer uses fp16.
    return (vectors / norms).astype(np.float16)


def verify_embeddings(path: Path, expected_rows: int, dimension: int, chunk_size: int) -> dict:
    vectors = np.load(path, mmap_mode="r")
    if vectors.shape != (expected_rows, dimension) or vectors.dtype != np.float16:
        raise ValueError("Stored embedding shape or dtype does not match the manifest")
    min_norm, max_norm, count = float("inf"), 0.0, 0
    for start in range(0, expected_rows, chunk_size):
        chunk = np.asarray(vectors[start:start + chunk_size], dtype=np.float32)
        if not np.isfinite(chunk).all():
            raise ValueError(f"Non-finite stored embeddings at row {start}")
        norms = np.linalg.norm(chunk, axis=1)
        min_norm, max_norm = min(min_norm, float(norms.min())), max(max_norm, float(norms.max()))
        if np.any(np.abs(norms - 1) > 0.01):
            raise ValueError(f"Non-normalized stored embeddings at row {start}")
        count += len(chunk)
    return {"rows_verified": count, "all_finite": True, "min_l2_norm": min_norm, "max_l2_norm": max_norm}


def run(args) -> dict:
    started = time.time()
    output = Path(args.output)
    reporter = Reporter(output, Path(args.durable_output) if args.durable_output else None)
    lock = (output / "encoding.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another encoder currently owns this output directory") from error
    inputs = []
    for path, expected in zip(args.inputs, args.expected_rows):
        path = Path(path).resolve()
        stat = path.stat()
        inputs.append({"path": str(path), "size_bytes": stat.st_size,
                       "mtime_ns": stat.st_mtime_ns, "expected_rows": expected})
    total = sum(args.expected_rows)
    target = min(total, args.max_rows) if args.max_rows else total
    reporter.write(f"Checkpoint identity verification; scope_rows={target:,} full_rows={total:,}")
    identity = {"schema": 1, "inputs_in_order": inputs,
                "checkpoint": checkpoint_identity(Path(args.model)),
                "prefix": args.prefix, "text_format": "business_name | business_address",
                "max_length": args.max_length, "dimension": args.dimension,
                "storage_dtype": "float16", "normalization": "float32_l2_then_float16",
                "target_rows": target, "full_dataset_rows": total,
                "scope": "prefix_smoke" if target < total else "full_pool"}
    progress_path = output / "progress.json"
    vectors_path = output / "pool_embeddings.npy"
    completed, chunks_completed = 0, 0
    progress = None
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        if progress["identity"] != identity:
            raise ValueError("Resume identity mismatch: inputs/order/checkpoint/encoding config changed")
        completed = int(progress["completed_rows"])
        chunks_completed = int(progress["chunks_completed"])
        if not 0 <= completed <= target:
            raise ValueError("Invalid completed row count in progress manifest")
        if not vectors_path.exists() and completed:
            raise ValueError("Progress manifest exists but its embedding file is missing")
        if (output / "complete.json").exists():
            complete = json.loads((output / "complete.json").read_text())
            if complete.get("identity") != identity or complete.get("completed_rows") != target:
                raise ValueError("Completion marker does not match the current experiment")
            reporter.write(f"Already complete: {completed:,} embeddings")
            return complete
    elif vectors_path.exists():
        raise ValueError("Embedding file exists without a progress manifest; refusing to overwrite it")
    required = (target - completed) * args.dimension * np.dtype(np.float16).itemsize
    if shutil.disk_usage(output).free < required + 512 * 1024 * 1024:
        raise ValueError(f"Insufficient output disk space for remaining {required / 1024**3:.2f} GiB")
    if progress is None:
        # Publish zero progress first. If allocation is interrupted, restarting is
        # safe because no rows have been declared durable yet.
        progress = {"identity": identity, "completed_rows": 0, "chunks_completed": 0,
                    "state": "encoding", "pid": os.getpid(), "run_started_unix": started,
                    "last_entity_id": None, "rows_per_second_last_chunk": None}
        reporter.manifest(progress)
    if not vectors_path.exists():
        vectors = np.lib.format.open_memmap(vectors_path, mode="w+", dtype=np.float16,
                                            shape=(target, args.dimension))
    else:
        vectors = np.load(vectors_path, mmap_mode="r+")
        if vectors.shape != (target, args.dimension) or vectors.dtype != np.float16:
            raise ValueError("Resume embedding shape/dtype mismatch")
    reporter.write(f"Loading local encoder; resume_rows={completed:,}; batch={args.batch_size}; threads={args.threads}")
    model = load_encoder(args)
    run_start_rows = completed
    source_base = 0
    observed_counts = []
    for input_index, metadata in enumerate(inputs):
        observed = 0
        if source_base >= target:
            break
        reader = pd.read_csv(metadata["path"], sep="\t", dtype=str, keep_default_na=False,
                             usecols=["entity_id", "business_name", "business_address"],
                             chunksize=min(args.chunk_size, target))
        for frame in reader:
            start = source_base + observed
            observed += len(frame)
            if observed > metadata["expected_rows"]:
                raise ValueError(f"Input contains more rows than expected: {metadata['path']}")
            stop = min(source_base + observed, target)
            if stop <= completed:
                continue
            if start < completed:
                frame = frame.iloc[completed - start:]
                start = completed
            frame = frame.iloc[:stop - start]
            if start != completed:
                raise ValueError("Non-contiguous resume row ordering")
            chunk_started = time.time()
            encoded = encode_chunk(model, embedding_texts(frame, args.prefix), args)
            vectors[start:stop] = encoded
            vectors.flush()
            with vectors_path.open("rb") as stream:
                os.fsync(stream.fileno())
            completed = stop
            chunks_completed += 1
            elapsed = time.time() - chunk_started
            throughput = len(frame) / max(elapsed, 1e-9)
            progress.update(completed_rows=completed, chunks_completed=chunks_completed,
                            state="encoding", pid=os.getpid(), last_input_index=input_index,
                            last_input_row=completed - source_base, last_entity_id=str(frame.entity_id.iloc[-1]),
                            rows_per_second_last_chunk=throughput, last_update_unix=time.time(),
                            run_rows_per_second=(completed - run_start_rows) / max(time.time() - started, 1e-9))
            reporter.manifest(progress)
            reporter.write(f"chunk={chunks_completed} rows={completed:,}/{target:,} "
                           f"source={input_index + 1} throughput={throughput:,.0f}/s "
                           f"chunk_seconds={elapsed:.2f}")
            del encoded, frame
            if completed >= target and target < total:
                break
        reader.close()
        observed_counts.append(observed)
        # A smoke prefix intentionally stops before EOF. A full run verifies each
        # source's exact count, including the S2/S3 boundary.
        if target == total and observed != metadata["expected_rows"]:
            raise ValueError(f"Input row count mismatch: observed {observed}, expected {metadata['expected_rows']}")
        source_base += metadata["expected_rows"]
    if completed != target:
        raise ValueError(f"Encoding stopped at {completed:,} rows; expected {target:,}")
    if target == total and observed_counts != args.expected_rows:
        raise ValueError("Full input row counts do not match their supplied counts")
    for metadata in inputs:
        stat = Path(metadata["path"]).stat()
        if stat.st_size != metadata["size_bytes"] or stat.st_mtime_ns != metadata["mtime_ns"]:
            raise ValueError("An input file changed during encoding")
    del vectors
    reporter.write("Verifying stored vector count, finiteness and normalization in bounded chunks")
    verification = verify_embeddings(vectors_path, target, args.dimension, args.chunk_size)
    progress.update(state="complete", completed_rows=target, full_dataset_complete=target == total,
                    verified_input_rows=observed_counts if target == total else None,
                    observed_input_rows=observed_counts, verification=verification,
                    finished_unix=time.time(), elapsed_this_run_seconds=time.time() - started)
    reporter.manifest(progress)
    reporter.manifest(progress, "complete.json")
    reporter.write(f"COMPLETE scope={identity['scope']} rows={target:,} "
                   f"dtype=float16 shape=({target},{args.dimension})")
    return progress


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--expected-rows", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--durable-output")
    parser.add_argument("--prefix", default="passage: ")
    parser.add_argument("--dimension", type=int, default=384)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=50000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--max-rows", type=int, help="Encode a smoke-test prefix only, clearly marked as partial scope")
    args = parser.parse_args()
    if len(args.inputs) != len(args.expected_rows) or any(value <= 0 for value in args.expected_rows):
        parser.error("Each input must have one positive expected row count")
    if not 0 < args.chunk_size <= 50000:
        parser.error("Chunk size must be between 1 and 50,000")
    if min(args.batch_size, args.dimension, args.max_length, args.threads) <= 0:
        parser.error("Batch size, dimension, max length and threads must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        parser.error("Max rows must be positive")
    run(args)


if __name__ == "__main__":
    main()
