r"""Controlled current-error BGE reranker continuation on sealed A-only pairs.

Examples (PYTHONPATH=code/business_entity_resolution/src):
  python -m ber.takeover_residual mine --train train.parquet --model INIT \
      --output mined --max-pairs 400000 --batch-size 256 --seed 27
  python -m ber.takeover_residual train --train train.parquet --model INIT \
      --output control --mode control --mined mined --steps 3000 \
      --batch-size 128 --microbatch 128 --seed 27
  python -m ber.takeover_residual train --train train.parquet --model INIT \
      --output residual --mode residual --mined mined --steps 3000 \
      --batch-size 128 --microbatch 128 --seed 27
  python -m ber.takeover_residual score --pairs brest_band.parquet \
      --model residual/checkpoint --output residual/brest.parquet --batch-size 256

Mining samples equal numbers of positives/negatives without replacement, keeping
physical input positions and original pair_row IDs. Labels are used for selection
and post-inference BCE only; the forward path accepts cached text tokens only.
No edges are created. The supplied pack, not an invented target pool, defines the
candidate distribution. Training REQUIRES its adjacent train_complete.json seal.

Both arms use 25% positives/75% negatives and the fast_trainer stake formula,
normalized by class over the ENTIRE optimizer batch, capped at 5, then weighted
BCE divided by the batch's weight sum. Control positives follow a shuffled cycle;
negative draws are half uniform and half the top --hard-fraction of negatives
ranked by v1_blend, then v1_tree (fixed v1-hard control, not the old typed sampler).
Residual draws in EACH class are half uniform and half high current BCE. Its hard
pool retains the top fraction within source/non-ASCII/empty-right-address strata;
these are coverage strata, not language classifications or fixed batch quotas.
The same mining positions are rescored once after floor(steps/2) updates, before
the next update. No B_REST labels, calibration, threshold, or evaluation is read.

Tokenization is lazy, once per visited row per invocation, in an in-memory ragged
cache. Each optimizer batch is length-bucketed into dynamically padded microbatches.
All HF loads are local; default CUDA execution is BF16 autocast with SDPA and FP32
AdamW parameters/state. Optional --compile wraps forward only. Standard HF model
files are saved alongside separate optimizer, sampler, RNG and sampled-index files.
The atomic `checkpoint` symlink names the sole retained generation. --save-every
defaults to 0 (save only at completion/cooperative stop); even periodic saves keep
only one generation plus one staging save, never a series of 6.8 GB checkpoints.

--resume resumes this output's latest committed optimizer boundary. Total --steps
and all mathematical settings are immutable, including the halfway refresh. Time
limits/signals finish the current optimizer step; setup/checkpoint I/O can overrun
the deadline. An interrupted refresh is discarded and retried from that boundary.
Bitwise replay depends on the same runtime/backend; compiled and eager dropout
streams are not interchangeable. Mining/scoring deadlines publish no partial result.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import sys
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


FORMAT = "ber.takeover_residual.v1"
MODEL_ID = "BAAI/bge-reranker-v2-m3"
PAIR_COLUMNS = ["pair_row", "s1_id", "rec_id", "left_text", "right_text"]
TRAIN_COLUMNS = [*PAIR_COLUMNS, "left_idx", "right_idx", "source", "label",
                 "true_count", "v1_tree", "v1_blend"]
TEXT_COLUMNS = ["left_text", "right_text"]
SCORE_COLUMNS = ["pair_row", "s1_id", "rec_id", "prob", "logit"]


def log(event, **fields):
    print(json.dumps({"event": event, **fields}, sort_keys=True, allow_nan=False), flush=True)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_digest(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(memoryview(values).cast("B")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def output_lock(root, resume=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume and any(p.name != ".lock" for p in root.iterdir()):
            raise FileExistsError(f"Output is not empty: {root}; use a new directory or --resume")
        yield root


class DeadlineExceeded(RuntimeError):
    pass


class Deadline:
    def __init__(self, minutes=None):
        self.started = time.monotonic()
        self.minutes = minutes
        self.signal = None

    def reason(self):
        if self.signal:
            return self.signal
        if self.minutes is not None and time.monotonic() - self.started >= self.minutes * 60:
            return "max_minutes"
        return None

    def check(self):
        if self.reason():
            raise DeadlineExceeded(self.reason())

    def __enter__(self):
        def handler(signum, _frame):
            self.signal = signal.Signals(signum).name
        self.previous = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
        return self

    def __exit__(self, *_):
        for s, handler in self.previous.items():
            signal.signal(s, handler)


def runtime_setup(args):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        count, period = quota.read_text().split()
        if count != "max":
            available = min(available, max(1, math.floor(int(count) / int(period))))
    threads = min(available, args.threads)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    torch.set_num_threads(threads)
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; CPU tests can use --device cpu --amp fp32")
    amp = ("bf16" if device.type == "cuda" else "fp32") if args.amp == "auto" else args.amp
    if amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires supported CUDA; use --amp fp32 on CPU")
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    import transformers

    memory = Path("/sys/fs/cgroup/memory.max")
    identity = {"torch": str(torch.__version__), "transformers": transformers.__version__,
                "numpy": np.__version__, "pyarrow": pa.__version__, "python": sys.version.split()[0],
                "cuda": torch.version.cuda, "device_type": device.type, "threads": threads,
                "cpu_limit": available, "memory_limit": memory.read_text().strip() if memory.exists() else None}
    return device, amp, identity


def autocast(device, amp):
    return torch.autocast(device.type, dtype=torch.bfloat16) if amp == "bf16" else nullcontext()


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count changed")
        torch.cuda.set_rng_state_all(state["cuda"])


def model_path(path):
    path = Path(path)
    if (path / "checkpoint").is_symlink():
        path = (path / "checkpoint").resolve()
    if not (path / "config.json").is_file():
        raise ValueError(f"Pass a complete local save_pretrained model directory: {path}")
    return path.resolve()


def model_identity(path):
    path = model_path(path)
    # Hash model/tokenizer inputs, not a previous trainer's multi-GB optimizer.
    files = {p.name: sha256(p) for p in sorted(path.iterdir()) if p.is_file() and (
        p.name in {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "added_tokens.json", "sentencepiece.bpe.model", "spiece.model", "vocab.txt", "merges.txt",
                   "vocab.json"} or p.name.startswith(("model", "pytorch_model"))
        and p.suffix in {".safetensors", ".bin", ".json"})}
    if not any(p.endswith((".safetensors", ".bin")) for p in files):
        raise ValueError("Model weights are missing")
    return {"path": str(path), "files": files, "sha256": digest(files)}


def load_model(path, device):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = model_path(path)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer.padding_side = "right"
    model = AutoModelForSequenceClassification.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa")
    if model.config.model_type != "xlm-roberta" or model.config.num_labels != 1:
        raise ValueError("This BGE adapter requires XLMRobertaForSequenceClassification with one logit")
    if sum(p.numel() for p in model.parameters()) > 600_000_000:
        raise ValueError("Model exceeds the BGE experiment's 600M parameter cap")
    if tokenizer.pad_token_id is None or model.config.pad_token_id != tokenizer.pad_token_id:
        raise ValueError("Model/tokenizer padding configuration differs")
    model.config.use_cache = False
    return model.to(device), tokenizer


def weights_digest(model):
    """Bind a refresh to live weights without writing a full midway checkpoint."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().contiguous().cpu()
        h.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        h.update(memoryview(tensor.reshape(-1).view(torch.uint8).numpy()).cast("B"))
    return h.hexdigest()


def validate_pairs(table):
    if set(PAIR_COLUMNS) - set(table.column_names):
        raise ValueError("Missing pair ID/text columns")
    for name in PAIR_COLUMNS:
        if table[name].null_count:
            raise ValueError(f"Null {name}")
        if name != "pair_row" and not (pa.types.is_string(table[name].type) or pa.types.is_large_string(table[name].type)):
            raise ValueError(f"{name} must be a string")
    rows = table["pair_row"].to_numpy()
    if rows.dtype.kind not in "iu" or np.any(rows < 0) or len(np.unique(rows)) != len(rows):
        raise ValueError("Duplicate/invalid pair_row")
    codes = []
    for name in ("s1_id", "rec_id"):
        encoded = table[name].combine_chunks().dictionary_encode()
        if pc.any(pc.equal(encoded.dictionary, "")).as_py():
            raise ValueError("Empty IDs")
        codes.append(encoded.indices.to_numpy().astype(np.int64))
    pair_keys = codes[0] * max(1, int(codes[1].max()) + 1 if len(rows) else 1) + codes[1]
    if len(np.unique(pair_keys)) != len(rows):
        raise ValueError("Duplicate (s1_id, rec_id)")
    return codes


@dataclass
class TrainingData:
    table: pa.Table
    labels: np.ndarray
    counts: np.ndarray
    pair_rows: np.ndarray
    identity: dict


def load_training(path):
    path = Path(path)
    seal_path = path.parent / "train_complete.json"
    if not seal_path.is_file():
        raise ValueError("Training requires adjacent train_complete.json proving sealed partition A")
    seal = read_json(seal_path)
    actual = sha256(path)
    if (seal.get("state") != "complete" or seal.get("partition") != "A"
            or type(seal.get("holdout2_rows_read")) is not int or seal["holdout2_rows_read"] != 0
            or seal.get("train_sha256") != actual):
        raise ValueError("Training seal must prove A-only, holdout2_rows_read=0 and matching dataset hash")
    pf = pq.ParquetFile(path)
    if "partition" in pf.schema_arrow.names:
        if any(value not in ("A", 0) for value in pf.read(columns=["partition"])["partition"].to_pylist()):
            raise ValueError("Training includes a non-A partition (B_REST is forbidden)")
    if set(TRAIN_COLUMNS) - set(pf.schema_arrow.names):
        raise ValueError(f"Missing training columns: {set(TRAIN_COLUMNS) - set(pf.schema_arrow.names)}")
    table = pf.read(columns=TRAIN_COLUMNS).combine_chunks()
    if not len(table) or any(table[k].null_count for k in TRAIN_COLUMNS):
        raise ValueError("Empty/null training pack")
    groups, records = validate_pairs(table)
    labels = table["label"].to_numpy()
    if labels.dtype.kind not in "biu" or not np.isin(labels, [0, 1]).all():
        raise ValueError("Training labels must be binary")
    labels = labels.astype(bool)
    counts = table["true_count"].to_numpy()
    first = np.unique(groups, return_index=True)[1]
    if (counts.dtype.kind not in "iu" or np.any(counts < 0)
            or not np.array_equal(counts, counts[first][groups])
            or np.any(np.bincount(groups[labels], minlength=len(first)) > counts[first])):
        raise ValueError("Invalid/inconsistent complete true_count")
    for key, codes in (("left_idx", groups), ("right_idx", records)):
        values = table[key].to_numpy()
        first_index = np.unique(codes, return_index=True)[1]
        if (values.dtype.kind not in "iu" or np.any(values < 0)
                or not np.array_equal(values, values[first_index][codes])
                or len(np.unique(values[first_index])) != len(first_index)):
            raise ValueError(f"{key} does not map one-to-one to supplied IDs")
    for key in ("v1_tree", "v1_blend"):
        values = table[key].to_numpy()
        if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
            raise ValueError(f"{key} must contain finite probabilities")
    source = table["source"].to_pylist()
    if any(s not in {"S2", "S3"} or not r.startswith(s + "-")
           for s, r in zip(source, table["rec_id"].to_pylist())):
        raise ValueError("Invalid source/record identity")
    for key, value in (("pairs", len(table)), ("positives", int(labels.sum())), ("train_s1", len(first))):
        if key in seal and seal[key] != value:
            raise ValueError(f"Training seal {key} differs from data")
    identity = {"path": str(path.resolve()), "sha256": actual, "seal_sha256": sha256(seal_path),
                "seal": seal, "rows": len(table), "s1s": len(first), "positives": int(labels.sum()),
                "partition": "A", "holdout2_rows_read": 0}
    return TrainingData(table, labels, counts, table["pair_row"].to_numpy(), identity)


def stake_weights(labels, true_counts, *, cap=5.0, normalize=True):
    """Exact ber.fast_trainer.stake_weights formula, without its heavy imports."""
    labels = np.asarray(labels, dtype=bool)
    counts = np.asarray(true_counts)
    if labels.shape != counts.shape or labels.ndim != 1 or np.any(counts < 0) or cap <= 0:
        raise ValueError("Stake labels/counts must be aligned, one-dimensional, and nonnegative")
    if np.any(labels & (counts < 1)):
        raise ValueError("A positive pair cannot belong to a no-match business")
    values = np.ones(len(labels), dtype=np.float32)
    matched_negative = ~labels & (counts >= 1)
    values[labels] = 1.0 / (5.0 * counts[labels] - 4.0)
    values[matched_negative] = 4.0 / (5.0 * counts[matched_negative] + 4.0)
    if normalize:
        for mask in (labels, ~labels):
            if mask.any():
                values[mask] /= float(values[mask].mean())
    return np.minimum(values, cap).astype(np.float32)


def mining_indices(labels, max_pairs, seed):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, [0, 1]).all() or max_pairs < 2 or max_pairs % 2:
        raise ValueError("Mining requires binary labels and a positive even --max-pairs")
    pools = [np.flatnonzero(labels == c) for c in (0, 1)]
    count = min(max_pairs // 2, *(len(p) for p in pools))
    if not count:
        raise ValueError("Mining needs both training classes")
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.choice(p, count, replace=False) for p in pools])).astype(np.int64)


class TokenCache:
    """Label-blind, lazy memory-only ragged cache, one tokenization per input row."""
    def __init__(self, texts, tokenizer, max_length=256, tokenize_batch_size=4096):
        if list(texts.column_names) != TEXT_COLUMNS:
            raise ValueError("TokenCache accepts only left_text/right_text")
        self.texts, self.tokenizer = texts, tokenizer
        self.max_length, self.chunk = max_length, tokenize_batch_size
        self.lengths = np.zeros(len(texts), dtype=np.int32)
        self.columns = ["input_ids"] + (["token_type_ids"] if "token_type_ids" in tokenizer.model_input_names else [])
        self.values = {k: [None] * len(texts) for k in self.columns}
        self.tokenized_rows = 0

    def ensure(self, indices, deadline=None):
        indices = np.asarray(indices, dtype=np.int64)
        missing = np.unique(indices[self.lengths[indices] == 0])
        for start in range(0, len(missing), self.chunk):
            if deadline:
                deadline.check()
            positions = missing[start:start + self.chunk]
            rows = self.texts.take(pa.array(positions))
            encoded = self.tokenizer(rows["left_text"].to_pylist(), rows["right_text"].to_pylist(),
                                     padding=False, truncation=True, max_length=self.max_length,
                                     return_attention_mask=False,
                                     return_token_type_ids="token_type_ids" in self.columns)
            for i, position in enumerate(positions):
                length = len(encoded["input_ids"][i])
                if not 0 < length <= self.max_length:
                    raise ValueError("Tokenizer exceeded the token-length cap")
                for key in self.columns:
                    self.values[key][position] = np.asarray(encoded[key][i], dtype=np.int32)
                self.lengths[position] = length
            self.tokenized_rows += len(positions)

    def batch(self, indices, device):
        indices = np.asarray(indices, dtype=np.int64)
        self.ensure(indices)
        width = min(self.max_length, math.ceil(int(self.lengths[indices].max()) / 8) * 8)
        arrays = {key: np.full((len(indices), width), self.tokenizer.pad_token_id if key == "input_ids" else 0,
                              dtype=np.int64) for key in self.columns}
        arrays["attention_mask"] = np.zeros((len(indices), width), dtype=np.int64)
        for i, position in enumerate(indices):
            length = self.lengths[position]
            for key in self.columns:
                arrays[key][i, :length] = self.values[key][position]
            arrays["attention_mask"][i, :length] = 1
        return {key: torch.from_numpy(value).to(device) for key, value in arrays.items()}


def bucket_positions(lengths, microbatch, boundaries):
    order = np.argsort(lengths, kind="stable")
    previous = 0
    for width in boundaries:
        end = int(np.searchsorted(lengths[order], width, side="right"))
        for start in range(previous, end, microbatch):
            yield order[start:min(end, start + microbatch)]
        previous = end
    if previous != len(order):
        raise ValueError("Length buckets do not cover the tokenized batch")


def predict(model, cache, indices, *, batch_size, device, amp, deadline=None):
    """Return logits in the caller's index order; labels never enter this API."""
    indices = np.asarray(indices, dtype=np.int64)
    cache.ensure(indices, deadline)
    order = np.argsort(cache.lengths[indices], kind="stable")
    result = np.empty(len(indices), dtype=np.float32)
    training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(order), batch_size):
                if deadline:
                    deadline.check()
                positions = order[start:start + batch_size]
                with autocast(device, amp):
                    logits = model(**cache.batch(indices[positions], device)).logits
                if logits.shape != (len(positions), 1):
                    raise ValueError("Expected one match logit per pair")
                values = logits[:, 0].float().cpu().numpy()
                if not np.isfinite(values).all():
                    raise FloatingPointError("Nonfinite inference logits")
                result[positions] = values
    finally:
        model.train(training)
    return result


def probabilities(logits):
    return np.exp(-np.logaddexp(0.0, -np.asarray(logits, dtype=np.float64))).astype(np.float32)


def bce_losses(logits, labels):
    return np.logaddexp(0.0, np.where(np.asarray(labels, bool), -np.asarray(logits, np.float64), logits)).astype(np.float32)


def coverage_strata(table):
    """Cheap coverage only: source x non-ASCII text x empty/missing right address."""
    return np.asarray([(int(s == "S3") * 4 + int(not (l.isascii() and r.isascii())) * 2
                        + int(not r.partition(" | ")[2].strip()))
                       for s, l, r in zip(table["source"].to_pylist(), table["left_text"].to_pylist(),
                                          table["right_text"].to_pylist())], dtype=np.uint8)


def commit_inventory(directory, kind, **metadata):
    files = {}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "complete.json":
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
            files[path.name] = sha256(path)
    atomic_json(directory / "complete.json", {"format": FORMAT, "kind": kind, "state": "complete",
                                               "files": files, **metadata})


def verify_inventory(directory, kind):
    directory = Path(directory)
    marker = read_json(directory / "complete.json")
    if marker.get("format") != FORMAT or marker.get("kind") != kind or marker.get("state") != "complete":
        raise ValueError(f"Not a complete residual {kind} artifact")
    required = ({"manifest.json", "indices.npz", "scores.parquet"} if kind == "mining" else
                {"config.json", "tokenizer_config.json", "training_config.json", "sampler.json", "state.json",
                 "optimizer.pt", "rng.pt", "sampled_indices.npz", "owner.json"})
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "complete.json"}
    if (not required <= marker.get("files", {}).keys() or actual != set(marker["files"])
            or (kind == "checkpoint" and not any(name.endswith(".safetensors") for name in actual))):
        raise ValueError("Artifact inventory is incomplete or changed")
    for name, expected in marker["files"].items():
        if Path(name).name != name or not (directory / name).is_file() or sha256(directory / name) != expected:
            raise ValueError(f"Artifact integrity failure: {name}")
    return marker


def write_mining(directory, data, indices, logits, reference, *, rng=None, sampler=None):
    """Publish one complete, aligned mining snapshot; no large model copy."""
    indices, logits = np.asarray(indices, dtype=np.int64), np.asarray(logits, dtype=np.float32)
    if (indices.ndim != 1 or not len(indices) or logits.shape != indices.shape
            or np.any(indices < 0) or np.any(indices >= len(data.table)) or np.any(np.diff(indices) <= 0)
            or not np.isfinite(logits).all()):
        raise ValueError("Mining requires finite logits aligned to unique increasing input positions")
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(f"Mining snapshot already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    stage = directory.with_name(f".{directory.name}.staging-{uuid.uuid4().hex}")
    stage.mkdir()
    try:
        selected = data.table.take(pa.array(indices))
        table = selected.select(["pair_row", "s1_id", "rec_id", "source", "label"])
        table = table.append_column("input_position", pa.array(indices, type=pa.int64()))
        for name, values in (("logit", logits), ("prob", probabilities(logits)),
                             ("loss", bce_losses(logits, data.labels[indices])),
                             ("stratum", coverage_strata(selected))):
            table = table.append_column(name, pa.array(values))
        pq.write_table(table, stage / "scores.parquet", compression="zstd")
        np.savez(stage / "indices.npz", input_position=indices, pair_row=data.pair_rows[indices])
        manifest = {"format": FORMAT, "train": data.identity, "reference": reference, "rows": len(indices),
                    "positives": int(data.labels[indices].sum()), "indices_sha256": array_digest(indices),
                    "labels_used_for_forward": False, "candidate_injection": False,
                    "loss": "unweighted binary cross entropy from current logits",
                    "strata": "4*(source==S3) + 2*(either text has non-ASCII) + empty/missing right address",
                    "coverage": {str(c): int(n) for c, n in zip(*np.unique(table["stratum"].to_numpy(), return_counts=True))}}
        atomic_json(stage / "manifest.json", manifest)
        if rng is not None:
            torch.save(rng, stage / "rng_before.pt")
            atomic_json(stage / "sampler_before.json", sampler)
        commit_inventory(stage, "mining")
        stage.rename(directory)
        sync_directory(directory.parent)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def load_mining(directory, data):
    directory = Path(directory)
    if (directory / "snapshot").is_dir():
        directory = directory / "snapshot"
    marker = verify_inventory(directory, "mining")
    if not {"manifest.json", "indices.npz", "scores.parquet"} <= marker["files"].keys():
        raise ValueError("Mining inventory is incomplete")
    manifest = read_json(directory / "manifest.json")
    if (manifest["train"]["sha256"] != data.identity["sha256"]
            or manifest["train"]["seal_sha256"] != data.identity["seal_sha256"]
            or manifest.get("labels_used_for_forward") is not False or manifest.get("candidate_injection") is not False):
        raise ValueError("Mining dataset identity or label/edge policy differs")
    table = pq.read_table(directory / "scores.parquet")
    indices = table["input_position"].to_numpy()
    if (indices.dtype.kind not in "iu" or not len(indices) or np.any(indices < 0)
            or np.any(indices >= len(data.table)) or np.any(np.diff(indices) <= 0)
            or array_digest(indices) != manifest["indices_sha256"] or len(indices) != manifest["rows"]):
        raise ValueError("Mining input positions are misaligned")
    with np.load(directory / "indices.npz", allow_pickle=False) as saved:
        if (not np.array_equal(saved["input_position"], indices)
                or not np.array_equal(saved["pair_row"], data.pair_rows[indices])):
            raise ValueError("Mining recorded indices are misaligned")
    selected = data.table.take(pa.array(indices))
    for name in ("pair_row", "s1_id", "rec_id", "source", "label"):
        if not pc.all(pc.equal(selected[name], table[name])).as_py():
            raise ValueError(f"Mining {name} is misaligned")
    logits, losses = (table[k].to_numpy() for k in ("logit", "loss"))
    if (not np.isfinite(logits).all() or not np.array_equal(losses, bce_losses(logits, data.labels[indices]))
            or not np.array_equal(table["prob"].to_numpy(), probabilities(logits))
            or not np.array_equal(table["stratum"].to_numpy(), coverage_strata(selected))):
        raise ValueError("Mining probabilities/loss/strata differ from aligned inputs")
    return {"directory": directory, "manifest": manifest, "sha256": sha256(directory / "complete.json"),
            "indices": indices, "losses": losses, "strata": table["stratum"].to_numpy()}


def high_loss_pools(data, mined, fraction, minimum):
    pools = {}
    indices, losses, strata = mined["indices"], mined["losses"], mined["strata"]
    for label in (0, 1):
        positions = np.flatnonzero(data.labels[indices] == label)
        if len(positions) < minimum[label]:
            raise ValueError("Mining subset is too small for the duplicate-free hard quota")
        ranked = positions[np.lexsort((data.pair_rows[indices[positions]], -losses[positions]))]
        chosen = []
        for stratum in np.unique(strata[ranked]):
            group = ranked[strata[ranked] == stratum]
            chosen.extend(group[:max(1, math.ceil(len(group) * fraction))].tolist())
        if len(chosen) < minimum[label]:
            chosen = list(dict.fromkeys([*chosen, *ranked[:minimum[label]].tolist()]))
        pools[label] = np.sort(indices[np.asarray(chosen, dtype=np.int64)])
    return pools


def v1_hard_pool(data, fraction, minimum):
    negative = np.flatnonzero(~data.labels)
    tree, blend = (data.table[k].to_numpy()[negative] for k in ("v1_tree", "v1_blend"))
    ranked = negative[np.lexsort((data.pair_rows[negative], -tree, -blend))]
    return np.sort(ranked[:max(minimum, math.ceil(len(ranked) * fraction))])


class PositiveCycle:
    def __init__(self, pool, seed, state=None):
        self.pool, self.seed = np.asarray(pool, dtype=np.int64), seed
        self.members = set(map(int, self.pool))
        self.epoch, self.cursor = (state["epoch"], state["cursor"]) if state else (0, 0)
        if not len(self.pool) or self.epoch < 0 or not 0 <= self.cursor <= len(self.pool):
            raise ValueError("Invalid positive cycle state")
        self._order()

    def _order(self):
        self.order = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(self.pool)

    def take(self, count, excluded=()):
        blocked = set(map(int, excluded))
        if len(self.pool) - len(blocked & self.members) < count:
            raise ValueError("Positive pool is too small for a duplicate-free batch")
        chosen = []
        while len(chosen) < count:
            if self.cursor == len(self.order):
                self.epoch, self.cursor = self.epoch + 1, 0
                self._order()
            value = int(self.order[self.cursor])
            self.cursor += 1
            if value not in blocked:
                chosen.append(value)
                blocked.add(value)
        return np.asarray(chosen, dtype=np.int64)

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}


class BatchSampler:
    """Exact 1:3 class quota, distinct rows within every optimizer batch."""
    def __init__(self, labels, hard, *, mode, batch_size, seed, state=None):
        self.labels = np.asarray(labels, bool)
        self.mode, self.batch_size, self.seed = mode, batch_size, seed
        if mode not in {"control", "residual"} or batch_size < 8 or batch_size % 8:
            raise ValueError("Batch size must be a multiple of eight for exact class/half quotas")
        self.counts = {1: batch_size // 4, 0: batch_size * 3 // 4}
        self.pools = {c: np.flatnonzero(self.labels == c) for c in (0, 1)}
        if any(len(self.pools[c]) < n for c, n in self.counts.items()):
            raise ValueError("Class pool is too small for a duplicate-free batch")
        self.rng = np.random.default_rng(seed + 1)
        self.step = 0
        self.set_hard(hard)
        self.cycle = PositiveCycle(self.pools[1], seed)
        if state is not None:
            for key, expected in (("mode", mode), ("seed", seed), ("batch_size", batch_size),
                                  ("labels_sha256", array_digest(self.labels)), ("hard_sha256", self.hard_sha256)):
                if state.get(key) != expected:
                    raise ValueError(f"Sampler resume {key} differs")
            self.rng.bit_generator.state = state["rng"]
            self.step = int(state["step"])
            if self.step < 0:
                raise ValueError("Invalid sampler step")
            self.cycle = PositiveCycle(self.pools[1], seed, state["positive_cycle"])

    def set_hard(self, hard):
        self.hard = {int(c): np.asarray(p, dtype=np.int64) for c, p in hard.items()}
        for c in ((0, 1) if self.mode == "residual" else (0,)):
            p = self.hard.get(c, np.empty(0, np.int64))
            if (len(p) < self.counts[c] // 2 or len(np.unique(p)) != len(p)
                    or np.any(p < 0) or np.any(p >= len(self.labels)) or np.any(self.labels[p] != c)):
                raise ValueError("Invalid/insufficient hard class pool")
        self.hard_sha256 = {str(c): array_digest(p) for c, p in self.hard.items()}

    def _distinct(self, pool, count, excluded=()):
        blocked = set(map(int, excluded))
        # Rejection sampling is O(batch), not O(1.8M), on the real pack.
        if len(pool) <= 4 * (len(blocked) + count):
            available = pool[~np.isin(pool, list(blocked))]
            if len(available) < count:
                raise ValueError("Pool is too small for a duplicate-free batch")
            return self.rng.choice(available, count, replace=False)
        chosen = []
        while len(chosen) < count:
            for value in pool[self.rng.integers(len(pool), size=count - len(chosen))]:
                value = int(value)
                if value not in blocked:
                    chosen.append(value)
                    blocked.add(value)
        return np.asarray(chosen, dtype=np.int64)

    def take(self):
        ids, kinds = [], []
        for c in (1, 0):
            count = self.counts[c]
            hard_count = 0 if c == 1 and self.mode == "control" else count // 2
            difficult = self._distinct(self.hard[c], hard_count) if hard_count else np.empty(0, np.int64)
            uniform = (self.cycle.take(count - hard_count, difficult) if c else
                       self._distinct(self.pools[c], count - hard_count, difficult))
            ids.extend([difficult, uniform])
            kinds.extend([np.ones(len(difficult), np.uint8), np.zeros(len(uniform), np.uint8)])
        ids, kinds = np.concatenate(ids), np.concatenate(kinds)
        order = self.rng.permutation(self.batch_size)
        self.step += 1
        return ids[order], kinds[order]

    def state(self):
        return {"mode": self.mode, "seed": self.seed, "batch_size": self.batch_size, "step": self.step,
                "labels_sha256": array_digest(self.labels), "hard_sha256": self.hard_sha256,
                "positive_cycle": self.cycle.state(), "rng": self.rng.bit_generator.state}


def train_step(model, forward, optimizer, cache, data, indices, *, microbatch, boundaries, device, amp, grad_clip):
    # Normalization MUST precede microbatching/bucketing, exactly as fast_trainer.
    weights = stake_weights(data.labels[indices], data.counts[indices])
    normalizer = float(weights.sum())
    cache.ensure(indices)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = torch.zeros((), device=device)
    for positions in bucket_positions(cache.lengths[indices], microbatch, boundaries):
        with autocast(device, amp):
            logits = forward(**cache.batch(indices[positions], device)).logits
        if logits.shape != (len(positions), 1):
            raise ValueError("Expected one training logit per pair")
        labels = torch.as_tensor(data.labels[indices[positions]], dtype=torch.float32, device=device)
        weight = torch.as_tensor(weights[positions], device=device)
        loss = (F.binary_cross_entropy_with_logits(logits[:, 0].float(), labels, reduction="none") * weight).sum() / normalizer
        loss.backward()
        total += loss.detach()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip, error_if_nonfinite=True)
    if not torch.isfinite(total).item():
        raise FloatingPointError("Nonfinite training loss")
    optimizer.step()
    return float(total), float(norm)


def checkpoint_path(root):
    root = Path(root)
    pointer = root / "checkpoint"
    if not pointer.is_symlink():
        raise ValueError("No committed residual checkpoint to resume")
    path = pointer.resolve()
    if path.parent != root.resolve() or not path.name.startswith("checkpoint-"):
        raise ValueError("Checkpoint pointer leaves its owned output")
    return path


def cleanup_checkpoints(root, run_sha):
    """Only this run's tagged inactive/staging generations may be removed."""
    root = Path(root)
    active = checkpoint_path(root) if (root / "checkpoint").is_symlink() else None
    for path in root.iterdir():
        if (path.is_symlink() or not path.is_dir() or path.resolve() == active
                or not path.name.startswith(("checkpoint-", ".checkpoint-"))):
            continue
        owner = path / "owner.json"
        if owner.is_file() and read_json(owner).get("run_sha256") == run_sha:
            shutil.rmtree(path)


def tensor_bytes(value):
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(tensor_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(v) for v in value)
    return 0


def save_checkpoint(root, model, tokenizer, optimizer, sampler, run, data, history, kinds, generations, *, refreshed):
    run_sha, step = digest(run), sampler.step
    cleanup_checkpoints(root, run_sha)
    state = optimizer.state_dict()
    needed = tensor_bytes(model.state_dict()) + tensor_bytes(state) + history[:step].nbytes * 3 + (32 << 20)
    if shutil.disk_usage(root).free < needed:
        raise OSError(f"Insufficient staging space: need approximately {needed / 2**30:.2f} GiB free")
    suffix = f"{step:08d}-{uuid.uuid4().hex[:12]}"
    stage, target = root / (".checkpoint-" + suffix), root / ("checkpoint-" + suffix)
    stage.mkdir()
    atomic_json(stage / "owner.json", {"run_sha256": run_sha})
    try:
        rng = capture_rng()
        model.save_pretrained(stage, safe_serialization=True)
        tokenizer.save_pretrained(stage)
        atomic_json(stage / "training_config.json", run)
        atomic_json(stage / "sampler.json", sampler.state())
        atomic_json(stage / "state.json", {"step": step, "refreshed": refreshed,
                    "active_mining_sha256": sha256(root / "refresh" / "complete.json") if refreshed else run["mining_sha256"]})
        torch.save(state, stage / "optimizer.pt")
        torch.save(rng, stage / "rng.pt")
        np.savez(stage / "sampled_indices.npz", input_position=history[:step],
                 pair_row=data.pair_rows[history[:step]], hard_draw=kinds[:step], generation=generations[:step])
        commit_inventory(stage, "checkpoint", step=step, run_sha256=run_sha)
        stage.rename(target)
        link = root / (".checkpoint-link-" + uuid.uuid4().hex)
        try:
            link.symlink_to(target.name, target_is_directory=True)
            os.replace(link, root / "checkpoint")
            sync_directory(root)
        finally:
            link.unlink(missing_ok=True)
        cleanup_checkpoints(root, run_sha)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    log("checkpoint", step=step, path=str(root / "checkpoint"))
    return root / "checkpoint"


def mine(args):
    with Deadline(args.max_minutes) as deadline, output_lock(args.output) as root:
        device, amp, runtime = runtime_setup(args)
        data = load_training(args.train)
        initial = model_identity(args.model)
        indices = mining_indices(data.labels, args.max_pairs, args.seed)
        model, tokenizer = load_model(args.model, device)
        cache = TokenCache(data.table.select(TEXT_COLUMNS), tokenizer, args.max_length, args.tokenize_batch_size)
        log("mine_start", rows=len(indices), positives=int(data.labels[indices].sum()), model=initial["path"])
        logits = predict(model, cache, indices, batch_size=args.batch_size, device=device, amp=amp, deadline=deadline)
        reference = {"initial_model": initial, "seed": args.seed, "max_pairs": args.max_pairs,
                     "max_length": args.max_length, "amp": amp, "batch_size": args.batch_size,
                     "runtime": runtime, "code_sha256": sha256(Path(__file__)), "selection": "uniform without replacement, 50/50 classes"}
        manifest = write_mining(root / "snapshot", data, indices, logits, reference)
        report = {"state": "complete", "rows": len(indices), "snapshot": str(root / "snapshot"),
                  "seconds": time.monotonic() - deadline.started, "manifest_sha256": sha256(root / "snapshot" / "manifest.json")}
        atomic_json(root / "mine_complete.json", report)
        log("mine_complete", **report)
        return manifest


def training_recipe(args, amp, device, runtime):
    keys = ("mode", "seed", "steps", "batch_size", "microbatch", "max_length", "hard_fraction", "learning_rate",
            "weight_decay", "warmup_steps", "grad_clip", "compile", "deterministic", "length_buckets", "mining_batch_size")
    recipe = {key: getattr(args, key) for key in keys}
    recipe.update(amp=amp, device_type=device.type, threads=runtime["threads"],
                  fused=(device.type == "cuda" if args.fused is None else args.fused),
                  positive_fraction=0.25, hard_share=0.5, refresh_step=args.steps // 2 if args.steps > 1 else None,
                  schedule="constant_after_warmup", stake="fast_trainer per-batch class-normalized, cap=5; divide by weight sum",
                  control="positive cycle; negatives half uniform, half top fraction by v1_blend/v1_tree")
    if recipe["fused"] and device.type != "cuda":
        raise ValueError("Fused AdamW requires CUDA")
    return recipe


def train(args):
    with Deadline(args.max_minutes) as deadline, output_lock(args.output, args.resume) as root:
        device, amp, runtime = runtime_setup(args)
        data = load_training(args.train)
        initial = model_identity(args.model)
        mined = load_mining(args.mined, data)
        reference = mined["manifest"]["reference"]
        if (reference["initial_model"]["sha256"] != initial["sha256"]
                or reference["max_length"] != args.max_length or reference["amp"] != amp):
            raise ValueError("Mining model, token length or precision differs from training initialization")
        recipe = training_recipe(args, amp, device, runtime)
        code_sha = sha256(Path(__file__))
        if args.resume:
            run = read_json(root / "run.json")
            if run.get("format") != FORMAT or run["recipe"] != recipe:
                raise ValueError("Resume training configuration differs (including total steps/refresh boundary)")
            if (run["data"]["sha256"] != data.identity["sha256"] or run["data"]["seal_sha256"] != data.identity["seal_sha256"]
                    or run["initial_model"]["sha256"] != initial["sha256"] or run["mining_sha256"] != mined["sha256"]
                    or run["code_sha256"] != code_sha or run["runtime"] != runtime):
                raise ValueError("Resume data/model/mining/code/runtime identity differs")
            checkpoint = checkpoint_path(root)
            marker = verify_inventory(checkpoint, "checkpoint")
            if marker["run_sha256"] != digest(run) or read_json(checkpoint / "training_config.json") != run:
                raise ValueError("Checkpoint belongs to another run")
            state = read_json(checkpoint / "state.json")
            step, refreshed = state["step"], state["refreshed"]
            if step != marker["step"] or not 0 <= step <= args.steps:
                raise ValueError("Checkpoint optimizer step differs")
            if (type(refreshed) is not bool or refreshed and (args.mode != "residual"
                    or recipe["refresh_step"] is None or step < recipe["refresh_step"])):
                raise ValueError("Checkpoint refresh boundary differs")
            model, tokenizer = load_model(checkpoint, device)
        else:
            model, tokenizer = load_model(args.model, device)
            common = {"recipe": {k: v for k, v in recipe.items() if k != "mode"},
                      "train_sha256": data.identity["sha256"], "initial_model_sha256": initial["sha256"],
                      "mining_sha256": mined["sha256"], "code_sha256": code_sha}
            run = {"format": FORMAT, "run_id": uuid.uuid4().hex, "recipe": recipe, "data": data.identity,
                   "initial_model": initial, "mining_sha256": mined["sha256"], "code_sha256": code_sha,
                   "runtime": runtime, "comparison_sha256": digest(common), "model_family": MODEL_ID,
                   "parameters": sum(p.numel() for p in model.parameters())}
            atomic_json(root / "run.json", run)
            # Small, self-contained mining provenance survives an upload of this run.
            shutil.copytree(mined["directory"], root / "initial_mining")
            step, refreshed = 0, False
        cleanup_checkpoints(root, digest(run))
        cached_mined = load_mining(root / "initial_mining", data)
        if cached_mined["sha256"] != run["mining_sha256"]:
            raise ValueError("Preserved initial mining snapshot changed")
        active = load_mining(root / "refresh", data) if refreshed else cached_mined
        if args.resume and state["active_mining_sha256"] != active["sha256"]:
            raise ValueError("Checkpoint active mining snapshot changed")
        minimum = {1: args.batch_size // 8, 0: args.batch_size * 3 // 8}
        hard = (high_loss_pools(data, active, args.hard_fraction, minimum) if args.mode == "residual" else
                {0: v1_hard_pool(data, args.hard_fraction, minimum[0])})
        sampler = BatchSampler(data.labels, hard, mode=args.mode, batch_size=args.batch_size, seed=args.seed,
                               state=read_json(checkpoint / "sampler.json") if args.resume else None)
        if sampler.step != step:
            raise ValueError("Checkpoint sampler and optimizer steps differ")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                     fused=recipe["fused"])
        history = np.empty((args.steps, args.batch_size), dtype=np.int64)
        kinds = np.empty_like(history, dtype=np.uint8)
        generations = np.empty(args.steps, dtype=np.uint8)
        cache = TokenCache(data.table.select(TEXT_COLUMNS), tokenizer, args.max_length, args.tokenize_batch_size)
        forward = torch.compile(model, dynamic=True) if args.compile else model
        if args.resume:
            with np.load(checkpoint / "sampled_indices.npz", allow_pickle=False) as saved:
                rows = saved["input_position"]
                if (rows.shape != (step, args.batch_size) or np.any(rows < 0) or np.any(rows >= len(data.table))
                        or not np.array_equal(data.pair_rows[rows], saved["pair_row"])):
                    raise ValueError("Checkpoint sampled indices differ")
                history[:step], kinds[:step], generations[:step] = rows, saved["hard_draw"], saved["generation"]
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location="cpu", weights_only=True))
            # This is our own hash-verified RNG file, not an external pickle.
            restore_rng(torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=False))
        log("train_start", step=step, total_steps=args.steps, mode=args.mode, comparison_sha256=run["comparison_sha256"],
            amp=amp, attention="sdpa", hard_pool_sizes={str(k): len(v) for k, v in hard.items()})
        last_saved = (step, refreshed) if args.resume else None
        reason = "steps"
        while sampler.step < args.steps:
            if deadline.reason():
                reason = deadline.reason()
                break
            if (args.mode == "residual" and not refreshed and recipe["refresh_step"] is not None
                    and sampler.step >= recipe["refresh_step"]):
                rng = capture_rng()
                try:
                    refresh_reference = {"initial_model": initial, "run_sha256": digest(run), "step": sampler.step,
                        "weights_sha256": weights_digest(model), "sampled_indices_sha256": array_digest(history[:sampler.step]),
                        "max_length": args.max_length, "amp": amp, "batch_size": args.mining_batch_size,
                        "initial_mining_sha256": run["mining_sha256"], "selection": "same fixed positions as initial mining"}
                    if (root / "refresh").exists():
                        active = load_mining(root / "refresh", data)
                        if active["manifest"]["reference"] != refresh_reference:
                            raise ValueError("Existing refresh is not from this exact optimizer boundary")
                    else:
                        log("refresh_start", step=sampler.step, rows=len(mined["indices"]))
                        logits = predict(model, cache, mined["indices"], batch_size=args.mining_batch_size,
                                         device=device, amp=amp, deadline=deadline)
                        write_mining(root / "refresh", data, mined["indices"], logits, refresh_reference,
                                     rng=rng, sampler=sampler.state())
                        active = load_mining(root / "refresh", data)
                    sampler.set_hard(high_loss_pools(data, active, args.hard_fraction, minimum))
                    refreshed = True
                    log("refresh_complete", step=sampler.step, snapshot_sha256=active["sha256"])
                except DeadlineExceeded as error:
                    reason = str(error)
                    break
                finally:
                    restore_rng(rng)
                if deadline.reason():
                    reason = deadline.reason()
                    break
            indices, draw_kind = sampler.take()
            lr = args.learning_rate * min(1.0, sampler.step / max(1, args.warmup_steps))
            for group in optimizer.param_groups:
                group["lr"] = lr
            loss, norm = train_step(model, forward, optimizer, cache, data, indices,
                microbatch=args.microbatch, boundaries=args.length_buckets, device=device, amp=amp, grad_clip=args.grad_clip)
            history[sampler.step - 1], kinds[sampler.step - 1], generations[sampler.step - 1] = indices, draw_kind, int(refreshed)
            if sampler.step == 1 or sampler.step % args.log_every == 0 or sampler.step == args.steps:
                log("loss", step=sampler.step, bce=loss, grad_norm=norm, lr=lr,
                    seconds=time.monotonic() - deadline.started, tokenized_rows=cache.tokenized_rows)
            if args.save_every and sampler.step % args.save_every == 0:
                save_checkpoint(root, model, tokenizer, optimizer, sampler, run, data, history, kinds, generations,
                                refreshed=refreshed)
                last_saved = (sampler.step, refreshed)
        if last_saved != (sampler.step, refreshed):
            save_checkpoint(root, model, tokenizer, optimizer, sampler, run, data, history, kinds, generations,
                            refreshed=refreshed)
        report = {"state": "complete" if sampler.step == args.steps else "paused", "reason": reason,
                  "step": sampler.step, "requested_steps": args.steps, "refresh_completed": refreshed,
                  "checkpoint": str(root / "checkpoint"), "comparison_sha256": run["comparison_sha256"],
                  "seconds": time.monotonic() - deadline.started, "tokenized_rows": cache.tokenized_rows,
                  "scope": "A-only supplied candidates; no evaluation labels"}
        atomic_json(root / "train_complete.json", report)
        log("train_complete", **report)
        return report


def score(args):
    output = Path(args.output)
    marker = output.with_name(output.name + ".complete.json")
    if output.exists() or marker.exists():
        raise FileExistsError(f"Score output already exists: {output}")
    with Deadline(args.max_minutes) as deadline:
        device, amp, runtime = runtime_setup(args)
        path = model_path(args.model)
        if (path / "complete.json").is_file() and read_json(path / "complete.json").get("format") == FORMAT:
            verify_inventory(path, "checkpoint")
        run = read_json(path / "training_config.json") if (path / "training_config.json").is_file() else None
        trained_length = run["recipe"]["max_length"] if run and run.get("format") == FORMAT else None
        length = args.max_length or trained_length or 256
        if trained_length and length != trained_length:
            raise ValueError("Score token length differs from the trained representation")
        identity = model_identity(path)
        model, tokenizer = load_model(path, device)
        pf = pq.ParquetFile(args.pairs)
        # Explicit column projection: never deserialize label/true_count/v1/baseline features.
        table = pf.read(columns=PAIR_COLUMNS).combine_chunks()
        validate_pairs(table)
        cache = TokenCache(table.select(TEXT_COLUMNS), tokenizer, length, args.tokenize_batch_size)
        logits = predict(model, cache, np.arange(len(table), dtype=np.int64), batch_size=args.batch_size,
                         device=device, amp=amp, deadline=deadline)
        result = table.select(SCORE_COLUMNS[:3]).append_column("prob", pa.array(probabilities(logits)))
        result = result.append_column("logit", pa.array(logits))
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
        try:
            pq.write_table(result, temporary, compression="zstd")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            # Exclusive publication even if another process raced the initial check.
            os.link(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        report = {"format": FORMAT, "state": "complete", "rows": len(table), "labels_read": False,
                  "input_sha256": sha256(args.pairs), "output_sha256": sha256(output), "model": identity,
                  "max_length": length, "amp": amp, "runtime": runtime, "seconds": time.monotonic() - deadline.started}
        atomic_json(marker, report)
        log("score_complete", rows=len(table), output=str(output), seconds=report["seconds"])
        return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("mine", "train", "score"):
        s = sub.add_parser(command)
        s.add_argument("--train" if command != "score" else "--pairs", type=Path, required=True)
        s.add_argument("--model", type=Path, required=True)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--batch-size", type=int, default=128 if command == "train" else 256)
        s.add_argument("--max-length", type=int, default=None if command == "score" else 256)
        s.add_argument("--tokenize-batch-size", type=int, default=4096)
        s.add_argument("--seed", type=int, default=20260927)
        s.add_argument("--device", default="auto")
        s.add_argument("--amp", choices=("auto", "bf16", "fp32"), default="auto")
        s.add_argument("--threads", type=int, default=4)
        s.add_argument("--deterministic", action="store_true")
        s.add_argument("--max-minutes", type=float)
        s.add_argument("--local-files-only", action="store_true", help="Compatibility flag; loads are always local-only")
        if command == "mine":
            s.add_argument("--max-pairs", type=int, default=400_000)
        if command == "train":
            s.add_argument("--mode", choices=("control", "residual"), required=True)
            s.add_argument("--mined", type=Path, required=True)
            s.add_argument("--steps", type=int, default=3000)
            s.add_argument("--microbatch", type=int, default=128)
            s.add_argument("--hard-fraction", type=float, default=0.25)
            s.add_argument("--mining-batch-size", type=int, default=256)
            s.add_argument("--learning-rate", "--lr", type=float, default=2e-5)
            s.add_argument("--weight-decay", type=float, default=0.01)
            s.add_argument("--warmup-steps", type=int, default=0)
            s.add_argument("--grad-clip", type=float, default=1.0)
            s.add_argument("--compile", action="store_true")
            s.add_argument("--fused", action=argparse.BooleanOptionalAction, default=None)
            s.add_argument("--length-buckets", help="Increasing ceilings ending at max-length; default 64,128,256 (clipped)")
            s.add_argument("--save-every", type=int, default=0, help="0: completion/stop only; retains one checkpoint at any interval")
            s.add_argument("--log-every", type=int, default=50)
            s.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if (min(args.batch_size, args.tokenize_batch_size, args.threads) < 1 or args.seed < 0
            or (args.max_length is not None and args.max_length < 8)
            or (args.max_minutes is not None and (not math.isfinite(args.max_minutes) or args.max_minutes <= 0))):
        p.error("Invalid positive runtime sizes, seed, token length or time budget")
    if args.command == "train":
        if (args.batch_size % 8 or not 1 <= args.microbatch <= args.batch_size
                or min(args.steps, args.log_every, args.mining_batch_size) < 1
                or args.save_every < 0 or args.warmup_steps < 0
                or not 0 < args.hard_fraction <= 1 or not math.isfinite(args.learning_rate) or args.learning_rate <= 0
                or not math.isfinite(args.weight_decay) or args.weight_decay < 0
                or not math.isfinite(args.grad_clip) or args.grad_clip <= 0):
            p.error("Invalid training counts, class ratio or optimizer settings")
        try:
            args.length_buckets = ([int(v) for v in args.length_buckets.split(",")] if args.length_buckets else
                                   [b for b in (64, 128) if b < args.max_length] + [args.max_length])
        except ValueError:
            p.error("--length-buckets must be comma-separated integers")
        if (not args.length_buckets or args.length_buckets != sorted(set(args.length_buckets))
                or args.length_buckets[0] < 1 or args.length_buckets[-1] != args.max_length):
            p.error("Length buckets must increase and end at --max-length")
    return {"mine": mine, "train": train, "score": score}[args.command](args)


if __name__ == "__main__":
    try:
        main()
    except DeadlineExceeded as error:
        log("stopped", reason=str(error), complete=False)
        raise SystemExit(2)
