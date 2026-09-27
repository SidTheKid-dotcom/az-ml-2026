"""Bounded E5-small retrieval experiment: train, encode, and exact full-pool search.

Train consumes the caller's A-only pair pack, never development/holdout labels.
Control is symmetric single-positive in-batch InfoNCE. Multi samples up to two
distinct positives and uses the supplied hard nonmatches. Its per-positive
InfoNCE excludes *all* known same-business copies from the negative denominator;
businesses, rather than positive records, receive equal weight in both directions.
Any extra known copy arriving through another business's negatives is positive.
Zero-positive businesses cannot be contrastive anchors and are recorded, not fit.

All HF loads are offline, without remote code. Both modes use identical stateless
S1 batch plans for a given seed/budget. --batch-size counts S1s per microbatch;
--grad-accum multiplies that number, but does not enlarge the contrastive matrix.
Use the same settings for the controlled comparison. --steps counts optimizer
updates; resuming cannot change the budget or learning-rate schedule.

Encode writes float16 vectors.npy, int8 countries.npy and ordered ID parquet
chunks. manifest.json describes their row ranges; complete.json is written last.
It streams parquet row groups, uses token-level truncation/dynamic padding, and
checks ID uniqueness with a bounded-cache on-disk SQLite index. --resume verifies
committed chunks and recomputes only uncommitted work. Put outputs on fast local
disk. Time limits/signals stop at optimizer-step/encoding-chunk boundaries, plus
checkpoint/flush time, rather than being hard process deadlines.

Search writes query_indices.npy, pool_indices.npy and scores.npy. Output row i
belongs to query_indices[i]; pool indices are original, uncompacted input rows.
Missing neighbors are -1/-inf, including countries without any targets. Search
reads no truth, normalizes stored vectors in FP32 and exhaustively tiles every
same-country target. Equal FP32 scores break ties by smaller raw pool index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_ID = "intfloat/multilingual-e5-small"
COUNTRIES = {"US": 0, "India": 1, "France": 2}
DEFAULT_ALLOWLIST = Path(__file__).resolve().parents[4] / "problem-statement/models.yaml"
TRAIN_COLUMNS = ["pair_row", "left_idx", "right_idx", "s1_id", "rec_id",
                 "left_text", "right_text", "label", "true_count"]


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def array_sha256(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value)).cast("B")).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


@contextmanager
def output_lock(output, resume=False):
    """Do not overwrite another run or race a second process."""
    import fcntl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not resume and any(p.name != ".lock" for p in output.iterdir()):
            raise FileExistsError(f"Output is not empty: {output}; use a new directory or --resume")
        yield output


@contextmanager
def stop_requested(max_minutes):
    started = time.monotonic()
    requested = [False]

    def handler(signum, frame):
        requested[0] = True

    previous = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: requested[0] or (max_minutes > 0 and time.monotonic() - started >= max_minutes * 60)
    finally:
        for s, old in previous.items():
            signal.signal(s, old)


def configure_runtime(threads, device):
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    quota = Path("/sys/fs/cgroup/cpu.max")
    if quota.exists():
        q, period = quota.read_text().split()
        if q != "max":
            available = min(available, max(1, math.floor(int(q) / int(period))))
    threads = min(threads, available)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import pyarrow as pa
    import torch

    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(threads)
    except RuntimeError:  # already initialized in an embedding caller/test
        pass
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU smoke tests can use --device cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    memory = Path("/sys/fs/cgroup/memory.max")
    return {"threads": threads, "cpu_limit": available,
            "cgroup_memory_max": memory.read_text().strip() if memory.exists() else None,
            "torch": str(torch.__version__), "numpy": np.__version__, "device": str(device)}


def with_prefix(text: str, kind: str) -> str:
    """Identical token input for raw training text and already-prefixed R/texts."""
    if kind not in {"query", "passage"} or not isinstance(text, str):
        raise ValueError("Expected text and kind=query|passage")
    for prefix in ("query:", "passage:"):
        if text.startswith(prefix):
            if prefix != kind + ":":
                raise ValueError(f"Wrong E5 prefix for {kind}: {prefix}")
            return text
    return f"{kind}: {text}"


def tokenize(tokenizer, texts, kind, max_length, device):
    tokens = tokenizer([with_prefix(text, kind) for text in texts], padding=True,
                       truncation=True, max_length=max_length,
                       pad_to_multiple_of=8 if max_length % 8 == 0 else None,
                       return_tensors="pt")
    if tokens["input_ids"].shape[1] > max_length:
        raise ValueError("Tokenizer exceeded the token-level length cap")
    return {key: value.to(device) for key, value in tokens.items()}


def mean_pool(hidden, attention_mask):
    import torch.nn.functional as F

    mask = attention_mask.bool().unsqueeze(-1)
    pooled = hidden.float().masked_fill(~mask, 0).sum(1) / mask.sum(1).clamp_min(1)
    return F.normalize(pooled, p=2, dim=1)


def autocast(device, amp):
    import torch

    if str(device).startswith("cuda") and amp == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 is unavailable; explicitly select --amp fp32")
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def embed(model, tokenizer, texts, kind, max_length, device, amp):
    tokens = tokenize(tokenizer, texts, kind, max_length, device)
    with autocast(device, amp):
        hidden = model(**tokens).last_hidden_state
    return mean_pool(hidden, tokens["attention_mask"])


def build_masks(query_groups, passage_ids, passage_owners):
    """Return positive/negative masks; duplicate record columns are ignored.

    passage_owners is aligned with passage_ids, using -1 for unknown ownership.
    It must be constructed from *all* input positives, not only sampled positives.
    Equal text with different record IDs is not deduplicated.
    """
    groups, ids, owners = map(np.asarray, (query_groups, passage_ids, passage_owners))
    if any(a.ndim != 1 for a in (groups, ids, owners)) or len(ids) != len(owners):
        raise ValueError("Mask inputs must be aligned one-dimensional arrays")
    if len(np.unique(groups)) != len(groups) or np.any(groups < 0):
        raise ValueError("Queries must be distinct known business groups")
    _, first, inverse = np.unique(ids, return_index=True, return_inverse=True)
    if not np.array_equal(owners, owners[first][inverse]):
        raise ValueError("Duplicate record has conflicting ownership")
    unique = np.zeros(len(ids), bool)
    unique[first] = True
    copies = groups[:, None] == owners[None, :]
    return copies & unique[None, :], ~copies & unique[None, :]


def contrastive_loss(query_vectors, passage_vectors, positive_mask, negative_mask, temperature=0.05):
    """Symmetric per-positive InfoNCE, equal weight per S1, no positive competition.

    With one positive per query this is exactly the old symmetric cross entropy.
    For each positive, the denominator contains that positive and true negatives
    only. Averaging over positives avoids the easy-copy-only logsumexp objective.
    """
    import torch
    import torch.nn.functional as F

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    scores = query_vectors.float() @ passage_vectors.float().T / temperature
    pos = torch.as_tensor(positive_mask, dtype=torch.bool, device=scores.device)
    neg = torch.as_tensor(negative_mask, dtype=torch.bool, device=scores.device)
    if pos.shape != scores.shape or neg.shape != scores.shape or (pos & neg).any():
        raise ValueError("Positive and negative masks must be disjoint and match scores")
    counts = pos.sum(1)
    if not len(counts) or (counts == 0).any():
        raise ValueError("Each training query needs at least one genuine positive")
    # A finite floor avoids all--inf logsumexp backward NaNs when no negatives exist.
    negative_scores = scores.masked_fill(~neg, torch.finfo(scores.dtype).min)
    forward = F.softplus(torch.logsumexp(negative_scores, dim=1)[:, None] - scores)
    reverse = F.softplus(torch.logsumexp(negative_scores, dim=0)[None, :] - scores)
    return (((forward + reverse) * pos).sum(1) / counts).mean() * 0.5


@dataclass
class TrainingData:
    s1_ids: object
    rec_ids: object
    queries: object
    passages: object
    left_indices: np.ndarray
    true_counts: np.ndarray
    pair_rows: np.ndarray
    owners: np.ndarray
    positive_records: list
    negative_records: list
    eligible: np.ndarray
    stats: dict


def load_training_data(path, max_rows=4_000_000):
    """Arrow text storage, compact group/record indices; no model-score features."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    if not 0 < parquet.metadata.num_rows <= max_rows:
        raise ValueError(f"Training pack must contain 1..{max_rows:,} rows")
    if set(TRAIN_COLUMNS) - set(parquet.schema_arrow.names):
        raise ValueError(f"Missing training columns: {set(TRAIN_COLUMNS) - set(parquet.schema_arrow.names)}")
    table = parquet.read(columns=TRAIN_COLUMNS, use_threads=False).combine_chunks()
    if any(table[name].null_count for name in TRAIN_COLUMNS):
        raise ValueError("Null training values are not allowed")
    dictionaries, codes, firsts = {}, {}, {}
    for key, text, index in (("s1_id", "left_text", "left_idx"), ("rec_id", "right_text", "right_idx")):
        column = table[key].combine_chunks()
        if not (pa.types.is_string(column.type) or pa.types.is_large_string(column.type)):
            raise ValueError("Business and record IDs must be strings")
        encoded = column.dictionary_encode()
        dictionaries[key] = encoded.dictionary
        code = encoded.indices.to_numpy()
        first = np.unique(code, return_index=True)[1]
        codes[key], firsts[key] = code, first
        if pc.any(pc.equal(encoded.dictionary, "")).as_py():
            raise ValueError("Empty IDs are not allowed")
        for name in (text, index):
            representative = table[name].take(pa.array(first))
            if not pc.all(pc.equal(table[name], representative.take(pa.array(code)))).as_py():
                raise ValueError(f"Inconsistent {name} for the same {key}")
        raw = table[index].take(pa.array(first)).to_numpy()
        if raw.dtype.kind not in "iu" or np.any(raw < 0) or len(np.unique(raw)) != len(raw):
            raise ValueError(f"{index} must identify records one-to-one")
    groups, records = codes["s1_id"], codes["rec_id"]
    ng, nr = len(dictionaries["s1_id"]), len(dictionaries["rec_id"])
    pair_rows = table["pair_row"].to_numpy()
    if pair_rows.dtype.kind not in "iu" or np.any(pair_rows < 0) or len(np.unique(pair_rows)) != len(pair_rows):
        raise ValueError("Duplicate/invalid pair_row values")
    pair_keys = groups.astype(np.int64) * nr + records
    if len(np.unique(pair_keys)) != len(pair_keys):
        raise ValueError("Duplicate training pairs, including duplicate positives, are not allowed")
    labels = table["label"].to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    labels = labels.astype(bool)
    owners = np.full(nr, -1, np.int32)
    if len(np.unique(records[labels])) != int(labels.sum()):
        raise ValueError("A positive record cannot belong to multiple S1 businesses")
    owners[records[labels]] = groups[labels]
    counts = table["true_count"].to_numpy()
    true_counts = counts[firsts["s1_id"]]
    if counts.dtype.kind not in "iu" or np.any(counts < 0) or not np.array_equal(counts, true_counts[groups]):
        raise ValueError("Inconsistent true_count for a business")
    positive_records, negative_records = [], []
    # One sort, not one whole-table scan per business.
    order = np.argsort(groups, kind="stable")
    offsets = np.r_[0, np.cumsum(np.bincount(groups, minlength=ng))]
    for g in range(ng):
        rows = order[offsets[g]:offsets[g + 1]]
        positive_records.append(records[rows[labels[rows]]].copy())
        negative_records.append(records[rows[~labels[rows]]].copy())
    npos = np.array([len(p) for p in positive_records])
    if np.any(npos > true_counts):
        raise ValueError("Retrieved positives exceed complete truth counts")
    eligible = np.flatnonzero(npos > 0).astype(np.int32)
    if len(eligible) < 2:
        raise ValueError("Need at least two businesses with retrieved positive records")
    stats = {"pairs": len(groups), "businesses": ng, "records": nr,
             "eligible_businesses": len(eligible), "positive_pairs": int(labels.sum()),
             "singleton_businesses": int((true_counts == 0).sum()),
             "nonempty_businesses_without_retrieved_positives": int(((true_counts > 0) & (npos == 0)).sum())}
    return TrainingData(dictionaries["s1_id"], dictionaries["rec_id"],
                        table["left_text"].take(pa.array(firsts["s1_id"])),
                        table["right_text"].take(pa.array(firsts["rec_id"])),
                        table["left_idx"].to_numpy()[firsts["s1_id"]], true_counts,
                        pair_rows, owners, positive_records, negative_records, eligible, stats)


def batch_plan(eligible, steps, batch_size, grad_accum, seed):
    effective = batch_size * grad_accum
    if effective > len(eligible):
        raise ValueError("Effective batch exceeds eligible distinct S1s; reduce batch-size/grad-accum")
    return np.stack([np.random.default_rng(np.random.SeedSequence([seed, step, 0])).choice(
        eligible, effective, replace=False) for step in range(steps)]).astype(np.int32)


def sample_passages(data, groups, step, seed, mode, negatives_per_query=6):
    chosen = []
    for group in groups:
        group = int(group)
        rng = np.random.default_rng(np.random.SeedSequence([seed, step, group, 1]))
        positives = rng.permutation(data.positive_records[group])
        chosen.extend(positives[:1 if mode == "control" else 2].tolist())
        if mode == "multi":
            negatives = data.negative_records[group]
            rng = np.random.default_rng(np.random.SeedSequence([seed, step, group, 2]))
            if len(negatives) > negatives_per_query:
                negatives = rng.choice(negatives, negatives_per_query, replace=False)
            chosen.extend(negatives.tolist())
    records = np.array(list(dict.fromkeys(chosen)), dtype=np.int64)
    positive, negative = build_masks(groups, records, data.owners[records])
    return records, positive, negative


def verify_checkpoint(path):
    path = Path(path)
    marker = read_json(path / "complete.json")
    if marker.get("state") != "complete":
        raise ValueError(f"Incomplete checkpoint: {path}")
    required = {"training_config.json", "optimizer.pt", "rng.pt", "progress.json", "config.json"}
    if not required.issubset(marker.get("files_sha256", {})):
        raise ValueError("Checkpoint marker is missing required state")
    for name, expected in marker["files_sha256"].items():
        if Path(name).name != name or file_sha256(path / name) != expected:
            raise ValueError(f"Checkpoint file changed: {name}")
    config = read_json(path / "training_config.json")
    if digest(config) != marker["config_sha256"]:
        raise ValueError("Checkpoint configuration digest mismatch")
    return config, marker


def latest_checkpoint(output):
    # A completed checkpoint may outlive a crash just before publishing latest.json.
    candidates = [p for p in (Path(output) / "checkpoints").glob("step_*")
                  if p.name[5:].isdigit() and (p / "complete.json").is_file()]
    if not candidates:
        raise ValueError("No complete checkpoint to resume")
    return max(candidates, key=lambda p: int(p.name[5:]))


def resolve_model(model):
    path = Path(model).expanduser()
    if not path.is_dir():
        if str(model) != MODEL_ID:
            raise ValueError("--model must be a local E5-small directory or its canonical cached HF ID")
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(MODEL_ID, local_files_only=True))
    if (path / "latest.json").exists() or (path / "checkpoints").is_dir():
        path = latest_checkpoint(path)
    if (path / "training_config.json").exists():
        verify_checkpoint(path)
    return path.resolve()


def model_identity(path, allowlist):
    import yaml

    inventory = yaml.safe_load(Path(allowlist).read_text())
    entry = next((x for x in inventory["models"] if x["id"] == MODEL_ID), None)
    if not entry or entry.get("role") != "embedding" or entry.get("licence") != "mit":
        raise ValueError("The allowlist must include MIT intfloat/multilingual-e5-small as an embedding model")
    if float(entry["params_b"]) > min(0.120, float(inventory["budget_total_params_b"])):
        raise ValueError("Unexpected E5-small parameter budget")
    names = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.txt", "vocab.json", "merges.txt", "spiece.model",
             "sentencepiece.bpe.model", "model.safetensors.index.json", "pytorch_model.bin.index.json"}
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.name in names or p.suffix == ".safetensors" or p.name.startswith("pytorch_model")))
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("No offline model weights found")
    return {"model_id": MODEL_ID, "licence": entry["licence"], "params_b": entry["params_b"],
            "files_sha256": {p.name: file_sha256(p) for p in files}}


def load_encoder(path, device):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    model = AutoModel.from_pretrained(str(path), local_files_only=True, trust_remote_code=False)
    if sum(p.numel() for p in model.parameters()) > 120_000_000:
        raise ValueError("Loaded model is larger than the declared E5-small experiment")
    return model.to(device), tokenizer


def rng_state():
    import torch

    name, values, position, gaussian, cached = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(), "numpy": [name, values.tolist(), position, gaussian, cached]}


def restore_rng(state):
    import torch

    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    random.setstate(state["python"])
    name, values, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.asarray(values, np.uint32), position, gaussian, cached))


def save_checkpoint(output, model, tokenizer, optimizer, config, step, elapsed):
    import torch

    checkpoints = output / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    target = checkpoints / f"step_{step:06d}"
    if target.exists():
        old_config, marker = verify_checkpoint(target)
        if old_config != config or marker["step"] != step:
            raise ValueError("Existing checkpoint does not belong to this run")
    else:
        temporary = checkpoints / f".step_{step:06d}.incomplete-{uuid.uuid4().hex}"
        temporary.mkdir()
        state = rng_state()
        model.save_pretrained(str(temporary), safe_serialization=True)
        tokenizer.save_pretrained(str(temporary))
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        torch.save(state, temporary / "rng.pt")
        write_json(temporary / "training_config.json", config)
        write_json(temporary / "progress.json", {"step": step, "elapsed_seconds": elapsed})
        files = {p.name: file_sha256(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        write_json(temporary / "complete.json", {"state": "complete", "step": step,
                   "config_sha256": digest(config), "files_sha256": files})
        temporary.rename(target)
        restore_rng(state)  # Saving must not perturb a resumed versus uninterrupted dropout stream.
    write_json(output / "latest.json", {"checkpoint": str(target.relative_to(output)), "step": step})
    return target


def train(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    started = time.monotonic()
    runtime = configure_runtime(args.threads, args.device)
    with output_lock(args.output, bool(args.resume)) as output, stop_requested(args.max_minutes) as stopping:
        source = Path(args.train).resolve()
        source_hash = file_sha256(source)
        sidecar = Path(args.train_manifest) if args.train_manifest else source.parent / "train_complete.json"
        provenance = {"partition": "A", "scope_source": "caller A-only input contract"}
        if sidecar.exists():
            provenance = read_json(sidecar)
            if (provenance.get("state") != "complete" or provenance.get("partition") != "A"
                    or provenance.get("holdout2_rows_read") != 0 or provenance.get("train_sha256") != source_hash):
                raise ValueError("Training manifest must seal this A-only file with holdout2_rows_read=0")
        elif args.train_manifest:
            raise FileNotFoundError(sidecar)
        initial_path = resolve_model(args.model)
        identity = model_identity(initial_path, args.allowlist)
        data = load_training_data(source, args.max_train_rows)
        plan = batch_plan(data.eligible, args.steps, args.batch_size, args.grad_accum, args.seed)
        if args.mode == "multi" and not any(len(data.negative_records[g]) for g in data.eligible):
            raise ValueError("Multi mode needs supplied explicit nonmatches")
        warmup = args.warmup_steps if args.warmup_steps is not None else max(1, args.steps // 10)
        if warmup > args.steps:
            raise ValueError("warmup-steps exceeds the fixed step budget")
        selection = pa.table({"group_idx": np.arange(len(data.s1_ids), dtype=np.int32),
                              "s1_id": data.s1_ids, "left_idx": data.left_indices,
                              "true_count": data.true_counts,
                              "retrieved_positives": [len(x) for x in data.positive_records],
                              "nonmatches": [len(x) for x in data.negative_records]})
        config = {"schema": 1, "command": "train", "model": identity,
                  "source_sha256": file_sha256(__file__), "train_sha256": source_hash,
                  "provenance": provenance, "data": data.stats, "runtime": runtime,
                  "mode": args.mode, "seed": args.seed, "steps": args.steps,
                  "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                  "batch_plan_sha256": array_sha256(plan), "max_length": args.max_length,
                  "negatives_per_query": args.negatives_per_query, "temperature": args.temperature,
                  "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
                  "warmup_steps": warmup, "amp": args.amp,
                  "gradient_checkpointing": args.gradient_checkpointing,
                  "deterministic": args.deterministic,
                  "sampler": "distinct S1 per optimizer update; SeedSequence(seed,step,stream)",
                  "objective": "symmetric per-positive InfoNCE, equal S1 weight, global known-copy mask"}
        if args.resume:
            if read_json(output / "run_config.json") != config:
                raise ValueError("Resume configuration/data/model/runtime changed; use the original arguments")
            checkpoint = latest_checkpoint(output)
            if args.resume != "latest" and Path(args.resume).resolve() != checkpoint.resolve():
                raise ValueError("Refusing to rewind/branch an existing output; resume its latest complete checkpoint")
            saved_config, _marker = verify_checkpoint(checkpoint)
            if saved_config != config:
                raise ValueError("Resume checkpoint belongs to a different experiment")
            if not np.array_equal(np.load(output / "batch_groups.npy", allow_pickle=False), plan):
                raise ValueError("Saved S1 batch plan changed")
        else:
            pq.write_table(selection, output / "selected_s1.parquet", compression="zstd")
            np.save(output / "batch_groups.npy", plan)
            np.save(output / "train_pair_rows.npy", data.pair_rows)
            write_json(output / "run_config.json", config)
            write_json(output / "selection.json", {"train_file": str(source), "initial_model": str(initial_path),
                       "seed": args.seed, "files_sha256": {name: file_sha256(output / name) for name in
                       ("selected_s1.parquet", "batch_groups.npy", "train_pair_rows.npy")}})
            checkpoint = None
        selection_manifest = read_json(output / "selection.json")
        for name, expected in selection_manifest["files_sha256"].items():
            if file_sha256(output / name) != expected:
                raise ValueError(f"Training selection changed: {name}")
        random.seed(args.seed)
        np.random.seed(args.seed % 2**32)
        torch.manual_seed(args.seed)
        torch.use_deterministic_algorithms(args.deterministic)
        model, tokenizer = load_encoder(checkpoint or initial_path, args.device)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        step, previous_elapsed = 0, 0.0
        if checkpoint:
            optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", map_location=args.device, weights_only=True))
            restore_rng(torch.load(checkpoint / "rng.pt", map_location="cpu", weights_only=True))
            progress = read_json(checkpoint / "progress.json")
            step, previous_elapsed = progress["step"], progress["elapsed_seconds"]
        last_saved_step = step if checkpoint else -1
        log(f"{args.mode}: {data.stats}; effective S1 batch={args.batch_size * args.grad_accum}, starting step={step}")
        while step < args.steps and not stopping():
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for groups in plan[step].reshape(args.grad_accum, args.batch_size):
                records, pos, neg = sample_passages(data, groups, step, args.seed, args.mode, args.negatives_per_query)
                q = embed(model, tokenizer, data.queries.take(pa.array(groups)).to_pylist(), "query",
                          args.max_length, args.device, args.amp)
                p = embed(model, tokenizer, data.passages.take(pa.array(records)).to_pylist(), "passage",
                          args.max_length, args.device, args.amp)
                loss = contrastive_loss(q, p, pos, neg, args.temperature)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite contrastive loss")
                (loss / args.grad_accum).backward()
                loss_value += loss.detach().item() / args.grad_accum
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            factor = min((step + 1) / max(1, warmup), (args.steps - step) / max(1, args.steps - warmup))
            for group in optimizer.param_groups:
                group["lr"] = args.learning_rate * factor
            optimizer.step()
            step += 1
            if step == 1 or step % args.log_every == 0:
                log(f"step={step}/{args.steps} loss={loss_value:.6f} elapsed={time.monotonic() - started:.1f}s")
            if step % args.save_every == 0:
                checkpoint = save_checkpoint(output, model, tokenizer, optimizer, config, step,
                                             previous_elapsed + time.monotonic() - started)
                last_saved_step = step
        if last_saved_step != step:
            checkpoint = save_checkpoint(output, model, tokenizer, optimizer, config, step,
                                         previous_elapsed + time.monotonic() - started)
        result = {"state": "complete" if step == args.steps else "paused", "step": step,
                  "steps": args.steps, "checkpoint": str(checkpoint.relative_to(output)),
                  "config_sha256": digest(config)}
        write_json(output / ("complete.json" if result["state"] == "complete" else "progress.json"), result)
        log(json.dumps(result))
        return result


def country_codes(values):
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Countries must be one-dimensional")
    if values.dtype.kind in "iu":
        if not np.isin(values, list(COUNTRIES.values())).all():
            raise ValueError("Expected R/texts country codes 0=US, 1=India, 2=France")
        return values.astype(np.int8, copy=False)
    try:
        return np.asarray([COUNTRIES[v] for v in values], dtype=np.int8)
    except (KeyError, TypeError) as exc:
        raise ValueError("Expected US/India/France countries or their R/texts codes") from exc


def parquet_chunks(path, columns, chunk_rows, start=0):
    """Skip completed row groups without decoding their text again."""
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    offset = 0
    for rg in range(parquet.num_row_groups):
        end = offset + parquet.metadata.row_group(rg).num_rows
        if end <= start:
            offset = end
            continue
        for batch in parquet.iter_batches(batch_size=chunk_rows, row_groups=[rg], columns=columns,
                                          use_threads=False):
            skip = min(len(batch), max(0, start - offset))
            if skip < len(batch):
                yield offset + skip, batch.slice(skip)
            offset += len(batch)


def open_array(path, shape, dtype, resume):
    if not resume:
        return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array = np.load(path, mmap_mode="r+", allow_pickle=False)
    if array.shape != shape or array.dtype != np.dtype(dtype):
        raise ValueError(f"Array shape/dtype changed: {path}")
    return array


def encode(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    configure_runtime(args.threads, args.device)
    with output_lock(args.output, args.resume) as output, stop_requested(args.max_minutes) as stopping:
        source = Path(args.input).resolve()
        parquet = pq.ParquetFile(source)
        if {"entity_id", "text", "cc"} - set(parquet.schema_arrow.names):
            raise ValueError("Encoding input needs entity_id,text,cc")
        n = parquet.metadata.num_rows
        model_path = resolve_model(args.model)
        identity = model_identity(model_path, args.allowlist)
        config = {"schema": 1, "command": "encode", "input_sha256": file_sha256(source),
                  "model": identity, "source_sha256": file_sha256(__file__), "rows": n,
                  "kind": args.kind, "max_length": args.max_length, "batch_size": args.batch_size,
                  "chunk_rows": args.chunk_rows, "amp": args.amp, "device": args.device}
        config_hash = digest(config)
        if args.resume:
            if read_json(output / "run_config.json") != config:
                raise ValueError("Encode resume input/model/tokenization configuration changed")
            progress = read_json(output / "progress.json")
        else:
            progress = {"rows_done": 0, "chunks": []}
            write_json(output / "run_config.json", config)
            write_json(output / "progress.json", progress)
        model, tokenizer = load_encoder(model_path, args.device)
        model.eval()
        dimension = int(model.config.hidden_size)
        # A crash before initialization is recoverable because no chunk was committed.
        committed = progress["rows_done"] > 0
        vectors = open_array(output / "vectors.npy", (n, dimension), np.float16, args.resume and committed)
        countries = open_array(output / "countries.npy", (n,), np.int8, args.resume and committed)
        ids_dir = output / "ids"
        ids_dir.mkdir(exist_ok=True)
        end = 0
        for chunk in progress["chunks"]:
            start, stop = chunk["start"], chunk["stop"]
            if start != end or not start < stop <= n:
                raise ValueError("Noncontiguous encoding chunk manifest")
            if (array_sha256(vectors[start:stop]) != chunk["vectors_sha256"]
                    or array_sha256(countries[start:stop]) != chunk["countries_sha256"]
                    or file_sha256(output / chunk["ids"]) != chunk["ids_sha256"]):
                raise ValueError(f"Committed encoding chunk changed: {start}:{stop}")
            end = stop
        if end != progress["rows_done"]:
            raise ValueError("Encoding progress does not match committed chunks")
        database = sqlite3.connect(output / "id_index.sqlite3")
        try:
            database.execute("PRAGMA cache_size=-16384")
            database.execute("CREATE TABLE IF NOT EXISTS ids (entity_id TEXT PRIMARY KEY, row_idx INTEGER NOT NULL) WITHOUT ROWID")
            database.execute("DELETE FROM ids WHERE row_idx >= ?", (end,))
            database.commit()
            if database.execute("SELECT COUNT(*) FROM ids").fetchone()[0] != end:
                raise ValueError("ID uniqueness index does not match committed encoding chunks")
            began = time.monotonic()
            resumed_at = end
            with torch.inference_mode():
                for start, batch in parquet_chunks(source, ["entity_id", "text", "cc"], args.chunk_rows, end):
                    if stopping():
                        break
                    if any(batch.column(i).null_count for i in range(3)):
                        raise ValueError("Null encoding input")
                    ids = batch.column(0).to_pylist()
                    if any(not isinstance(x, str) or not x for x in ids):
                        raise ValueError("Encoding IDs must be nonempty strings")
                    texts = batch.column(1).to_pylist()
                    codes = country_codes(batch.column(2).to_numpy(zero_copy_only=False))
                    values = np.empty((len(batch), dimension), dtype=np.float16)
                    for b in range(0, len(batch), args.batch_size):
                        result = embed(model, tokenizer, texts[b:b + args.batch_size], args.kind,
                                       args.max_length, args.device, args.amp)
                        if not torch.isfinite(result).all() or (result.norm(dim=1) < 0.99).any():
                            raise ValueError("Encoder produced nonfinite/zero embeddings")
                        values[b:b + len(result)] = result.cpu().numpy().astype(np.float16)
                    stop = start + len(batch)
                    try:
                        database.executemany("INSERT INTO ids VALUES (?, ?)",
                                             ((entity, start + i) for i, entity in enumerate(ids)))
                    except sqlite3.IntegrityError as exc:
                        database.rollback()
                        raise ValueError("Duplicate encoding entity_id") from exc
                    vectors[start:stop], countries[start:stop] = values, codes
                    vectors.flush()
                    countries.flush()
                    ids_path = ids_dir / f"{start:012d}.parquet"
                    temp = ids_path.with_suffix(".parquet.tmp")
                    pq.write_table(pa.table({"row_idx": np.arange(start, stop, dtype=np.int64),
                                             "entity_id": ids, "cc": codes}), temp, compression="zstd")
                    temp.replace(ids_path)
                    database.commit()
                    progress["chunks"].append({"start": start, "stop": stop,
                        "ids": str(ids_path.relative_to(output)), "ids_sha256": file_sha256(ids_path),
                        "vectors_sha256": array_sha256(values), "countries_sha256": array_sha256(codes)})
                    end = progress["rows_done"] = stop
                    write_json(output / "progress.json", progress)
                    log(f"encoded {end:,}/{n:,}; {(end - resumed_at) / max(time.monotonic() - began, 1e-6):,.0f} rows/s")
        finally:
            database.close()
        result = {"state": "complete" if end == n else "paused", "rows_done": end, "rows": n,
                  "dimension": dimension, "config_sha256": config_hash}
        if end == n:
            manifest = {**result, "vectors": "vectors.npy", "dtype": "float16",
                        "countries": "countries.npy", "country_codes": COUNTRIES,
                        "order": "exact input parquet row order; ids chunks carry explicit row_idx",
                        "input": str(source), "chunks": progress["chunks"]}
            write_json(output / "manifest.json", manifest)
            write_json(output / "complete.json", {**result, "manifest_sha256": file_sha256(output / "manifest.json")})
        log(json.dumps(result))
        return result


def stable_topk(scores, raw_indices, k):
    """Exact lexicographic top-k: FP32 score descending, raw uint32 row ascending.

    Pack the ordered float32 bits and reversed row index into signed int64 keys.
    This resolves boundary ties without adding epsilon to/rounding cosine scores
    or sorting the entire score matrix. The full competition pool fits uint32.
    """
    import torch

    indices = raw_indices.expand_as(scores)
    bits = scores.contiguous().view(torch.int32).to(torch.int64)
    ordered = torch.where(bits < 0, -(bits & 0x7fffffff), bits)
    keys = ordered * (1 << 32) + ((1 << 32) - 1 - indices)
    chosen = torch.topk(keys, min(k, scores.shape[1]), dim=1, sorted=True).indices
    return scores.gather(1, chosen), indices.gather(1, chosen)


def iter_search_tiles(queries, pool, query_countries, pool_countries, *, query_rows=None,
                      top_k=50, query_tile=512, pool_tile=50000, device="cuda") -> Iterator:
    """Yield (output row positions, raw pool indices, FP32 scores), bounded tiles."""
    import torch
    import torch.nn.functional as F

    if queries.ndim != 2 or pool.ndim != 2 or queries.shape[1] != pool.shape[1]:
        raise ValueError("Query/pool embedding dimensions differ")
    if queries.shape[1] == 0 or len(pool) >= 2**32 or min(top_k, query_tile, pool_tile) < 1:
        raise ValueError("Invalid dimensions, tile sizes, top-k, or uint32 pool size")
    if queries.dtype.kind != "f" or pool.dtype.kind != "f":
        raise ValueError("Vectors must have floating dtype")
    qc, pc = country_codes(query_countries), country_codes(pool_countries)
    if len(qc) != len(queries) or len(pc) != len(pool):
        raise ValueError("Country arrays must align with complete vector arrays")
    rows = np.arange(len(queries), dtype=np.int64) if query_rows is None else np.asarray(query_rows)
    if (rows.ndim != 1 or rows.dtype.kind not in "iu" or np.any(rows < 0)
            or np.any(rows >= len(queries)) or len(np.unique(rows)) != len(rows)):
        raise ValueError("query-rows must be unique valid original query indices")
    torch.backends.cuda.matmul.allow_tf32 = False

    def tensor(array, selected):
        value = torch.from_numpy(np.array(array[selected], dtype=np.float32, copy=True)).to(device)
        if not torch.isfinite(value).all() or (value.norm(dim=1) == 0).any():
            raise ValueError("Search requires finite nonzero vectors")
        return F.normalize(value, p=2, dim=1)

    with torch.inference_mode():
        for country in np.unique(qc[rows]):
            positions = np.flatnonzero(qc[rows] == country)
            pool_rows = np.flatnonzero(pc == country)
            for start in range(0, len(positions), query_tile):
                out_rows = positions[start:start + query_tile]
                indices = np.full((len(out_rows), top_k), -1, np.int64)
                values = np.full((len(out_rows), top_k), -np.inf, np.float32)
                q = tensor(queries, rows[out_rows])
                best_scores = torch.empty((len(q), 0), dtype=torch.float32, device=device)
                best_ids = torch.empty((len(q), 0), dtype=torch.int64, device=device)
                for pstart in range(0, len(pool_rows), pool_tile):
                    raw = pool_rows[pstart:pstart + pool_tile]
                    p = tensor(pool, raw)
                    similarity = q @ p.T
                    block_scores, block_ids = stable_topk(similarity, torch.as_tensor(raw, device=device)[None, :], top_k)
                    best_scores, best_ids = stable_topk(torch.cat((best_scores, block_scores), dim=1),
                                                       torch.cat((best_ids, block_ids), dim=1), top_k)
                    del p, similarity, block_scores, block_ids
                if best_scores.shape[1]:
                    k = best_scores.shape[1]
                    indices[:, :k], values[:, :k] = best_ids.cpu().numpy(), best_scores.cpu().numpy()
                yield out_rows, indices, values


def search(args):
    configure_runtime(args.threads, args.device)
    with output_lock(args.out) as output:
        paths = {key: Path(getattr(args, key)) for key in
                 ("queries", "pool", "query_countries", "pool_countries")}
        arrays = {key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in paths.items()}
        identities = []
        for name, kind, cc_name in (("queries", "query", "query_countries"), ("pool", "passage", "pool_countries")):
            folder = paths[name].parent
            config_path = folder / "run_config.json"
            encoded = read_json(config_path) if config_path.exists() else {}
            if encoded.get("command") == "encode":
                complete_path = folder / "complete.json"
                if not complete_path.is_file():
                    raise ValueError(f"Unfinished encoding: {folder}")
                marker = read_json(complete_path)
                if (marker.get("state") != "complete" or marker.get("config_sha256") != digest(encoded)
                        or marker.get("manifest_sha256") != file_sha256(folder / "manifest.json")):
                    raise ValueError(f"Invalid encoding completion marker: {folder}")
                if encoded["kind"] != kind:
                    raise ValueError(f"Expected {kind} embeddings, got {encoded['kind']}")
                if (paths[name].resolve() != (folder / "vectors.npy").resolve()
                        or file_sha256(paths[cc_name]) != file_sha256(folder / "countries.npy")):
                    raise ValueError("Vectors/countries do not match the encoding order manifest")
                identities.append((encoded["model"], encoded["max_length"]))
        if len(identities) == 2 and identities[0] != identities[1]:
            raise ValueError("Query and pool encodings use different models or token caps")
        rows = (np.load(args.query_rows, allow_pickle=False) if args.query_rows
                else np.arange(len(arrays["queries"]), dtype=np.int64))
        config = {"schema": 1, "command": "search", "files_sha256": {key: file_sha256(p) for key, p in paths.items()},
                  "query_rows_sha256": array_sha256(rows), "top_k": args.top_k,
                  "query_tile": args.query_tile, "pool_tile": args.pool_tile,
                  "device": args.device, "score": "FP32 cosine; exhaustive same-country targets",
                  "ties": "smaller raw pool row first", "source_sha256": file_sha256(__file__)}
        write_json(output / "run_config.json", config)
        np.save(output / "query_indices.npy", rows)
        indices = open_array(output / "pool_indices.npy", (len(rows), args.top_k), np.int64, False)
        scores = open_array(output / "scores.npy", (len(rows), args.top_k), np.float32, False)
        done = 0
        for positions, ix, sc in iter_search_tiles(**arrays, query_rows=rows, top_k=args.top_k,
                                                   query_tile=args.query_tile, pool_tile=args.pool_tile,
                                                   device=args.device):
            indices[positions], scores[positions] = ix, sc
            done += len(positions)
            log(f"searched {done:,}/{len(rows):,} queries against full same-country pool")
        if done != len(rows):
            raise RuntimeError("Incomplete search coverage")
        indices.flush()
        scores.flush()
        result = {"state": "complete", "queries": len(rows), "pool_rows": len(arrays["pool"]),
                  "top_k": args.top_k, "missing_index": -1, "missing_score": "-inf",
                  "config_sha256": digest(config), "files_sha256": {name: file_sha256(output / name) for name in
                  ("query_indices.npy", "pool_indices.npy", "scores.npy")}}
        write_json(output / "complete.json", result)
        return result


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train", help="A-only fixed-business control or hard-negative multi-positive experiment")
    t.add_argument("--train", required=True)
    t.add_argument("--train-manifest", help="Optional A-only seal; adjacent train_complete.json is checked if present")
    t.add_argument("--output", required=True)
    t.add_argument("--mode", choices=["control", "multi"], required=True)
    t.add_argument("--steps", type=int, default=2000)
    t.add_argument("--batch-size", type=int, default=64, help="Distinct S1s per microbatch")
    t.add_argument("--grad-accum", "--gradient-accumulation", type=int, default=1)
    t.add_argument("--seed", type=int, default=20260927)
    t.add_argument("--learning-rate", type=float, default=2e-5)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--temperature", type=float, default=0.05)
    t.add_argument("--warmup-steps", type=int)
    t.add_argument("--negatives-per-query", type=int, default=6, help="Use all supplied nonmatches up to this cap")
    t.add_argument("--save-every", type=int, default=500)
    t.add_argument("--log-every", type=int, default=20)
    t.add_argument("--max-train-rows", type=int, default=4_000_000)
    t.add_argument("--gradient-checkpointing", action="store_true")
    t.add_argument("--deterministic", action="store_true", help="Require deterministic Torch kernels; may reduce throughput")
    t.add_argument("--resume", nargs="?", const="latest", help="Resume this output's latest checkpoint, or a specified checkpoint")
    e = sub.add_parser("encode", help="Streaming, chunk-resumable offline E5 encoding in exact parquet row order")
    e.add_argument("--input", required=True)
    e.add_argument("--output", required=True)
    e.add_argument("--kind", choices=["query", "passage"], required=True)
    e.add_argument("--batch-size", type=int, default=256)
    e.add_argument("--chunk-rows", type=int, default=16384)
    e.add_argument("--resume", action="store_true")
    for command in (t, e):
        command.add_argument("--model", required=True, help=f"Offline {MODEL_ID} or local pretrained/checkpoint directory")
        command.add_argument("--allowlist", default=str(DEFAULT_ALLOWLIST))
        command.add_argument("--max-length", type=int, default=128, help="Token cap including prefix and special tokens")
        command.add_argument("--amp", choices=["bf16", "fp32"], default="bf16", help="BF16 on CUDA; CPU always FP32")
        command.add_argument("--max-minutes", type=float, default=0, help="Cooperative wall cap; 0 means step/input bound only")
    s = sub.add_parser("search", help="Exact country-blocked FP32 cosine, preserving raw query/pool row indices")
    s.add_argument("--queries", required=True, help="Query vectors.npy")
    s.add_argument("--pool", required=True, help="Full pool vectors.npy")
    s.add_argument("--query-countries", required=True, help="Aligned full-query .npy, R/texts codes or string countries")
    s.add_argument("--pool-countries", required=True, help="Aligned full-pool .npy, R/texts codes or string countries")
    s.add_argument("--query-rows", help="Optional unique original-query indices .npy; preserves the supplied order")
    s.add_argument("--out", required=True)
    s.add_argument("--top-k", type=int, default=50)
    s.add_argument("--query-tile", "--query-batch-size", type=int, default=512)
    s.add_argument("--pool-tile", "--pool-batch-size", type=int, default=50000)
    for command in (t, e, s):
        command.add_argument("--device", default="cuda")
        command.add_argument("--threads", type=int, default=2)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ("steps", "batch_size", "grad_accum", "negatives_per_query", "max_length", "threads",
                 "save_every", "log_every", "max_train_rows", "chunk_rows", "top_k", "query_tile", "pool_tile"):
        if hasattr(args, name) and getattr(args, name) <= 0:
            p.error(f"{name.replace('_', '-')} must be positive")
    for name in ("learning_rate", "temperature", "weight_decay", "max_minutes"):
        if hasattr(args, name):
            value = getattr(args, name)
            if not math.isfinite(value) or value < 0 or (name in {"learning_rate", "temperature"} and value == 0):
                p.error(f"Invalid {name}")
    if getattr(args, "seed", 0) < 0 or (getattr(args, "warmup_steps", None) is not None and args.warmup_steps < 0):
        p.error("Seed/warmup must be nonnegative")
    if args.command == "train" and args.batch_size < 2:
        p.error("Training needs at least two S1s per microbatch for the in-batch control")
    return {"train": train, "encode": encode, "search": search}[args.command](args)


if __name__ == "__main__":
    main()
