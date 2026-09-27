"""ByT5 copy-versus-nonmatch likelihoods on supplied, real candidate pairs.

Train consumes an A-only parquet, caps a seeded 1:1 sample at 500,000 pairs,
and predicts observed right_text from ``COPY S2: left_text`` (positive) or
``NONMATCH S2: left_text`` (negative), analogously for S3. After MLE warmup,
the optional BCE objective differentiates both teacher-forced likelihoods.
No generation, synthetic negatives, thresholds, or development labels are used.

Example (from the repository root, with PYTHONPATH=code/business_entity_resolution/src):
  python -m ber.takeover_byte train --train train.parquet --model /models/byt5-small \
      --output /runs/byte1 --steps 1000 --batch-size 64 --microbatch 8 --seed 27 \
      --max-minutes 30 --local-files-only
  python -m ber.takeover_byte score --pairs brest_screen.parquet \
      --model /runs/byte1 --output /runs/byte1/brest_scores.parquet --batch-size 16

NLL columns are SUMS in natural-log units, including EOS and excluding padding.
log_ratio = nonmatch_nll - copy_nll; log_ratio_normalized divides by token_count.
Lengths include EOS. Both conditions see the SAME UTF-8-boundary-clipped left
text (budgeted for the longer NONMATCH prefix). Clipped targets get an EOS;
therefore their score is a likelihood of the clipped string, not the full string.
Microbatch counts PAIRS: discriminative training/scoring forwards 2x that many
conditioned sequences. Dynamic padding and deterministic length buckets are used.

--resume [checkpoint] resumes an owned run's latest complete optimizer boundary;
--steps is a total and can be extended (constant learning rate). Other training
hyperparameters must agree. Save directories are committed atomically with hashes
and complete markers. SIGINT/SIGTERM and --max-minutes finish the current step and
save; setup and final checkpoint I/O can exceed a very short wall-time budget.
An optional adjacent train_complete.json is checked for A-only/holdout provenance;
without it the A-only guarantee is the caller's input contract.
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
import signal
import sys
import tempfile
import time
import uuid

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F


FORMAT = "ber.takeover_byte.v1"
PAIR_COLUMNS = ("pair_row", "s1_id", "rec_id", "left_text", "right_text", "source")
TRAIN_COLUMNS = (*PAIR_COLUMNS, "label")
SCORE_NAMES = ("copy_nll", "nonmatch_nll", "token_count", "log_ratio", "log_ratio_normalized")


def log(event: str, **values) -> None:
    print(json.dumps({"event": event, **values}, allow_nan=False, sort_keys=True), flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path: Path, value: dict, *, replace: bool = False) -> None:
    """Exclusive publication by default; replacement is only for our run pointers."""
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)  # Atomic and never clobbers an existing file.
    finally:
        temporary.unlink(missing_ok=True)
    sync_directory(path.parent)


def validate_table(table: pa.Table, *, training: bool = False) -> None:
    required = TRAIN_COLUMNS if training else PAIR_COLUMNS
    missing = set(required) - set(table.column_names)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    if table["pair_row"].type != pa.int64():
        raise ValueError("pair_row must be int64")
    for name in required:
        if table[name].null_count:
            raise ValueError(f"Null {name}")
    if training and not pa.types.is_integer(table["label"].type):
        raise ValueError("Training label must be an integer 0/1")
    for row in table.select(required).to_pylist():
        if row["pair_row"] < 0:
            raise ValueError("Negative pair_row")
        if any(not isinstance(row[k], str) for k in PAIR_COLUMNS[1:]):
            raise ValueError("IDs, raw texts and source must be strings")
        if (not row["s1_id"] or not row["rec_id"] or row["source"] not in ("S2", "S3")
                or not row["rec_id"].startswith(row["source"] + "-")):
            raise ValueError("Empty IDs or source/rec_id mismatch")
        if training and row["label"] not in (0, 1):
            raise ValueError("Training label must be 0/1")


def unique_pairs(rows, seen_rows: set, seen_pairs: set) -> None:
    for row in rows:
        key = (row["s1_id"], row["rec_id"])
        if row["pair_row"] in seen_rows or key in seen_pairs:
            raise ValueError("Duplicate pair_row or (s1_id, rec_id)")
        seen_rows.add(row["pair_row"])
        seen_pairs.add(key)


def select_training(path: Path, max_pairs: int, seed: int):
    """Read only IDs/labels globally; gather selected text through bounded batches."""
    if max_pairs < 2 or max_pairs % 2:
        raise ValueError("--max-pairs must be positive and even (at least 2)")
    pf = pq.ParquetFile(path)
    if not set(TRAIN_COLUMNS) <= set(pf.schema_arrow.names):
        raise ValueError(f"Training requires {TRAIN_COLUMNS}")
    identity = {"sha256": sha256(path), "rows": pf.metadata.num_rows,
                "path": str(path.resolve()), "partition_check": "caller_A_only_contract"}
    seal_path = path.parent / "train_complete.json"
    if seal_path.exists():
        seal = read_json(seal_path)
        if (seal.get("state") != "complete" or seal.get("partition") != "A"
                or seal.get("holdout2_rows_read") != 0 or seal.get("train_sha256") != identity["sha256"]):
            raise ValueError("Training pack is not a verified A-only, holdout-excluded pack")
        identity.update(partition_check="train_complete.json", pack_manifest_sha256=sha256(seal_path))
    if "partition" in pf.schema_arrow.names:
        if any(x not in ("A", 0) for x in pf.read(columns=["partition"])["partition"].to_pylist()):
            raise ValueError("Training contains a non-A partition")
    index = pf.read(columns=["pair_row", "label"])
    if (index["pair_row"].type != pa.int64() or index["pair_row"].null_count
            or index["label"].null_count or not pa.types.is_integer(index["label"].type)):
        raise ValueError("Invalid pair_row/label schema")
    pair_rows, labels = (index[k].to_numpy() for k in ("pair_row", "label"))
    if np.any(pair_rows < 0) or len(np.unique(pair_rows)) != len(pair_rows):
        raise ValueError("Duplicate/negative training pair_row")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Training label must be 0/1")
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    per_class = min(max_pairs // 2, *(len(group) for group in groups))
    if not per_class:
        raise ValueError("Both real training classes are required")
    rng = np.random.default_rng(seed)
    positions = np.sort(np.concatenate([rng.choice(g, per_class, replace=False) for g in groups]))
    tables, offset = [], 0
    for batch in pf.iter_batches(batch_size=65_536, columns=list(TRAIN_COLUMNS)):
        lo, hi = np.searchsorted(positions, [offset, offset + len(batch)])
        if hi > lo:
            tables.append(pa.Table.from_batches([batch]).take(positions[lo:hi] - offset))
        offset += len(batch)
    selected = pa.concat_tables(tables)
    validate_table(selected, training=True)
    rows = selected.to_pylist()
    unique_pairs(rows, set(), set())
    h = hashlib.sha256()
    for row in rows:
        encoded = json.dumps([row[k] for k in TRAIN_COLUMNS], ensure_ascii=False,
                             separators=(",", ":")).encode("utf-8")
        h.update(len(encoded).to_bytes(8, "little"))
        h.update(encoded)
    identity.update(selected_rows=len(rows), positives=per_class, negatives=per_class,
                    selected_data_sha256=h.hexdigest(),
                    selected_pair_rows_sha256=hashlib.sha256(pair_rows[positions].astype("<i8").tobytes()).hexdigest(),
                    selection="seeded uniform without replacement per class; retained in input order")
    return rows, positions, identity


def clip_utf8(text: str, byte_budget: int) -> tuple[str, bool]:
    if byte_budget < 0:
        raise ValueError("Input length cannot fit the condition prefix and EOS")
    raw = text.encode("utf-8", errors="strict")
    # Ignore only an incomplete terminal codepoint; no replacement characters.
    return (raw[:byte_budget].decode("utf-8", errors="ignore"), True) if len(raw) > byte_budget else (text, False)


@dataclass
class ClipStats:
    rows: int = 0
    left_clipped: int = 0
    right_clipped: int = 0

    def add(self, other: ClipStats) -> None:
        self.rows += other.rows
        self.left_clipped += other.left_clipped
        self.right_clipped += other.right_clipped

    def report(self) -> dict:
        return {"rows": self.rows, "left_clipped": self.left_clipped, "right_clipped": self.right_clipped,
                "left_clip_rate": self.left_clipped / max(self.rows, 1),
                "right_clip_rate": self.right_clipped / max(self.rows, 1)}


def collate_pairs(tokenizer, rows, *, max_input_length=384, max_target_length=384,
                  pad_to_multiple_of=8, conditions=None):
    """No access to label/IDs/features. None emits COPY rows then NONMATCH rows."""
    if not rows or max_target_length < 1 or pad_to_multiple_of < 1:
        raise ValueError("Nonempty rows and positive target/padding lengths required")
    left, right, sources = [], [], []
    stats = ClipStats(rows=len(rows))
    for row in rows:
        source = row["source"]
        if source not in ("S2", "S3"):
            raise ValueError("source must be S2/S3")
        text, clipped = clip_utf8(row["left_text"], max_input_length - 1 - len(f"NONMATCH {source}: "))
        target, target_clipped = clip_utf8(row["right_text"], max_target_length - 1)
        left.append(text)
        right.append(target)
        sources.append(source)
        stats.left_clipped += int(clipped)
        stats.right_clipped += int(target_clipped)
    if conditions is None:
        left, right, sources = left * 2, right * 2, sources * 2
        conditions = ["COPY"] * len(rows) + ["NONMATCH"] * len(rows)
    if len(conditions) != len(left) or any(c not in ("COPY", "NONMATCH") for c in conditions):
        raise ValueError("One COPY/NONMATCH condition per row required")
    kwargs = dict(padding=True, pad_to_multiple_of=pad_to_multiple_of,
                  return_tensors="pt", truncation=False, split_special_tokens=True)
    # split_special_tokens keeps literal '<pad>'/'</s>' business text as bytes.
    encoder = tokenizer([f"{c} {s}: {t}" for c, s, t in zip(conditions, sources, left)], **kwargs)
    target = tokenizer(right, **kwargs)
    batch = {"input_ids": encoder["input_ids"], "attention_mask": encoder["attention_mask"],
             "target_ids": target["input_ids"], "target_mask": target["attention_mask"].bool()}
    return batch, stats


def masked_nll(logits: torch.Tensor, target_ids: torch.Tensor, mask: torch.Tensor):
    """FP32 summed token NLL and count; mask, NOT token ID, defines padding."""
    if logits.ndim != 3 or logits.shape[:2] != target_ids.shape or mask.shape != target_ids.shape:
        raise ValueError("Expected logits[B,T,V], target_ids[B,T], mask[B,T]")
    if mask.dtype != torch.bool or target_ids.dtype != torch.long:
        raise ValueError("mask must be bool and target_ids int64")
    counts = mask.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError("Every target must contain at least EOS")
    targets = target_ids.masked_fill(~mask, -100)
    tokens = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction="none", ignore_index=-100)
    return tokens.sum(dim=1), counts


def teacher_forced_nll(model, batch):
    targets, mask = batch["target_ids"], batch["target_mask"]
    decoder_ids = model.prepare_decoder_input_ids_from_labels(labels=targets.masked_fill(~mask, -100))
    # The decoder-start token equals pad for T5 but MUST attend as a real BOS.
    decoder_mask = torch.cat([torch.ones_like(mask[:, :1]), mask[:, :-1]], dim=1)
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                    decoder_input_ids=decoder_ids, decoder_attention_mask=decoder_mask,
                    use_cache=False, return_dict=True)
    return masked_nll(outputs.logits, targets, mask)


def discriminative_objectives(copy_nll, nonmatch_nll, counts, labels, *, temperature=0.1, normalization="mean"):
    if (copy_nll.shape != nonmatch_nll.shape or counts.shape != labels.shape
            or labels.shape != copy_nll.shape or temperature <= 0 or normalization not in ("mean", "sum")):
        raise ValueError("Invalid objective shapes/temperature/normalization")
    chosen = torch.where(labels.bool(), copy_nll, nonmatch_nll)
    ratio = nonmatch_nll - copy_nll
    if normalization == "mean":
        ratio = ratio / counts
    return (chosen / counts).mean(), F.binary_cross_entropy_with_logits(ratio / temperature, labels.float())


def lengths_for(rows, max_input_length, max_target_length):
    return np.asarray([max(min(len(r["left_text"].encode("utf-8")) + 14, max_input_length),
                           min(len(r["right_text"].encode("utf-8")) + 1, max_target_length))
                       for r in rows], dtype=np.int64)


class LengthBatches:
    """Reconstructable shuffle/sortish order; state is at optimizer boundaries."""
    def __init__(self, lengths, batch_size, seed, bucket_multiplier=32, state=None):
        self.lengths, self.batch_size, self.seed = lengths, batch_size, seed
        self.bucket_multiplier = bucket_multiplier
        self.epoch, self.cursor = (state["epoch"], state["cursor"]) if state else (0, 0)
        if not len(lengths) or self.epoch < 0 or not 0 <= self.cursor <= len(lengths):
            raise ValueError("Invalid saved sampler position")
        self._order()

    def _order(self):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch]))
        order = rng.permutation(len(self.lengths))
        width = self.batch_size * self.bucket_multiplier
        for start in range(0, len(order), width):
            ids = order[start:start + width]
            order[start:start + width] = ids[np.argsort(self.lengths[ids], kind="stable")]
        full = len(order) // self.batch_size * self.batch_size
        blocks = order[:full].reshape(-1, self.batch_size)
        self.order = np.concatenate([blocks[rng.permutation(len(blocks))].ravel(), order[full:]])

    def take(self):
        if self.cursor == len(self.order):
            self.epoch, self.cursor = self.epoch + 1, 0
            self._order()
        ids = self.order[self.cursor:self.cursor + self.batch_size]
        self.cursor += len(ids)
        return ids

    def state(self):
        return {"epoch": self.epoch, "cursor": self.cursor}


def runtime_setup(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(args.threads)
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    amp = args.amp
    if amp == "auto":
        amp = "bf16" if device.type == "cuda" and torch.cuda.is_bf16_supported() else "fp32"
    if amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 requires a BF16-capable CUDA device; use --amp fp32 on CPU")
    torch.use_deterministic_algorithms(args.deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    return device, amp


def runtime_identity():
    import transformers

    return {"torch": str(torch.__version__), "transformers": transformers.__version__,
            "numpy": np.__version__, "pyarrow": pa.__version__, "python": sys.version.split()[0],
            "cuda": torch.version.cuda}


def autocast(device, amp):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if amp == "bf16" else nullcontext()


def load_model(path, device, local_files_only=False):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, ByT5Tokenizer

    kwargs = dict(local_files_only=local_files_only, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(path), **kwargs)
    if not isinstance(tokenizer, ByT5Tokenizer):
        raise ValueError("This UTF-8 byte scorer requires a ByT5Tokenizer")
    tokenizer.padding_side = "right"
    model = AutoModelForSeq2SeqLM.from_pretrained(str(path), dtype=torch.float32, **kwargs)
    if (model.config.model_type != "t5" or model.config.vocab_size < len(tokenizer)
            or model.config.decoder_start_token_id != tokenizer.pad_token_id
            or model.config.eos_token_id != tokenizer.eos_token_id):
        raise ValueError("Incompatible T5/ByT5 vocabulary or decoder special-token configuration")
    return model.to(device), tokenizer


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device count differs")
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_path(path: Path) -> Path:
    if (path / "complete.json").is_file():
        return path
    candidates = sorted(p for p in path.glob("checkpoint-*") if (p / "complete.json").is_file())
    if not candidates:
        raise ValueError(f"No complete checkpoint at {path}")
    return candidates[-1]


def verify_checkpoint(path: Path) -> dict:
    complete = read_json(path / "complete.json")
    if complete.get("format") != FORMAT or complete.get("state") != "complete":
        raise ValueError("Not a complete takeover_byte checkpoint")
    files = complete.get("files", {})
    required = {"config.json", "tokenizer_config.json", "training_state.pt", "run.json", "selected_rows.npz"}
    if not required <= files.keys() or not any(k.endswith(".safetensors") for k in files):
        raise ValueError("Incomplete checkpoint inventory")
    for name, digest in files.items():
        if Path(name).name != name or not (path / name).is_file() or sha256(path / name) != digest:
            raise ValueError(f"Checkpoint integrity failure: {name}")
    if json_hash(read_json(path / "run.json")) != complete["run_sha256"]:
        raise ValueError("Checkpoint run identity changed")
    return complete


def save_checkpoint(root, model, tokenizer, optimizer, sampler, step, run, positions, pair_rows):
    target = root / f"checkpoint-{step:08d}"
    if target.exists():
        raise FileExistsError(f"Checkpoint already exists: {target}")
    temporary = Path(tempfile.mkdtemp(prefix=f".checkpoint-{step:08d}-", dir=root))
    state = {"step": step, "sampler": sampler.state(), "optimizer": optimizer.state_dict(), "rng": capture_rng()}
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save(state, temporary / "training_state.pt")
    np.savez(temporary / "selected_rows.npz", input_position=positions, pair_row=pair_rows)
    atomic_json(temporary / "run.json", run)
    files = {}
    for file in sorted(temporary.iterdir()):
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
        files[file.name] = sha256(file)
    atomic_json(temporary / "complete.json", {"format": FORMAT, "state": "complete", "step": step,
                                             "run_sha256": json_hash(run), "files": files})
    os.rename(temporary, target)
    sync_directory(root)
    atomic_json(root / "latest.json", {"checkpoint": target.name, "step": step}, replace=True)
    log("checkpoint", step=step, path=str(target))
    return target


@contextmanager
def stop_signals():
    requested = []
    def handler(signum, _frame):
        requested.append(signal.Signals(signum).name)
    previous = {s: signal.signal(s, handler) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield requested
    finally:
        for sig, old in previous.items():
            signal.signal(sig, old)


def train(args) -> dict:
    started = time.monotonic()
    root, source = Path(args.output), Path(args.train)
    if args.batch_size % args.microbatch:
        raise ValueError("--batch-size must be divisible by --microbatch")
    device, amp = runtime_setup(args)
    recipe_keys = ("model", "seed", "batch_size", "microbatch", "max_pairs", "max_input_length",
                   "max_target_length", "pad_to_multiple_of", "lr", "weight_decay", "grad_clip",
                   "mle_warmup_steps", "discriminative_weight", "discriminative_temperature",
                   "discriminative_normalization", "gradient_checkpointing", "bucket_multiplier", "deterministic",
                   "threads")
    recipe = {k: getattr(args, k) for k in recipe_keys}
    recipe.update(amp=amp, device_type=device.type)
    if args.resume:
        run = read_json(root / "run.json")
        if run.get("format") != FORMAT or run.get("recipe") != recipe:
            raise ValueError("Resume training configuration differs from this run")
    else:
        root.mkdir(parents=True, exist_ok=False)
    with (root / ".train.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another trainer owns this output") from error
        rows, positions, identity = select_training(source, args.max_pairs, args.seed)
        pair_rows = np.asarray([r["pair_row"] for r in rows], dtype=np.int64)
        runtime = runtime_identity()
        code_hash = sha256(Path(__file__))
        state = None
        if args.resume:
            if (run["input"]["sha256"] != identity["sha256"]
                    or run["input"]["selected_data_sha256"] != identity["selected_data_sha256"]
                    or run["code_sha256"] != code_hash or run["runtime"] != runtime):
                raise ValueError("Resume data/code/runtime identity differs")
            latest = checkpoint_path(root)
            checkpoint = latest if args.resume == "latest" else Path(args.resume).resolve()
            if checkpoint.resolve() != latest.resolve():
                raise ValueError("Resume must use this output's latest complete checkpoint")
            complete = verify_checkpoint(checkpoint)
            if complete["run_sha256"] != json_hash(run):
                raise ValueError("Checkpoint belongs to a different run")
            with np.load(checkpoint / "selected_rows.npz", allow_pickle=False) as saved:
                if not (np.array_equal(saved["input_position"], positions) and np.array_equal(saved["pair_row"], pair_rows)):
                    raise ValueError("Saved selection differs")
            # Only load our own, hash-verified full optimizer/RNG checkpoint.
            state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
            if state["step"] != complete["step"] or state["step"] > args.steps:
                raise ValueError("Resume step exceeds --steps or mismatches marker")
            model, tokenizer = load_model(checkpoint, device, local_files_only=True)
        else:
            model, tokenizer = load_model(args.model, device, args.local_files_only)
            initial_files = {}
            local_model = Path(args.model)
            if local_model.is_dir():
                initial_files = {p.name: sha256(p) for p in sorted(local_model.iterdir())
                                 if p.is_file() and p.suffix in (".json", ".safetensors", ".bin")}
            run = {"format": FORMAT, "run_id": uuid.uuid4().hex, "recipe": recipe, "input": identity,
                   "code_sha256": code_hash, "runtime": runtime,
                   "initial_model": {"name": args.model, "commit": getattr(model.config, "_commit_hash", None),
                                     "files": initial_files, "parameters": sum(p.numel() for p in model.parameters())}}
            atomic_json(root / "run.json", run)
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        sampler = LengthBatches(lengths_for(rows, args.max_input_length, args.max_target_length),
                                args.batch_size, args.seed, args.bucket_multiplier,
                                state=state["sampler"] if state else None)
        step = state["step"] if state else 0
        if state:
            optimizer.load_state_dict(state["optimizer"])
            restore_rng(state["rng"])
        log("train_start", step=step, total_steps=args.steps, selected=identity, amp=amp, device=str(device))
        clips, last_saved = ClipStats(), step if state else -1
        reason = "steps"
        with stop_signals() as stopped:
            while step < args.steps:
                if stopped or (args.max_minutes is not None and time.monotonic() - started >= 60 * args.max_minutes):
                    reason = stopped[-1] if stopped else "max_minutes"
                    break
                ids = sampler.take()
                active = args.discriminative_weight > 0 and step >= args.mle_warmup_steps
                optimizer.zero_grad(set_to_none=True)
                totals = {"loss": 0.0, "mle_nll_per_token": 0.0, "discriminative_bce": 0.0}
                for start in range(0, len(ids), args.microbatch):
                    micro = [rows[i] for i in ids[start:start + args.microbatch]]
                    labels = torch.tensor([r["label"] for r in micro], device=device)
                    conditions = None if active else ["COPY" if r["label"] else "NONMATCH" for r in micro]
                    batch, stat = collate_pairs(tokenizer, micro, max_input_length=args.max_input_length,
                                               max_target_length=args.max_target_length,
                                               pad_to_multiple_of=args.pad_to_multiple_of, conditions=conditions)
                    clips.add(stat)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    with autocast(device, amp):
                        nll, counts = teacher_forced_nll(model, batch)
                        if active:
                            n = len(micro)
                            mle, disc = discriminative_objectives(nll[:n], nll[n:], counts[:n], labels,
                                temperature=args.discriminative_temperature,
                                normalization=args.discriminative_normalization)
                        else:
                            mle, disc = (nll / counts).mean(), nll.new_zeros(())
                        loss = mle + args.discriminative_weight * disc
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError(f"Nonfinite loss at next step {step + 1}")
                    weight = len(micro) / len(ids)
                    (loss * weight).backward()
                    for key, value in zip(totals, (loss, mle, disc)):
                        totals[key] += float(value.detach()) * weight
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                if step % args.log_every == 0 or step == 1 or step == args.steps:
                    log("loss", step=step, discriminative_active=active,
                        **{**totals, "discriminative_bce": totals["discriminative_bce"] if active else None},
                        grad_norm=float(grad_norm), elapsed_seconds=time.monotonic() - started, clipping=clips.report())
                if step % args.save_every == 0:
                    checkpoint = save_checkpoint(root, model, tokenizer, optimizer, sampler, step, run, positions, pair_rows)
                    last_saved = step
            if last_saved != step:
                checkpoint = save_checkpoint(root, model, tokenizer, optimizer, sampler, step, run, positions, pair_rows)
        summary = {"state": "complete", "reason": reason, "step": step, "requested_steps": args.steps,
                   "checkpoint": str(checkpoint), "elapsed_seconds": time.monotonic() - started,
                   "clipping_this_invocation": clips.report()}
        atomic_json(root / "run_complete.json", summary, replace=True)
        log("train_complete", **summary)
        return summary


def score(args) -> dict:
    started = time.monotonic()
    source, output = Path(args.pairs), Path(args.output)
    marker = output.with_name(output.name + ".complete.json")
    if output.exists() or marker.exists():
        raise FileExistsError(f"Score output already exists: {output}")
    device, amp = runtime_setup(args)
    checkpoint = checkpoint_path(Path(args.model))
    complete = verify_checkpoint(checkpoint)
    run = read_json(checkpoint / "run.json")
    recipe = run["recipe"]
    max_input = args.max_input_length or recipe["max_input_length"]
    max_target = args.max_target_length or recipe["max_target_length"]
    model, tokenizer = load_model(checkpoint, device, local_files_only=True)
    model.eval()
    pf = pq.ParquetFile(source)
    missing = set(PAIR_COLUMNS) - set(pf.schema_arrow.names)
    if missing:
        raise ValueError(f"Missing pair columns: {sorted(missing)}")
    if pf.schema_arrow.field("pair_row").type != pa.int64():
        raise ValueError("pair_row must be int64")
    # Output IDs retain their Arrow types; no label/true_count/tree/blend column is read.
    id_schema = pa.schema([pf.schema_arrow.field(k) for k in PAIR_COLUMNS[:3]])
    schema = pa.schema([*id_schema, *[pa.field(k, pa.int64() if k == "token_count" else pa.float32())
                                    for k in SCORE_NAMES]])
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    os.close(fd)
    temporary = Path(name)
    clips, seen_rows, seen_pairs, total = ClipStats(), set(), set(), 0
    try:
        with pq.ParquetWriter(temporary, schema, compression="zstd") as writer, torch.inference_mode():
            for raw in pf.iter_batches(batch_size=args.chunk_size, columns=list(PAIR_COLUMNS)):
                table = pa.Table.from_batches([raw])
                validate_table(table)
                rows = table.to_pylist()
                unique_pairs(rows, seen_rows, seen_pairs)
                order = np.argsort(lengths_for(rows, max_input, max_target), kind="stable")
                values = {k: np.empty(len(rows), dtype=np.int64 if k == "token_count" else np.float32) for k in SCORE_NAMES}
                for start in range(0, len(rows), args.batch_size):
                    ids = order[start:start + args.batch_size]
                    batch, stat = collate_pairs(tokenizer, [rows[i] for i in ids], max_input_length=max_input,
                                               max_target_length=max_target, pad_to_multiple_of=args.pad_to_multiple_of)
                    clips.add(stat)
                    with autocast(device, amp):
                        nll, counts = teacher_forced_nll(model, {k: v.to(device) for k, v in batch.items()})
                    n = len(ids)
                    copy, nonmatch, count = nll[:n], nll[n:], counts[:n]
                    ratio = nonmatch - copy
                    for key, value in zip(SCORE_NAMES, (copy, nonmatch, count, ratio, ratio / count)):
                        if not bool(torch.isfinite(value).all()):
                            raise FloatingPointError("Nonfinite score")
                        values[key][ids] = value.cpu().numpy()
                result = pa.Table.from_arrays([table[k] for k in PAIR_COLUMNS[:3]] +
                                             [pa.array(values[k]) for k in SCORE_NAMES], schema=schema)
                writer.write_table(result)
                total += len(rows)
                log("score_progress", rows=total, clipping=clips.report(), elapsed_seconds=time.monotonic() - started)
        if total != pf.metadata.num_rows:
            raise ValueError("Scoring did not preserve the full input row count")
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        summary = {"format": FORMAT, "state": "complete", "rows": total,
                   "input": {"path": str(source.resolve()), "sha256": sha256(source)},
                   "checkpoint": str(checkpoint.resolve()), "checkpoint_manifest_sha256": sha256(checkpoint / "complete.json"),
                   "training_run_sha256": complete["run_sha256"], "output_sha256": sha256(temporary),
                   "code_sha256": sha256(Path(__file__)), "amp": amp, "device": str(device),
                   "runtime": runtime_identity(), "deterministic": args.deterministic,
                   "threads": args.threads, "chunk_size": args.chunk_size,
                   "batch_size": args.batch_size, "max_input_length": max_input, "max_target_length": max_target,
                   "pad_to_multiple_of": args.pad_to_multiple_of, "clipping": clips.report(),
                   "nll": "sum of negative natural log probabilities; includes EOS; excludes masked padding",
                   "log_ratio": "nonmatch_nll - copy_nll", "log_ratio_normalized": "log_ratio / token_count",
                   "labels_used": False, "elapsed_seconds": time.monotonic() - started}
        os.link(temporary, output)
        atomic_json(marker, summary)
    finally:
        temporary.unlink(missing_ok=True)
    log("score_complete", **summary)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subs = p.add_subparsers(dest="command", required=True)
    for command in ("train", "score"):
        q = subs.add_parser(command)
        q.add_argument("--model", required=True)
        q.add_argument("--output", required=True)
        q.add_argument("--batch-size", type=int, default=64 if command == "train" else 16)
        q.add_argument("--seed", type=int, default=27)
        q.add_argument("--max-input-length", type=int, default=384 if command == "train" else None)
        q.add_argument("--max-target-length", type=int, default=384 if command == "train" else None)
        q.add_argument("--pad-to-multiple-of", type=int, default=8)
        q.add_argument("--device", default="auto")
        q.add_argument("--amp", choices=("auto", "fp32", "bf16"), default="auto")
        q.add_argument("--threads", type=int, default=4)
        q.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
        if command == "score":
            q.add_argument("--pairs", required=True)
            q.add_argument("--chunk-size", type=int, default=4096)
            continue
        q.add_argument("--train", required=True)
        q.add_argument("--steps", required=True, type=int)
        q.add_argument("--microbatch", type=int, default=8)
        q.add_argument("--max-pairs", type=int, default=500_000)
        q.add_argument("--lr", type=float, default=1e-4)
        q.add_argument("--weight-decay", type=float, default=0.01)
        q.add_argument("--grad-clip", type=float, default=1.0)
        q.add_argument("--mle-warmup-steps", type=int, default=50)
        q.add_argument("--discriminative-weight", type=float, default=0.1, help="0 disables the BCE objective")
        q.add_argument("--discriminative-temperature", type=float, default=0.1)
        q.add_argument("--discriminative-normalization", choices=("mean", "sum"), default="mean")
        q.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
        q.add_argument("--bucket-multiplier", type=int, default=32)
        q.add_argument("--save-every", type=int, default=500)
        q.add_argument("--log-every", type=int, default=10)
        q.add_argument("--max-minutes", type=float)
        q.add_argument("--resume", nargs="?", const="latest", help="latest or an explicit latest checkpoint of --output")
        q.add_argument("--local-files-only", action="store_true")
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    positive = ("batch_size", "threads", "pad_to_multiple_of", "max_input_length", "max_target_length",
                "steps", "microbatch", "max_pairs", "lr", "grad_clip", "discriminative_temperature",
                "save_every", "log_every", "max_minutes", "chunk_size", "bucket_multiplier")
    for name in positive:
        value = getattr(args, name, None)
        if value is not None and (not math.isfinite(value) or value <= 0):
            p.error(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("weight_decay", "mle_warmup_steps", "discriminative_weight", "seed"):
        value = getattr(args, name, 0)
        if not math.isfinite(value) or value < 0:
            p.error(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.max_input_length is not None and args.max_input_length < 14:
        p.error("--max-input-length must fit NONMATCH S2: plus EOS (at least 14)")
    return train(args) if args.command == "train" else score(args)


if __name__ == "__main__":
    main()
