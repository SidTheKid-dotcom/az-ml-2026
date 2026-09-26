"""Fast, deterministic C2 arena reranker training.

The lane consumes the label-free arena candidates plus separately aligned TRAIN
labels. It never accepts HOLDOUT-2 truth. Sampling happens on the main thread;
the bounded worker queue only tokenizes immutable batches, so discarded prefetched
batches cannot advance RNG state across resume.
"""
from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import os
from dataclasses import asdict, dataclass, replace
from enum import IntFlag
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .cross_encoder_experiment import _amp_context, load_reranker
from .experiment import json_write
from .rerank_inference import minibatch_order
from .text_views import canonicalize, record_texts
from .v2_ce_train import PositiveCycle, _pair_order_sha256, file_sha256, sealed_ids


MODEL_ID = "BAAI/bge-reranker-v2-m3"
DEFAULT_SEED = 20260925
_DIGITS = re.compile(r"\d+")


class NegativeType(IntFlag):
    FALSE_ACCEPT = 1
    CONTEST_LOSER = 2
    UNOWNED_LOOKALIKE = 4
    DIGIT_SUBSTITUTION = 8
    EMPTY_ADDRESS_SIBLING = 16


@dataclass(frozen=True)
class ModeConfig:
    mode: str
    steps: int
    batch_size: int
    positive_fraction: float
    canonical_text: bool
    field_tagged: bool
    schedule: str
    init: str


def mode_config(mode: str, *, batch_size: int | None = None,
                steps: int | None = None) -> ModeConfig:
    """Return locked comparison semantics for C2, G0, or G0b."""
    if mode == "g0b":
        if batch_size not in (None, 32) or steps not in (None, 1800):
            raise ValueError("G0b is fixed at 1,800 steps and batch 32")
        return ModeConfig(mode, 1800, 32, .5, False, False, "legacy_linear", "pretrained")
    if mode == "g0":
        size = 64 if batch_size is None else batch_size
        if size not in (32, 64, 128) or steps not in (None, 300):
            raise ValueError("G0 uses 300 steps and batch 32, 64, or 128")
        return ModeConfig(mode, 300, size, .25, True, True, "constant", "checkpoint")
    if mode == "c2":
        size = 64 if batch_size is None else batch_size
        count = 15_000 if steps is None else steps
        if size < 4 or size % 4 or count < 1:
            raise ValueError("C2 batch size must be a positive multiple of four")
        return ModeConfig(mode, count, size, .25, True, True, "constant", "checkpoint")
    raise ValueError("Mode must be c2, g0, or g0b")


def g0_matrix() -> list[dict]:
    """The prescribed throughput screen, represented as independently runnable jobs."""
    variants = (("fused", True, False), ("frozen", True, True))
    return [{"variant": name, "batch_size": size, "fused": fused,
             "freeze_embeddings": frozen, "steps": 300}
            for size in (32, 64, 128) for name, fused, frozen in variants]


def model_identity(path: Path, allowlist: Path, model_id: str = MODEL_ID) -> dict:
    """Hash every initial-model file and bind it to an allowed reranker license."""
    import yaml

    path, allowlist = Path(path).resolve(), Path(allowlist).resolve()
    if not path.is_dir():
        raise ValueError("Initial model must be a complete local directory")
    document = yaml.safe_load(allowlist.read_text(encoding="utf-8"))
    entries = {row["id"]: row for row in document["models"]}
    row = entries.get(model_id)
    if (row is None or row.get("role") != "reranker" or row.get("licence") not in {"mit", "apache-2.0"}
            or float(row["params_b"]) > float(document.get("budget_total_params_b", 8.0))):
        raise ValueError("Initial model is absent from the allowed MIT/Apache reranker inventory")
    files = {str(file.relative_to(path)): file_sha256(file)
             for file in sorted(path.rglob("*")) if file.is_file()}
    if "config.json" not in files or not any(name.endswith((".safetensors", ".bin")) for name in files):
        raise ValueError("Initial model directory lacks config or weight files")
    return {"model_id": model_id, "path": str(path), "licence": row["licence"],
            "params_b": float(row["params_b"]), "allowlist_sha256": file_sha256(allowlist),
            "files_sha256": files}


def stake_weights(labels, true_counts, *, cap: float = 5.0,
                  normalize: bool = True) -> np.ndarray:
    """C2 pair stakes, normalized to mean one within each represented class.

    Normalization precedes the explicit 5x cap. ``true_counts`` must be the
    complete per-business count, not the number of retrieved positive pairs.
    """
    labels = np.asarray(labels, dtype=bool)
    counts = np.asarray(true_counts)
    if labels.shape != counts.shape or labels.ndim != 1 or np.any(counts < 0) or cap <= 0:
        raise ValueError("Stake labels/counts must be aligned, one-dimensional, and nonnegative")
    if np.any(labels & (counts < 1)):
        raise ValueError("A positive pair cannot belong to a no-match business")
    values = np.ones(len(labels), dtype=np.float32)
    positive = labels
    matched_negative = ~labels & (counts >= 1)
    values[positive] = 1.0 / (5.0 * counts[positive] - 4.0)
    values[matched_negative] = 4.0 / (5.0 * counts[matched_negative] + 4.0)
    if normalize:
        for mask in (positive, ~positive):
            if mask.any():
                values[mask] /= float(values[mask].mean())
    return np.minimum(values, cap).astype(np.float32)


def _same_length_digit_substitution(left: str, right: str) -> bool:
    a, b = _DIGITS.findall(left), _DIGITS.findall(right)
    return bool(a and len(a) == len(b) and a != b and all(len(x) == len(y) for x, y in zip(a, b)))


def classify_negative_types(reference: pd.DataFrame, pool: pd.DataFrame,
                            left_idx, right_idx, labels, tree_scores, blend_scores,
                            record_owner, *, accept_threshold: float = .775,
                            lookalike_threshold: float = .3,
                            tree_gate: float = .01) -> np.ndarray:
    """Assign overlapping typed-negative flags in exact candidate row order."""
    left = np.asarray(left_idx, dtype=np.int64)
    right = np.asarray(right_idx, dtype=np.int64)
    label = np.asarray(labels, dtype=bool)
    tree = np.asarray(tree_scores, dtype=np.float32)
    blend = np.asarray(blend_scores, dtype=np.float32)
    owner = np.asarray(record_owner, dtype=np.int64)
    n = len(left)
    if any(array.shape != (n,) for array in (right, label, tree, blend)):
        raise ValueError("Arena pair arrays are not aligned")
    if (n and (left.min() < 0 or left.max() >= len(reference) or right.min() < 0 or right.max() >= len(pool))
            or owner.shape != (len(pool),)):
        raise ValueError("Arena indices or record-owner array are invalid")
    flags = np.zeros(n, dtype=np.uint8)
    negative = ~label
    flags[negative & (blend >= accept_threshold)] |= np.uint8(NegativeType.FALSE_ACCEPT)
    flags[negative & (blend >= lookalike_threshold) & (owner[right] >= 0) & (owner[right] != left)] |= np.uint8(NegativeType.CONTEST_LOSER)
    flags[negative & (blend >= lookalike_threshold) & (owner[right] < 0)] |= np.uint8(NegativeType.UNOWNED_LOOKALIKE)
    names_left = reference.business_name.fillna("").astype(str).map(canonicalize).to_numpy()
    names_right = pool.business_name.fillna("").astype(str).map(canonicalize).to_numpy()
    addr_left = reference.business_address.fillna("").astype(str).to_numpy()
    addr_right = pool.business_address.fillna("").astype(str).to_numpy()
    # Restrict expensive text checks to the tree-gated training band while
    # retaining overlapping flags for auditing and balanced typed sampling.
    text_candidates = np.flatnonzero(negative & (tree >= tree_gate))
    for position in text_candidates:
        li, ri = left[position], right[position]
        same_name = bool(names_left[li]) and names_left[li] == names_right[ri]
        if same_name and _same_length_digit_substitution(addr_left[li], addr_right[ri]):
            flags[position] |= np.uint8(NegativeType.DIGIT_SUBSTITUTION)
        if same_name and not canonicalize(addr_right[ri]):
            flags[position] |= np.uint8(NegativeType.EMPTY_ADDRESS_SIBLING)
    return flags


class ArenaBatchSampler:
    """Exactly resumable 1:3 sampler with a 50/50 typed/random negative split."""

    def __init__(self, positive, negative_flags, gated_negative, *, batch_size: int = 64,
                 positive_fraction: float = .25, seed: int = DEFAULT_SEED,
                 state: dict | None = None):
        self.positive = np.asarray(positive, dtype=np.int64)
        self.flags = np.asarray(negative_flags, dtype=np.uint8)
        self.gated = np.asarray(gated_negative, dtype=np.int64)
        if batch_size < 2 or positive_fraction not in (.25, .5) or batch_size * positive_fraction % 1:
            raise ValueError("Batch must support the requested 1:3 or 1:1 class ratio")
        self.positive_count = int(batch_size * positive_fraction)
        self.negative_count = batch_size - self.positive_count
        if self.negative_count % 2:
            raise ValueError("Negative count must split equally into typed and random halves")
        self.typed_count = self.negative_count // 2
        self.random_count = self.negative_count - self.typed_count
        if not len(self.positive) or not len(self.gated):
            raise ValueError("C2 needs positive and gated-negative pools")
        self.typed = {kind: np.flatnonzero(self.flags & int(kind)) for kind in NegativeType}
        self.typed_all = np.flatnonzero(self.flags != 0)
        if not len(self.typed_all):
            raise ValueError("C2 needs at least one typed arena negative")
        self.cycle = PositiveCycle(len(self.positive), seed)
        self.rng = np.random.default_rng(seed + 1)
        self.step = 0
        if state is not None:
            self._restore(state, seed)

    def _restore(self, state: dict, seed: int) -> None:
        if state.get("kind") != "arena_1_to_3" or state.get("seed") != seed:
            raise ValueError("Sampler checkpoint seed differs")
        if (state.get("positive_count") != self.positive_count
                or state.get("negative_count") != self.negative_count):
            raise ValueError("Sampler checkpoint batch composition differs")
        self.cycle = PositiveCycle(**state["positive_cycle"])
        self.rng.bit_generator.state = state["rng_state"]
        self.step = int(state["step"])

    def state(self) -> dict:
        return {"kind": "arena_1_to_3", "seed": self.cycle.seed, "step": self.step,
                "positive_count": self.positive_count, "negative_count": self.negative_count,
                "positive_cycle": self.cycle.state(), "rng_state": self.rng.bit_generator.state}

    def _distinct(self, pool: np.ndarray, count: int, excluded=()) -> np.ndarray:
        """Draw a tiny unique sample without copying/scanning an arena-scale pool."""
        blocked = {int(value) for value in excluded}
        if len(pool) <= 10_000:
            available = np.asarray([value for value in pool if int(value) not in blocked], dtype=np.int64)
            if len(available) < count:
                raise ValueError("Negative pool is too small for a duplicate-free batch")
            return self.rng.choice(available, count, replace=False)
        chosen: list[int] = []
        attempts = 0
        limit = max(100, count * 100)
        while len(chosen) < count and attempts < limit:
            candidate = int(pool[int(self.rng.integers(len(pool)))])
            if candidate not in blocked:
                chosen.append(candidate)
                blocked.add(candidate)
            attempts += 1
        if len(chosen) != count:
            raise ValueError("Negative pool is too small for a duplicate-free batch")
        return np.asarray(chosen, dtype=np.int64)

    def _typed_sample(self) -> np.ndarray:
        available = [kind for kind, values in self.typed.items() if len(values)]
        chosen: list[int] = []
        start = self.step % len(available)
        attempts = 0
        while len(chosen) < self.typed_count and attempts < self.typed_count * len(available) * 8:
            kind = available[(start + attempts) % len(available)]
            candidate = int(self.rng.choice(self.typed[kind]))
            if candidate not in chosen:
                chosen.append(candidate)
            attempts += 1
        if len(chosen) < self.typed_count:
            need = self.typed_count - len(chosen)
            chosen.extend(self._distinct(self.typed_all, need, chosen).tolist())
        return np.asarray(chosen, dtype=np.int64)

    def take(self) -> tuple[np.ndarray, np.ndarray]:
        positives = self.positive[self.cycle.take(self.positive_count)]
        typed = self._typed_sample()
        random_ids = self._distinct(self.gated, self.random_count, typed)
        pair_ids = np.concatenate((positives, typed, random_ids))
        labels = np.r_[np.ones(self.positive_count, dtype=np.float32),
                       np.zeros(self.negative_count, dtype=np.float32)]
        order = self.rng.permutation(len(pair_ids))
        self.step += 1
        return pair_ids[order], labels[order]


class LegacyBatchSampler:
    """Historical G0b 1:1 shuffled class cycles, without C2 negative typing."""

    def __init__(self, positive, negative, *, batch_size: int = 32,
                 seed: int = DEFAULT_SEED, state: dict | None = None):
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Legacy parity batch size must be even")
        self.positive = np.asarray(positive, dtype=np.int64)
        self.negative = np.asarray(negative, dtype=np.int64)
        if not len(self.positive) or not len(self.negative):
            raise ValueError("Legacy parity needs both classes")
        self.rng = np.random.default_rng(seed)
        self.positive_order = self.rng.permutation(len(self.positive))
        self.negative_order = self.rng.permutation(len(self.negative))
        self.positive_offset = self.negative_offset = 0
        self.batch_size, self.seed, self.step = batch_size, seed, 0
        if state is not None:
            if (state.get("kind") != "legacy_1_to_1" or state.get("seed") != seed
                    or state.get("batch_size") != batch_size):
                raise ValueError("Legacy sampler checkpoint differs")
            target = int(state["step"])
            for _ in range(target):
                self.take()

    def _take(self, values: np.ndarray, order_name: str, offset_name: str, count: int) -> np.ndarray:
        chunks = []
        while count:
            order = getattr(self, order_name)
            offset = getattr(self, offset_name)
            if offset == len(order):
                order = self.rng.permutation(len(values))
                offset = 0
                setattr(self, order_name, order)
            size = min(count, len(order) - offset)
            chunks.append(values[order[offset:offset + size]])
            setattr(self, offset_name, offset + size)
            count -= size
        return np.concatenate(chunks)

    def state(self) -> dict:
        return {"kind": "legacy_1_to_1", "seed": self.seed, "step": self.step,
                "batch_size": self.batch_size}

    def take(self) -> tuple[np.ndarray, np.ndarray]:
        half = self.batch_size // 2
        pair_ids = np.r_[self._take(self.positive, "positive_order", "positive_offset", half),
                         self._take(self.negative, "negative_order", "negative_offset", half)]
        labels = np.r_[np.ones(half, dtype=np.float32), np.zeros(half, dtype=np.float32)]
        order = self.rng.permutation(self.batch_size)
        self.step += 1
        return pair_ids[order], labels[order]


@dataclass
class TokenBatch:
    pair_ids: np.ndarray
    labels: np.ndarray
    weights: np.ndarray
    microbatches: list[dict]
    microbatch_positions: list[np.ndarray]
    sampler_state: dict | None = None


def tokenize_batch(tokenizer, left_text: list[str], right_text: list[str], pair_ids,
                   labels, weights, left_idx, right_idx, *, max_length: int,
                   microbatch: int, pin_memory: bool = False,
                   sampler_state: dict | None = None,
                   length_buckets: tuple[int, ...] | None = None) -> TokenBatch:
    """Tokenize once without padding, then stable-bucket and pad each microbatch.

    With ``length_buckets`` every microbatch is padded to the smallest listed
    width that holds it, so ``torch.compile`` sees a few static shapes.
    """
    ids = np.asarray(pair_ids, dtype=np.int64)
    left = np.asarray(left_idx, dtype=np.int64)
    right = np.asarray(right_idx, dtype=np.int64)
    encoded = tokenizer([left_text[left[i]] for i in ids], [right_text[right[i]] for i in ids],
                        padding=False, truncation=True, max_length=max_length,
                        return_attention_mask=True)
    columns = [name for name in tokenizer.model_input_names if name in encoded]
    lengths = np.fromiter((len(row) for row in encoded["input_ids"]), dtype=np.int32, count=len(ids))
    batches, positions_out = [], []
    for positions in minibatch_order(lengths, microbatch, True):
        rows = [{name: encoded[name][int(i)] for name in columns} for i in positions]
        if length_buckets:
            longest = int(lengths[positions].max())
            width = next(size for size in length_buckets if size >= longest)
            tokens = tokenizer.pad(rows, padding="max_length", max_length=width, return_tensors="pt")
        else:
            tokens = tokenizer.pad(rows, padding=True, pad_to_multiple_of=8, return_tensors="pt")
        if pin_memory:
            tokens = {name: value.pin_memory() for name, value in tokens.items()}
        batches.append(tokens)
        positions_out.append(positions)
    return TokenBatch(ids, np.asarray(labels, dtype=np.float32), np.asarray(weights, dtype=np.float32),
                      batches, positions_out, sampler_state)


class TokenizationPrefetcher:
    """Bounded single-worker tokenization queue; input sampling stays synchronous."""

    def __init__(self, batches: Iterable[tuple], tokenize, *, depth: int = 4):
        if depth < 1:
            raise ValueError("Prefetch depth must be positive")
        self.batches, self.tokenize, self.depth = iter(batches), tokenize, depth

    def __iter__(self) -> Iterator[TokenBatch]:
        workers = int(os.environ.get("BER_TOKENIZER_WORKERS", "1"))
        if workers < 1:
            raise ValueError("BER_TOKENIZER_WORKERS must be positive")
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="c2-tokenizer") as executor:
            pending: list[Future] = []
            for _ in range(self.depth):
                try:
                    pending.append(executor.submit(self.tokenize, *next(self.batches)))
                except StopIteration:
                    break
            while pending:
                future = pending.pop(0)
                yield future.result()
                try:
                    pending.append(executor.submit(self.tokenize, *next(self.batches)))
                except StopIteration:
                    pass


def freeze_word_embeddings(model, freeze: bool) -> list[str]:
    """Freeze only the input word-embedding weight, leaving positional layers trainable."""
    embedding = model.get_input_embeddings()
    if embedding is None or not hasattr(embedding, "weight"):
        raise ValueError("Model does not expose input word embeddings")
    names = [name for name, parameter in model.named_parameters() if parameter is embedding.weight]
    if len(names) != 1:
        raise ValueError("Input embedding weight must map to exactly one named parameter")
    embedding.weight.requires_grad_(not freeze)
    return names if freeze else []


def build_adamw(torch, model, *, learning_rate: float, weight_decay: float,
                fused: bool, device: str):
    """Create AdamW over trainable parameters and verify requested fused support."""
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Optimizer has no trainable parameters")
    supports = "fused" in inspect.signature(torch.optim.AdamW).parameters
    if fused and (not supports or not device.startswith("cuda")):
        raise ValueError("Fused AdamW was requested but is unsupported on this device/runtime")
    options = {"lr": learning_rate, "weight_decay": weight_decay}
    if supports:
        options["fused"] = fused
    return torch.optim.AdamW(parameters, **options)


def learning_rate(step: int, *, base: float, warmup_steps: int,
                  total_steps: int, schedule: str) -> float:
    if not 1 <= step <= total_steps or warmup_steps < 0:
        raise ValueError("Learning-rate step or schedule bounds are invalid")
    if step <= warmup_steps:
        return base * step / max(1, warmup_steps)
    if schedule == "constant":
        return base
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    if schedule == "cosine":
        return base * .5 * (1 + math.cos(math.pi * progress))
    if schedule == "legacy_linear":
        return base * (1 - progress)
    raise ValueError(f"Unknown learning-rate schedule: {schedule}")


_DEFERRED_HASH_MIN_BYTES = 64 * 1024 * 1024


def _checkpoint_marker(directory: Path) -> dict:
    marker = json.loads((directory / "complete.json").read_text(encoding="utf-8"))
    files = marker.get("files_sha256", {})
    sizes = marker.get("files_bytes", {})
    actual = {path.name for path in directory.iterdir() if path.name != "complete.json"}
    if (marker.get("state") != "complete" or set(files) & set(sizes) or set(files) | set(sizes) != actual
            or any(file_sha256(directory / name) != digest for name, digest in files.items())
            or any((directory / name).stat().st_size != size for name, size in sizes.items())):
        raise ValueError("Checkpoint is incomplete or changed")
    return marker


def save_checkpoint(model, tokenizer, optimizer, sampler,
                    *, step: int, config: dict, config_sha256: str, root: Path, torch,
                    defer_hash: bool = False) -> Path:
    """Atomically commit model, optimizer, and all RNG/sampling state.

    ``defer_hash`` records large weight/optimizer files by byte size only; a
    network volume otherwise stalls training for minutes re-reading them.
    """
    destination = Path(root) / f"step_{step:06d}"
    stage = Path(root) / f".step_{step:06d}.incomplete"
    Path(root).mkdir(parents=True, exist_ok=True)
    if destination.exists() or stage.exists():
        raise FileExistsError(f"Checkpoint path already exists: {destination}")
    stage.mkdir()
    model.save_pretrained(stage, safe_serialization=True)
    tokenizer.save_pretrained(stage)
    json_write(stage / "training_config.json", config)
    torch.save({"optimizer": optimizer.state_dict(), "torch_cpu_rng": torch.get_rng_state(),
                "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
               stage / "optimizer.pt")
    sampler_state = sampler.state() if hasattr(sampler, "state") else sampler
    if sampler_state.get("step") != step:
        raise ValueError("Checkpoint sampler state must end at the committed optimizer step")
    json_write(stage / "state.json", {"step": step, "config_sha256": config_sha256,
                                       "sampler": sampler_state})
    written = [path for path in stage.iterdir() if path.is_file()]
    deferred = {path.name: path.stat().st_size for path in written
                if defer_hash and path.stat().st_size >= _DEFERRED_HASH_MIN_BYTES}
    files = {path.name: file_sha256(path) for path in written if path.name not in deferred}
    marker = {"state": "complete", "step": step, "config_sha256": config_sha256, "files_sha256": files}
    if deferred:
        marker["files_bytes"] = deferred
    json_write(stage / "complete.json", marker)
    stage.replace(destination)
    return destination


def restore_checkpoint(directory: Path, model, optimizer, *, config_sha256: str,
                       positive, negative_flags, gated_negative, batch_size: int,
                       positive_fraction: float = .25, seed: int, torch):
    marker = _checkpoint_marker(Path(directory))
    if marker["config_sha256"] != config_sha256:
        raise ValueError("Checkpoint belongs to a different trainer configuration")
    saved = torch.load(Path(directory) / "optimizer.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(saved["optimizer"])
    torch.set_rng_state(saved["torch_cpu_rng"])
    if saved["torch_cuda_rng"] is not None:
        torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
    state = json.loads((Path(directory) / "state.json").read_text(encoding="utf-8"))
    if state["sampler"].get("kind") == "legacy_1_to_1":
        sampler = LegacyBatchSampler(positive, gated_negative, batch_size=batch_size,
                                     seed=seed, state=state["sampler"])
    else:
        sampler = ArenaBatchSampler(positive, negative_flags, gated_negative,
                                    batch_size=batch_size, positive_fraction=positive_fraction,
                                    seed=seed, state=state["sampler"])
    if sampler.step != marker["step"]:
        raise ValueError("Checkpoint sampler and optimizer steps differ")
    return marker["step"], sampler


@dataclass
class ArenaData:
    reference: pd.DataFrame
    pool: pd.DataFrame
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    true_count: np.ndarray
    record_owner: np.ndarray
    tree: np.ndarray
    blend: np.ndarray
    partition: np.ndarray
    identity: dict


@dataclass
class ScoreData:
    reference: pd.DataFrame
    pool: pd.DataFrame
    pairs: pd.DataFrame
    partition: np.ndarray
    identity: dict


def _partition_values(path: Path | None, count: int) -> np.ndarray:
    partition = (np.load(path, mmap_mode="r") if path else np.full(count, "A", dtype="U1"))
    if partition.dtype.kind == "b":
        partition = np.where(partition, "A", "B")
    elif partition.dtype.kind in "iu":
        # finalize_v1 stores 0=A, 1=B_REST, 2=SLICE (arena_build.PARTITION_NAMES).
        # A file that also uses code 3 keeps the older 0=A, 1=B, 2=SLICE, 3=B_REST order.
        if len(partition) and (int(partition.min()) < 0 or int(partition.max()) > 3):
            raise ValueError("Integer arena partition codes must be in 0..3")
        names = (np.asarray(["A", "B_REST", "SLICE"])
                 if not len(partition) or int(partition.max()) <= 2
                 else np.asarray(["A", "B", "SLICE", "B_REST"]))
        partition = names[partition]
    elif partition.dtype.kind == "S":
        partition = partition.astype("U")
    if partition.shape != (count,) or not np.isin(partition, ["A", "B", "SLICE", "B_REST"]).all():
        raise ValueError("Arena partition must assign every S1 to A/B/SLICE/B_REST")
    return partition


def _validate_pair_order(left: np.ndarray, right: np.ndarray, manifest: dict) -> None:
    """Validate C2's sorted order or G0b's bounded grouped historical order."""
    semantics = manifest.get("pair_order", "left_then_right")
    if len(left) > 1 and np.any(left[1:] < left[:-1]):
        raise ValueError("Candidate pairs must be nondecreasing by left_idx")
    if semantics == "left_then_right":
        if len(left) > 1 and np.any((left[1:] == left[:-1]) & (right[1:] <= right[:-1])):
            raise ValueError("Real C2 arena pairs must be strictly sorted by left_idx/right_idx")
        return
    if semantics != "left_grouped_original_right_order":
        raise ValueError(f"Unsupported pair-order semantics: {semantics!r}")
    maximum = manifest.get("max_pairs_per_reference")
    if type(maximum) is not int or not 1 <= maximum <= 10_000:
        raise ValueError("Grouped historical pair order needs a bounded max_pairs_per_reference")
    starts = np.r_[0, np.flatnonzero(left[1:] != left[:-1]) + 1, len(left)]
    for start, stop in zip(starts[:-1], starts[1:]):
        if stop - start > maximum:
            raise ValueError("Historical candidate group exceeds its declared bound")
        group = right[start:stop]
        if len(np.unique(group)) != len(group):
            raise ValueError("Duplicate candidate pair in historical left_idx group")


def load_score_data(arena: Path, *, partition_file: Path | None = None,
                    sealed_reservation: Path | None = None, load_text: bool = True) -> ScoreData:
    """Verify label-free arena inputs and preserve their immutable pair order."""
    arena = Path(arena)
    manifest_path = arena / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("holdout2_excluded") is not True or manifest.get("labels_read") is not False:
        raise ValueError("Scoring arena must be label-free and exclude HOLDOUT-2")
    hashes = manifest.get("sha256", {})
    required_files = {"reference.parquet", "pool.parquet", "pairs.parquet"}
    if not required_files <= set(hashes):
        raise ValueError("Arena manifest must hash every record and pair input")
    for name, expected in hashes.items():
        path = arena / name
        if not path.is_file() or file_sha256(path) != expected:
            raise ValueError(f"Arena manifest input changed: {name}")
    reference_path, pool_path, pair_path = (arena / name for name in
                                            ("reference.parquet", "pool.parquet", "pairs.parquet"))
    reference_columns = None if load_text else ["entity_id", "country"]
    pool_columns = None if load_text else ["entity_id"]
    reference = pd.read_parquet(reference_path, columns=reference_columns)
    pool = pd.read_parquet(pool_path, columns=pool_columns)
    pairs = pd.read_parquet(pair_path, columns=["left_idx", "right_idx"])
    left, right = (pairs[name].to_numpy(dtype=np.int64) for name in ("left_idx", "right_idx"))
    _validate_pair_order(left, right, manifest)
    if (len(reference) != manifest.get("references") or len(pool) != manifest.get("pool_records")
            or len(pairs) != manifest.get("pairs")
            or (len(pairs) and (left.min() < 0 or left.max() >= len(reference)
                                or right.min() < 0 or right.max() >= len(pool)))):
        raise ValueError("Arena records, pair order, or manifest counts differ")
    partition = _partition_values(partition_file, len(reference))
    sealed_sha = None
    if sealed_reservation is not None:
        forbidden, sealed_sha = sealed_ids(sealed_reservation)
        if set(reference.entity_id) & forbidden:
            raise ValueError("Arena reference contains a sealed TRAIN/TUNE/HOLDOUT-2 ID")
    identity = {"manifest_sha256": file_sha256(manifest_path),
                "reference_sha256": file_sha256(reference_path),
                "pool_sha256": file_sha256(pool_path), "pairs_sha256": file_sha256(pair_path),
                "pair_order_sha256": _pair_order_sha256(pair_path),
                "partition_sha256": file_sha256(partition_file) if partition_file else None,
                "sealed_reservation_sha256": sealed_sha,
                "references": len(reference), "pool_records": len(pool), "pairs": len(pairs)}
    return ScoreData(reference, pool, pairs, partition, identity)


def load_arena(arena: Path, labels: Path, *, partition_file: Path | None = None,
               sealed_reservation: Path | None = None, load_text: bool = True) -> ArenaData:
    """Load and strictly align the C2 arena inputs without any holdout-label path."""
    arena, labels = Path(arena), Path(labels)
    scored = load_score_data(arena, partition_file=partition_file,
                             sealed_reservation=sealed_reservation, load_text=load_text)
    reference, pool, pairs = scored.reference, scored.pool, scored.pairs
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    summary = json.loads((labels / "labels_summary.json").read_text(encoding="utf-8"))
    if summary.get("holdout2_rows_read") != 0:
        raise ValueError("Arena labels include HOLDOUT-2 rows")
    def aligned(path: Path, shape: tuple[int, ...], dtype_kind: str | None = None):
        values = np.load(path, mmap_mode="r")
        if values.shape != shape or (dtype_kind and values.dtype.kind not in dtype_kind):
            raise ValueError(f"Misaligned arena array: {path}")
        return values
    label = aligned(labels / "label.npy", (len(left),), "b")
    true_count = aligned(labels / "true_count.npy", (len(reference),), "iu")
    owner = aligned(labels / "rec_owner.npy", (len(pool),), "i")
    tree_path = arena / "v1_tree.npy" if (arena / "v1_tree.npy").exists() else arena / "base_probabilities.npy"
    blend_path = arena / "v1_blend.npy" if (arena / "v1_blend.npy").exists() else arena / "probabilities.npy"
    tree = aligned(tree_path, (len(left),), "f")
    blend = aligned(blend_path, (len(left),), "f")
    retrieved = np.bincount(left[label], minlength=len(reference))
    if (np.any(retrieved > true_count) or np.any(owner < -1) or np.any(owner >= len(reference))
            or not np.array_equal(label, owner[right] == left)
            or ("true_links" in summary and int(true_count.sum()) != summary["true_links"])
            or ("positive_pairs" in summary and int(label.sum()) != summary["positive_pairs"])
            or ("singletons" in summary and int((true_count == 0).sum()) != summary["singletons"])):
        raise ValueError("Arena labels, complete true counts, owners, or summary disagree")
    partition = scored.partition
    identity = {**scored.identity,
                "labels_summary_sha256": file_sha256(labels / "labels_summary.json"),
                "label_sha256": file_sha256(labels / "label.npy"),
                "true_count_sha256": file_sha256(labels / "true_count.npy"),
                "record_owner_sha256": file_sha256(labels / "rec_owner.npy"),
                "tree_scores_sha256": file_sha256(tree_path),
                "blend_scores_sha256": file_sha256(blend_path)}
    return ArenaData(reference, pool, left, right, label, true_count, owner, tree, blend, partition, identity)


def _config_digest(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def verified_checkpoint(path: Path) -> tuple[dict, dict]:
    """Verify a committed trainer checkpoint and recover its immutable text mode."""
    path = Path(path)
    marker = _checkpoint_marker(path)
    config_path = path / "training_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (file_sha256(config_path) != marker["files_sha256"].get("training_config.json")
            or _config_digest(config) != marker.get("config_sha256")
            or config.get("schema") != 2 or "initial_model" not in config):
        raise ValueError("Checkpoint training configuration or initial-model identity is invalid")
    profile = config.get("profile", {})
    if (type(profile.get("canonical_text")) is not bool
            or type(profile.get("field_tagged")) is not bool
            or not isinstance(config.get("max_length"), int)):
        raise ValueError("Checkpoint does not contain a complete saved text-mode configuration")
    initial = config["initial_model"]
    if (initial.get("model_id") != MODEL_ID or initial.get("licence") not in {"mit", "apache-2.0"}
            or not isinstance(initial.get("files_sha256"), dict) or not initial["files_sha256"]):
        raise ValueError("Checkpoint lacks the bound licensed BGE initial-model identity")
    identity = {"path": str(path.resolve()), "marker_sha256": file_sha256(path / "complete.json"),
                "step": marker["step"], "config_sha256": marker["config_sha256"],
                "model_files_sha256": marker["files_sha256"]}
    if marker.get("files_bytes"):
        identity["deferred_files_sha256_at_verify"] = {name: file_sha256(path / name)
                                                       for name in marker["files_bytes"]}
    return config, identity


def _atomic_npy(path: Path, values) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, values)
    temporary.replace(path)


def _population(partition: np.ndarray, name: str) -> np.ndarray:
    if name == "all":
        return np.ones(len(partition), dtype=bool)
    if name == "B":
        return np.isin(partition, ["B", "SLICE", "B_REST"])
    return partition == name


def _score_chunk(model, tokenizer, left_idx: np.ndarray, right_idx: np.ndarray,
                 ids: np.ndarray, left_text: list[str], right_text: list[str], args, torch) -> np.ndarray:
    batch = tokenize_batch(tokenizer, left_text, right_text, ids,
                           np.zeros(len(ids), dtype=np.float32), np.ones(len(ids), dtype=np.float32),
                           left_idx, right_idx,
                           max_length=args.max_length, microbatch=args.inference_batch,
                           pin_memory=args.pin_memory and args.device.startswith("cuda"))
    output = np.empty(len(ids), dtype=np.float32)
    model.eval()
    with torch.inference_mode():
        for tokens, positions in zip(batch.microbatches, batch.microbatch_positions):
            tokens = {name: value.to(args.device, non_blocking=args.pin_memory) for name, value in tokens.items()}
            with _amp_context(torch, args.device, args.amp):
                logits = model(**tokens).logits
            if logits.shape != (len(positions), 1):
                raise ValueError("Checkpoint scorer requires one match logit per pair")
            output[positions] = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
    return output


def score_checkpoint(args) -> dict:
    """Score one arena partition in immutable pair order and commit aligned arrays."""
    import torch

    torch.set_num_threads(args.threads)
    config, checkpoint = verified_checkpoint(args.checkpoint)
    needs_seal = config["profile"].get("mode") in {"c2", "g0"}
    if needs_seal and args.sealed_reservation is None:
        raise ValueError("C2/G0 scoring requires the sealed ID reservation")
    data = load_score_data(args.arena, partition_file=args.partition,
                           sealed_reservation=args.sealed_reservation if needs_seal else None)
    trained = config.get("arena", {})
    # G0b trains from precomputed historical pools without a partition file; its
    # TUNE partition is only needed to select the rows being scored.
    partition_free = config["profile"].get("mode") == "g0b" and trained.get("partition_sha256") is None
    if any(trained.get(key) != value for key, value in data.identity.items()
           if not (partition_free and key == "partition_sha256")):
        raise ValueError("Checkpoint was trained against different arena records, pair order, or partition")
    profile = config["profile"]
    args.max_length = config["max_length"]
    if args.device.startswith("cuda") and args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("BF16 scoring requires a supported CUDA device")
    model, tokenizer = load_reranker(str(args.checkpoint), args.device)
    left_text = record_texts(data.reference, canonical=profile["canonical_text"],
                             field_tagged=profile["field_tagged"])
    right_text = record_texts(data.pool, canonical=profile["canonical_text"],
                              field_tagged=profile["field_tagged"])
    population = _population(data.partition, args.score_partition)
    left = data.pairs.left_idx.to_numpy(dtype=np.int64)
    right = data.pairs.right_idx.to_numpy(dtype=np.int64)
    selected = population[left]
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Scoring output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    score_tmp = output / "scores.npy.tmp"
    scores = np.lib.format.open_memmap(score_tmp, mode="w+", dtype=np.float32, shape=(len(data.pairs),))
    scores[:] = 0
    started = time.perf_counter()
    selected_ids = np.flatnonzero(selected)
    for begin in range(0, len(selected_ids), args.score_chunk):
        ids = selected_ids[begin:begin + args.score_chunk]
        scores[ids] = _score_chunk(model, tokenizer, left, right, ids, left_text, right_text, args, torch)
    scores.flush()
    del scores
    score_tmp.replace(output / "scores.npy")
    _atomic_npy(output / "scored_mask.npy", selected.astype(bool))
    if args.device.startswith("cuda"):
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    manifest = {"state": "complete", "schema": 1, "scope": args.score_partition,
                "pairs": len(data.pairs), "scored_pairs": int(selected.sum()),
                "pair_order": "immutable arena pairs.parquet row order", "inputs": data.identity,
                "checkpoint": checkpoint,
                "text_mode": {"canonical": profile["canonical_text"],
                              "field_tagged": profile["field_tagged"],
                              "max_length": config["max_length"]},
                "runtime": {"amp": args.amp, "inference_batch": args.inference_batch,
                            "score_chunk": args.score_chunk, "seconds": elapsed,
                            "pairs_per_second": len(selected_ids) / elapsed if elapsed else None},
                "files": {"scores.npy": file_sha256(output / "scores.npy"),
                          "scored_mask.npy": file_sha256(output / "scored_mask.npy")}}
    json_write(output / "complete.json", manifest)
    return manifest


def _parse_choice(value: str) -> tuple[str, float, float]:
    try:
        name, weight, threshold = value.split(":")
        weight, threshold = float(weight), float(threshold)
    except ValueError as error:
        raise ValueError("Choice must be NAME:CE_WEIGHT:THRESHOLD") from error
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or not 0 <= weight <= 1 or not 0 <= threshold <= 1:
        raise ValueError("Choice name, CE weight, or threshold is invalid")
    return name, weight, threshold


def evaluate_scores(args) -> dict:
    """Evaluate fixed choices on complete S1 accounting; never tune a threshold."""
    from .arena_decision import evaluate, one_owner

    score_dir = Path(args.scores)
    score_manifest = json.loads((score_dir / "complete.json").read_text(encoding="utf-8"))
    checkpoint_config, checkpoint_identity = verified_checkpoint(Path(score_manifest["checkpoint"]["path"]))
    if checkpoint_identity != score_manifest["checkpoint"] or checkpoint_config.get("schema") != 2:
        raise ValueError("Score artifact checkpoint changed after inference")
    needs_seal = checkpoint_config["profile"].get("mode") in {"c2", "g0"}
    if needs_seal and args.sealed_reservation is None:
        raise ValueError("C2/G0 evaluation requires the sealed ID reservation")
    data = load_arena(args.arena, args.labels, partition_file=args.partition,
                      sealed_reservation=args.sealed_reservation if needs_seal else None,
                      load_text=False)
    input_keys = {"manifest_sha256", "reference_sha256", "pool_sha256", "pairs_sha256",
                  "pair_order_sha256", "partition_sha256", "sealed_reservation_sha256",
                  "references", "pool_records", "pairs"}
    if (score_manifest.get("state") != "complete" or set(score_manifest.get("inputs", {})) != input_keys
            or score_manifest["inputs"] != {key: data.identity[key] for key in input_keys}
            or set(score_manifest.get("files", {})) != {"scores.npy", "scored_mask.npy"}):
        raise ValueError("Score artifact does not belong to these exact arena inputs")
    for name, digest in score_manifest["files"].items():
        if file_sha256(score_dir / name) != digest:
            raise ValueError(f"Scored artifact changed: {name}")
    scores = np.load(score_dir / "scores.npy", mmap_mode="r")
    scored = np.load(score_dir / "scored_mask.npy", mmap_mode="r")
    if scores.shape != data.label.shape or scored.shape != data.label.shape or scored.dtype.kind != "b":
        raise ValueError("C2 scores/mask do not align with immutable arena pairs")
    population = _population(data.partition, args.eval_partition)
    pair_population = population[data.left]
    needed = pair_population & (data.tree >= args.tree_gate)
    choices = [_parse_choice(value) for value in args.choice]
    if any(weight > 0 for _, weight, _ in choices) and np.any(needed & ~scored):
        raise ValueError("Score artifact does not cover every gated pair in the evaluation population")
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Evaluation output must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, weight, threshold in choices:
        values = np.where(data.tree >= args.tree_gate,
                          (1 - weight) * data.tree + weight * scores, 0.).astype(np.float32)
        accept = pair_population & (values.astype(np.float64) >= threshold)
        if args.one_owner:
            accept = one_owner(accept, data.left, data.right, values)
        metrics = evaluate(accept, data.left, data.label, data.true_count,
                           population=population, country=data.reference.country.to_numpy(),
                           rec_idx=data.right, return_per_s1=True)
        per_s1 = metrics.pop("per_s1")
        entity_path = output / f"{name}_entities.tsv"
        entity_tmp = Path(str(entity_path) + ".tmp")
        pd.DataFrame({"source1_entity_id": data.reference.entity_id[population].to_numpy(),
                      "macro_f05": per_s1[population]}).to_csv(
                          entity_tmp, sep="\t", index=False, lineterminator="\n")
        entity_tmp.replace(entity_path)
        results.append({"name": name, "ce_weight": weight, "threshold": threshold,
                        "one_owner": args.one_owner, "metrics": metrics})
    report = {"state": "complete", "scope": args.eval_partition,
              "selection": "fixed choices only; no threshold tuning", "arena": data.identity,
              "scores_manifest_sha256": file_sha256(score_dir / "complete.json"),
              "tree_gate": args.tree_gate, "results": results,
              "files": {f"{name}_entities.tsv": file_sha256(output / f"{name}_entities.tsv")
                        for name, _, _ in choices}}
    json_write(output / "results.json", report)
    return report


def adapt_g0b(args) -> dict:
    """Materialize the historical medium BGE1800 TRAIN/TUNE comparison as an arena."""
    from .cross_encoder_experiment import (conflict_priorities, load_base_bundle,
                                           training_pool)
    from .experiment import load_data
    from .metrics import pair_labels

    data_path, base_path, output = Path(args.medium_data), Path(args.base_dir), Path(args.output)
    if output.exists():
        raise FileExistsError(f"G0b adapter output already exists: {output}")
    reference, pool, truth = load_data(data_path)
    _, original_pairs, base_scores, features, provenance = load_base_bundle(base_path, "tabular")
    keep_ref = reference.split.isin(["train", "tune"]).to_numpy()
    keep_pair = keep_ref[original_pairs.left_idx.to_numpy(dtype=np.int64)]
    remap = np.full(len(reference), -1, dtype=np.int64)
    remap[np.flatnonzero(keep_ref)] = np.arange(int(keep_ref.sum()))
    pairs = original_pairs.loc[keep_pair, ["left_idx", "right_idx"]].copy().reset_index(drop=True)
    pairs["left_idx"] = remap[pairs.left_idx.to_numpy(dtype=np.int64)]
    reference = reference.loc[keep_ref].reset_index(drop=True)
    scores = np.asarray(base_scores[keep_pair], dtype=np.float32)
    labels = pair_labels(reference, pool, pairs, {key: truth[key] for key in reference.entity_id}).astype(bool)
    focus = conflict_priorities(features.loc[keep_pair].reset_index(drop=True))
    positive, negative = training_pool(pairs, reference, labels, scores, 3.,
                                       conflicts=focus, conflict_fraction=.5)
    arena, label_dir = output / "arena", output / "labels"
    arena.mkdir(parents=True)
    label_dir.mkdir()
    reference.to_parquet(arena / "reference.parquet", index=False)
    pool.to_parquet(arena / "pool.parquet", index=False)
    pairs.to_parquet(arena / "pairs.parquet", index=False)
    _atomic_npy(arena / "v1_tree.npy", scores)
    _atomic_npy(arena / "v1_blend.npy", scores)
    _atomic_npy(arena / "g0b_positive.npy", positive.astype(np.int64))
    _atomic_npy(arena / "g0b_negative.npy", negative.astype(np.int64))
    partition = np.where(reference.split.eq("train"), "A", "B")
    _atomic_npy(output / "partition.npy", partition)
    true_count = np.asarray([len(truth[entity]) for entity in reference.entity_id], dtype=np.int16)
    owner_lookup = {}
    for left, entity in enumerate(reference.entity_id):
        for target in truth[entity]:
            if target in owner_lookup and owner_lookup[target] != left:
                raise ValueError("Historical medium truth assigns one target to multiple businesses")
            owner_lookup[target] = left
    owner = np.asarray([owner_lookup.get(entity, -1) for entity in pool.entity_id], dtype=np.int32)
    _atomic_npy(label_dir / "label.npy", labels)
    _atomic_npy(label_dir / "true_count.npy", true_count)
    _atomic_npy(label_dir / "rec_owner.npy", owner)
    json_write(label_dir / "labels_summary.json", {"holdout2_rows_read": 0,
               "scope": "historical medium TRAIN/TUNE only; eval excluded", "positive_pairs": int(labels.sum()),
               "true_links": int(true_count.sum()), "singletons": int((true_count == 0).sum())})
    names = ("reference.parquet", "pool.parquet", "pairs.parquet", "v1_tree.npy",
             "v1_blend.npy", "g0b_positive.npy", "g0b_negative.npy")
    manifest = {"schema": 1, "references": len(reference), "pool_records": len(pool),
                "pairs": len(pairs), "labels_read": False, "holdout2_excluded": True,
                "pair_order": "left_grouped_original_right_order",
                "max_pairs_per_reference": int(pairs.groupby("left_idx", sort=False).size().max()),
                "sha256": {name: file_sha256(arena / name) for name in names},
                "g0b": {"steps": 1800, "batch_size": 32, "positive_fraction": .5,
                         "negative_ratio": 3., "conflict_fraction": .5,
                         "text": "raw name | address", "source_base": provenance,
                         "source_data_sha256": {name: file_sha256(data_path / name)
                                                for name in ("reference.parquet", "pool.parquet", "truth.json")},
                         "source_pairs_sha256": file_sha256(base_path / "pairs.parquet")}}
    json_write(arena / "manifest.json", manifest)
    report = {"state": "complete", "arena": str(arena.resolve()), "labels": str(label_dir.resolve()),
              "partition": str((output / "partition.npy").resolve()), "positive_pool": len(positive),
              "negative_pool": len(negative), "tune_references": int(reference.split.eq("tune").sum()),
              "manifest_sha256": file_sha256(arena / "manifest.json")}
    json_write(output / "complete.json", report)
    return report


def train(args) -> dict:
    """Run one C2/G0/G0b job. G0 matrix jobs are intentionally independent."""
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(args.threads)
    profile = mode_config(args.mode, batch_size=args.batch_size, steps=args.steps)
    if args.schedule is not None:
        if profile.mode != "c2":
            raise ValueError("Only C2 may override its screen schedule")
        profile = replace(profile, schedule=args.schedule)
    text_view = getattr(args, "text_view", None)
    if text_view == "plain":
        if profile.mode != "c2":
            raise ValueError("Only C2 may switch to the plain 'name | address' view")
        profile = replace(profile, canonical_text=False, field_tagged=False)
    if profile.mode in {"c2", "g0"} and getattr(args, "sealed_reservation", None) is None:
        raise ValueError("C2/G0 training requires the sealed ID reservation")
    data = load_arena(args.arena, args.labels, partition_file=args.partition,
                      sealed_reservation=args.sealed_reservation if profile.mode in {"c2", "g0"} else None)
    if profile.mode in {"c2", "g0"} and (not np.any(data.partition == "A")
                                            or np.all(data.partition == "A")):
        raise ValueError("C2/G0 partition must separate training A from evaluation businesses")
    in_train = data.partition[data.left] == "A"
    positives = np.flatnonzero(in_train & data.label)
    negative = in_train & ~data.label
    if profile.mode == "g0b":
        flags = np.zeros(len(data.left), dtype=np.uint8)
        flags[negative] = int(NegativeType.FALSE_ACCEPT)
        exact_positive, exact_negative = (Path(args.arena) / name for name in
                                          ("g0b_positive.npy", "g0b_negative.npy"))
        if not exact_positive.is_file() or not exact_negative.is_file():
            raise ValueError("G0b requires an adapt-g0b bundle with exact historical sampling pools")
        positives = np.load(exact_positive, mmap_mode="r")
        gated = np.load(exact_negative, mmap_mode="r")
        if (positives.dtype.kind not in "iu" or gated.dtype.kind not in "iu"
                or np.any(~data.label[positives]) or np.any(data.label[gated])
                or np.any(~in_train[positives]) or np.any(~in_train[gated])):
            raise ValueError("G0b sampling pools are not aligned TRAIN positive/negative rows")
    else:
        flags = classify_negative_types(data.reference, data.pool, data.left, data.right, data.label,
                                        data.tree, data.blend, data.record_owner,
                                        accept_threshold=args.accept_threshold,
                                        lookalike_threshold=args.lookalike_threshold,
                                        tree_gate=args.tree_gate)
        flags[~in_train] = 0
        gated = np.flatnonzero(negative & (data.tree >= args.tree_gate) & (flags == 0))
    sampler = (LegacyBatchSampler(positives, gated, batch_size=profile.batch_size, seed=args.seed)
               if profile.mode == "g0b" else
               ArenaBatchSampler(positives, flags, gated, batch_size=profile.batch_size,
                                 positive_fraction=profile.positive_fraction, seed=args.seed))
    left_text = record_texts(data.reference, canonical=profile.canonical_text, field_tagged=profile.field_tagged)
    right_text = record_texts(data.pool, canonical=profile.canonical_text, field_tagged=profile.field_tagged)
    initial_model = model_identity(args.model, args.allowlist, args.model_id)
    attn = getattr(args, "attn_implementation", None)
    buckets = getattr(args, "length_buckets", None)
    model, tokenizer = load_reranker(args.model, args.device, attn)
    frozen_names = freeze_word_embeddings(model, args.freeze_embeddings)
    optimizer = build_adamw(torch, model, learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay, fused=args.fused, device=args.device)
    config = {"schema": 2, "profile": asdict(profile), "arena": data.identity,
              "initial_model": initial_model,
              "seed": args.seed, "max_length": args.max_length, "microbatch": args.microbatch,
              "prefetch": args.prefetch, "threads": args.threads, "tree_gate": args.tree_gate,
              "accept_threshold": args.accept_threshold, "lookalike_threshold": args.lookalike_threshold,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay,
              "warmup_fraction": args.warmup_fraction, "fused": args.fused,
              "frozen_parameters": frozen_names, "amp": args.amp}
    if buckets:
        config["length_buckets"] = list(buckets)
    digest = _config_digest(config)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "training_config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Output belongs to a different fast-trainer configuration")
    json_write(config_path, config)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 requires a supported CUDA device")
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()
    start = 0
    if args.resume:
        pointer = json.loads((output / "latest_checkpoint.json").read_text())
        checkpoint = Path(pointer["checkpoint"])
        if (pointer.get("step") != _checkpoint_marker(checkpoint)["step"]
                or pointer.get("marker_sha256") != file_sha256(checkpoint / "complete.json")):
            raise ValueError("Latest checkpoint pointer no longer matches its committed marker")
        model, tokenizer = load_reranker(str(checkpoint), args.device, attn)
        frozen_names = freeze_word_embeddings(model, args.freeze_embeddings)
        optimizer = build_adamw(torch, model, learning_rate=args.learning_rate,
                                weight_decay=args.weight_decay, fused=args.fused, device=args.device)
        start, sampler = restore_checkpoint(checkpoint, model, optimizer, config_sha256=digest,
                                            positive=positives, negative_flags=flags,
                                            gated_negative=gated, batch_size=profile.batch_size,
                                            positive_fraction=profile.positive_fraction,
                                            seed=args.seed, torch=torch)
    attn_used = getattr(model.config, "_attn_implementation", None)
    print(json.dumps({"attn_implementation": attn_used, "compile": bool(getattr(args, "compile", False)),
                      "length_buckets": list(buckets) if buckets else None,
                      "text_view": "plain" if not profile.field_tagged else "tagged"}), flush=True)
    forward = torch.compile(model) if getattr(args, "compile", False) else model
    warmup = round(profile.steps * args.warmup_fraction)
    pair_digest = hashlib.sha256()
    losses: list[dict] = []

    def descriptors():
        for _ in range(start + 1, profile.steps + 1):
            pair_ids, targets = sampler.take()
            counts = data.true_count[data.left[pair_ids]]
            weights = (np.ones(len(targets), dtype=np.float32) if profile.mode == "g0b" else
                       stake_weights(targets.astype(bool), counts))
            pair_digest.update(pair_ids.astype("<i8").tobytes())
            yield pair_ids, targets, weights, sampler.state()

    def tokenized(pair_ids, targets, weights, sampler_state):
        return tokenize_batch(tokenizer, left_text, right_text, pair_ids, targets, weights,
                              data.left, data.right, max_length=args.max_length,
                              microbatch=args.microbatch,
                              pin_memory=args.pin_memory and args.device.startswith("cuda"),
                              sampler_state=sampler_state, length_buckets=buckets)

    started = time.perf_counter()
    for step, batch in enumerate(TokenizationPrefetcher(descriptors(), tokenized, depth=args.prefetch), start + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        normalizer = float(batch.weights.sum())
        loss_value = 0.0
        for tokens, positions in zip(batch.microbatches, batch.microbatch_positions):
            tokens = {name: value.to(args.device, non_blocking=args.pin_memory) for name, value in tokens.items()}
            target = torch.as_tensor(batch.labels[positions], dtype=torch.float32, device=args.device)
            weight = torch.as_tensor(batch.weights[positions], dtype=torch.float32, device=args.device)
            with _amp_context(torch, args.device, args.amp):
                logits = forward(**tokens).logits
            if logits.shape != (len(positions), 1):
                raise ValueError("Fast trainer requires one match logit per pair")
            loss = (F.binary_cross_entropy_with_logits(logits[:, 0].float(), target, reduction="none")
                    * weight).sum() / normalizer
            loss.backward()
            loss_value += float(loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        lr = learning_rate(step, base=args.learning_rate, warmup_steps=warmup,
                           total_steps=profile.steps, schedule=profile.schedule)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()
        if step == 1 or step % args.log_every == 0:
            losses.append({"step": step, "bce": loss_value, "lr": lr,
                            "elapsed_s": round(time.perf_counter() - started, 1)})
            print(json.dumps(losses[-1]), flush=True)
        if step % args.save_every == 0 or step == profile.steps:
            checkpoint = save_checkpoint(model, tokenizer, optimizer, batch.sampler_state, step=step,
                                         config=config,
                                         config_sha256=digest, root=args.checkpoint_root or output / "checkpoints",
                                         torch=torch,
                                         defer_hash=(getattr(args, "defer_checkpoint_hash", False)
                                                     and step != profile.steps))
            json_write(output / "latest_checkpoint.json", {"checkpoint": str(checkpoint.resolve()),
                                                             "step": step,
                                                             "marker_sha256": file_sha256(checkpoint / "complete.json")})
    elapsed = time.perf_counter() - started
    report = {"mode": profile.mode, "completed_steps": profile.steps,
              "resumed_from_step": start,
              "pairs_this_invocation": (profile.steps - start) * profile.batch_size,
              "seconds": elapsed, "pairs_per_second": ((profile.steps - start) * profile.batch_size / elapsed
                                                        if elapsed else None),
              "invocation_sampled_pair_sha256": pair_digest.hexdigest(), "trace": losses,
              "fused": args.fused, "freeze_embeddings": args.freeze_embeddings,
              "attn_implementation": attn_used, "compile": bool(getattr(args, "compile", False)),
              "scope": "arena partition A only; HOLDOUT-2 excluded"}
    if args.device.startswith("cuda"):
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 1024**3
    json_write(output / "results.json", report)
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--command", choices=("train", "score", "evaluate", "adapt-g0b"), default="train")
    parser.add_argument("--mode", choices=("c2", "g0", "g0b"), default="c2")
    parser.add_argument("--arena", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--partition", type=Path, help="Per-S1 A/B/SLICE/B_REST .npy; omitted only for compatibility smokes")
    parser.add_argument("--sealed-reservation", type=Path,
                        help="ID-only reservation used to prove C2/G0 excludes sealed businesses")
    parser.add_argument("--model", help="Local BGE checkpoint (pretrained snapshot for G0b)")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--allowlist", type=Path,
                        default=Path(__file__).resolve().parents[4] / "problem-statement/models.yaml")
    parser.add_argument("--checkpoint", type=Path, help="Completed fast-trainer checkpoint to score")
    parser.add_argument("--scores", type=Path, help="Completed score artifact to evaluate")
    parser.add_argument("--score-partition", choices=("A", "B", "SLICE", "B_REST", "all"), default="all")
    parser.add_argument("--eval-partition", choices=("A", "B", "SLICE", "B_REST", "all"), default="SLICE")
    parser.add_argument("--choice", action="append", help="Fixed NAME:CE_WEIGHT:THRESHOLD; repeatable")
    parser.add_argument("--one-owner", action="store_true")
    parser.add_argument("--medium-data", type=Path, help="Historical medium dense bundle for adapt-g0b")
    parser.add_argument("--base-dir", type=Path, help="Historical completed tabular base for adapt-g0b")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--inference-batch", type=int, default=128)
    parser.add_argument("--score-chunk", type=int, default=100_000)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fused", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--freeze-embeddings", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--warmup-fraction", type=float, default=.05)
    parser.add_argument("--schedule", choices=("constant", "cosine"),
                        help="C2 screen defaults to constant; use cosine for a final run")
    parser.add_argument("--tree-gate", type=float, default=.01)
    parser.add_argument("--accept-threshold", type=float, default=.775)
    parser.add_argument("--lookalike-threshold", type=float, default=.3)
    parser.add_argument("--text-view", choices=("tagged", "plain"),
                        help="C2 record text: canonical field-tagged (default) or raw 'name | address'")
    parser.add_argument("--length-buckets", help="Comma-separated static pad widths, e.g. 64,128,256")
    parser.add_argument("--attn-implementation", choices=("sdpa", "flash_attention_2", "eager"),
                        help="Override the transformers attention kernel; default is the library's choice")
    parser.add_argument("--compile", action="store_true", help="torch.compile the training forward pass")
    parser.add_argument("--defer-checkpoint-hash", action="store_true",
                        help="Record large intermediate checkpoint files by size; the final step is fully hashed")
    parser.add_argument("--save-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--print-g0-matrix", action="store_true")
    args = parser.parse_args(argv)
    if args.print_g0_matrix:
        print(json.dumps(g0_matrix(), indent=2))
        return
    if args.command == "adapt-g0b":
        if any(value is None for value in (args.medium_data, args.base_dir, args.output)):
            parser.error("adapt-g0b requires --medium-data, --base-dir, and --output")
        print(json.dumps(adapt_g0b(args), indent=2))
        return
    if args.command == "score":
        if any(value is None for value in (args.arena, args.checkpoint, args.output)):
            parser.error("score requires --arena, --checkpoint, and --output")
        if args.score_partition != "all" and args.partition is None:
            parser.error("partition scoring requires --partition")
        if min(args.inference_batch, args.score_chunk, args.threads) < 1:
            parser.error("Scoring batch, chunk, and thread counts must be positive")
        print(json.dumps(score_checkpoint(args), indent=2))
        return
    if args.command == "evaluate":
        if any(value is None for value in (args.arena, args.labels, args.scores, args.output)) or not args.choice:
            parser.error("evaluate requires --arena, --labels, --scores, --output, and at least one --choice")
        if args.eval_partition != "all" and args.partition is None:
            parser.error("partition evaluation requires --partition")
        print(json.dumps(evaluate_scores(args), indent=2))
        return
    if any(value is None for value in (args.arena, args.labels, args.model, args.output)):
        parser.error("train requires --arena, --labels, --model, and --output")
    if (min(args.microbatch, args.max_length, args.prefetch, args.save_every, args.log_every) < 1
            or args.threads < 1 or args.learning_rate <= 0 or args.weight_decay < 0
            or not 0 <= args.warmup_fraction < 1):
        parser.error("Batch/runtime sizes and learning rate must be positive; schedule values are invalid")
    if args.length_buckets:
        try:
            buckets = tuple(int(value) for value in args.length_buckets.split(","))
        except ValueError:
            parser.error("--length-buckets must be comma-separated integers")
        if list(buckets) != sorted(set(buckets)) or buckets[0] < 8 or buckets[-1] < args.max_length:
            parser.error("--length-buckets must be increasing and end at or above --max-length")
        args.length_buckets = buckets
    if args.text_view == "plain" and args.mode != "c2":
        parser.error("--text-view plain is only available for C2")
    profile = mode_config(args.mode, batch_size=args.batch_size, steps=args.steps)
    if args.fused is None:
        args.fused = args.mode != "g0b"
    if args.freeze_embeddings is None:
        args.freeze_embeddings = False
    if args.mode in {"c2", "g0"} and args.partition is None:
        parser.error("C2/G0 require an explicit arena partition; only G0b compatibility may omit it")
    if args.mode in {"c2", "g0"} and args.sealed_reservation is None:
        parser.error("C2/G0 require --sealed-reservation for an ID-level exclusion check")
    if args.mode == "g0b":
        if args.model_id != MODEL_ID:
            parser.error(f"G0b is locked to {MODEL_ID}")
        args.microbatch = 32
        args.max_length = 256
        args.learning_rate = 2e-5
        args.weight_decay = .01
        args.warmup_fraction = .10
        args.amp = "bf16"
        args.seed = DEFAULT_SEED
    if profile.batch_size % args.microbatch:
        parser.error("Microbatch must divide the effective batch size")
    if args.amp == "bf16" and not args.device.startswith("cuda"):
        parser.error("BF16 is available only on CUDA; use --amp fp32 for CPU checks")
    if args.mode == "g0b" and (args.freeze_embeddings or args.fused):
        parser.error("G0b changes trainer mechanics one at a time; use --no-fused and unfrozen embeddings")
    train(args)


if __name__ == "__main__":
    main()
