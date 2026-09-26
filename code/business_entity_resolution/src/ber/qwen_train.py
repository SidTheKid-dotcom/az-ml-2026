"""Qwen3-Reranker-0.6B trained in C2's sampling world, then scored on gated pairs.

Commands (CPU prep, GPU train/score)::

    python -m ber.qwen_train prepare-train --model M --out DIR [--steps 40000]
    python -m ber.qwen_train prepare-score --kind arena|test --model M --out DIR
    python -m ber.qwen_train train --data DIR --model M --output RUN [--time-budget-min 75]
    python -m ber.qwen_train score --data DIR --checkpoint RUN/checkpoints/step_XXXXXX --output DIR

``prepare-train`` reuses ``fast_trainer.classify_negative_types`` (flags cached
to disk) and ``fast_trainer.ArenaBatchSampler`` (1 positive : 3 negatives, half
typed, half random tree-gated, arena partition A only) and ``stake_weights``
(per-batch class-normalized, cap 5). It materializes the sampler's batches in
order, so step ``s`` trains on rows ``[s*B, (s+1)*B)``: resume is exact and needs
no sampler state. Text is C2's plain view (raw ``name | address``).

The prompt keeps Qwen3-Reranker's chat/answer format (user turn, empty think
block, answer token) but drops the long system message to cut ~40 tokens per
pair. Score = final-token ``h . (W_yes - W_no)`` (= ``logit(yes) - logit(no)``),
probability = sigmoid. Sequences are right-padded and run with pure causal
attention (no padding mask): real tokens never attend to padding, positions
stay 0..len-1, and SDPA can use its causal kernel. The final real token is
gathered per row. Training keeps FP32 master weights with BF16 autocast.

Score outputs match C2's layout (``scores.npy`` over every pair, 0 where
unscored, ``scored_mask.npy``, ``complete.json``) so ``ber.pair_tree_c2`` helpers
read them unchanged. No SLICE or HOLDOUT-2 label is indexed here; partition-A
negatives are typed with the full record-owner array exactly as C2 did.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_SEED = 20260926  # C2 used 20260925; same sampling world, different draw
BATCH = 128
PROMPT_PREFIX = ("<|im_start|>user\n<Instruct>: Do the Query and the Document refer to the same "
                 "real-world business?\n<Query>: ")
PROMPT_MIDDLE = "\n<Document>: "
PROMPT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RECORD_CAP = 100      # tokens per record; arena p99 is 53, max 152
MAX_LENGTH = 256
BUCKETS = (48, 64, 80, 96, 112, 128, 160, 192, 256)
FLAG_NAMES = ("false_accept", "contest_loser", "unowned_lookalike", "digit_substitution",
              "empty_address_sibling")
TEXT_COLUMNS = ("left_business_name", "left_business_address", "right_business_name",
                "right_business_address")
T0 = time.time()


def log(message) -> None:
    print(f"[{time.time() - T0:8.1f}s] {message}", flush=True)


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def json_dump(path: Path, payload) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value))


def save_npy(path: Path, values) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as handle:
        np.save(handle, values)
    tmp.replace(path)


# ----------------------------------------------------------------------------- tokens

def encode_many(tokenizer, texts: list[str], chunk: int = 200_000) -> tuple[np.ndarray, np.ndarray]:
    """Batch-encode ``texts`` without special tokens into a flat int32 array plus offsets."""
    pieces, lengths = [], []
    for start in range(0, len(texts), chunk):
        encoded = tokenizer(texts[start:start + chunk], add_special_tokens=False)["input_ids"]
        lengths.append(np.fromiter(map(len, encoded), dtype=np.int64, count=len(encoded)))
        pieces.append(np.fromiter((token for row in encoded for token in row), dtype=np.int32,
                                  count=int(lengths[-1].sum())))
    lens = np.concatenate(lengths) if lengths else np.zeros(0, dtype=np.int64)
    flat = np.concatenate(pieces) if pieces else np.zeros(0, dtype=np.int32)
    offsets = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=offsets[1:])
    return flat, offsets


def fixed_ids(tokenizer, text: str) -> np.ndarray:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError("Prompt piece tokenized to nothing")
    return np.asarray(ids, dtype=np.int32)


def answer_ids(tokenizer) -> tuple[int, int]:
    yes, no = (int(tokenizer.convert_tokens_to_ids(word)) for word in ("yes", "no"))
    unk = getattr(tokenizer, "unk_token_id", None)
    if yes == no or (unk is not None and unk in (yes, no)):
        raise ValueError("yes/no are not distinct single tokens")
    return yes, no


def _ragged_copy(out, dst_start, src, src_start, lens) -> None:
    lens = np.asarray(lens, dtype=np.int64)
    total = int(lens.sum())
    if not total:
        return
    within = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(lens) - lens, lens)
    out[np.repeat(np.asarray(dst_start, dtype=np.int64), lens) + within] = \
        src[np.repeat(np.asarray(src_start, dtype=np.int64), lens) + within]


def assemble(prefix, middle, suffix, left_flat, left_off, left_pos, right_flat, right_off, right_pos,
             cap: int = RECORD_CAP) -> tuple[np.ndarray, np.ndarray]:
    """``prefix + left[:cap] + middle + right[:cap] + suffix`` per pair, as flat tokens + offsets."""
    left_pos, right_pos = np.asarray(left_pos, np.int64), np.asarray(right_pos, np.int64)
    ll = np.minimum(left_off[left_pos + 1] - left_off[left_pos], cap)
    rl = np.minimum(right_off[right_pos + 1] - right_off[right_pos], cap)
    lp, lm, ls = len(prefix), len(middle), len(suffix)
    lengths = lp + ll + lm + rl + ls
    if int(lengths.max(initial=0)) > MAX_LENGTH:
        raise ValueError("Assembled prompt exceeds the maximum length")
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    out = np.empty(int(offsets[-1]), dtype=np.int32)
    start = offsets[:-1]
    n = len(lengths)
    for piece, at in ((prefix, start), (middle, start + lp + ll), (suffix, start + lp + ll + lm + rl)):
        _ragged_copy(out, at, np.tile(piece, 1), np.zeros(n, np.int64), np.full(n, len(piece)))
    _ragged_copy(out, start + lp, left_flat, left_off[left_pos], ll)
    _ragged_copy(out, start + lp + ll + lm, right_flat, right_off[right_pos], rl)
    return out, offsets


def tokenize_pairs(tokenizer, left_text: list[str], right_text: list[str], left_idx, right_idx):
    """Tokenize each distinct record once, then assemble every pair's prompt."""
    left_idx, right_idx = np.asarray(left_idx, np.int64), np.asarray(right_idx, np.int64)
    left_u, left_pos = np.unique(left_idx, return_inverse=True)
    right_u, right_pos = np.unique(right_idx, return_inverse=True)
    log(f"tokenizing {len(left_u)} left and {len(right_u)} right records")
    lf, lo = encode_many(tokenizer, [left_text[i] for i in left_u])
    rf, ro = encode_many(tokenizer, [right_text[i] for i in right_u])
    pieces = [fixed_ids(tokenizer, text) for text in (PROMPT_PREFIX, PROMPT_MIDDLE, PROMPT_SUFFIX)]
    tokens, offsets = [], [np.zeros(1, np.int64)]
    base = 0
    for start in range(0, len(left_idx), 2_000_000):
        stop = start + 2_000_000
        chunk, off = assemble(*pieces, lf, lo, left_pos[start:stop], rf, ro, right_pos[start:stop])
        tokens.append(chunk)
        offsets.append(off[1:] + base)
        base += int(off[-1])
    return np.concatenate(tokens), np.concatenate(offsets)


def token_meta(tokenizer) -> dict:
    yes, no = answer_ids(tokenizer)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return {"yes_id": yes, "no_id": no, "pad_id": int(pad), "prompt_prefix": PROMPT_PREFIX,
            "prompt_middle": PROMPT_MIDDLE, "prompt_suffix": PROMPT_SUFFIX, "record_cap": RECORD_CAP,
            "max_length": MAX_LENGTH, "text_view": "plain raw 'name | address' (C2 plain)",
            "prompt_sha256": sha256_bytes((PROMPT_PREFIX + "\0" + PROMPT_MIDDLE + "\0" + PROMPT_SUFFIX).encode())}


def load_tokenizer(model_dir):
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)


# ----------------------------------------------------------------------------- CPU prep

def _texts(frame) -> list[str]:
    from .text_views import record_texts

    return record_texts(frame, canonical=False, field_tagged=False)


def _raw(frame, column) -> np.ndarray:
    return frame[column].fillna("").astype(str).to_numpy(dtype=object)


def arena_flags(args) -> dict:
    """C2 negative flags on partition-A pairs, cached (flags.npy covers every arena pair, 0 off A)."""
    import pandas as pd
    import pyarrow.parquet as pq

    from .fast_trainer import classify_negative_types
    from .v2_ce_train import sealed_ids

    flag_dir = Path(args.flags)
    arena, labels_dir = Path(args.arena), Path(args.labels)
    table = pq.read_table(arena / "pairs.parquet", columns=["left_idx", "right_idx"])
    left = table.column("left_idx").to_numpy().astype(np.int64)
    right = table.column("right_idx").to_numpy().astype(np.int64)
    del table
    partition = np.load(args.partition)
    if partition.dtype.kind not in "iu" or int(partition.max()) > 2:
        raise ValueError("Expected finalize_v1 codes 0=A, 1=B_REST, 2=SLICE")
    in_a = partition[left] == 0
    rows_a = np.flatnonzero(in_a)
    tree = np.load(arena / "v1_tree.npy", mmap_mode="r")
    blend = np.load(arena / "v1_blend.npy", mmap_mode="r")
    true_count = np.load(labels_dir / "true_count.npy")
    summary = json.loads((labels_dir / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise ValueError("Arena labels include HOLDOUT-2 rows")
    label_a = np.asarray(np.load(labels_dir / "label.npy", mmap_mode="r")[rows_a], dtype=bool)
    meta_path = flag_dir / "meta.json"
    reference = pd.read_parquet(arena / "reference.parquet", columns=["entity_id", "business_name",
                                                                        "business_address"])
    pool = pd.read_parquet(arena / "pool.parquet", columns=["business_name", "business_address"])
    forbidden, sealed_sha = sealed_ids(Path(args.sealed_reservation))
    if set(reference.entity_id) & forbidden:
        raise ValueError("Arena reference contains a sealed ID")
    if meta_path.exists():
        log(f"load cached flags {flag_dir}")
        flags = np.load(flag_dir / "flags.npy")
        meta = json.loads(meta_path.read_text())
    else:
        flag_dir.mkdir(parents=True, exist_ok=True)
        owner = np.load(labels_dir / "rec_owner.npy")
        log(f"classifying {len(rows_a)} partition-A pairs")
        flags_a = classify_negative_types(reference, pool, left[rows_a], right[rows_a], label_a,
                                          np.asarray(tree[rows_a]), np.asarray(blend[rows_a]), owner,
                                          accept_threshold=.775, lookalike_threshold=.3, tree_gate=.01)
        flags = np.zeros(len(left), dtype=np.uint8)
        flags[rows_a] = flags_a
        save_npy(flag_dir / "flags.npy", flags)
        meta = {"scope": "arena partition A pairs only; flags 0 elsewhere",
                "function": "ber.fast_trainer.classify_negative_types",
                "accept_threshold": .775, "lookalike_threshold": .3, "tree_gate": .01,
                "record_owner": "full rec_owner.npy, as C2", "sealed_reservation_sha256": sealed_sha,
                "pairs": len(left), "partition_a_pairs": int(len(rows_a)),
                "flag_counts": {name: int(((flags & (1 << bit)) != 0).sum())
                                for bit, name in enumerate(FLAG_NAMES)},
                "flags_sha256": sha256_file(flag_dir / "flags.npy")}
        json_dump(meta_path, meta)
    positive = rows_a[label_a]
    tree_a = np.asarray(tree[rows_a])
    gated = rows_a[~label_a & (tree_a >= .01) & (flags[rows_a] == 0)]
    return {"left": left, "right": right, "flags": flags, "positive": positive, "gated": gated,
            "true_count": true_count, "reference": reference, "pool": pool, "meta": meta,
            "partition_sha256": sha256_file(args.partition)}


def prepare_train(args) -> dict:
    import pandas as pd

    from .fast_trainer import ArenaBatchSampler, stake_weights

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = arena_flags(args)
    left, right, flags = data["left"], data["right"], data["flags"]
    log(f"positives {len(data['positive'])}, random-gated {len(data['gated'])}, "
        f"typed {int((flags != 0).sum())}")
    sampler = ArenaBatchSampler(data["positive"], flags, data["gated"], batch_size=BATCH,
                                positive_fraction=.25, seed=args.seed)
    ids = np.empty(args.steps * BATCH, dtype=np.int64)
    labels = np.empty(args.steps * BATCH, dtype=np.int8)
    weights = np.empty(args.steps * BATCH, dtype=np.float32)
    for step in range(args.steps):
        pair_ids, targets = sampler.take()
        rows = slice(step * BATCH, (step + 1) * BATCH)
        ids[rows] = pair_ids
        labels[rows] = targets.astype(np.int8)
        weights[rows] = stake_weights(targets.astype(bool), data["true_count"][left[pair_ids]])
        if (step + 1) % 5000 == 0:
            log(f"sampled {step + 1} steps")
    li, ri = left[ids], right[ids]
    pair_flags = flags[ids]
    kind = np.full(len(ids), "positive", dtype=object)
    negative = labels == 0
    kind[negative & (pair_flags == 0)] = "random_gated"
    for bit in reversed(range(len(FLAG_NAMES))):  # lowest set bit wins
        kind[negative & ((pair_flags & (1 << bit)) != 0)] = FLAG_NAMES[bit]
    reference, pool = data["reference"], data["pool"]
    frame = pd.DataFrame({
        "pair_row": np.arange(len(ids), dtype=np.int64), "step": np.arange(len(ids)) // BATCH,
        "arena_pair": ids, "left_idx": li, "right_idx": ri, "label": labels,
        "true_count": data["true_count"][li].astype(np.int32), "negative_type": kind,
        "negative_flags": pair_flags, "weight": weights, "partition": "A",
        "left_business_name": _raw(reference, "business_name")[li],
        "left_business_address": _raw(reference, "business_address")[li],
        "right_business_name": _raw(pool, "business_name")[ri],
        "right_business_address": _raw(pool, "business_address")[ri]})
    frame.to_parquet(out / "train_pairs.parquet.tmp", index=False)
    (out / "train_pairs.parquet.tmp").replace(out / "train_pairs.parquet")
    log("wrote train_pairs.parquet; tokenizing")
    tokenizer = load_tokenizer(args.model)
    tokens, offsets = tokenize_pairs(tokenizer, _texts(reference), _texts(pool), li, ri)
    save_npy(out / "tokens.npy", tokens)
    save_npy(out / "offsets.npy", offsets)
    save_npy(out / "labels.npy", labels)
    save_npy(out / "weights.npy", weights)
    lengths = np.diff(offsets)
    manifest = {
        "kind": "train", "state": "complete", "model_id": MODEL_ID, "tokenizer_dir": str(Path(args.model).resolve()),
        **token_meta(tokenizer), "seed": args.seed, "batch_size": BATCH, "steps": args.steps,
        "pairs": len(ids), "positive_fraction": float((labels == 1).mean()),
        "negative_types": {str(k): int(v) for k, v in zip(*np.unique(kind, return_counts=True))},
        "sampler": "ber.fast_trainer.ArenaBatchSampler (1:3, 50% typed / 50% random tree-gated, partition A)",
        "weights": "ber.fast_trainer.stake_weights per batch (class-normalized, cap 5)",
        "arena": str(Path(args.arena).resolve()), "partition_sha256": data["partition_sha256"],
        "flags": data["meta"], "pools": {"positive": len(data["positive"]), "gated": len(data["gated"])},
        "token_length": {"mean": float(lengths.mean()), "p50": float(np.percentile(lengths, 50)),
                         "p99": float(np.percentile(lengths, 99)), "max": int(lengths.max())},
        "files": {name: sha256_file(out / name) for name in
                  ("tokens.npy", "offsets.npy", "labels.npy", "weights.npy", "train_pairs.parquet")},
        "labels_read": "partition-A pair labels and true counts only", "holdout2_rows_read": 0}
    json_dump(out / "manifest.json", manifest)
    log(json.dumps({k: manifest[k] for k in ("pairs", "negative_types", "token_length")}))
    return manifest


def prepare_score(args) -> dict:
    """Tree-gated pairs (arena: B_REST S1s only) in ascending pair order, tokenized."""
    import pandas as pd
    import pyarrow.parquet as pq

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.kind == "arena":
        arena = Path(args.arena)
        table = pq.read_table(arena / "pairs.parquet", columns=["left_idx", "right_idx"])
        left = table.column("left_idx").to_numpy().astype(np.int64)
        right = table.column("right_idx").to_numpy().astype(np.int64)
        del table
        tree = np.load(arena / "v1_tree.npy", mmap_mode="r")
        partition = np.load(args.partition)
        selected = (np.asarray(tree) >= args.gate) & (partition[left] == 1)
        ref_path, pool_path = arena / "reference.parquet", arena / "pool.parquet"
        inputs = {"pairs": str(arena / "pairs.parquet"), "tree": str(arena / "v1_tree.npy"),
                  "partition": str(args.partition), "partition_sha256": sha256_file(args.partition),
                  "population": "B_REST (code 1)"}
    else:
        base = Path(args.test_inputs)
        left = np.load(base / "index" / "left.npy").astype(np.int64)
        right = np.load(base / "index" / "right.npy").astype(np.int64)
        tree = np.load(base / "base_probabilities.npy", mmap_mode="r")
        selected = np.asarray(tree) >= args.gate
        ref_path, pool_path = base / "dense_test" / "reference.parquet", base / "dense_test" / "pool.parquet"
        inputs = {"left": str(base / "index" / "left.npy"), "right": str(base / "index" / "right.npy"),
                  "tree": str(base / "base_probabilities.npy"), "population": "all test S1s"}
    if len(tree) != len(left):
        raise ValueError("Tree scores and pair indices are not aligned")
    rows = np.flatnonzero(selected)
    li, ri = left[rows], right[rows]
    reference = pd.read_parquet(ref_path, columns=["business_name", "business_address"])
    pool = pd.read_parquet(pool_path, columns=["business_name", "business_address"])
    frame = pd.DataFrame({"pair_row": np.arange(len(rows), dtype=np.int64), "pair_index": rows,
                          "left_idx": li, "right_idx": ri,
                          "left_business_name": _raw(reference, "business_name")[li],
                          "left_business_address": _raw(reference, "business_address")[li],
                          "right_business_name": _raw(pool, "business_name")[ri],
                          "right_business_address": _raw(pool, "business_address")[ri]})
    frame.to_parquet(out / "pairs.parquet.tmp", index=False)
    (out / "pairs.parquet.tmp").replace(out / "pairs.parquet")
    tokenizer = load_tokenizer(args.model)
    tokens, offsets = tokenize_pairs(tokenizer, _texts(reference), _texts(pool), li, ri)
    save_npy(out / "tokens.npy", tokens)
    save_npy(out / "offsets.npy", offsets)
    save_npy(out / "pair_index.npy", rows.astype(np.int64))
    lengths = np.diff(offsets)
    manifest = {"kind": args.kind, "state": "complete", "model_id": MODEL_ID, **token_meta(tokenizer),
                "gate": args.gate, "total_pairs": int(len(left)), "scored_pairs": int(len(rows)),
                "inputs": inputs, "reference": str(ref_path), "pool": str(pool_path),
                "alignment": "row k scores pair pair_index[k] of the full pair arrays (C2 scores.npy order)",
                "token_length": {"mean": float(lengths.mean()), "p99": float(np.percentile(lengths, 99)),
                                 "max": int(lengths.max())},
                "files": {name: sha256_file(out / name) for name in
                          ("tokens.npy", "offsets.npy", "pair_index.npy", "pairs.parquet")},
                "labels_read": False}
    json_dump(out / "manifest.json", manifest)
    log(json.dumps({k: manifest[k] for k in ("kind", "total_pairs", "scored_pairs", "token_length")}))
    return manifest


# ----------------------------------------------------------------------------- batching

def bucket_width(longest: int, buckets=BUCKETS) -> int:
    for width in buckets:
        if width >= longest:
            return width
    raise ValueError(f"Sequence of {longest} tokens exceeds the largest bucket")


def pad_right(tokens, offsets, rows, pad_id: int, buckets=BUCKETS) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad ``rows`` to one bucket width. Returns ids and each row's last real position."""
    starts, stops = offsets[rows], offsets[np.asarray(rows) + 1]
    lengths = (stops - starts).astype(np.int64)
    width = bucket_width(int(lengths.max()), buckets)
    ids = np.full((len(rows), width), pad_id, dtype=np.int64)
    within = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    ids[np.repeat(np.arange(len(rows)), lengths), within] = tokens[np.repeat(starts, lengths) + within]
    return ids, lengths - 1


def step_microbatches(offsets, step: int, batch: int, microbatches: int):
    """Rows of one step, length-sorted and split into ``microbatches`` contiguous groups."""
    rows = np.arange(step * batch, (step + 1) * batch)
    order = rows[np.argsort(np.diff(offsets)[rows], kind="stable")]
    return [part for part in np.array_split(order, microbatches) if len(part)]


# ----------------------------------------------------------------------------- model

class YesNoScorer:
    """``h_last . (W_yes - W_no)`` over the backbone of a Qwen3 causal LM (tied or untied head)."""

    def __init__(self, model, yes_id: int, no_id: int, backbone=None):
        self.model, self.yes_id, self.no_id = model, int(yes_id), int(no_id)
        self.backbone = model.model if backbone is None else backbone

    def __call__(self, torch, input_ids, last):
        hidden = self.backbone(input_ids=input_ids, use_cache=False).last_hidden_state
        final = hidden[torch.arange(hidden.shape[0], device=hidden.device), last]
        weight = self.model.lm_head.weight
        direction = (weight[self.yes_id] - weight[self.no_id]).float()
        score = final.float() @ direction
        bias = getattr(self.model.lm_head, "bias", None)
        if bias is not None:
            score = score + (bias[self.yes_id] - bias[self.no_id]).float()
        return score


def load_model(path, device: str, torch, dtype=None):
    from transformers import AutoModelForCausalLM

    dtype = torch.float32 if dtype is None else dtype
    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True, trust_remote_code=False,
                                                 dtype=dtype, attn_implementation="sdpa")
    if model.config.model_type != "qwen3":
        raise ValueError("Expected a Qwen3 causal LM")
    model.config.use_cache = False
    return model.to(device)


def lr_at(step: int, *, base: float, warmup: int, total: int | None, schedule: str, floor: float = .1) -> float:
    if step <= warmup:
        return base * step / max(1, warmup)
    if schedule == "constant" or total is None:
        return base
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * .5 * (1 + math.cos(math.pi * progress)))


def _checkpoints(root: Path) -> list[Path]:
    return sorted(path for path in root.glob("step_*") if (path / "complete.json").exists())


def _to_cpu(value):
    """Deep copy of a (nested) state dict with every tensor copied to CPU."""
    if hasattr(value, "detach") and hasattr(value, "to"):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_to_cpu(item) for item in value)
    return value


def save_checkpoint(model, optimizer, torch, *, root: Path, step: int, state: dict, model_dir,
                    keep_optimizers: int = 1, background=None):
    """Snapshot weights + optimizer to CPU now; write them (optionally on ``background``) atomically.

    Returns the final path, or a future of it when ``background`` is an executor.
    Tied ``lm_head.weight`` is dropped (``from_pretrained`` re-ties it).
    """
    root.mkdir(parents=True, exist_ok=True)
    final, stage = root / f"step_{step:06d}", root / f".step_{step:06d}.incomplete"
    if final.exists():
        raise FileExistsError(final)
    model_state = {key: value.detach().to("cpu", copy=True) for key, value in model.state_dict().items()}
    if getattr(model.config, "tie_word_embeddings", False):
        model_state.pop("lm_head.weight", None)
    optimizer_state = _to_cpu(optimizer.state_dict())
    state = {**state, "step": step}

    def write() -> Path:
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        model.save_pretrained(stage, state_dict=model_state, safe_serialization=True)
        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json",
                     "chat_template.jinja", "generation_config.json", "LICENSE"):
            if (Path(model_dir) / name).is_file() and not (stage / name).exists():
                shutil.copy2(Path(model_dir) / name, stage / name)
        torch.save({"optimizer": optimizer_state}, stage / "optimizer.pt")
        json_dump(stage / "state.json", state)
        sizes = {path.name: path.stat().st_size for path in stage.iterdir() if path.is_file()}
        json_dump(stage / "complete.json", {"state": "complete", "step": step, "files_bytes": sizes,
                                            "config_sha256": state["config_sha256"]})
        stage.replace(final)
        for old in _checkpoints(root)[:-keep_optimizers]:
            if (old / "optimizer.pt").exists():
                (old / "optimizer.pt").unlink()  # keep weights for averaging, drop stale optimizer state
        log(f"checkpoint written {final}")
        return final

    return write() if background is None else background.submit(write)


def train(args) -> dict:
    import torch
    from torch.nn import functional as F

    torch.set_num_threads(args.threads)
    data = Path(args.data)
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("kind") != "train" or manifest.get("state") != "complete":
        raise ValueError("Training data manifest is not a complete train set")
    tokens = np.load(data / "tokens.npy")
    offsets = np.load(data / "offsets.npy")
    labels = np.load(data / "labels.npy").astype(np.float32)
    weights = np.load(data / "weights.npy").astype(np.float32)
    batch = int(manifest["batch_size"])
    available = len(labels) // batch
    max_steps = min(args.steps or available, available)
    config = {"schema": 1, "model_id": MODEL_ID, "data_files": manifest["files"],
              "prompt_sha256": manifest["prompt_sha256"], "batch_size": batch, "microbatches": args.microbatches,
              "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "warmup": args.warmup,
              "schedule": args.schedule, "lr_floor": args.lr_floor, "max_steps": max_steps,
              "time_budget_min": args.time_budget_min, "amp": args.amp, "compile": bool(args.compile),
              "model_dir": str(Path(args.model).resolve()), "seed": args.seed}
    config_sha = sha256_bytes(json.dumps(config, sort_keys=True).encode())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    root = output / "checkpoints"
    config_path = output / "train_config.json"
    if config_path.exists() and json.loads(config_path.read_text()).get("config_sha256") != config_sha:
        raise ValueError("Output belongs to a different training configuration; use a new --output")
    json_dump(config_path, {**config, "config_sha256": config_sha})
    torch.manual_seed(args.seed)
    device = args.device
    cuda = device.startswith("cuda")
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    start, elapsed_before = 0, 0.0
    total_steps = None if args.time_budget_min else max_steps
    resume = _checkpoints(root)[-1] if (args.resume and _checkpoints(root)) else None
    model = load_model(resume or args.model, device, torch)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate, weight_decay=args.weight_decay,
                                  betas=(.9, .999), eps=1e-8, fused=cuda)
    if resume is not None:
        state = json.loads((resume / "state.json").read_text())
        if state["config_sha256"] != config_sha:
            raise ValueError("Checkpoint belongs to a different configuration")
        optimizer.load_state_dict(torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)["optimizer"])
        start, total_steps = int(state["step"]), state.get("total_steps")
        elapsed_before = float(state.get("train_seconds", 0.0))
        log(f"resumed {resume} at step {start}, total_steps {total_steps}")
    backbone = torch.compile(model.model, dynamic=True) if args.compile else None
    forward = YesNoScorer(model, manifest["yes_id"], manifest["no_id"], backbone)
    pad_id = int(manifest["pad_id"])
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if (cuda and args.amp == "bf16") else torch.autocast(
        "cpu", enabled=False)

    def prepared(step):
        parts = []
        for rows in step_microbatches(offsets, step, batch, args.microbatches):
            ids, last = pad_right(tokens, offsets, rows, pad_id)
            tensors = [torch.from_numpy(ids), torch.from_numpy(last), torch.from_numpy(labels[rows]),
                       torch.from_numpy(weights[rows])]
            if cuda:
                tensors = [t.pin_memory() for t in tensors]
            parts.append((tensors, int(ids.size), int((ids != pad_id).sum())))
        return parts

    status_path = output / "status.json"
    window = {"loss": 0.0, "steps": 0, "pairs": 0, "tokens": 0, "padded": 0, "t": time.perf_counter()}
    started = time.perf_counter()
    calibrated_at = None
    budget_s = args.time_budget_min * 60 if args.time_budget_min else None
    step = start
    stop_reason = "max_steps"
    executor = ThreadPoolExecutor(max_workers=2)
    writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="qwen-ckpt")
    saving = None
    pending = {}
    last_saved = start
    model.train()
    try:
        while True:
            target = total_steps or max_steps
            if step >= target:
                break
            step += 1
            for ahead in range(step, min(step + 4, target + 1)):
                if ahead not in pending:
                    pending[ahead] = executor.submit(prepared, ahead - 1)
            parts = pending.pop(step).result()
            normalizer = float(weights[(step - 1) * batch: step * batch].sum())
            loss_total = 0.0
            for (ids, last, target_y, weight), size, real in parts:
                ids, last = ids.to(device, non_blocking=True), last.to(device, non_blocking=True)
                target_y, weight = target_y.to(device, non_blocking=True), weight.to(device, non_blocking=True)
                with amp:
                    score = forward(torch, ids, last)
                loss = (F.binary_cross_entropy_with_logits(score.float(), target_y, reduction="none") * weight).sum() / normalizer
                loss.backward()
                loss_total += float(loss.detach())
                window["tokens"] += real
                window["padded"] += size
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            lr = lr_at(step, base=args.learning_rate, warmup=args.warmup, total=total_steps,
                       schedule=args.schedule, floor=args.lr_floor)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            window["loss"] += loss_total
            window["steps"] += 1
            window["pairs"] += batch
            elapsed = elapsed_before + time.perf_counter() - started
            if (total_steps is None and budget_s and step - start == args.calibrate_steps):
                rate = (step - start - 20) / max(1e-9, time.perf_counter() - calibrated_at)
                remaining = budget_s - elapsed - args.reserve_min * 60
                total_steps = int(min(max_steps, step + max(0, rate * remaining)))
                total_steps = max(step + 1, total_steps - total_steps % 250 if total_steps > 500 else total_steps)
                log(f"calibrated {rate:.2f} steps/s -> total_steps {total_steps}")
            if step - start == 20:
                calibrated_at = time.perf_counter()
            if step % args.log_every == 0 or step == 1:
                now = time.perf_counter()
                span = max(1e-9, now - window["t"])
                target = total_steps or max_steps
                status = {"step": step, "total_steps": total_steps, "max_steps": max_steps,
                          "loss": window["loss"] / max(1, window["steps"]), "lr": lr,
                          "pairs_per_s": window["pairs"] / span, "tokens_per_s": window["tokens"] / span,
                          "pad_efficiency": window["tokens"] / max(1, window["padded"]),
                          "elapsed_min": elapsed / 60,
                          "eta_min": (target - step) / max(1e-9, window["steps"] / span) / 60}
                if cuda:
                    status["max_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
                print(json.dumps(status), flush=True)
                json_dump(status_path, status)
                window.update(loss=0.0, steps=0, pairs=0, tokens=0, padded=0, t=now)
            over_budget = budget_s is not None and elapsed > budget_s * 1.1
            if step % args.checkpoint_every == 0 or step == (total_steps or max_steps) or over_budget:
                state = {"config_sha256": config_sha, "total_steps": total_steps, "train_seconds": elapsed,
                         "last_loss": loss_total}
                if saving is not None:
                    saving.result()  # surface a failed write before starting the next
                saving = save_checkpoint(model, optimizer, torch, root=root, step=step, state=state,
                                         model_dir=args.model, background=writer)
                last_saved = step
            if over_budget:
                stop_reason = "time_budget_exceeded"
                break
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        writer.shutdown(wait=True)
    if saving is not None:
        saving.result()
    if last_saved != step:
        state = {"config_sha256": config_sha, "total_steps": total_steps,
                 "train_seconds": elapsed_before + time.perf_counter() - started, "last_loss": None}
        save_checkpoint(model, optimizer, torch, root=root, step=step, state=state, model_dir=args.model)
    result = {"state": "complete", "completed_steps": step, "total_steps": total_steps, "stop_reason": stop_reason,
              "pairs_seen": step * batch, "train_seconds": elapsed_before + time.perf_counter() - started,
              "final_checkpoint": str(_checkpoints(root)[-1]), "config_sha256": config_sha,
              "scope": "arena partition A only; HOLDOUT-2 excluded"}
    if cuda:
        result["max_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
    json_dump(output / "result.json", result)
    log(json.dumps(result))
    return result


# ----------------------------------------------------------------------------- scoring

def score(args) -> dict:
    import torch

    torch.set_num_threads(args.threads)
    data, out = Path(args.data), Path(args.output)
    manifest = json.loads((data / "manifest.json").read_text())
    if manifest.get("kind") not in ("arena", "test") or manifest.get("state") != "complete":
        raise ValueError("Score data manifest is not a complete arena/test set")
    checkpoint = Path(args.checkpoint)
    if not (checkpoint / "complete.json").exists():
        raise ValueError("Checkpoint is not complete")
    ck_state = json.loads((checkpoint / "state.json").read_text())
    train_cfg_path = checkpoint.parent.parent / "train_config.json"
    if train_cfg_path.exists():
        train_cfg = json.loads(train_cfg_path.read_text())
        if train_cfg.get("prompt_sha256") != manifest["prompt_sha256"]:
            raise ValueError("Score inputs were tokenized with a different prompt than training")
    tokens = np.load(data / "tokens.npy")
    offsets = np.load(data / "offsets.npy")
    pair_index = np.load(data / "pair_index.npy")
    n = len(pair_index)
    lengths = np.diff(offsets)
    out.mkdir(parents=True, exist_ok=True)
    progress_path = out / "progress.json"
    logit_path = out / "logits_gated.npy"
    if progress_path.exists():
        done = int(json.loads(progress_path.read_text())["done"])
        logits = np.load(logit_path, mmap_mode="r+")
    else:
        done = 0
        logits = np.lib.format.open_memmap(logit_path, mode="w+", dtype=np.float32, shape=(n,))
        logits[:] = np.nan
        logits.flush()
    device = args.device
    cuda = device.startswith("cuda")
    model = load_model(checkpoint, device, torch)
    if cuda and args.amp == "bf16":
        amp = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        amp = torch.autocast("cpu", enabled=False)
    model.eval()
    scorer = YesNoScorer(model, manifest["yes_id"], manifest["no_id"])
    pad_id = int(manifest["pad_id"])
    order = np.argsort(lengths, kind="stable")
    batches = []
    position = 0
    while position < n:  # token-budget batches over length-sorted rows
        width = bucket_width(int(lengths[order[min(n - 1, position + args.batch_size - 1)]]))
        size = max(1, min(args.batch_size, args.token_budget // width))
        batches.append(order[position:position + size])
        position += size
    chunk_rows = np.cumsum([len(b) for b in batches])
    first = int(np.searchsorted(chunk_rows, done, side="right"))
    started, begin = time.perf_counter(), done

    def prepared(rows):
        ids, last = pad_right(tokens, offsets, rows, pad_id)
        tensors = (torch.from_numpy(ids), torch.from_numpy(last))
        return rows, tuple(t.pin_memory() for t in tensors) if cuda else tensors

    executor = ThreadPoolExecutor(max_workers=2)
    futures = {i: executor.submit(prepared, batches[i]) for i in range(first, min(first + 6, len(batches)))}
    with torch.inference_mode():
        for i in range(first, len(batches)):
            rows, (ids, last) = futures.pop(i).result()
            if i + 6 < len(batches):
                futures[i + 6] = executor.submit(prepared, batches[i + 6])
            with amp:
                value = scorer(torch, ids.to(device, non_blocking=True), last.to(device, non_blocking=True))
            logits[rows] = value.float().cpu().numpy()
            done = int(chunk_rows[i])
            if (i + 1) % args.flush_every == 0 or i == len(batches) - 1:
                logits.flush()
                rate = (done - begin) / max(1e-9, time.perf_counter() - started)
                status = {"done": done, "total": n, "pairs_per_second": round(rate, 1),
                          "eta_min": round((n - done) / max(rate, 1e-9) / 60, 1)}
                json_dump(progress_path, status)
                print(json.dumps(status), flush=True)
    executor.shutdown(wait=True)
    values = np.asarray(np.load(logit_path, mmap_mode="r"))
    if not np.isfinite(values).all():
        raise ValueError("Some gated pairs were not scored")
    prob = (1.0 / (1.0 + np.exp(-values.astype(np.float64)))).astype(np.float32)
    full = np.zeros(int(manifest["total_pairs"]), dtype=np.float32)
    full[pair_index] = prob
    mask = np.zeros(int(manifest["total_pairs"]), dtype=bool)
    mask[pair_index] = True
    save_npy(out / "scores.npy", full)
    save_npy(out / "scored_mask.npy", mask)
    result = {"state": "complete", "kind": manifest["kind"], "model_id": MODEL_ID,
              "checkpoint": {"path": str(checkpoint.resolve()), "step": ck_state.get("step"),
                             "config_sha256": ck_state.get("config_sha256")},
              "score": "sigmoid(logit(yes) - logit(no)) at the final real token",
              "pairs": int(manifest["total_pairs"]), "scored_pairs": n, "gate": manifest["gate"],
              "inputs": {"data": str(data.resolve()), "manifest_sha256": sha256_file(data / "manifest.json")},
              "runtime": {"amp": args.amp, "batch_size": args.batch_size, "token_budget": args.token_budget,
                          "seconds": time.perf_counter() - started},
              "files": {name: sha256_file(out / name) for name in ("scores.npy", "scored_mask.npy")},
              "labels_read": False}
    json_dump(out / "complete.json", result)
    log(json.dumps({k: result[k] for k in ("kind", "scored_pairs", "runtime")}))
    return result


# ----------------------------------------------------------------------------- CLI

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare-train")
    p.add_argument("--model", required=True, help="Local Qwen3-Reranker-0.6B snapshot (tokenizer)")
    p.add_argument("--out", required=True)
    p.add_argument("--flags", help="Flag cache dir (default OUT/flags)")
    p.add_argument("--arena", default="/workspace/az-ml-new/arena/c2_input")
    p.add_argument("--labels", default="/workspace/az-ml-new/arena/labels")
    p.add_argument("--partition", default="/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy")
    p.add_argument("--sealed-reservation", default="/workspace/az-ml-new/v2_holdout2")
    p.add_argument("--steps", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p = sub.add_parser("prepare-score")
    p.add_argument("--kind", choices=("arena", "test"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--arena", default="/workspace/az-ml-new/arena/c2_input")
    p.add_argument("--partition", default="/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy")
    p.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
    p.add_argument("--gate", type=float, default=.01)
    p = sub.add_parser("train")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--steps", type=int, help="Upper bound on steps (default: every materialized step)")
    p.add_argument("--time-budget-min", type=float, help="Size total steps from measured speed to fit this budget")
    p.add_argument("--reserve-min", type=float, default=3.0, help="Budget minutes kept for saving/loading")
    p.add_argument("--calibrate-steps", type=int, default=220)
    p.add_argument("--microbatches", type=int, default=2)
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--warmup", type=int, default=400)
    p.add_argument("--schedule", choices=("cosine", "constant"), default="cosine")
    p.add_argument("--lr-floor", type=float, default=.1)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p = sub.add_parser("score")
    p.add_argument("--data", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=("bf16", "fp32"), default="bf16")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--token-budget", type=int, default=98_304)
    p.add_argument("--flush-every", type=int, default=200)
    p.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "prepare-train":
        args.flags = args.flags or str(Path(args.out) / "flags")
        prepare_train(args)
    elif args.command == "prepare-score":
        prepare_score(args)
    elif args.command == "train":
        if args.time_budget_min is not None and args.calibrate_steps <= 20:
            parser.error("--calibrate-steps must exceed 20")
        train(args)
    else:
        score(args)


if __name__ == "__main__":
    main()
