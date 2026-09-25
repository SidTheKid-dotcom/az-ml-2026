"""Qwen3-Reranker-0.6B yes/no adapter.

The BGE trainer scores a single sequence-classification logit. This model is a
decoder: the match score is the final-position ``logit(yes) - logit(no)``, and
the probability is a stable sigmoid of that difference. It is an optional
resolver feature. This module does not compute macro F0.5 and does not read
HOLDOUT-2 labels.

Resource figures in ``resource_estimates`` are estimates, not measurements.
Promotion still requires at least +0.0005 resolver gain on the fixed arena SLICE.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import time
import unicodedata

import numpy as np
import pandas as pd

from .experiment import json_write


MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
EXPECTED_LICENSE = "apache-2.0"
EXPECTED_PARAMS_B = 0.596
PARAM_TOLERANCE_B = 0.0005
MAX_LENGTH = 256
PAD_MULTIPLE = 8
NEGATIVE_RATIO = 3
STAKE_CAP = 5.0
SCREEN_MAX_STEPS = 20_000
SCREEN_MAX_EFFECTIVE_BATCH = 128
DEFAULT_LR = 2e-5
DEFAULT_WEIGHT_DECAY = 0.01
WARMUP_FRACTION = 0.05
DEFAULT_SEED = 20260925

SYSTEM_PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the '
    'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
ASSISTANT_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
DEFAULT_INSTRUCTION = (
    "Determine whether these two business records refer to the same real-world business. "
    "Allow spelling, transliteration, and address-format variation, but reject distinct "
    "businesses even at similar addresses. Judge the identity of the business, not topical relevance."
)

TYPED_NEGATIVE_TYPES = (
    "false_accept",
    "contest_loser",
    "unowned_lookalike",
    "digit_substitution",
    "empty_address_sibling",
)
RANDOM_NEGATIVE_TYPE = "random_gated"
NEGATIVE_TYPES = TYPED_NEGATIVE_TYPES + (RANDOM_NEGATIVE_TYPE,)
_HOLDOUT_MARKERS = ("holdout-2", "holdout2", "v2_holdout", "reserved_ids")
_JUNK = ("<NULL>", "##", ">>", "<<")
_TEXT_COLUMNS = (
    "left_business_name",
    "left_business_address",
    "right_business_name",
    "right_business_address",
)
_Scorer = Callable[[np.ndarray, np.ndarray], np.ndarray]


def resource_estimates() -> dict:
    """Planning figures only. Nothing here was measured on a GPU."""
    params = EXPECTED_PARAMS_B * 1e9
    return {
        "kind": "estimate",
        "measured": False,
        "model_id": MODEL_ID,
        "params_b": EXPECTED_PARAMS_B,
        "weight_bf16_gib": params * 2 / 1024**3,
        "adamw_moments_fp32_gib": params * 8 / 1024**3,
        "assumed_train_vram_gib_effective_batch_64_len_256": 12.0,
        "assumed_infer_vram_gib_batch_32_len_256_bf16": 4.0,
        "assumed_a40_pairs_per_second": "80-150",
        "assumed_screen_wall_hours_at_15k_steps_batch_64": "1.8-3.3",
        "plan_queue_budget_hours": 1.0,
        "assumptions": (
            "BF16 weights, FP32 Adam moments, sequence length 256, no activation "
            "checkpointing. The 1 hour Q7 budget is the plan's allocation, not a timed run."
        ),
    }


def refuse_holdout_path(path: Path | str) -> None:
    lowered = str(path).lower().replace("\\", "/")
    if any(marker in lowered for marker in _HOLDOUT_MARKERS):
        raise ValueError(f"Refusing to read a HOLDOUT-2 path: {path}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(payload: Mapping) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _jsonify(value):
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _is_latin_letter(char: str) -> bool:
    return char.isalpha() and unicodedata.name(char, "").startswith("LATIN ")


def fold_latin_accents(text: str) -> str:
    """Drop combining marks on Latin letters. Indic and other scripts stay intact."""
    decomposed = unicodedata.normalize("NFD", text)
    output: list[str] = []
    index = 0
    while index < len(decomposed):
        char = decomposed[index]
        if unicodedata.category(char).startswith("M"):
            output.append(char)
            index += 1
            continue
        nxt = index + 1
        marks: list[str] = []
        while nxt < len(decomposed) and unicodedata.category(decomposed[nxt]).startswith("M"):
            marks.append(decomposed[nxt])
            nxt += 1
        output.append(char)
        if not _is_latin_letter(char):
            output.extend(marks)
        index = nxt
    return unicodedata.normalize("NFC", "".join(output))


def canonical_field(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        text = ""
    else:
        text = str(value)
    for junk in _JUNK:
        text = text.replace(junk, " ")
    return " ".join(fold_latin_accents(text).lower().split())


def field_tagged_business(name, address) -> str:
    """Field tags only. Callers must not pass ids or label-derived fields."""
    return f"business name: {canonical_field(name)} ; business address: {canonical_field(address)}"


def prompt_parts(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> tuple[str, str, str]:
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    return SYSTEM_PREFIX, body, ASSISTANT_SUFFIX


def render_prompt(query: str, document: str, instruction: str = DEFAULT_INSTRUCTION) -> str:
    prefix, body, suffix = prompt_parts(query, document, instruction)
    return prefix + body + suffix


def encode_text(tokenizer, text: str) -> list[int]:
    if not isinstance(text, str):
        raise ValueError("Tokenizer input must be a string")
    try:
        encoded = tokenizer.encode(text, add_special_tokens=False)
    except TypeError:
        encoded = tokenizer.encode(text)
    ids = [int(token) for token in list(encoded)]
    if any(token < 0 for token in ids):
        raise ValueError("Tokenizer returned a negative token id")
    return ids


def decode_ids(tokenizer, ids: list[int]) -> str:
    try:
        return tokenizer.decode(list(ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(ids))


def verify_answer_token(tokenizer, word: str) -> int:
    """Accept only an exact single-token encoding of ``yes`` or ``no``."""
    if word not in {"yes", "no"}:
        raise ValueError("Answer token must be yes or no")
    unk = getattr(tokenizer, "unk_token_id", None)
    ids = encode_text(tokenizer, word)
    if len(ids) != 1:
        raise ValueError(f"{word!r} is not a single tokenizer token: {ids}")
    token_id = ids[0]
    if unk is not None and token_id == int(unk):
        raise ValueError(f"{word!r} encodes to the unknown token")
    decoded = decode_ids(tokenizer, [token_id])
    if decoded != word:
        raise ValueError(f"Token {token_id} decodes to {decoded!r}, expected {word!r}")
    try:
        converted = tokenizer.convert_tokens_to_ids(word)
    except TypeError as exc:
        raise ValueError(f"Tokenizer cannot map {word!r} to an id") from exc
    if converted is None or int(converted) != token_id:
        raise ValueError(f"convert_tokens_to_ids({word!r}) disagrees with encode")
    if unk is not None and int(converted) == int(unk):
        raise ValueError(f"{word!r} is the unknown token")
    return token_id


def verify_yes_no_tokens(tokenizer) -> tuple[int, int]:
    yes_id = verify_answer_token(tokenizer, "yes")
    no_id = verify_answer_token(tokenizer, "no")
    if yes_id == no_id:
        raise ValueError("yes and no resolved to the same token id")
    return yes_id, no_id


def assemble_token_ids(tokenizer, query: str, document: str, *, max_length: int = MAX_LENGTH,
                       instruction: str = DEFAULT_INSTRUCTION) -> list[int]:
    """Truncate the body only, so the assistant suffix stays in the final positions."""
    if type(max_length) is not int or max_length < 8:
        raise ValueError("max_length must be an integer of at least 8")
    prefix_text, body, suffix_text = prompt_parts(query, document, instruction)
    prefix = encode_text(tokenizer, prefix_text)
    suffix = encode_text(tokenizer, suffix_text)
    if not prefix or not suffix:
        raise ValueError("Reranker prefix or suffix tokenized to empty")
    room = max_length - len(prefix) - len(suffix)
    if room < 1 or (max_length >= MAX_LENGTH and room < 32):
        raise ValueError("max_length leaves no room for the business fields")
    ids = prefix + encode_text(tokenizer, body)[:room] + suffix
    if ids[-len(suffix):] != suffix or len(ids) > max_length:
        raise ValueError("Prompt assembly dropped the assistant suffix or exceeded max_length")
    return ids


def pad_left(sequences: list[list[int]], pad_id: int, multiple: int = PAD_MULTIPLE) -> tuple[np.ndarray, np.ndarray]:
    if not sequences or any(len(seq) < 1 for seq in sequences):
        raise ValueError("Every sequence needs at least one token")
    if type(pad_id) is not int or pad_id < 0:
        raise ValueError("pad_id must be a non-negative integer")
    if type(multiple) is not int or multiple < 1:
        raise ValueError("Padding multiple must be a positive integer")
    width = max(len(seq) for seq in sequences)
    width = (width + multiple - 1) // multiple * multiple
    input_ids = np.full((len(sequences), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(sequences), width), dtype=np.int64)
    for row, seq in enumerate(sequences):
        input_ids[row, width - len(seq):] = seq
        attention[row, width - len(seq):] = 1
    if not np.all(attention[:, -1] == 1) or width % multiple != 0:
        raise AssertionError("Left padding failed to keep the final token real")
    return input_ids, attention


def length_batch_positions(lengths: np.ndarray, batch_size: int):
    lengths = np.asarray(lengths)
    if lengths.ndim != 1 or lengths.size == 0 or np.any(lengths <= 0):
        raise ValueError("Token lengths must be positive")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    order = np.argsort(lengths, kind="stable")
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def stable_sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    out = np.empty(scores.shape, dtype=np.float64)
    positive = scores >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
    exp_scores = np.exp(scores[~positive])
    out[~positive] = exp_scores / (1.0 + exp_scores)
    return out


def yes_no_from_logits(logits: np.ndarray, yes_id: int, no_id: int) -> tuple[np.ndarray, np.ndarray]:
    logits = np.asarray(logits)
    if logits.ndim != 2:
        raise ValueError(f"Final-position logits must have shape (batch, vocab), got {logits.shape}")
    if yes_id < 0 or no_id < 0 or yes_id == no_id or yes_id >= logits.shape[1] or no_id >= logits.shape[1]:
        raise ValueError("yes/no token ids are missing from the final-position vocabulary")
    values = np.asarray(logits, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite final-position logits")
    score = values[:, yes_id] - values[:, no_id]
    probability = stable_sigmoid(score)
    return np.ascontiguousarray(score, dtype=np.float32), np.ascontiguousarray(probability, dtype=np.float32)


def score_sequences(sequences: list[list[int]], scorer: _Scorer, *, yes_id: int, no_id: int,
                    pad_id: int, batch_size: int, multiple: int = PAD_MULTIPLE) -> tuple[np.ndarray, np.ndarray]:
    """Length-sort, left-pad, score the final position, then restore input order."""
    if yes_id == no_id:
        raise ValueError("yes and no token ids collide")
    lengths = np.fromiter((len(seq) for seq in sequences), dtype=np.int32, count=len(sequences))
    scores = np.empty(len(sequences), dtype=np.float32)
    probabilities = np.empty(len(sequences), dtype=np.float32)
    for positions in length_batch_positions(lengths, batch_size):
        input_ids, attention = pad_left([sequences[int(pos)] for pos in positions], pad_id, multiple)
        logits = np.asarray(scorer(input_ids, attention))
        if logits.shape[0] != len(positions):
            raise ValueError("Scorer changed the batch size")
        batch_scores, batch_probabilities = yes_no_from_logits(logits, yes_id, no_id)
        scores[positions] = batch_scores
        probabilities[positions] = batch_probabilities
    if scores.dtype != np.float32 or probabilities.dtype != np.float32:
        raise AssertionError("Scores must stay float32")
    return scores, probabilities


def pair_rows(frame: pd.DataFrame) -> np.ndarray:
    if "pair_row" not in frame.columns:
        raise ValueError("Pairs need an immutable pair_row column")
    rows = np.asarray(frame["pair_row"])
    if rows.dtype.kind not in "iu":
        coerced = np.asarray(rows, dtype=np.int64)
        if not np.array_equal(rows, coerced):
            raise ValueError("pair_row must be integers")
        rows = coerced
    rows = np.ascontiguousarray(rows, dtype=np.int64)
    count = len(frame)
    if count == 0 or rows.shape != (count,) or int(rows.min()) != 0 or int(rows.max()) != count - 1:
        raise ValueError("pair_row must be a permutation of 0..n-1")
    if len(np.unique(rows)) != count:
        raise ValueError("pair_row values must be unique")
    return rows


def _text_column(frame: pd.DataFrame, column: str) -> list[str]:
    if column not in frame.columns:
        raise ValueError(f"Missing business field column: {column}")
    return frame[column].fillna("").astype(str).tolist()


def pair_field_texts(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Render name and address only. Ids, country, and labels are not read."""
    left_names = _text_column(frame, "left_business_name")
    left_addresses = _text_column(frame, "left_business_address")
    right_names = _text_column(frame, "right_business_name")
    right_addresses = _text_column(frame, "right_business_address")
    queries = [field_tagged_business(name, address) for name, address in zip(left_names, left_addresses)]
    documents = [field_tagged_business(name, address) for name, address in zip(right_names, right_addresses)]
    return queries, documents


def score_pair_frame(frame: pd.DataFrame, tokenizer, scorer: _Scorer, *, yes_id: int, no_id: int,
                     pad_id: int, batch_size: int, max_length: int = MAX_LENGTH,
                     multiple: int = PAD_MULTIPLE,
                     instruction: str = DEFAULT_INSTRUCTION) -> tuple[np.ndarray, np.ndarray]:
    rows = pair_rows(frame)
    queries, documents = pair_field_texts(frame)
    sequences = [assemble_token_ids(tokenizer, query, document, max_length=max_length, instruction=instruction)
                 for query, document in zip(queries, documents)]
    scores, probabilities = score_sequences(
        sequences, scorer, yes_id=yes_id, no_id=no_id, pad_id=pad_id,
        batch_size=batch_size, multiple=multiple)
    ordered_scores = np.empty(len(frame), dtype=np.float32)
    ordered_probabilities = np.empty(len(frame), dtype=np.float32)
    ordered_scores[rows] = scores
    ordered_probabilities[rows] = probabilities
    return ordered_scores, ordered_probabilities


def alignment_text_sha256(frame: pd.DataFrame) -> str:
    rows = pair_rows(frame)
    queries, documents = pair_field_texts(frame)
    ordered_queries = [""] * len(frame)
    ordered_documents = [""] * len(frame)
    for position, row in enumerate(rows):
        ordered_queries[int(row)] = queries[position]
        ordered_documents[int(row)] = documents[position]
    payload = "\n".join(f"{query}\t{document}" for query, document in zip(ordered_queries, ordered_documents))
    return _sha256_bytes(payload.encode())


def raw_stake_weight(label: int, true_count: int) -> float:
    """Entity-value stake from the C2 recipe.

    Positive: ``1/(5T-4)`` for ``T >= 1``. Negative: ``4/(5T+4)`` for ``T >= 1``,
    and ``1`` for a no-match business (``T == 0``).
    """
    label = int(label)
    true_count = int(true_count)
    if true_count < 0 or label not in (0, 1):
        raise ValueError("Stake weight needs a 0/1 label and a non-negative true count")
    if label == 1:
        if true_count < 1:
            raise ValueError("A positive pair requires true count T >= 1")
        return 1.0 / (5 * true_count - 4)
    if true_count == 0:
        return 1.0
    return 4.0 / (5 * true_count + 4)


def class_normalize_weights(raw: np.ndarray, labels: np.ndarray, cap: float = STAKE_CAP) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if (raw.ndim != 1 or raw.shape != labels.shape or not np.isfinite(raw).all() or np.any(raw < 0)
            or not np.all((labels == 0) | (labels == 1))):
        raise ValueError("Stake weights must be finite, non-negative, and aligned with 0/1 labels")
    if cap < 1:
        raise ValueError("Stake cap must be at least 1")
    normalized = raw.copy()
    for cls in (0, 1):
        mask = labels == cls
        if not np.any(mask):
            raise ValueError("Class normalization needs both positives and negatives")
        mean = float(normalized[mask].mean())
        if mean <= 0:
            raise ValueError("Class stake mean must be positive")
        normalized[mask] /= mean
    return np.minimum(normalized, cap)


def weighted_bce_with_logits(scores: np.ndarray, labels: np.ndarray, weights: np.ndarray,
                             normalizer: float | None = None) -> float:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if scores.ndim != 1 or scores.shape != labels.shape or scores.shape != weights.shape:
        raise ValueError("Loss inputs must be aligned 1-D arrays")
    if not np.isfinite(scores).all() or not np.isfinite(weights).all() or np.any((labels != 0) & (labels != 1)):
        raise ValueError("Loss inputs must be finite 0/1 labels with finite weights")
    per_example = np.maximum(scores, 0) - scores * labels + np.log1p(np.exp(-np.abs(scores)))
    denom = float(weights.sum()) if normalizer is None else float(normalizer)
    if not math.isfinite(denom) or denom <= 0:
        raise ValueError("Loss weight normalizer must be positive")
    return float((weights * per_example).sum() / denom)


def screen_learning_rate(step: int, total_steps: int, base: float = DEFAULT_LR,
                         warmup_fraction: float = WARMUP_FRACTION) -> float:
    if type(step) is not int or type(total_steps) is not int or step < 1 or total_steps < 1 or step > total_steps:
        raise ValueError("Learning-rate step is outside the bounded screen")
    if base <= 0 or not 0 < warmup_fraction < 1:
        raise ValueError("Learning rate and warmup fraction must be positive")
    warmup = max(1, int(round(warmup_fraction * total_steps)))
    if step <= warmup:
        return base * step / warmup
    return base


class ArenaSampler:
    """1 positive : 3 negatives. Half of the negative slots are typed, half random-gated."""

    def __init__(self, positive: np.ndarray, typed: Mapping[str, np.ndarray], random_idx: np.ndarray,
                 seed: int, *, epoch: int = 0, offset: int = 0, rng: np.random.Generator | None = None):
        self.positive = np.ascontiguousarray(positive, dtype=np.int64)
        self.typed = {name: np.ascontiguousarray(typed[name], dtype=np.int64) for name in TYPED_NEGATIVE_TYPES}
        self.random_idx = np.ascontiguousarray(random_idx, dtype=np.int64)
        if self.positive.size < 1 or self.random_idx.size < 1 or any(self.typed[name].size < 1 for name in TYPED_NEGATIVE_TYPES):
            raise ValueError("Arena-A sampling needs positives, all five typed pools, and random-gated negatives")
        if type(seed) is not int or seed < 0 or epoch < 0 or not 0 <= offset <= self.positive.size:
            raise ValueError("Invalid sampler seed or cursor")
        self.seed = seed
        self.epoch = epoch
        self.offset = offset
        self.rng = np.random.default_rng(seed) if rng is None else rng
        self._order = self._permutation()

    def _permutation(self) -> np.ndarray:
        return np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch])).permutation(self.positive.size)

    def _take_positive(self, count: int) -> np.ndarray:
        chunks = []
        remaining = count
        while remaining:
            if self.offset == self.positive.size:
                self.epoch += 1
                self.offset = 0
                self._order = self._permutation()
            take = min(remaining, self.positive.size - self.offset)
            chunks.append(self.positive[self._order[self.offset:self.offset + take]])
            self.offset += take
            remaining -= take
        return np.concatenate(chunks)

    def _take_typed(self, count: int) -> np.ndarray:
        chosen = np.empty(count, dtype=np.int64)
        for index in range(count):
            pool = self.typed[TYPED_NEGATIVE_TYPES[index % len(TYPED_NEGATIVE_TYPES)]]
            chosen[index] = pool[int(self.rng.integers(pool.size))]
        return chosen

    def take(self, n_positive: int) -> np.ndarray:
        if type(n_positive) is not int or n_positive < 1:
            raise ValueError("Positive count per step must be a positive integer")
        n_negative = n_positive * NEGATIVE_RATIO
        n_typed = n_negative // 2
        n_random = n_negative - n_typed
        positive = self._take_positive(n_positive)
        typed = self._take_typed(n_typed)
        random_ids = self.rng.choice(self.random_idx, size=n_random, replace=self.random_idx.size < n_random)
        mixed = np.concatenate([positive, typed, np.asarray(random_ids, dtype=np.int64)])
        return mixed[self.rng.permutation(mixed.size)]

    def state(self) -> dict:
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "offset": self.offset,
            "positive_count": int(self.positive.size),
            "typed_counts": {name: int(self.typed[name].size) for name in TYPED_NEGATIVE_TYPES},
            "random_count": int(self.random_idx.size),
            "rng_state": _jsonify(self.rng.bit_generator.state),
        }

    @classmethod
    def restore(cls, positive: np.ndarray, typed: Mapping[str, np.ndarray], random_idx: np.ndarray,
                state: Mapping) -> "ArenaSampler":
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        sampler = cls(positive, typed, random_idx, int(state["seed"]), epoch=int(state["epoch"]),
                      offset=int(state["offset"]), rng=rng)
        if (sampler.positive.size != int(state["positive_count"])
                or sampler.random_idx.size != int(state["random_count"])
                or any(sampler.typed[name].size != int(state["typed_counts"][name]) for name in TYPED_NEGATIVE_TYPES)):
            raise ValueError("Checkpoint sampler pools do not match these arena-A pairs")
        return sampler


def training_index_pools(frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    if "partition" not in frame.columns:
        raise ValueError("Training pairs need partition")
    partition = frame["partition"].fillna("").astype(str)
    if not partition.eq("A").all():
        raise ValueError("Training reads arena-A pairs only")
    labels = np.asarray(frame["label"], dtype=np.int64)
    true_count = np.asarray(frame["true_count"], dtype=np.int64)
    kinds = frame["negative_type"].fillna("").astype(str).to_numpy()
    if labels.shape != (len(frame),) or np.any((labels != 0) & (labels != 1)) or np.any(true_count < 0):
        raise ValueError("Training labels must be 0/1 and true_count must be non-negative")
    positive = np.flatnonzero(labels == 1)
    negative = np.flatnonzero(labels == 0)
    if positive.size < 1 or negative.size < 1 or np.any(true_count[positive] < 1):
        raise ValueError("Arena-A training needs positives with T >= 1 and negatives")
    if np.any(~np.isin(kinds[positive], ("", "positive"))):
        raise ValueError("Positive rows cannot carry a negative type")
    if np.any(~np.isin(kinds[negative], NEGATIVE_TYPES)):
        raise ValueError("Negative rows must use a typed or random-gated negative_type")
    typed = {name: negative[kinds[negative] == name] for name in TYPED_NEGATIVE_TYPES}
    random_idx = negative[kinds[negative] == RANDOM_NEGATIVE_TYPE]
    missing = [name for name, pool in typed.items() if pool.size < 1]
    if missing or random_idx.size < 1:
        raise ValueError(f"Missing typed or random-gated negatives: {missing or [RANDOM_NEGATIVE_TYPE]}")
    return positive, typed, random_idx


def _checkpoint_files(directory: Path) -> dict[str, str]:
    files = {}
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.name != "complete.json":
            files[path.relative_to(directory).as_posix()] = _sha256_file(path)
    return files


def verify_checkpoint(directory: Path) -> dict:
    marker_path = directory / "complete.json"
    if not marker_path.is_file():
        raise ValueError("Checkpoint has no completed-step marker")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    files = _checkpoint_files(directory)
    recorded = marker.get("files_sha256")
    if (marker.get("state") != "complete" or not isinstance(recorded, dict) or files != recorded
            or int(marker.get("step", -1)) < 1):
        raise ValueError("Checkpoint is not complete or one of its files changed")
    state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
    if int(state.get("completed_steps", -1)) != int(marker["step"]) or state.get("config_sha256") != marker.get("config_sha256"):
        raise ValueError("Completed-step marker disagrees with checkpoint state")
    return marker


def save_checkpoint(root: Path, *, step: int, config: Mapping, config_sha: str, sampler_state: Mapping,
                    backend, train_seconds: float, last_loss: float) -> Path:
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint step must be positive")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"step_{step:06d}"
    stage = root / f".step_{step:06d}.incomplete"
    if destination.exists() or stage.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint step {step}")
    stage.mkdir()
    backend.save(stage)
    json_write(stage / "training_config.json", dict(config))
    state = {
        "completed_steps": step,
        "config_sha256": config_sha,
        "sampler": _jsonify(sampler_state),
        "train_seconds": train_seconds,
        "last_loss": last_loss,
    }
    json_write(stage / "state.json", state)
    marker = {"state": "complete", "step": step, "config_sha256": config_sha, "files_sha256": _checkpoint_files(stage)}
    json_write(stage / "complete.json", marker)
    stage.replace(destination)
    verify_checkpoint(destination)
    return destination


def load_checkpoint(directory: Path, backend, config_sha: str) -> dict:
    marker = verify_checkpoint(directory)
    if marker["config_sha256"] != config_sha:
        raise ValueError("Resume checkpoint configuration does not match this screen")
    backend.load(directory)
    return json.loads((directory / "state.json").read_text(encoding="utf-8"))


def _frame_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.asarray(frame["label"], dtype=np.int64)
    true_count = np.asarray(frame["true_count"], dtype=np.int64)
    return labels, true_count, pair_rows(frame)


def training_config(frame: pd.DataFrame, *, steps: int, positives_per_step: int, grad_accum: int,
                    seed: int, max_length: int, learning_rate: float, weight_decay: float,
                    checkpoint_every: int, input_sha256: str | None) -> dict:
    rows = pair_rows(frame)
    labels, true_count, _ = _frame_arrays(frame)
    return {
        "schema": 1,
        "model_id": MODEL_ID,
        "license": EXPECTED_LICENSE,
        "params_b": EXPECTED_PARAMS_B,
        "score": "logit(yes)-logit(no)",
        "probability": "sigmoid(score)",
        "instruction": DEFAULT_INSTRUCTION,
        "text": "canonical field-tagged name and address; no ids or label-derived fields",
        "partition": "A",
        "negative_ratio": NEGATIVE_RATIO,
        "typed_negative_share": 0.5,
        "stake": "positive 1/(5T-4); negative 4/(5T+4) or 1 if T=0; class-mean normalized; cap 5",
        "steps": steps,
        "positives_per_step": positives_per_step,
        "grad_accum": grad_accum,
        "effective_batch": positives_per_step * (1 + NEGATIVE_RATIO),
        "max_length": max_length,
        "pad_to_multiple_of": PAD_MULTIPLE,
        "padding_side": "left",
        "seed": seed,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_fraction": WARMUP_FRACTION,
        "schedule": "linear warmup then constant",
        "checkpoint_every": checkpoint_every,
        "pair_count": len(frame),
        "pair_row_sha256": _sha256_bytes(rows.tobytes()),
        "label_sha256": _sha256_bytes(np.ascontiguousarray(labels).tobytes()),
        "true_count_sha256": _sha256_bytes(np.ascontiguousarray(true_count).tobytes()),
        "alignment_text_sha256": alignment_text_sha256(frame),
        "input_sha256": input_sha256,
        "holdout2_labels_read": False,
    }


def run_training(frame: pd.DataFrame, tokenizer, backend, *, steps: int, positives_per_step: int,
                 grad_accum: int, seed: int, max_length: int = MAX_LENGTH,
                 learning_rate: float = DEFAULT_LR, weight_decay: float = DEFAULT_WEIGHT_DECAY,
                 checkpoint_every: int, checkpoint_root: Path, config: Mapping, resume: Path | None = None) -> dict:
    """Bounded arena-A screen loop. ``backend`` owns the model and optimizer."""
    config = dict(config)
    config_sha = _sha256_bytes(_canonical_json(config).encode())
    positive, typed, random_idx = training_index_pools(frame)
    labels, true_count, _ = _frame_arrays(frame)
    queries, documents = pair_field_texts(frame)
    yes_id, no_id = verify_yes_no_tokens(tokenizer)
    if resume is None:
        sampler = ArenaSampler(positive, typed, random_idx, seed)
        start = 0
        backend.prepare(total_steps=steps, learning_rate=learning_rate, weight_decay=weight_decay)
    else:
        state = load_checkpoint(resume, backend, config_sha)
        sampler = ArenaSampler.restore(positive, typed, random_idx, state["sampler"])
        start = int(state["completed_steps"])
        if start > steps:
            raise ValueError("Checkpoint is already past the requested screen steps")
    step_size = positives_per_step * (1 + NEGATIVE_RATIO)
    if step_size % grad_accum != 0:
        raise ValueError("Gradient accumulation must divide the 1:3 step")
    micro = step_size // grad_accum
    started = time.perf_counter()
    last_loss = None
    checkpoint = None
    for step in range(start + 1, steps + 1):
        indices = sampler.take(positives_per_step)
        step_labels = labels[indices]
        raw = np.array([raw_stake_weight(int(label), int(count))
                        for label, count in zip(step_labels, true_count[indices])], dtype=np.float64)
        weights = class_normalize_weights(raw, step_labels)
        normalizer = float(weights.sum())
        loss_sum = 0.0
        for micro_index in range(grad_accum):
            chosen = indices[micro_index * micro:(micro_index + 1) * micro]
            sequences = [assemble_token_ids(tokenizer, queries[int(i)], documents[int(i)], max_length=max_length)
                         for i in chosen]
            input_ids, attention = pad_left(sequences, int(tokenizer.pad_token_id), PAD_MULTIPLE)
            loss_sum += float(backend.micro_backward(
                input_ids, attention, step_labels[micro_index * micro:(micro_index + 1) * micro],
                weights[micro_index * micro:(micro_index + 1) * micro], normalizer, yes_id, no_id))
        backend.optimizer_step(step, steps)
        last_loss = loss_sum
        if step % checkpoint_every == 0 or step == steps:
            checkpoint = save_checkpoint(
                checkpoint_root, step=step, config=config, config_sha=config_sha, sampler_state=sampler.state(),
                backend=backend, train_seconds=time.perf_counter() - started, last_loss=last_loss)
    return {"completed_steps": steps, "config_sha256": config_sha, "last_loss": last_loss,
            "checkpoint": None if checkpoint is None else str(checkpoint), "yes_token_id": yes_id,
            "no_token_id": no_id, "holdout2_labels_read": False}


def verify_allowlist(allowlist: Path) -> dict:
    refuse_holdout_path(allowlist)
    import yaml

    document = yaml.safe_load(Path(allowlist).read_text(encoding="utf-8"))
    models = {row["id"]: row for row in document.get("models", [])}
    row = models.get(MODEL_ID)
    budget = float(document.get("budget_total_params_b", 0))
    if row is None or row.get("role") != "reranker":
        raise ValueError(f"{MODEL_ID} is absent from the reranker allowlist")
    if row.get("licence") != EXPECTED_LICENSE:
        raise ValueError(f"{MODEL_ID} license is not Apache-2.0")
    params = float(row.get("params_b", -1))
    if abs(params - EXPECTED_PARAMS_B) > 1e-12 or params > budget:
        raise ValueError(f"{MODEL_ID} parameter count is not the allowlisted {EXPECTED_PARAMS_B}B within the 8B budget")
    return {"model_id": MODEL_ID, "license": EXPECTED_LICENSE, "params_b": EXPECTED_PARAMS_B,
            "role": "reranker", "allowlist_sha256": _sha256_file(Path(allowlist)),
            "budget_total_params_b": budget}


def safetensors_parameter_count(path: Path) -> int:
    with path.open("rb") as stream:
        header_len = int.from_bytes(stream.read(8), "little")
        header = json.loads(stream.read(header_len))
    total = 0
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        shape = meta["shape"]
        count = 1
        for dim in shape:
            count *= int(dim)
        total += count
    return total


def assert_parameter_count(count: int, expected_b: float = EXPECTED_PARAMS_B) -> None:
    actual_b = count / 1e9
    if abs(actual_b - expected_b) > PARAM_TOLERANCE_B:
        raise ValueError(
            f"Cached weight parameter count {actual_b:.6f}B is not {expected_b:.3f}B ± {PARAM_TOLERANCE_B}")


def snapshot_license_status(snapshot: Path) -> str:
    chunks = []
    for name in ("LICENSE", "LICENSE.txt", "README.md"):
        path = snapshot / name
        if path.is_file():
            chunks.append(path.read_text(encoding="utf-8", errors="replace")[:8000])
    if not chunks:
        return "allowlist-only"
    blob = "\n".join(chunks).lower()
    if "apache" not in blob or any(banned in blob for banned in ("cc-by-nc", "non-commercial", "llama license", "gpl-3.0")):
        raise ValueError("Cached model license file is not Apache-2.0")
    return "apache-2.0"


def inspect_snapshot(snapshot: Path) -> dict:
    snapshot = Path(snapshot)
    config_path = snapshot / "config.json"
    if not config_path.is_file():
        raise ValueError("Cached snapshot has no config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    architectures = list(config.get("architectures") or [])
    if config.get("model_type") != "qwen3" or "Qwen3ForCausalLM" not in architectures:
        raise ValueError("Cached model is not a Qwen3 causal LM; yes/no logits would be undefined")
    if any("SequenceClassification" in name for name in architectures):
        raise ValueError("Refusing a sequence-classification checkpoint for yes/no scoring")
    name = str(config.get("_name_or_path") or "")
    if "Qwen3-Reranker" in name and "0.6B" not in name:
        raise ValueError(f"Cached model identity is {name}, not {MODEL_ID}")
    license_status = snapshot_license_status(snapshot)
    weights = sorted(snapshot.glob("*.safetensors"))
    if not weights:
        raise ValueError("Cached snapshot has no safetensors weights")
    count = sum(safetensors_parameter_count(path) for path in weights)
    assert_parameter_count(count)
    return {
        "snapshot": str(snapshot.resolve()),
        "license_file": license_status,
        "parameter_count": count,
        "params_b_counted": count / 1e9,
        "config_sha256": _sha256_file(config_path),
        "weights_sha256": {path.name: _sha256_file(path) for path in weights},
        "model_type": config.get("model_type"),
        "architectures": architectures,
    }


def cached_snapshot(allowlist: Path) -> tuple[Path, dict]:
    """Resolve the local snapshot. Never downloads."""
    identity = verify_allowlist(allowlist)
    previous = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise FileNotFoundError(
                f"{MODEL_ID} is not available locally (huggingface_hub is not installed); refusing to download"
            ) from exc
        try:
            snapshot = Path(snapshot_download(repo_id=MODEL_ID, local_files_only=True))
        except Exception as exc:
            raise FileNotFoundError(
                f"{MODEL_ID} is not in the local Hugging Face cache; refusing to download"
            ) from exc
    finally:
        if previous is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous
    return snapshot, {**identity, **inspect_snapshot(snapshot)}


def _save_array(path: Path, values: np.ndarray) -> None:
    temporary = Path(str(path) + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, np.ascontiguousarray(values, dtype=np.float32))
    temporary.replace(path)


def write_score_outputs(output: Path, *, scores: np.ndarray, probabilities: np.ndarray, frame: pd.DataFrame,
                        identity: Mapping, yes_id: int, no_id: int, input_path: Path, batch_size: int,
                        max_length: int = MAX_LENGTH) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Choose a new empty score output directory")
    output.mkdir(parents=True, exist_ok=True)
    rows = pair_rows(frame)
    if scores.shape != (len(frame),) or probabilities.shape != scores.shape:
        raise ValueError("Score arrays must align with pair_row")
    if scores.dtype != np.float32 or probabilities.dtype != np.float32:
        raise ValueError("Score arrays must be float32")
    if not np.isfinite(scores).all() or not np.isfinite(probabilities).all():
        raise ValueError("Non-finite scores")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Sigmoid probabilities must lie in [0, 1]")
    _save_array(output / "scores.npy", scores)
    _save_array(output / "probabilities.npy", probabilities)
    manifest = {
        "schema": 1,
        "model_id": identity["model_id"],
        "license": identity["license"],
        "params_b": identity["params_b"],
        "snapshot": identity.get("snapshot"),
        "config_sha256": identity.get("config_sha256"),
        "weights_sha256": identity.get("weights_sha256"),
        "parameter_count": identity.get("parameter_count"),
        "allowlist_sha256": identity.get("allowlist_sha256"),
        "yes_token_id": yes_id,
        "no_token_id": no_id,
        "score_definition": "logit(yes)-logit(no) at the final real token",
        "probability_definition": "stable sigmoid(score)",
        "dtype": "float32",
        "max_length": max_length,
        "pad_to_multiple_of": PAD_MULTIPLE,
        "padding_side": "left",
        "length_sorted": True,
        "batch_size": batch_size,
        "pair_count": len(frame),
        "alignment": "scores[i] and probabilities[i] belong to pair_row == i",
        "pair_row_sha256": _sha256_bytes(rows.tobytes()),
        "alignment_text_sha256": alignment_text_sha256(frame),
        "input_path": str(Path(input_path).resolve()),
        "input_sha256": _sha256_file(Path(input_path)),
        "scores_sha256": _sha256_file(output / "scores.npy"),
        "probabilities_sha256": _sha256_file(output / "probabilities.npy"),
        "scores_bytes_sha256": _sha256_bytes(np.ascontiguousarray(scores, dtype=np.float32).tobytes()),
        "labels_read": False,
        "holdout2_labels_read": False,
        "official_f05": None,
    }
    json_write(output / "manifest.json", manifest)
    return manifest


def load_pair_frame(path: Path, *, training: bool) -> pd.DataFrame:
    refuse_holdout_path(path)
    frame = pd.read_parquet(path)
    pair_rows(frame)
    pair_field_texts(frame)
    if training:
        training_index_pools(frame)
    return frame


def validate_screen_args(args) -> None:
    for path in (args.pairs, args.output, args.checkpoint_root, args.resume, args.allowlist):
        if path is not None:
            refuse_holdout_path(path)
    if type(args.steps) is not int or not 1 <= args.steps <= SCREEN_MAX_STEPS:
        raise ValueError(f"Screen steps must be an integer from 1 to {SCREEN_MAX_STEPS}")
    if args.max_length != MAX_LENGTH:
        raise ValueError("Qwen screen max_length is fixed at 256")
    if type(args.positives_per_step) is not int or args.positives_per_step < 1:
        raise ValueError("positives-per-step must be a positive integer")
    if type(args.grad_accum) is not int or args.grad_accum < 1:
        raise ValueError("grad-accum must be a positive integer")
    effective = args.positives_per_step * (1 + NEGATIVE_RATIO)
    if effective > SCREEN_MAX_EFFECTIVE_BATCH or effective % args.grad_accum != 0:
        raise ValueError("Effective 1:3 batch must be divisible by grad-accum and at most 128")
    if type(args.checkpoint_every) is not int or args.checkpoint_every < 1:
        raise ValueError("checkpoint-every must be a positive integer")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if type(args.seed) is not int or args.seed < 0:
        raise ValueError("Seed must be a non-negative integer")
    if not str(args.device).startswith("cuda"):
        raise ValueError("The bounded screen trains on CUDA; this command does not launch a CPU substitute")


def validate_score_args(args) -> None:
    refuse_holdout_path(args.pairs)
    refuse_holdout_path(args.output)
    refuse_holdout_path(args.allowlist)
    if args.max_length != MAX_LENGTH:
        raise ValueError("Qwen scoring max_length is fixed at 256")
    if type(args.batch_size) is not int or not 1 <= args.batch_size <= 128:
        raise ValueError("Score batch size must be an integer from 1 to 128")
    if not str(args.device).startswith("cuda"):
        raise ValueError("BF16 Qwen scoring requires CUDA; refusing a silent FP32 score")


class TorchTrainBackend:
    """CUDA trainer. Constructed only by the screen command."""

    def __init__(self, model, optimizer, device: str):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.base_lr = DEFAULT_LR

    def prepare(self, *, total_steps: int, learning_rate: float, weight_decay: float) -> None:
        del total_steps, weight_decay
        self.base_lr = learning_rate
        self.optimizer.zero_grad(set_to_none=True)

    def micro_backward(self, input_ids: np.ndarray, attention: np.ndarray, labels: np.ndarray,
                       weights: np.ndarray, normalizer: float, yes_id: int, no_id: int) -> float:
        import torch
        from torch.nn import functional as F

        batch = {
            "input_ids": torch.from_numpy(np.ascontiguousarray(input_ids)).to(self.device),
            "attention_mask": torch.from_numpy(np.ascontiguousarray(attention)).to(self.device),
        }
        target = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32)).to(self.device)
        weight = torch.from_numpy(np.ascontiguousarray(weights, dtype=np.float32)).to(self.device)
        self.model.train()
        autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        with autocast:
            try:
                logits = self.model(**batch, logits_to_keep=1).logits
            except TypeError:
                logits = self.model(**batch).logits
        if logits.ndim != 3:
            raise ValueError("Causal LM logits must have shape (batch, positions, vocab)")
        final = logits[:, -1, :].float()
        score = final[:, yes_id] - final[:, no_id]
        loss = F.binary_cross_entropy_with_logits(score, target, weight=weight, reduction="sum") / float(normalizer)
        loss.backward()
        return float(loss.detach().cpu())

    def optimizer_step(self, step: int, total_steps: int) -> None:
        import torch

        lr = screen_learning_rate(step, total_steps, self.base_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

    def save(self, directory: Path) -> None:
        import torch

        model_dir = directory / "model"
        model_dir.mkdir()
        self.model.save_pretrained(model_dir, safe_serialization=True)
        torch.save({
            "optimizer": self.optimizer.state_dict(),
            "torch_cpu_rng": torch.get_rng_state(),
            "torch_cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }, directory / "optimizer.pt")

    def load(self, directory: Path) -> None:
        import torch
        from safetensors.torch import load_file

        saved = torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["torch_cpu_rng"])
        if saved["torch_cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved["torch_cuda_rng"])
        files = sorted((directory / "model").glob("*.safetensors"))
        if not files:
            raise ValueError("Checkpoint is missing causal-LM weights")
        state = {}
        for path in files:
            state.update(load_file(path, device="cpu"))
        self.model.load_state_dict(state, strict=True)


def load_causal_lm(snapshot: Path, device: str):
    """Load the cached causal LM in BF16 on CUDA. Does not download."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not str(device).startswith("cuda"):
        raise ValueError("BF16 Qwen scoring requires CUDA; refusing a silent FP32 score")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        if getattr(tokenizer, "eos_token", None) is None:
            raise ValueError("Tokenizer has no pad or eos token")
        tokenizer.pad_token = tokenizer.eos_token
    verify_yes_no_tokens(tokenizer)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16)
    return model.to(device), tokenizer


def score_cuda(model, tokenizer, frame: pd.DataFrame, *, device: str, batch_size: int,
               max_length: int = MAX_LENGTH) -> tuple[np.ndarray, np.ndarray, int, int]:
    import torch

    if not str(device).startswith("cuda"):
        raise ValueError("BF16 Qwen scoring requires CUDA; refusing a silent FP32 score")
    yes_id, no_id = verify_yes_no_tokens(tokenizer)
    pad_id = int(tokenizer.pad_token_id)

    def scorer(input_ids: np.ndarray, attention: np.ndarray) -> np.ndarray:
        batch = {
            "input_ids": torch.from_numpy(np.ascontiguousarray(input_ids)).to(device),
            "attention_mask": torch.from_numpy(np.ascontiguousarray(attention)).to(device),
        }
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            try:
                logits = model(**batch, logits_to_keep=1).logits
            except TypeError:
                logits = model(**batch).logits
        if logits.ndim == 3:
            final = logits[:, -1, :]
        elif logits.ndim == 2:
            final = logits
        else:
            raise ValueError(f"Unexpected logit rank {logits.ndim}")
        return final.float().detach().cpu().numpy()

    model.eval()
    scores, probabilities = score_pair_frame(
        frame, tokenizer, scorer, yes_id=yes_id, no_id=no_id, pad_id=pad_id,
        batch_size=batch_size, max_length=max_length)
    return scores, probabilities, yes_id, no_id


def default_allowlist() -> Path:
    return Path(__file__).resolve().parents[4] / "problem-statement" / "models.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen3-Reranker-0.6B yes/no scoring adapter")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Label-free BF16 scores aligned to pair_row")
    score.add_argument("--pairs", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.add_argument("--allowlist", type=Path, default=default_allowlist())
    score.add_argument("--batch-size", type=int, default=32)
    score.add_argument("--max-length", type=int, default=MAX_LENGTH)
    score.add_argument("--device", default="cuda:0")
    screen = commands.add_parser("screen", help="Bounded arena-A fine-tune")
    screen.add_argument("--pairs", type=Path, required=True)
    screen.add_argument("--output", type=Path, required=True)
    screen.add_argument("--checkpoint-root", type=Path)
    screen.add_argument("--allowlist", type=Path, default=default_allowlist())
    screen.add_argument("--steps", type=int, required=True)
    screen.add_argument("--positives-per-step", type=int, default=16)
    screen.add_argument("--grad-accum", type=int, default=16)
    screen.add_argument("--seed", type=int, default=DEFAULT_SEED)
    screen.add_argument("--max-length", type=int, default=MAX_LENGTH)
    screen.add_argument("--checkpoint-every", type=int, default=2000)
    screen.add_argument("--resume", type=Path)
    screen.add_argument("--device", default="cuda:0")
    screen.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    screen.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    return parser


def run_score_command(args) -> dict:
    validate_score_args(args)
    frame = load_pair_frame(args.pairs, training=False)
    snapshot, identity = cached_snapshot(args.allowlist)
    model, tokenizer = load_causal_lm(snapshot, args.device)
    scores, probabilities, yes_id, no_id = score_cuda(
        model, tokenizer, frame, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
    return write_score_outputs(
        args.output, scores=scores, probabilities=probabilities, frame=frame, identity=identity,
        yes_id=yes_id, no_id=no_id, input_path=args.pairs, batch_size=args.batch_size,
        max_length=args.max_length)


def run_screen_command(args) -> dict:
    if args.checkpoint_root is None:
        args.checkpoint_root = Path(args.output) / "checkpoints"
    validate_screen_args(args)
    frame = load_pair_frame(args.pairs, training=True)
    snapshot, identity = cached_snapshot(args.allowlist)
    model, tokenizer = load_causal_lm(snapshot, args.device)
    import torch

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay, fused=True)
    config = training_config(
        frame, steps=args.steps, positives_per_step=args.positives_per_step, grad_accum=args.grad_accum,
        seed=args.seed, max_length=args.max_length, learning_rate=args.learning_rate,
        weight_decay=args.weight_decay, checkpoint_every=args.checkpoint_every,
        input_sha256=_sha256_file(Path(args.pairs)))
    config["optimizer_fused"] = True
    config["snapshot"] = identity["snapshot"]
    config["weights_sha256"] = identity["weights_sha256"]
    config["parameter_count"] = identity["parameter_count"]
    config["estimates"] = resource_estimates()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    json_write(Path(args.output) / "screen_config.json", config)
    backend = TorchTrainBackend(model, optimizer, args.device)
    result = run_training(
        frame, tokenizer, backend, steps=args.steps, positives_per_step=args.positives_per_step,
        grad_accum=args.grad_accum, seed=args.seed, max_length=args.max_length,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        checkpoint_every=args.checkpoint_every, checkpoint_root=Path(args.checkpoint_root),
        config=config, resume=args.resume)
    json_write(Path(args.output) / "screen_result.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "score":
        run_score_command(args)
    elif args.command == "screen":
        run_screen_command(args)
    else:
        raise ValueError(f"Unknown command {args.command}")


if __name__ == "__main__":
    main()
