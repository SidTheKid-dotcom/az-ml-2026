"""CPU inputs for the Qwen3-Reranker second model family (arena A training, B_REST/test scoring).

`train`: the exact C2 sampling world. Typed-negative flags come from
`fast_trainer.classify_negative_types` on partition-A rows (cached to disk), and
batches are drawn by `fast_trainer.ArenaBatchSampler` with C2's seed, batch 128
and 1:3 ratio, so rows 128*(s-1)..128*s-1 of the parquet are C2 step s. Stake
weights are C2's per-step `stake_weights`. Only partition-A labels are read
(record owners are used for contest/unowned typing exactly as C2 does).

`brest` / `test`: label-free tree-gated pairs (tree >= 0.01) in source pair
order; `arena_row` is the row in the source pair order, so scores map back to
the same rows as C2's full-length scores.npy. SLICE and HOLDOUT-2 are not read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .fast_trainer import ArenaBatchSampler, NegativeType, classify_negative_types, stake_weights

C2_SEED = 20260925
C2_BATCH = 128
A, B_REST = 0, 1
TEXT = ("business_name", "business_address")
KIND_NAMES = {NegativeType.FALSE_ACCEPT: "false_accept", NegativeType.CONTEST_LOSER: "contest_loser",
              NegativeType.UNOWNED_LOOKALIKE: "unowned_lookalike",
              NegativeType.DIGIT_SUBSTITUTION: "digit_substitution",
              NegativeType.EMPTY_ADDRESS_SIBLING: "empty_address_sibling"}


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _json(path: Path, payload: dict) -> None:
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str) + "\n")
    tmp.replace(path)


def _pairs(path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(path, columns=["left_idx", "right_idx"])
    return (table.column("left_idx").to_numpy().astype(np.int64),
            table.column("right_idx").to_numpy().astype(np.int64))


def negative_type_names(labels: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """positive / random_gated / lowest typed flag (the flags column keeps every bit)."""
    names = np.full(len(labels), "random_gated", dtype=object)
    for kind in sorted(KIND_NAMES, key=int, reverse=True):
        names[(flags & int(kind)) != 0] = KIND_NAMES[kind]
    names[np.asarray(labels, dtype=bool)] = "positive"
    return names


def draw_c2_stream(sampler: ArenaBatchSampler, steps: int, left: np.ndarray, true_count: np.ndarray,
                   digest_at: int = 40_000) -> dict:
    """Replay C2's sampler: per step the pair ids, labels and C2 stake weights."""
    ids, labels, weights, step_no = [], [], [], []
    digest, snapshot = hashlib.sha256(), None
    for step in range(1, steps + 1):
        pair_ids, targets = sampler.take()
        counts = true_count[left[pair_ids]]
        ids.append(pair_ids)
        labels.append(targets)
        weights.append(stake_weights(targets.astype(bool), counts))
        step_no.append(np.full(len(pair_ids), step, dtype=np.int32))
        digest.update(pair_ids.astype("<i8").tobytes())
        if step == digest_at:
            snapshot = digest.hexdigest()
    return {"ids": np.concatenate(ids), "labels": np.concatenate(labels).astype(np.int8),
            "weights": np.concatenate(weights).astype(np.float32), "step": np.concatenate(step_no),
            "digest_all": digest.hexdigest(), "digest_at": snapshot}


def _text_table(ref: pa.Table, pool: pa.Table, left_idx: np.ndarray, right_idx: np.ndarray) -> dict:
    left_part, right_part = ref.take(pa.array(left_idx)), pool.take(pa.array(right_idx))
    return {f"left_{name}": left_part.column(name) for name in TEXT} | {
        f"right_{name}": right_part.column(name) for name in TEXT}


def _write(path: Path, columns: dict) -> str:
    tmp = Path(str(path) + ".tmp")
    pq.write_table(pa.table(columns), tmp, compression="zstd", row_group_size=500_000)
    tmp.replace(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def typed_flags(arena: Path, labels: Path, partition: np.ndarray, left: np.ndarray, right: np.ndarray,
                cache: Path, *, gate: float, accept: float, lookalike: float) -> np.ndarray:
    """Full-length uint8 flags, nonzero only on partition-A negatives (C2's flags[~in_train] = 0)."""
    key = {"arena_manifest_sha256": hashlib.sha256((arena.parent / "c2_input" / "manifest.json").read_bytes()).hexdigest()
           if (arena.parent / "c2_input" / "manifest.json").exists() else None,
           "pairs": int(len(left)), "tree_gate": gate, "accept_threshold": accept,
           "lookalike_threshold": lookalike, "partition": "A (code 0)"}
    meta = cache.with_suffix(".json")
    if cache.exists() and meta.exists() and json.loads(meta.read_text()).get("key") == key:
        log(f"typed-negative flags from cache {cache}")
        return np.load(cache)
    rows = np.flatnonzero(partition[left] == A)
    label = np.asarray(np.load(labels / "label.npy", mmap_mode="r")[rows], dtype=bool)
    tree = np.asarray(np.load(arena / "base_probabilities.npy", mmap_mode="r")[rows], dtype=np.float32)
    blend = np.asarray(np.load(arena / "probabilities.npy", mmap_mode="r")[rows], dtype=np.float32)
    owner = np.load(labels / "rec_owner.npy")
    ref = pq.read_table(arena / "reference.parquet", columns=list(TEXT)).to_pandas()
    pool = pq.read_table(arena / "pool.parquet", columns=list(TEXT)).to_pandas()
    log(f"classifying {len(rows)} partition-A pairs")
    flags_a = classify_negative_types(ref, pool, left[rows], right[rows], label, tree, blend, owner,
                                      accept_threshold=accept, lookalike_threshold=lookalike, tree_gate=gate)
    flags = np.zeros(len(left), dtype=np.uint8)
    flags[rows] = flags_a
    np.save(cache, flags)
    _json(meta, {"key": key, "counts": {KIND_NAMES[k]: int(((flags & int(k)) != 0).sum()) for k in KIND_NAMES},
                 "labels_read": "partition A rows of label.npy; rec_owner.npy as in C2"})
    return flags


def build_train(args) -> dict:
    arena, labels, out = Path(args.arena), Path(args.labels), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((labels / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("labels include HOLDOUT-2 rows")
    left, right = _pairs(arena / "pairs.parquet")
    partition = np.load(args.partition)
    if int(partition.max()) > 2:
        raise SystemExit("expected finalize_v1 partition codes 0=A, 1=B_REST, 2=SLICE")
    flags = typed_flags(arena, labels, partition, left, right, out / "c2_typed_flags.npy",
                        gate=args.tree_gate, accept=args.accept_threshold, lookalike=args.lookalike_threshold)
    rows = np.flatnonzero(partition[left] == A)
    label_a = np.asarray(np.load(labels / "label.npy", mmap_mode="r")[rows], dtype=bool)
    tree_a = np.asarray(np.load(arena / "base_probabilities.npy", mmap_mode="r")[rows], dtype=np.float32)
    positives = rows[label_a]
    gated = rows[~label_a & (tree_a >= args.tree_gate) & (flags[rows] == 0)]
    true_count = np.load(labels / "true_count.npy")
    log(f"A pairs {len(rows)}, positives {len(positives)}, random-gated pool {len(gated)}")
    sampler = ArenaBatchSampler(positives, flags, gated, batch_size=C2_BATCH, positive_fraction=.25, seed=args.seed)
    started = time.perf_counter()
    stream = draw_c2_stream(sampler, args.steps, left, true_count)
    log(f"drew {args.steps} C2 steps in {time.perf_counter() - started:.0f}s")
    ids = stream["ids"]
    if not np.all(partition[left[ids]] == A):
        raise AssertionError("sampled a non-A pair")
    ref = pq.read_table(arena / "reference.parquet", columns=list(TEXT))
    pool = pq.read_table(arena / "pool.parquet", columns=list(TEXT))
    row_flags = flags[ids]
    columns = {"pair_row": np.arange(len(ids), dtype=np.int64), "step": stream["step"], "arena_row": ids,
               "left_idx": left[ids], "right_idx": right[ids],
               **_text_table(ref, pool, left[ids], right[ids]),
               "label": stream["labels"], "true_count": true_count[left[ids]].astype(np.int32),
               "stake_weight": stream["weights"], "flags": row_flags,
               "negative_type": negative_type_names(stream["labels"], row_flags).astype(str),
               "partition": np.full(len(ids), "A")}
    sha = _write(out / "train_pairs.parquet", columns)
    types, counts = np.unique(columns["negative_type"], return_counts=True)
    report = {"state": "complete", "rows": int(len(ids)), "steps": args.steps, "batch": C2_BATCH,
              "seed": args.seed, "tree_gate": args.tree_gate, "accept_threshold": args.accept_threshold,
              "lookalike_threshold": args.lookalike_threshold, "positives_pool": int(len(positives)),
              "random_gated_pool": int(len(gated)), "typed_pool": int((flags != 0).sum()),
              "negative_type_counts": dict(zip(types.tolist(), counts.tolist())),
              "unique_pairs": int(len(np.unique(ids))), "positive_share": float(stream["labels"].mean()),
              "sampled_pair_sha256_first_40000_steps": stream["digest_at"],
              "sampled_pair_sha256_all": stream["digest_all"],
              "note": "rows 128*(s-1)..128*s-1 are C2 step s; compare the 40000-step digest with C2 "
                      "results.json invocation_sampled_pair_sha256",
              "parquet_sha256": sha, "labels_read": "partition A only", "holdout2_labels_read": False}
    _json(out / "train_pairs.json", report)
    log(json.dumps({k: report[k] for k in ("rows", "positive_share", "negative_type_counts")}))
    return report


def build_scoring(args) -> dict:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.command == "brest":
        arena = Path(args.arena)
        left, right = _pairs(arena / "pairs.parquet")
        tree = np.load(arena / "base_probabilities.npy", mmap_mode="r")
        partition = np.load(args.partition)
        selected = (partition[left] == B_REST) & (np.asarray(tree) >= args.tree_gate)
        ref_path, pool_path, name = arena / "reference.parquet", arena / "pool.parquet", "brest_pairs"
        scope = "arena B_REST (partition code 1), tree-gated; no labels read"
    else:
        root = Path(args.test_inputs)
        left = np.load(root / "index" / "left.npy").astype(np.int64)
        right = np.load(root / "index" / "right.npy").astype(np.int64)
        tree = np.load(root / "base_probabilities.npy", mmap_mode="r")
        selected = np.asarray(tree) >= args.tree_gate
        ref_path, pool_path = root / "dense_test" / "reference.parquet", root / "dense_test" / "pool.parquet"
        name, scope = "test_pairs", "test pairs, tree-gated; no labels exist"
    rows = np.flatnonzero(selected)
    ref = pq.read_table(ref_path, columns=list(TEXT))
    pool = pq.read_table(pool_path, columns=list(TEXT))
    columns = {"pair_row": np.arange(len(rows), dtype=np.int64), "arena_row": rows,
               "left_idx": left[rows], "right_idx": right[rows], **_text_table(ref, pool, left[rows], right[rows]),
               "tree": np.asarray(tree[rows], dtype=np.float32)}
    sha = _write(out / f"{name}.parquet", columns)
    report = {"state": "complete", "scope": scope, "rows": int(len(rows)), "source_pairs": int(len(left)),
              "tree_gate": args.tree_gate, "alignment": "arena_row = row in the source pair order; "
              "score arrays are written full-length in that order", "parquet_sha256": sha,
              "labels_read": False}
    _json(out / f"{name}.json", report)
    log(json.dumps(report))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "brest", "test"))
    parser.add_argument("--arena", default="/workspace/az-ml-new/arena/v1")
    parser.add_argument("--labels", default="/workspace/az-ml-new/arena/labels")
    parser.add_argument("--partition", default="/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy")
    parser.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
    parser.add_argument("--output", default="/workspace/az-ml-new/qwen")
    parser.add_argument("--steps", type=int, default=46_875, help="C2 steps of 128 pairs (46,875 = 6.0M pairs)")
    parser.add_argument("--seed", type=int, default=C2_SEED)
    parser.add_argument("--tree-gate", type=float, default=.01)
    parser.add_argument("--accept-threshold", type=float, default=.775)
    parser.add_argument("--lookalike-threshold", type=float, default=.3)
    args = parser.parse_args(argv)
    for value in (args.arena, args.labels, args.output, args.partition):
        if any(marker in str(value).lower() for marker in ("holdout2", "holdout-2", "reserved_ids")):
            raise SystemExit("refusing a HOLDOUT-2 path")
    build_train(args) if args.command == "train" else build_scoring(args)


if __name__ == "__main__":
    main()
