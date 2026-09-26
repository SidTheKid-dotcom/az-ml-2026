"""Shared, label-safe loaders for Track D diagnostics (RESET 27 Sep 2026). Measurement only.

Labels: B_REST (partition 1) pair labels and partition-A ground truth only. SLICE (partition 2)
true counts and record owners are masked on load; HOLDOUT-2 S1s are not in the arena and their
ground-truth rows are skipped unparsed. Nothing here writes predictions or submission files.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

pa.set_cpu_count(1)
pa.set_io_thread_count(1)

ARENA = Path(os.environ.get("ARENA", "/workspace/az-ml-new/arena"))
REF_MAP = Path("/workspace/az-ml-new/artifacts/arena/v1/reference_map.parquet")
TREE = Path("/opt/az-ml-new/agents/pair_tree_c2/step040000")
C2_SCORES = Path("/opt/az-ml-new/runs/c2_step040000/b_rest_scores")
RAW = Path(os.environ.get("RAW", "/workspace/az-ml-new/union/raw"))
TEST_INPUTS = Path("/opt/az-ml-new/test_inputs")
OUT = Path(os.environ.get("D_OUT", "/workspace/az-ml-new/reset/D"))
CACHE = OUT / "cache"
TRAIN_POOL_ROWS = (5_034_616, 5_285_603)
THRESHOLD = 0.725
EXPECTED_B_REST = 0.985511
US_INDIA_WEIGHT = 0.850248
COLUMNS = ["entity_id", "business_name", "business_address", "country"]
BOOT_N = int(os.environ.get("D_BOOT", "1000"))
BOOT_SEED = 20260927
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:8.1f}s] {msg}", flush=True)


def _default(x):
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    raise TypeError(type(x))


def save_json(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_default) + "\n")
    tmp.replace(path)
    log(f"wrote {path}")


def rate(num, den):
    return float(num) / float(den) if den else None


def read_raw_tsv(path, columns=COLUMNS):
    """Same parser settings as ber.arena_build.read_raw (so row numbers match raw_row_idx)."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, usecols=columns)


def raw_sources(split):
    """Source-2 then Source-3 rows of ``split`` with source (2/3), file_row and global raw_idx (arena order)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{split}_s23.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frames, offset = [], 0
    for source in (2, 3):
        frame = read_raw_tsv(RAW / f"{split}_source{source}.tsv")
        frame["source"] = np.int8(source)
        frame["file_row"] = np.arange(len(frame), dtype=np.int64)
        frame["raw_idx"] = frame["file_row"] + offset
        offset += len(frame)
        frames.append(frame)
        log(f"{split} source{source}: {len(frame):,} rows")
    out = pd.concat(frames, ignore_index=True)
    if split == "train" and tuple(len(f) for f in frames) != TRAIN_POOL_ROWS:
        raise SystemExit("train S2/S3 row counts differ from the arena build")
    tmp = path.with_suffix(".tmp")
    out.to_parquet(tmp, index=False)
    tmp.replace(path)
    return out


def partition():
    return np.load(ARENA / "finalize_v1" / "s1_partition.npy")


def arena_reference(columns=COLUMNS):
    return pq.read_table(ARENA / "v1" / "reference.parquet", columns=columns).to_pandas()


def arena_pool(columns=COLUMNS + ["raw_row_idx"]):
    return pq.read_table(ARENA / "v1" / "pool.parquet", columns=columns).to_pandas()


def true_count_safe(part):
    """true_count with SLICE entries zeroed (never used)."""
    tc = np.load(ARENA / "labels" / "true_count.npy").astype(np.int64)
    tc[part == 2] = 0
    return tc


def read_truth(allowed):
    """Ground truth for S1 ids in ``allowed`` (dict id -> arena index); every other row is skipped unparsed.

    Returns (s1 arena index int64, target entity id object array).
    """
    s1_pos, targets = [], []
    with open(RAW / "train_ground_truth.tsv", encoding="utf-8", newline="") as handle:
        if handle.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise SystemExit("unexpected ground-truth header")
        for line in handle:
            sid, tab, payload = line.rstrip("\r\n").partition("\t")
            pos = allowed.get(sid)
            if pos is None:
                continue
            if payload:
                for target in payload.split(","):
                    s1_pos.append(pos)
                    targets.append(target)
    return np.asarray(s1_pos, dtype=np.int64), np.asarray(targets, dtype=object)


def truth_for_partitions(ref_ids, part, keep):
    """Truth links (s1 index, target id) for S1s whose partition is in ``keep`` (never 2)."""
    if 2 in keep:
        raise SystemExit("SLICE labels are sealed for Track D")
    idx = np.flatnonzero(np.isin(part, list(keep)))
    allowed = dict(zip(ref_ids[idx].tolist(), idx.tolist()))
    return read_truth(allowed)


def all_pairs():
    table = pq.read_table(ARENA / "v1" / "pairs.parquet", columns=["left_idx", "right_idx"])
    return table.column("left_idx").to_numpy(), table.column("right_idx").to_numpy()


def brest_decision(left_all=None, right_all=None):
    """Current best B_REST decision: C2 pair-tree OOF, one owner, cap 11 at 0.725 (reproduces 0.985511)."""
    from ber.arena_decision import evaluate
    from ber.pair_tree_arena import decide

    part = partition()
    component = np.load(ARENA / "finalize_v1" / "component_ids.npy")
    tc = true_count_safe(part)
    rows = np.load(TREE / "b_rest_pair_rows.npy")
    oof = np.load(TREE / "oof.npy").astype(np.float32)
    if left_all is None:
        left_all, right_all = all_pairs()
    left = left_all[rows].astype(np.int64)
    right = right_all[rows].astype(np.int64)
    if not (part[left] == 1).all():
        raise SystemExit("B_REST pair rows include non-B_REST S1s")
    label = np.asarray(np.load(ARENA / "labels" / "label.npy", mmap_mode="r")[rows], dtype=bool)
    n_s1 = len(part)
    accept = decide(oof, left, right, THRESHOLD, n_s1)
    pop = part == 1
    ref_country = arena_reference(["country"]).country.to_numpy()
    base = evaluate(accept, left, label, tc, population=pop, country=ref_country, return_per_s1=True)
    log(f"B_REST current decision macro F0.5 {base['macro_f05']:.6f} P {base['precision']:.5f} "
        f"R {base['recall']:.5f} false {base['false_links']}")
    if abs(base["macro_f05"] - EXPECTED_B_REST) > 5e-6:
        raise SystemExit(f"decision does not reproduce {EXPECTED_B_REST}")
    return {"part": part, "component": component, "tc": tc, "rows": rows, "oof": oof, "left": left,
            "right": right, "label": label, "accept": accept, "pop": pop, "base": base, "n_s1": n_s1,
            "country": ref_country}


def slim(result):
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "true_positive_links", "false_links",
            "true_links", "singleton_accuracy")
    out = {k: result[k] for k in keys if k in result}
    if "by_country" in result:
        out["by_country"] = {c: {k: r[k] for k in ("s1", "macro_f05", "precision", "recall", "false_links",
                                                   "true_positive_links")}
                             for c, r in result["by_country"].items()}
    return out
