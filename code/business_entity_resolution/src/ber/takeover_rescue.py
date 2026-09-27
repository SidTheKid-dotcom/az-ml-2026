"""Label-blind candidate preparation for a future additive rescue scorer.

The existing takeover_residual.py scores pairs. This module never trains a
transformer. New pair_row values index the complete new-candidate export, NOT
the historical world. Only preparation is implemented here; gating and final TSV
replay have not yet been implemented or measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


BASE_THRESHOLD = 0.725
THRESHOLDS = (0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99)
CAP = 11
SOURCE_CAPS = {2: 5, 3: 6}
FORMAT = "ber.takeover_rescue.v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def log(event, **values):
    print(json.dumps({"event": event, **values}, allow_nan=False), flush=True)


def new_dir(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing nonempty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def integers(values, name, upper=None, unique=False):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu":
        raise ValueError(f"{name}: expected one-dimensional integer indices")
    if len(a) and (a.min() < 0 or (upper is not None and a.max() >= upper)):
        raise ValueError(f"{name}: out-of-range indices")
    if unique and len(np.unique(a)) != len(a):
        raise ValueError(f"{name}: duplicate indices")
    return a.astype(np.int64, copy=False)


def source_codes(ids):
    result = np.fromiter((2 if str(x).startswith("S2-") else
                          3 if str(x).startswith("S3-") else -1 for x in ids),
                         dtype=np.int8, count=len(ids))
    if np.any(result < 0):
        raise ValueError("Record IDs must identify Source 2 or Source 3")
    return result


def baseline_decode(left, right, prob):
    """Exact fixed-.725, one-owner, then cap-11 historical decision."""
    left, right, prob = np.asarray(left), np.asarray(right), np.asarray(prob)
    if left.shape != right.shape or prob.shape != left.shape or not np.isfinite(prob).all():
        raise ValueError("Unaligned/nonfinite baseline")
    rows = np.flatnonzero(prob >= np.asarray(BASE_THRESHOLD, dtype=prob.dtype))
    if not len(rows):
        return rows
    order = np.lexsort((rows, left[rows], -prob[rows].astype(float), right[rows]))
    rows = rows[order]
    rows = rows[np.r_[True, right[rows][1:] != right[rows][:-1]]]
    rows = rows[np.lexsort((rows, -prob[rows].astype(float), left[rows]))]
    starts = np.maximum.accumulate(np.where(np.r_[True, left[rows][1:] != left[rows][:-1]],
                                           np.arange(len(rows)), 0))
    return np.sort(rows[np.arange(len(rows)) - starts < CAP])


def take_rows(path, rows, columns):
    """Read compact rows without copying a whole text dataset into RAM."""
    rows = integers(rows, "text rows", pq.ParquetFile(path).metadata.num_rows)
    wanted = np.unique(rows)
    pieces, offset = [], 0
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=131072, columns=columns):
        start, end = np.searchsorted(wanted, [offset, offset + len(batch)])
        if end > start:
            pieces.append(pa.Table.from_batches([batch]).take(pa.array(wanted[start:end] - offset)))
        offset += len(batch)
    if not len(wanted):
        return pa.Table.from_batches([], schema=pa.schema([pf.schema_arrow.field(c) for c in columns]))
    compact = pa.concat_tables(pieces)
    if len(compact) != len(wanted):
        raise ValueError("Incomplete raw text lookup")
    return compact.take(pa.array(np.searchsorted(wanted, rows)))


def search_features(search, left, right, n_pool):
    search = Path(search)
    marker = json.loads((search / "complete.json").read_text())
    if marker.get("state") != "complete":
        raise ValueError("Retrieval must have a complete marker")
    q = integers(np.load(search / "query_indices.npy"), "query indices", unique=True)
    r = np.load(search / "pool_indices.npy", mmap_mode="r")
    scores = np.load(search / "scores.npy", mmap_mode="r")
    if r.ndim != 2 or len(r) != len(q) or r.shape != scores.shape or r.dtype.kind not in "iu":
        raise ValueError("Retrieval indices/scores are unaligned")
    if np.any((r < -1) | (r >= n_pool)):
        raise ValueError("Raw pool search index out of bounds")
    keep = r.ravel() >= 0
    keys = (np.repeat(q, r.shape[1]) * n_pool + r.ravel())[keep]
    order = np.argsort(keys, kind="stable")
    keys = keys[order]
    if np.any(keys[1:] == keys[:-1]):
        raise ValueError("Duplicate search pair")
    requested = np.asarray(left, np.int64) * n_pool + right
    pos = np.searchsorted(keys, requested)
    if np.any(pos >= len(keys)) or not np.array_equal(keys[pos], requested):
        raise ValueError("Candidate absent from the complete label-blind search")
    rank = np.tile(np.arange(1, r.shape[1] + 1, dtype=np.int16), len(q))[keep][order][pos]
    cosine = np.asarray(scores).ravel()[keep][order][pos].astype(np.float32)
    if not np.isfinite(cosine).all():
        raise ValueError("Nonfinite retained search score")
    for name, digest in marker.get("files_sha256", {}).items():
        if sha256(search / name) != digest:
            raise ValueError(f"Search hash mismatch: {name}")
    return cosine, rank, q, marker


def prepare_dev(args):
    """Project only label-free world arrays; truth is not loaded here."""
    started = time.monotonic()
    out = new_dir(args.output)
    marker = json.loads(args.world.with_name("brest_complete.json").read_text())
    if marker.get("state") != "complete" or marker.get("holdout2_rows_read") != 0:
        raise ValueError("Complete, H2-unread B_REST world required")
    if sha256(args.world) != marker["world_sha256"]:
        raise ValueError("World hash mismatch")
    with np.load(args.world, allow_pickle=False) as world:
        left, right, prob = (world[k] for k in ("left", "right", "baseline_p"))
        population, component, folds = (world[k] for k in ("population", "component", "folds"))
    selected = integers(np.load(args.s1_rows), "selected S1", len(population), unique=True)
    if not population[selected].all():
        raise ValueError("Selected S1s must all be B_REST")
    accepted = baseline_decode(left, right, prob)
    log("baseline_decoded_label_blind", pairs=len(left), accepted=len(accepted))
    raw_ids = pq.read_table(args.raw_pool, columns=["entity_id"])["entity_id"].to_numpy()
    raw_index = pd.Index(raw_ids)
    if not raw_index.is_unique or raw_index.hasnans:
        raise ValueError("Raw record IDs must be nonnull and unique")
    modern = pq.read_table(args.modern_pool, columns=["entity_id", "raw_row_idx"])
    mapping = integers(modern["raw_row_idx"].to_numpy(), "modern pool map", len(raw_ids), unique=True)
    if not np.array_equal(raw_ids[mapping], modern["entity_id"].to_numpy()):
        raise ValueError("Modern pool raw indices and IDs disagree")
    extra_ids = json.loads(args.extra_rec_ids.read_text())
    extra_map = raw_index.get_indexer(extra_ids)
    integers(extra_map, "extra record map", len(raw_ids), unique=True)
    mapping = np.r_[mapping, extra_map]
    integers(mapping, "complete modern map", len(raw_ids), unique=True)
    integers(right, "world right indices", len(mapping))
    owned = mapping[right[accepted]]
    if len(np.unique(owned)) != len(owned):
        raise ValueError("Baseline ownership is not unique in raw ID space")
    base_source = source_codes(raw_ids[owned])
    np.savez_compressed(out / "baseline_label_blind.npz", world_rows=accepted, left=left[accepted],
                        raw_right=owned, source=base_source)
    extra = pq.read_table(args.extra_pairs, columns=["left_idx", "raw_right_idx"])
    el = integers(extra["left_idx"].to_numpy(), "extra left", len(population))
    er = integers(extra["raw_right_idx"].to_numpy(), "extra raw right", len(raw_ids))
    keys = el * len(raw_ids) + er
    if len(np.unique(keys)) != len(keys) or not np.isin(el, selected).all():
        raise ValueError("Extra candidate pairs duplicated or outside selected S1s")
    cosine, rank, queries, retrieval = search_features(args.search, el, er, len(raw_ids))
    if not np.array_equal(np.sort(queries), np.sort(selected)):
        raise ValueError("Search and selected S1 identities differ")
    current = left * len(raw_ids) + mapping[right]
    if np.isin(keys, current).any():
        raise ValueError("Extra candidate already present in current world")
    # This filter is the deployed unowned-only policy, never a truth filter.
    unowned = ~np.isin(er, owned)
    refs = take_rows(args.reference, selected, ["entity_id", "business_name", "business_address", "country"])
    ref_lookup = pd.Index(selected).get_indexer(el)
    ref_ids = refs["entity_id"].to_numpy()
    if len(np.unique(ref_ids)) != len(ref_ids):
        raise ValueError("Duplicate reference IDs")
    candidates = pa.table({"pair_row": np.arange(len(extra), dtype=np.int64), "left_idx": el,
        "raw_right_idx": er, "s1_id": ref_ids[ref_lookup], "rec_id": raw_ids[er],
        "source": source_codes(raw_ids[er]), "dense_cosine": cosine, "dense_rank": rank,
        "baseline_unowned": unowned, "provenance": ["e5_control_2000:new_outside_u8t"] * len(extra)})
    pq.write_table(candidates, out / "all_extra_candidates.parquet", compression="zstd")
    eligible = candidates.filter(pa.array(unowned))
    plain_left = pc.binary_join_element_wise(pc.cast(pc.fill_null(refs["business_name"], ""), pa.string()),
                                            pc.cast(pc.fill_null(refs["business_address"], ""), pa.string()), " | ")
    text_right = take_rows(args.raw_pool, er[unowned], ["entity_id", "text"])
    if not np.array_equal(text_right["entity_id"].to_numpy(), eligible["rec_id"].to_numpy()):
        raise ValueError("Raw pair text/ID alignment mismatch")
    if not pc.all(pc.starts_with(text_right["text"], pattern="passage: ")).as_py():
        raise ValueError("Unexpected E5 passage representation")
    plain_right = pc.utf8_slice_codeunits(text_right["text"], start=len("passage: "))
    pairs = eligible.select(["pair_row", "s1_id", "rec_id"]).append_column(
        "left_text", plain_left.take(pa.array(ref_lookup[unowned]))).append_column("right_text", plain_right)
    pq.write_table(pairs, out / "pairs.parquet", compression="zstd")
    pq.write_table(pa.table({"left_idx": selected, "s1_id": ref_ids,
                             "component": component[selected], "inherited_fold": folds[selected],
                             "country": refs["country"]}), out / "selected_s1.parquet", compression="zstd")
    report = {"format": FORMAT, "state": "complete", "labels_read": False, "holdout2_rows_read": 0,
        "world": str(args.world), "world_sha256": marker["world_sha256"], "selected_s1": len(selected),
        "selected_sha256": array_sha(selected), "raw_pool_rows": len(raw_ids), "baseline_links": len(accepted),
        "all_extra_pairs": len(extra), "owned_record_pairs_removed": int((~unowned).sum()),
        "inference_pairs": len(pairs), "membership_policy": "ALL new search pairs outside U8t; unowned records only",
        "baseline_policy": ">=.725, one_owner(score desc,left asc,row asc), then cap11; no source pruning",
        "retrieval": retrieval, "source_sha256": sha256(__file__),
        "inputs": {str(p): sha256(p) for p in (args.extra_pairs, args.s1_rows, args.extra_rec_ids)},
        "files_sha256": {p.name: sha256(p) for p in out.iterdir() if p.is_file()},
        "seconds": time.monotonic() - started}
    write_json(out / "prepared.complete.json", report)
    log("prepared", **report)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-dev", help="Freeze unowned new pairs without reading truth")
    for name in ("world", "s1-rows", "extra-pairs", "search", "reference", "raw-pool", "modern-pool", "extra-rec-ids", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--threads", type=int, default=6)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    if args.command == "prepare-dev":
        prepare_dev(args)


if __name__ == "__main__":
    main()
