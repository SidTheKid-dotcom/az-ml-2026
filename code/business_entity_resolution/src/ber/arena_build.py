"""Arena: every eligible training S1 scored against its dense top-50 of the full pool.

`prepare` writes reference/pool/pairs in exactly the layout the frozen v1
streaming predictor consumed on test (dense-only retrieval columns, compact pool,
pairs sorted by left then right), from P2's completed dense mining. It reads no
labels. HOLDOUT-2 S1s are dropped. `labels` aligns ground truth to the arena
pairs afterwards and never reads HOLDOUT-2 rows.
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
import pyarrow.parquet as pq

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
ELIGIBLE = "train_tune_eligible"
POOL_ROWS = (5_034_616, 5_285_603)


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_raw(path: Path, chunksize: int = 200_000):
    """Same parser settings as v1's full_dense_candidates pool construction."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       usecols=RECORD_COLUMNS, chunksize=chunksize)


def prepare(args) -> dict:
    p2, raw, out = Path(args.p2), Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    p2_reference = pd.read_parquet(p2 / "reference.parquet", columns=["entity_id", "population", "raw_source1_row"])
    eligible = (p2_reference.population == ELIGIBLE).to_numpy()
    if int(eligible.sum()) != args.expected_references:
        raise ValueError(f"Expected {args.expected_references} eligible S1s, found {int(eligible.sum())}")
    p2_to_arena = np.full(len(p2_reference), -1, dtype=np.int64)
    p2_to_arena[eligible] = np.arange(int(eligible.sum()))
    keep = p2_reference.loc[eligible].reset_index(drop=True)

    # Reference text re-read from the raw file with v1's parser, matched by row and ID.
    rows = keep.raw_source1_row.to_numpy(dtype=np.int64)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("P2 reference order is not raw Source 1 order")
    pieces, start = [], 0
    for frame in read_raw(raw / "train_source1.tsv"):
        stop = start + len(frame)
        low, high = np.searchsorted(rows, [start, stop])
        if high > low:
            pieces.append(frame.iloc[rows[low:high] - start])
        start = stop
    reference = pd.concat(pieces, ignore_index=True)
    if not np.array_equal(reference.entity_id.to_numpy(), keep.entity_id.to_numpy()):
        raise ValueError("Raw Source 1 rows do not match P2 reference IDs")
    reference[RECORD_COLUMNS].to_parquet(out / "reference.parquet", index=False)
    keep.assign(arena_left_idx=np.arange(len(keep)))[["arena_left_idx", "entity_id", "raw_source1_row"]].to_parquet(
        out / "reference_map.parquet", index=False)
    log(f"reference: {len(reference):,} eligible S1s")

    table = pq.read_table(p2 / "candidate_pairs.parquet",
                          columns=["left_idx", "right_raw_idx", "dense_score", "dense_rank"])
    left = p2_to_arena[table.column("left_idx").to_numpy()]
    mask = left >= 0
    right_raw = table.column("right_raw_idx").to_numpy().astype(np.int64)[mask]
    dense_score = table.column("dense_score").to_numpy()[mask].astype(np.float32)
    dense_rank = table.column("dense_rank").to_numpy()[mask].astype(np.int32)
    left = left[mask]
    del table
    log(f"pairs kept: {len(left):,} (dropped {int((~mask).sum()):,} HOLDOUT-2 pairs)")

    requested = np.unique(right_raw)
    pieces, start = [], 0
    for name, expected in zip(("train_source2.tsv", "train_source3.tsv"), POOL_ROWS):
        seen = 0
        for frame in read_raw(raw / name):
            stop = start + len(frame)
            low, high = np.searchsorted(requested, [start, stop])
            if high > low:
                piece = frame.iloc[requested[low:high] - start].copy()
                piece["raw_row_idx"] = requested[low:high]
                pieces.append(piece)
            start, seen = stop, seen + len(frame)
        if seen != expected:
            raise ValueError(f"{name}: {seen} rows, expected {expected}")
    pool = pd.concat(pieces, ignore_index=True)
    if not np.array_equal(pool.raw_row_idx.to_numpy(), requested) or pool.entity_id.duplicated().any():
        raise ValueError("Compact pool lost or duplicated raw rows")
    pool.to_parquet(out / "pool.parquet", index=False)
    log(f"pool: {len(pool):,} compact records")

    right = np.searchsorted(requested, right_raw)
    order = np.lexsort((right, left))
    left, right = left[order].astype(np.int32), right[order].astype(np.int32)
    right_raw, dense_score, dense_rank = right_raw[order].astype(np.int32), dense_score[order], dense_rank[order]
    if np.any((left[1:] == left[:-1]) & (right[1:] == right[:-1])):
        raise ValueError("Duplicate arena pairs")
    columns = {"left_idx": left, "right_idx": right, "right_raw_idx": right_raw,
               "dense_score": np.clip(dense_score, -1, 1).astype(np.float32), "dense_rank": dense_rank,
               "name_score": np.zeros(len(left), np.float32), "address_score": np.zeros(len(left), np.float32),
               "retrieval_score": np.clip(dense_score, -1, 1).astype(np.float32),
               "retrieval_rank": dense_rank.astype(np.int32), "dense_only": np.ones(len(left), np.float32)}
    pq.write_table(pa.table(columns), out / "pairs.parquet", row_group_size=1_000_000)
    manifest = {"schema": 1, "p2": str(p2), "raw": str(raw), "references": len(reference),
                "pool_records": len(pool), "pairs": int(len(left)), "labels_read": False,
                "holdout2_excluded": True, "population": ELIGIBLE,
                "feature_semantics": {"name_score": "zero", "address_score": "zero",
                                      "retrieval_score": "dense cosine", "dense_only": "one"},
                "sha256": {name: sha256(out / name) for name in ("reference.parquet", "pool.parquet", "pairs.parquet")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(json.dumps({key: manifest[key] for key in ("references", "pool_records", "pairs")}))
    return manifest


def labels(args) -> dict:
    """Align ground truth to arena pairs; only eligible S1 rows of the truth file are used."""
    data, raw, out = Path(args.data), Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(data / "reference.parquet", columns=["entity_id"])
    pool = pd.read_parquet(data / "pool.parquet", columns=["entity_id"])
    s1_index = pd.Series(np.arange(len(reference)), index=reference.entity_id)
    truth = pd.read_csv(raw / "train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False)
    truth = truth[truth.source1_entity_id.isin(s1_index.index)]
    if len(truth) != len(reference):
        raise ValueError("Every eligible S1 must have one ground-truth row")
    exploded = truth.assign(target=truth.matched_entity_ids.str.split(",")).explode("target")
    exploded = exploded[exploded.target.notna() & (exploded.target != "")]
    true_count = np.bincount(s1_index[exploded.source1_entity_id].to_numpy(), minlength=len(reference)).astype(np.int16)
    owner = pd.Series(s1_index[exploded.source1_entity_id].to_numpy(), index=exploded.target.to_numpy())
    rec_owner = owner.reindex(pool.entity_id).fillna(-1).to_numpy().astype(np.int32)
    pairs = pq.read_table(data / "pairs.parquet", columns=["left_idx", "right_idx"])
    left, right = pairs.column("left_idx").to_numpy(), pairs.column("right_idx").to_numpy()
    label = rec_owner[right] == left
    np.save(out / "label.npy", label)
    np.save(out / "true_count.npy", true_count)
    np.save(out / "rec_owner.npy", rec_owner)
    summary = {"pairs": int(len(label)), "positive_pairs": int(label.sum()), "true_links": int(true_count.sum()),
               "link_recall": float(label.sum() / max(1, true_count.sum())), "singletons": int((true_count == 0).sum()),
               "pool_records_owned": int((rec_owner >= 0).sum()), "holdout2_rows_read": 0}
    (out / "labels_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    log(json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--p2", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expected-references", type=int, default=2_072_821)
    q = commands.add_parser("labels")
    q.add_argument("--data", required=True)
    q.add_argument("--raw", required=True)
    q.add_argument("--out", required=True)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else labels)(args)


if __name__ == "__main__":
    main()
