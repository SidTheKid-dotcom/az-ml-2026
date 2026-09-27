"""HOLDOUT-2 final check, step 2 (label-free): v1 scoring inputs in the arena's layout.

From the GPU step A forward top-50 (raw train pool rows + float32 cosines) writes H2/v1data/{reference,pool,pairs}
.parquet exactly as `ber.arena_build.prepare` did for the arena (compact pool of referenced raw rows, pairs sorted
by (left, right), dense_rank 1-based, retrieval columns = dense). Also:
  * a forward-fidelity report (GPU step A's recomputed top-50 for 20k arena S1s vs the stored P2 arena pairs);
  * H2/v1parity/: 2,000 arena S1s with their stored arena pairs in the same layout, so GPU step B can re-score them
    and compare with the stored arena blend/tree (v1 scoring parity).
No labels are read.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/ubuntu/run27/U")
H2 = ROOT / os.environ.get("U_H2DIR", "h2")
G = H2 / "gpu_out"
COLS = ["entity_id", "business_name", "business_address", "country"]
POOL_ROWS = (5_034_616, 5_285_603)


def read_pool_rows(requested):
    pieces, start = [], 0
    for name, expected in zip(("train_source2.tsv", "train_source3.tsv"), POOL_ROWS):
        seen = 0
        for frame in pd.read_csv(H2 / "raw" / name, sep="\t", dtype=str, keep_default_na=False, usecols=COLS,
                                 chunksize=200_000):
            stop = start + len(frame)
            lo, hi = np.searchsorted(requested, [start, stop])
            if hi > lo:
                piece = frame.iloc[requested[lo:hi] - start].copy()
                piece["raw_row_idx"] = requested[lo:hi]
                pieces.append(piece)
            start, seen = stop, seen + len(frame)
        if seen != expected:
            raise SystemExit(f"{name}: {seen} rows, expected {expected}")
    pool = pd.concat(pieces, ignore_index=True)
    if not np.array_equal(pool.raw_row_idx.to_numpy(), requested):
        raise SystemExit("compact pool lost raw rows")
    return pool


def write_layout(out, reference, left, right_raw, score, rank):
    out.mkdir(parents=True, exist_ok=True)
    requested = np.unique(right_raw)
    pool = read_pool_rows(requested)
    right = np.searchsorted(requested, right_raw)
    order = np.lexsort((right, left))
    left, right, right_raw, score, rank = left[order], right[order], right_raw[order], score[order], rank[order]
    if np.any((left[1:] == left[:-1]) & (right[1:] == right[:-1])):
        raise SystemExit("duplicate pairs")
    reference[COLS].to_parquet(out / "reference.parquet", index=False)
    pool.to_parquet(out / "pool.parquet", index=False)
    s = np.clip(score, -1, 1).astype(np.float32)
    cols = {"left_idx": left.astype(np.int32), "right_idx": right.astype(np.int32),
            "right_raw_idx": right_raw.astype(np.int32), "dense_score": s, "dense_rank": rank.astype(np.int32),
            "name_score": np.zeros(len(left), np.float32), "address_score": np.zeros(len(left), np.float32),
            "retrieval_score": s, "retrieval_rank": rank.astype(np.int32), "dense_only": np.ones(len(left), np.float32)}
    pq.write_table(pa.table(cols), out / "pairs.parquet", row_group_size=1_000_000)
    return {"references": int(len(reference)), "pool_records": int(len(pool)), "pairs": int(len(left))}


def main():
    ref = pd.read_parquet(H2 / "reference.parquet")
    fi, fs = np.load(G / "h2_fwd_idx.npy"), np.load(G / "h2_fwd_score.npy")
    if fi.shape != (len(ref), 50) or np.any(fi < 0):
        raise SystemExit("forward top-50 incomplete")
    left = np.repeat(np.arange(len(ref)), 50)
    rank = np.tile(np.arange(1, 51), len(ref))
    rep = {"h2": write_layout(H2 / "v1data", ref, left, fi.reshape(-1).astype(np.int64), fs.reshape(-1), rank)}
    # forward fidelity vs stored P2 arena pairs
    chk_rows, chk_idx = np.load(G / "check_arena_rows.npy"), np.load(G / "check_arena_fwd_idx.npy")
    t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_raw_idx", "dense_rank"])
    al = t.column("left_idx").to_numpy()
    ar = t.column("right_raw_idx").to_numpy()
    sel = np.isin(al, chk_rows)
    stored = pd.DataFrame({"l": al[sel], "r": ar[sel]})
    got = pd.DataFrame({"l": np.repeat(chk_rows, 50), "r": chk_idx.reshape(-1)})
    inter = stored.merge(got, on=["l", "r"]).shape[0]
    rep["forward_fidelity_vs_p2"] = {"arena_s1": int(len(chk_rows)), "stored_pairs": int(len(stored)),
                                     "overlap": round(inter / max(len(stored), 1), 5)}
    # v1 parity sample: 2,000 arena S1s (stored pairs) for GPU step B
    par = chk_rows[:2000]
    aref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=COLS).to_pandas()
    psel = np.flatnonzero(np.isin(al, par))
    sub_left = np.searchsorted(par, al[psel])
    t2 = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["dense_score", "dense_rank"])
    rep["v1parity"] = write_layout(H2 / "v1parity", aref.iloc[par].reset_index(drop=True), sub_left,
                                   ar[psel].astype(np.int64), t2.column("dense_score").to_numpy()[psel],
                                   t2.column("dense_rank").to_numpy()[psel])
    np.save(H2 / "v1parity" / "arena_pair_rows.npy", psel[np.lexsort((ar[psel], sub_left))])
    rep["labels_read"] = False
    (H2 / "pairs_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
