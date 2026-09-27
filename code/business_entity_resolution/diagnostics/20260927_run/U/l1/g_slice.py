"""Idea 4 (27 Sep, Agent U): slice Agent G's v1 string-feature tables (out1/{arena,test}/X.npy, 60 features)
onto the L1 target rows. Runs on the G box (read-only on G's files). Label-free.

S1/record identity is checked by entity id against the arena / dense-test parquets; test route records beyond the
dense pool are matched by entity id (extra_rec_ids.json). Writes OUT/g_{arena,test}.npy (float32, NaN = no G row)
and g_meta.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

G = Path("/home/ubuntu/run27/G/out1")
U = Path("/home/ubuntu/run27/U")
T = U / "l1targets"
OUT = U / "l1g"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def ids(path):
    return pq.read_table(path, columns=["entity_id"]).column("entity_id").to_numpy(zero_copy_only=False)


def slice_split(split, s1_path, rec_path, left, right_ids_or_idx):
    z = np.load(G / split / "pairs.npz")
    gl, gr = z["left"].astype(np.int64), z["right"].astype(np.int64)
    g_s1, g_rec = ids(G / split / "s1.parquet"), ids(G / split / "rec.parquet")
    my_s1 = ids(s1_path)
    if not np.array_equal(g_s1, my_s1):
        raise SystemExit(f"{split}: G S1 order differs")
    my_rec = ids(rec_path)
    n_dense = len(my_rec)
    if not np.array_equal(g_rec[:n_dense], my_rec):
        raise SystemExit(f"{split}: G record order differs on the dense pool")
    right = right_ids_or_idx
    if split == "test":
        extra = json.loads((T / "extra_rec_ids.json").read_text())
        pos = {v: i for i, v in enumerate(g_rec[n_dense:].tolist())}
        emap = np.asarray([n_dense + pos.get(v, -10**9) for v in extra], np.int64)
        right = np.where(right < n_dense, right, emap[np.maximum(right - n_dense, 0)] if len(emap) else -1)
        log(f"test extra records {len(extra)}; unmatched {int((emap < 0).sum())}")
    nrec = len(g_rec) + 1
    gk = gl * nrec + gr
    order = np.argsort(gk, kind="stable")
    gk = gk[order]
    k = left * nrec + right
    p = np.minimum(np.searchsorted(gk, k), len(gk) - 1)
    hit = (gk[p] == k) & (right >= 0)
    grow = np.where(hit, order[p], -1)
    log(f"{split}: {int(hit.sum())} of {len(k)} target rows found in G")
    X = np.load(G / split / "X.npy", mmap_mode="r")
    out = np.full((len(k), X.shape[1]), np.nan, np.float32)
    idx = np.flatnonzero(hit)
    srt = idx[np.argsort(grow[idx])]
    step = 2_000_000
    for s in range(0, len(srt), step):
        sub = srt[s:s + step]
        out[sub] = X[grow[sub]]
        log(f"  {split} {s + len(sub)}/{len(srt)}")
    np.save(OUT / f"g_{split}.npy", out)
    return {"rows": int(len(k)), "found": int(hit.sum()), "found_route": None}


meta = json.loads((G / "arena" / "meta.json").read_text())
res = {"features": meta["features"], "source": str(G)}
res["arena"] = slice_split("arena", U / "data/arena/reference.parquet", U / "data/arena/pool.parquet",
                           np.load(T / "targets_arena_left.npy"), np.load(T / "targets_arena_right.npy"))
res["test"] = slice_split("test", U / "data/test/dense_test/reference.parquet", U / "data/test/dense_test/pool.parquet",
                          np.load(T / "targets_test_left.npy"), np.load(T / "targets_test_right.npy"))
(OUT / "g_meta.json").write_text(json.dumps(res, indent=1) + "\n")
log("done")
