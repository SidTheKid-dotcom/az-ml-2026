"""Build string-feature pair tables for the arena (all dense pairs + B_REST eureka routes) or test.

    python -m ber.string_gbdt_features arena --data D --out O
    python -m ber.string_gbdt_features test --data D --out O [--country France]

Outputs in O/<set>/: pairs.npz (left, right, route flag, [label]), X.npy (float32 features,
ber.string_gbdt.FEATURES order), s1.parquet / rec.parquet (entity ids, country, core name, empty flag).
Arena labels come from label.npy (dense) and rec_owner (routes); they are only stored, never features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.string_gbdt import FEATURES, Table, log, pair_features, s1_multiplicity, source_s3

COLS = ["entity_id", "business_name", "business_address", "country"]


def build(ref, pool, left, right, route, out, workers, label=None, resume=False):
    out.mkdir(parents=True, exist_ok=True)
    order = np.lexsort((right, left))
    left, right, route = left[order], right[order], route[order]
    if label is not None:
        label = label[order]
    key = left.astype(np.int64) * len(pool) + right
    if np.any(key[1:] == key[:-1]):
        raise SystemExit("duplicate pairs")
    table = Table(ref.business_name.to_numpy(object), ref.business_address.to_numpy(object),
                  pool.business_name.to_numpy(object), pool.business_address.to_numpy(object), workers)
    mult = s1_multiplicity(table, ref.country.to_numpy(object))
    rec_s3 = source_s3(pool.entity_id)
    extra = {} if label is None else {"label": label}
    np.savez(out / "pairs.npz", left=left.astype(np.int32), right=right.astype(np.int32), route=route, **extra)
    pd.DataFrame({"entity_id": ref.entity_id.to_numpy(), "country": ref.country.to_numpy(),
                  "core": table.n_core[table.s1_n], "empty": table.a_kind[table.s1_a] == 0,
                  "mult": mult}).to_parquet(out / "s1.parquet", index=False)
    pd.DataFrame({"entity_id": pool.entity_id.to_numpy(), "country": pool.country.to_numpy(),
                  "core": table.n_core[table.rec_n], "empty": table.a_kind[table.rec_a] == 0}
                 ).to_parquet(out / "rec.parquet", index=False)
    log(f"pairs {len(left)} (routes {int(route.sum())}); computing features")
    pair_features(table, left, right, mult, rec_s3, out / "X.npy", workers, resume=resume)
    (out / "meta.json").write_text(json.dumps({"features": FEATURES, "pairs": int(len(left)),
                                               "s1": len(ref), "records": len(pool)}, indent=1))


def arena(args):
    d = Path(args.data)
    ref = pq.read_table(d / "arena_reference.parquet", columns=COLS).to_pandas()
    pool = pq.read_table(d / "arena_pool.parquet", columns=COLS).to_pandas()
    left = np.load(d / "arena/arena_left.npy").astype(np.int64)
    right = np.load(d / "arena/arena_right.npy").astype(np.int64)
    label = np.load(d / "arena/labels/label.npy")
    owner = np.load(d / "arena/labels/rec_owner.npy")
    partition = np.load(d / "arena/finalize_v1/s1_partition.npy")
    if args.limit_s1:
        keep = left < args.limit_s1
        left, right, label = left[keep], right[keep], label[keep]
    rl = np.load(d / "eureka/retrieval/b_rest/left.npy").astype(np.int64)
    rr = np.load(d / "eureka/retrieval/b_rest/right.npy").astype(np.int64)
    if args.limit_s1:
        m = rl < args.limit_s1
        rl, rr = rl[m], rr[m]
    assert np.all(partition[rl] == 1)
    dense_key = np.unique(left * len(pool) + right)
    rkey = rl * len(pool) + rr
    new = ~np.isin(rkey, dense_key)
    rl, rr = rl[new], rr[new]
    rlabel = owner[rr] == rl
    log(f"dense {len(left)}, routes new {len(rl)} (true {int(rlabel.sum())})")
    L = np.concatenate([left, rl])
    R = np.concatenate([right, rr])
    route = np.concatenate([np.zeros(len(left), bool), np.ones(len(rl), bool)])
    lab = np.concatenate([label, rlabel])
    if args.limit_s1:
        ref = ref.iloc[:args.limit_s1]
    build(ref, pool, L, R, route, Path(args.out) / "arena", args.workers, label=lab, resume=args.resume)


def test(args):
    d = Path(args.data)
    ref = pq.read_table(d / "test/dense_test/reference.parquet", columns=COLS).to_pandas()
    dpool = pq.read_table(d / "test/dense_test/pool.parquet", columns=COLS + ["raw_row_idx"]).to_pandas()
    epool = pq.read_table(d / "eureka/retrieval/test/pool.parquet", columns=COLS + ["raw_row_idx"]).to_pandas()
    pool = pd.concat([dpool, epool], ignore_index=True).drop_duplicates("raw_row_idx").reset_index(drop=True)
    raw_to = pd.Series(np.arange(len(pool)), index=pool.raw_row_idx.to_numpy())
    left = np.load(d / "test/index/left.npy").astype(np.int64)
    right = raw_to.reindex(dpool.raw_row_idx.to_numpy()).to_numpy()[np.load(d / "test/index/right.npy")]
    e = d / "eureka/retrieval/test"
    el, er = np.load(e / "left.npy"), np.load(e / "right.npy")
    score, rank, branch = np.load(e / "score.npy"), np.load(e / "rank.npy"), np.load(e / "branch.npy")
    kept = ((branch == 1) & (score >= 0.6) & (rank <= 99)) | ((branch == 2) & (score >= 0.8) & (rank <= 2))
    el = el[kept].astype(np.int64)
    er = raw_to.reindex(epool.raw_row_idx.to_numpy()).to_numpy()[er[kept]]
    key = np.unique(left * len(pool) + right)
    new = ~np.isin(el * len(pool) + er, key)
    el, er = el[new], er[new]
    log(f"test dense {len(left)}, routes kept new {len(el)}")
    L = np.concatenate([left, el])
    R = np.concatenate([right, er])
    route = np.concatenate([np.zeros(len(left), bool), np.ones(len(el), bool)])
    name = "test"
    if args.country:
        m = ref.country.to_numpy()[L] == args.country
        L, R, route = L[m], R[m], route[m]
        name = f"test_{args.country}"
    build(ref, pool, L, R, route, Path(args.out) / name, args.workers)


def arena_fr(args):
    """Agent S Frenchified arena (option A, train-derived): features for all its pairs."""
    d = Path(args.data) / "arena_fr_v3"
    ref = pq.read_table(d / "arena/reference.parquet", columns=COLS).to_pandas()
    pool = pq.read_table(d / "arena/pool.parquet", columns=COLS).to_pandas()
    t = pq.read_table(d / "arena/pairs.parquet", columns=["left_idx", "right_idx"])
    left = t.column(0).to_numpy().astype(np.int64)
    right = t.column(1).to_numpy().astype(np.int64)
    label = np.load(d / "labels/label.npy")
    build(ref, pool, left, right, np.zeros(len(left), bool), Path(args.out) / "arena_fr", args.workers, label=label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["arena", "test", "arena_fr"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--country")
    ap.add_argument("--limit-s1", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    {"arena": arena, "test": test, "arena_fr": arena_fr}[args.what](args)


if __name__ == "__main__":
    main()
