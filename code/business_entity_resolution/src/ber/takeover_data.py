"""Bounded, ID-aligned data packs for the 27 Sep takeover experiments.

Training is partition A only. Development is B_REST only; its full truth counts
and complete saved candidate world remain in the evaluation pack. No HOLDOUT-2
ground-truth payload is opened. Text always comes from the supplied records.
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


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def plain_text(frame):
    return (frame.business_name.fillna("") + " | " + frame.business_address.fillna("")).to_numpy(object)


def write_pairs(path, left, right, ref, pool, *, labels=None, pair_rows=None, extra=None):
    """Write compact text pairs in exact caller order; input IDs are provenance only."""
    left, right = np.asarray(left), np.asarray(right)
    assert len(left) == len(right)
    assert np.all(left >= 0) and np.all(right >= 0)
    st = ref.text.to_numpy(object) if "text" in ref else plain_text(ref)
    rt = pool.text.to_numpy(object) if "text" in pool else plain_text(pool)
    if "text" in ref:
        st = np.asarray([x.removeprefix("query: ") for x in st], object)
    if "text" in pool:
        rt = np.asarray([x.removeprefix("passage: ") for x in rt], object)
    sid, rid = ref.entity_id.to_numpy(object), pool.entity_id.to_numpy(object)
    writer = None
    try:
        for s in range(0, len(left), 100_000):
            l, r = left[s:s + 100_000], right[s:s + 100_000]
            cols = {"pair_row": np.arange(s, s + len(l), dtype=np.int64) if pair_rows is None else pair_rows[s:s + len(l)],
                    "left_idx": l.astype(np.int32), "right_idx": r.astype(np.int32),
                    "s1_id": sid[l], "rec_id": rid[r], "left_text": st[l], "right_text": rt[r],
                    "source": np.asarray([x[:2] for x in rid[r]], object)}
            if labels is not None:
                cols["label"] = np.asarray(labels[s:s + len(l)], np.int8)
            if extra:
                cols.update({k: v[s:s + len(l)] for k, v in extra.items()})
            table = pa.Table.from_pydict(cols)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer:
            writer.close()
    log(f"wrote {path.name}: {len(left):,} pairs")


def prepare_train(a):
    out, arena = Path(a.out), Path(a.arena)
    if (out / "train_complete.json").exists():
        raise FileExistsError("Training pack is already complete")
    out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((arena / "labels/labels_summary.json").read_text())
    assert summary["holdout2_rows_read"] == 0
    part = np.load(arena / "finalize_v1/s1_partition.npy")
    count = np.load(arena / "labels/true_count.npy")
    ref = pd.read_parquet(arena / "v1/reference.parquet")
    pool = pd.read_parquet(arena / "v1/pool.parquet")
    rng = np.random.default_rng(a.seed)
    selected = np.sort(rng.choice(np.flatnonzero(part == 0), a.train_s1, replace=False))
    chosen = np.zeros(len(part), bool)
    chosen[selected] = True
    pair = pq.read_table(arena / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
    l = pair.column(0).to_numpy()
    r = pair.column(1).to_numpy()
    label = np.load(arena / "labels/label.npy", mmap_mode="r")
    tree = np.load(arena / "v1/base_probabilities.npy", mmap_mode="r")
    blend = np.load(arena / "v1/probabilities.npy", mmap_mode="r")
    rows = np.flatnonzero(chosen[l])
    pos = rows[np.asarray(label[rows], bool)]
    neg = rows[~np.asarray(label[rows], bool)]
    # Four strongest v1 negatives and two randomized competitors per business.
    # This is a fixed mining pack; the new sampler later measures current losses.
    score = np.asarray(tree[neg], np.float32)
    order = np.lexsort((rng.random(len(neg)), -score, l[neg]))
    sorted_neg = neg[order]
    group = l[sorted_neg]
    first = np.maximum.accumulate(np.where(np.r_[True, group[1:] != group[:-1]], np.arange(len(group)), 0))
    hard = sorted_neg[np.arange(len(group)) - first < 4]
    order = np.lexsort((rng.random(len(neg)), l[neg]))
    sorted_neg = neg[order]
    group = l[sorted_neg]
    first = np.maximum.accumulate(np.where(np.r_[True, group[1:] != group[:-1]], np.arange(len(group)), 0))
    random_neg = sorted_neg[np.arange(len(group)) - first < 2]
    keep = np.sort(np.unique(np.r_[pos, hard, random_neg]))
    assert np.all(part[l[keep]] == 0)
    np.save(out / "train_s1.npy", selected)
    np.save(out / "train_pair_rows.npy", keep)
    write_pairs(out / "train.parquet", l[keep], r[keep], ref, pool, labels=label[keep], pair_rows=keep,
                extra={"true_count": count[l[keep]], "v1_tree": np.asarray(tree[keep]),
                       "v1_blend": np.asarray(blend[keep])})
    save_json(out / "train_complete.json", {"state": "complete", "seed": a.seed, "train_s1": len(selected),
              "pairs": len(keep), "positives": int(np.asarray(label[keep]).sum()), "partition": "A",
              "holdout2_rows_read": 0, "selection": "all retrieved positives; top4 tree and 2 random negatives/S1",
              "partition_sha256": digest(arena / "finalize_v1/s1_partition.npy"),
              "train_sha256": digest(out / "train.parquet"), "pair_rows_sha256": digest(out / "train_pair_rows.npy")})


def prepare_brest(a):
    out, u = Path(a.out), Path(a.u)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "brest_complete.json").exists():
        raise FileExistsError("B_REST pack already complete")
    d = u / "prep/brest_u8t"
    model = u / "u4/u8t__base_q_coll_ed_route_l1v2_rev3_nat"
    names = ("left", "right", "label", "true_count", "population", "component", "country_code", "partition")
    arrays = {n: np.load(d / (n + ".npy"), mmap_mode="r") for n in names}
    p = np.load(model / "oof.npy")
    left, right = arrays["left"], arrays["right"]
    assert len(p) == len(left) == len(right) == len(arrays["label"])
    assert np.all(arrays["partition"][left] == 1)
    ref = pd.read_parquet(u / "data/arena/reference.parquet")
    pool = pd.read_parquet(u / "data/arena/pool.parquet")
    extra = json.loads((d / "extra_rec_ids.json").read_text())
    if extra:
        raw = pd.read_parquet(Path(a.texts) / "train_pool.parquet")
        ix = pd.Index(raw.entity_id).get_indexer(extra)
        assert np.all(ix >= 0)
        text = raw.text.iloc[ix].str.removeprefix("passage: ")
        parts = text.str.partition(" | ")
        ep = pd.DataFrame({"entity_id": extra, "business_name": parts[0].to_numpy(),
                           "business_address": parts[2].to_numpy()})
        pool = pd.concat([pool, ep], ignore_index=True)
        del raw
    assert len(pool) > int(right.max()) and pool.entity_id.is_unique
    band = np.flatnonzero((p > a.band_low) & (p < a.band_high))
    rng = np.random.default_rng(a.seed)
    screen_s1 = rng.choice(np.flatnonzero(arrays["population"]), a.screen_s1, replace=False)
    screen = band[np.isin(left[band], screen_s1)]
    np.save(out / "screen_s1.npy", np.sort(screen_s1))
    folds = np.load(u / "folds/brest_fold5_u4-20260927.npy")
    np.savez(out / "brest_world.npz", **arrays, baseline_p=p, folds=folds)
    extra_cols = {"baseline_p": p[band]}
    write_pairs(out / "brest_band.parquet", left[band], right[band], ref, pool,
                labels=arrays["label"][band], pair_rows=band, extra=extra_cols)
    write_pairs(out / "brest_screen.parquet", left[screen], right[screen], ref, pool,
                labels=arrays["label"][screen], pair_rows=screen, extra={"baseline_p": p[screen]})
    save_json(out / "brest_complete.json", {"state": "complete", "scope": "B_REST only; all 521846 S1s in world",
              "baseline": str(model), "baseline_oof_sha256": digest(model / "oof.npy"),
              "rows": len(p), "band_rows": len(band), "screen_rows": len(screen), "screen_s1": a.screen_s1,
              "band_low": a.band_low, "band_high": a.band_high, "holdout2_rows_read": 0,
              "world_sha256": digest(out / "brest_world.npz")})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train", "brest"])
    p.add_argument("--out", required=True)
    p.add_argument("--arena", default="/opt/az-ml-new/artifacts/arena")
    p.add_argument("--u", default="/home/ubuntu/run27/U")
    p.add_argument("--texts", default="/home/ubuntu/run27/R/texts")
    p.add_argument("--train-s1", type=int, default=200_000)
    p.add_argument("--screen-s1", type=int, default=30_000)
    p.add_argument("--band-low", type=float, default=0.001)
    p.add_argument("--band-high", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=202609271)
    a = p.parse_args()
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    {"train": prepare_train, "brest": prepare_brest}[a.command](a)


if __name__ == "__main__":
    main()
