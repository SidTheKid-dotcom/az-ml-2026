"""Extend the fixed A pack with existing reverse/native candidate edges.

Candidate membership is label-free. Only the selected A businesses' raw truth
payloads are decoded, after the candidate union is fixed. This writes training
examples, never evaluation candidates or predictions.
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


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/takeover_20260927_1000"))
    p.add_argument("--u", type=Path, default=Path("/home/ubuntu/run27/U"))
    p.add_argument("--r", type=Path, default=Path("/home/ubuntu/run27/R"))
    p.add_argument("--arena", type=Path, default=Path("/opt/az-ml-new/artifacts/arena"))
    a = p.parse_args()
    pa.set_cpu_count(4)
    pa.set_io_thread_count(4)
    out = a.root / "augmented_data"
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(out)
    old = pd.read_parquet(a.root / "data/train.parquet")
    seal = json.loads((a.root / "data/train_complete.json").read_text())
    assert seal["partition"] == "A" and seal["holdout2_rows_read"] == 0
    assert sha(a.root / "data/train.parquet") == seal["train_sha256"]
    selected = np.load(a.root / "data/train_s1.npy")
    part = np.load(a.arena / "finalize_v1/s1_partition.npy")
    assert np.all(part[selected] == 0)
    sel = np.zeros(len(part), bool)
    sel[selected] = True
    ref = pd.read_parquet(a.arena / "v1/reference.parquet").fillna("")
    legacy_raw = pq.read_table(a.arena / "v1/pool.parquet", columns=["raw_row_idx"])["raw_row_idx"].to_numpy()
    raw = pd.read_parquet(a.r / "texts/train_pool.parquet", columns=["entity_id", "text"])
    n_pool = len(raw)
    assert n_pool == 10_320_219
    old_right = legacy_raw[old.right_idx.to_numpy()]
    assert np.array_equal(raw.entity_id.to_numpy()[old_right], old.rec_id)
    old_keys = old.left_idx.to_numpy(dtype=np.int64) * n_pool + old_right
    reverse_path = a.r / "out/r_reverse_pairs_a.parquet"
    rv = pd.read_parquet(reverse_path, columns=["left_idx", "raw_row_idx"])
    rv = rv[sel[rv.left_idx.to_numpy()]]
    native_path = a.u / "nat/arena_candidates.parquet"
    nt = pd.read_parquet(native_path, columns=["s1", "raw", "rank"])
    nt = nt[sel[nt.s1.to_numpy()] & (nt["rank"].to_numpy() == 0)]
    candidate_keys = np.union1d(rv.left_idx.to_numpy(dtype=np.int64) * n_pool + rv.raw_row_idx.to_numpy(),
                              nt.s1.to_numpy(dtype=np.int64) * n_pool + nt.raw.to_numpy())
    new_keys = np.setdiff1d(candidate_keys, old_keys)
    l, r = new_keys // n_pool, new_keys % n_pool
    assert np.all(sel[l])
    # Membership is fixed before opening the allowed A truth payloads.
    candidate_digest = hashlib.sha256(new_keys.astype("<i8").tobytes()).hexdigest()
    id_to_left = {ref.entity_id.iloc[i]: int(i) for i in selected}
    owner, counts, decoded = {}, np.zeros(len(ref), np.int32), 0
    truth_path = a.u / "nat/gt/train_ground_truth.tsv"
    with truth_path.open() as f:
        assert f.readline().strip() == "source1_entity_id\tmatched_entity_ids"
        for line in f:
            sid, _, payload = line.rstrip("\r\n").partition("\t")
            if sid not in id_to_left:
                continue
            li = id_to_left[sid]
            ids = [x for x in payload.split(",") if x]
            counts[li] = len(ids)
            decoded += 1
            for rid in ids:
                assert rid not in owner
                owner[rid] = li
    assert decoded == len(selected)
    assert np.array_equal(counts[old.left_idx], old.true_count)
    rec_ids = raw.entity_id.to_numpy()[r]
    y = np.fromiter((owner.get(rid, -1) == li for rid, li in zip(rec_ids, l)), bool, len(l))
    assert len(new_keys) == 186_722 and int(y.sum()) == 4_444
    names = (ref.business_name + " | " + ref.business_address).to_numpy()
    extra = pd.DataFrame({
        "pair_row": int(old.pair_row.max()) + 1 + np.arange(len(l), dtype=np.int64),
        "left_idx": l.astype(np.int32), "right_idx": r.astype(np.int32),
        "s1_id": ref.entity_id.to_numpy()[l], "rec_id": rec_ids,
        "left_text": names[l],
        "right_text": [t.removeprefix("passage: ") for t in raw.text.to_numpy()[r]],
        "source": [rid[:2] for rid in rec_ids], "label": y.astype(np.int8),
        "true_count": counts[l], "v1_tree": np.full(len(l), .5, np.float32),
        "v1_blend": np.zeros(len(l), np.float32)})
    old["right_idx"] = old_right.astype(np.int32)
    old["from_union"], extra["from_union"] = False, True
    combined = pd.concat([old, extra], ignore_index=True)
    assert not combined.duplicated(["s1_id", "rec_id"]).any()
    assert combined.pair_row.is_unique and np.all(part[combined.left_idx] == 0)
    combined.to_parquet(out / "train.parquet", index=False, compression="zstd")
    report = {"state": "complete", "partition": "A", "holdout2_rows_read": 0,
              "train_s1": len(selected), "pairs": len(combined), "positives": int(combined.label.sum()),
              "added_pairs": len(extra), "added_positives": int(y.sum()), "added_negatives": int((~y).sum()),
              "parent_train_sha256": seal["train_sha256"], "train_sha256": sha(out / "train.parquet"),
              "partition_sha256": seal["partition_sha256"], "candidate_keys_sha256": candidate_digest,
              "right_idx_semantics": "original full raw S2-then-S3 row order",
              "candidate_sources": {str(p): sha(p) for p in (reverse_path, native_path)},
              "labels": "Only the same 200000 A businesses' raw truth; membership fixed first",
              "new_negative_v1_proxy": "tree=.5, blend=0 (not a measured v1 score)"}
    (out / "train_complete.json").write_text(json.dumps(report, indent=2) + "\n")
    print(time.strftime("%H:%M:%S"), json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
