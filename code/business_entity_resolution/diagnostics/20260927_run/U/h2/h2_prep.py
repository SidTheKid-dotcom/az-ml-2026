"""HOLDOUT-2 final check, step 1 (label-free): reference rows and retrieval tokens (27 Sep, Agent U).

Reads the ID-only reservation (reserved_ids.txt; sorted-ID sha256 must equal reservation.json) and the raw
train_source1.tsv with v1's parser (dtype=str, keep_default_na=False, raw order). Writes H2/reference.parquet
(entity_id, business_name, business_address, country, raw_source1_row) and the joint S1 token set for the encoder
(R's layout): tokens/h2_s1.{ids,off,cc}.npy = arena train_s1 tokens followed by the 50,000 HOLDOUT-2 S1 tokens
("query: {name} | {address}", truncation 128). No ground-truth file is opened.

    python h2_prep.py
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path("/home/ubuntu/run27/U")
H2 = ROOT / "h2"
EXPECTED_SORTED_SHA = "7c98dcaef35d5a392a051e0525f854a2a6ceb55c366616ff2184abd98aa4ae63"
COLS = ["entity_id", "business_name", "business_address", "country"]
CC = {"US": 0, "India": 1, "France": 2}


def main():
    ids = [s.strip() for s in (H2 / "reserved_ids.txt").read_text().split("\n") if s.strip()]
    digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    if len(ids) != 50_000 or digest != EXPECTED_SORTED_SHA:
        raise SystemExit(f"reservation mismatch: {len(ids)} ids, digest {digest}")
    want = set(ids)
    pieces, start = [], 0
    for frame in pd.read_csv(H2 / "raw/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                             usecols=COLS, chunksize=200_000):
        sel = frame.entity_id.isin(want).to_numpy()
        if sel.any():
            piece = frame.loc[sel, COLS].copy()
            piece["raw_source1_row"] = start + np.flatnonzero(sel)
            pieces.append(piece)
        start += len(frame)
    ref = pd.concat(pieces, ignore_index=True)
    if len(ref) != 50_000 or ref.entity_id.duplicated().any():
        raise SystemExit("HOLDOUT-2 S1s not found exactly once in raw source 1")
    arena_ids = set(pq.read_table(ROOT / "data/arena/reference.parquet", columns=["entity_id"]).column(
        "entity_id").to_pylist())
    if want & arena_ids:
        raise SystemExit("HOLDOUT-2 overlaps the arena")
    ref.to_parquet(H2 / "reference.parquet", index=False)
    counts = ref.country.value_counts().to_dict()
    # tokens: arena S1 tokens (R) + HOLDOUT-2 S1 tokens
    tok = Tokenizer.from_file(str(ROOT / "rmodel/tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length=128)
    texts = ("query: " + ref.business_name.fillna("").astype(str) + " | "
             + ref.business_address.fillna("").astype(str)).tolist()
    enc = tok.encode_batch(texts, add_special_tokens=True)
    h_len = np.asarray([len(e.ids) for e in enc], np.int64)
    h_ids = np.fromiter((i for e in enc for i in e.ids), np.int32)
    h_cc = ref.country.map(CC).to_numpy(np.int8)
    t = ROOT / "rtokens"
    a_ids, a_off, a_cc = np.load(t / "train_s1.ids.npy"), np.load(t / "train_s1.off.npy"), np.load(t / "train_s1.cc.npy")
    out = H2 / "tokens"
    out.mkdir(parents=True, exist_ok=True)
    off = np.concatenate([a_off, a_off[-1] + np.cumsum(h_len)])
    np.save(out / "h2_s1.ids.npy", np.concatenate([a_ids, h_ids]))
    np.save(out / "h2_s1.off.npy", off)
    np.save(out / "h2_s1.cc.npy", np.concatenate([a_cc, h_cc]))
    for x in ("ids", "off", "cc"):
        p = out / f"h2_pool.{x}.npy"
        if not p.exists():
            p.symlink_to(t / f"train_pool.{x}.npy")
    man = {"holdout2_s1": len(ref), "country_counts": counts, "sorted_id_sha256": digest,
           "arena_s1_in_token_set": int(len(a_cc)), "joint_s1": int(len(a_cc) + len(h_cc)),
           "h2_rows_in_joint_s1": [int(len(a_cc)), int(len(a_cc) + len(h_cc))],
           "raw_source1_rows_scanned": int(start), "labels_read": False}
    (H2 / "prep_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
