"""Precision of exact-key moves and unowned adds on the labeled arena.

The key is the manual's exact key, matched within country, Source-1 keys kept
only when unique. Labels are used only to score a predeclared rule. SLICE is
not used. HOLDOUT-2 is refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import one_owner
from ber.forensic_features import exact_keys, s1_multiplicities

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("KEY_OUT", "/opt/az-ml-new/runs/key_residual"))


def rates(true_owner, key_owner, mask):
    hit = mask & (key_owner >= 0)
    correct = hit & (true_owner == key_owner)
    n = int(hit.sum())
    return {"n": n, "correct": int(correct.sum()), "precision": float(correct.sum() / n) if n else None}


def main() -> None:
    summary = json.loads((ART / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("refusing HOLDOUT-2 labels")
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(ART / "v1" / "reference.parquet",
                        columns=["business_name", "business_address", "country"]).to_pandas()
    pool = pq.read_table(ART / "v1" / "pool.parquet",
                         columns=["business_name", "business_address", "country"]).to_pandas()
    mult = s1_multiplicities(ref.business_name, ref.business_address, ref.country)
    s1_key = np.array([f"{c}|{k}" if k else "" for c, k in zip(ref.country.to_numpy(), mult.exact_key.to_numpy())],
                      dtype=object)
    unique = (mult.key_count.to_numpy() == 1) & (s1_key != "")
    rec_key = np.empty(len(pool), dtype=object)
    for code, sub in pool.groupby("country", sort=False):
        keys = exact_keys(sub.business_name, sub.business_address).astype(object)
        rec_key[sub.index.to_numpy()] = np.array([f"{code}|{k}" if k else "" for k in keys], dtype=object)
    import pandas as pd
    codes, _ = pd.factorize(np.concatenate([s1_key, rec_key]), sort=False)
    s1_code, rec_code = codes[:len(ref)], codes[len(ref):]
    lookup = np.full(int(codes.max()) + 1, -1, np.int32)
    lookup[s1_code[unique]] = np.flatnonzero(unique).astype(np.int32)
    key_owner = lookup[rec_code]
    key_owner[rec_key == ""] = -1

    true_owner = np.load(ART / "labels" / "rec_owner.npy")
    if len(true_owner) != len(pool):
        raise SystemExit("rec_owner is not aligned to the pool")
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    # Drop key owners that sit in SLICE so the confirmation is not a SLICE read.
    in_slice = partition == 2
    key_owner[in_slice[np.maximum(key_owner, 0)] & (key_owner >= 0)] = -1

    left = np.load(INDEX / "left.npy", mmap_mode="r")
    right = np.load(INDEX / "right.npy", mmap_mode="r")
    blend = np.load(ART / "v1" / "probabilities.npy", mmap_mode="r")
    accept = one_owner(np.asarray(blend) >= 0.775, left, right, blend)
    model_owner = np.full(len(pool), -1, np.int32)
    rows = np.flatnonzero(accept)
    model_owner[np.asarray(right[rows])] = np.asarray(left[rows])
    list_len = np.bincount(np.asarray(left[rows]), minlength=len(ref))

    has_key = key_owner >= 0
    move = has_key & (model_owner >= 0) & (model_owner != key_owner)
    add = has_key & (model_owner < 0)
    add_cap = add & (list_len[np.maximum(key_owner, 0)] < 11)
    country = ref.country.to_numpy()
    report = {
        "population": "arena US/India, SLICE key-owners excluded, holdout2 unread",
        "overall_key_precision": rates(true_owner, key_owner, has_key),
        "moves": rates(true_owner, key_owner, move),
        "unowned_adds": rates(true_owner, key_owner, add),
        "unowned_adds_list_under_11": rates(true_owner, key_owner, add_cap),
        "by_country": {},
        "holdout2_rows_read": 0,
        "slice_used_for_selection": False,
    }
    for code in ("US", "India"):
        here = country[np.maximum(key_owner, 0)] == code
        report["by_country"][code] = {
            "overall": rates(true_owner, key_owner, has_key & here),
            "moves": rates(true_owner, key_owner, move & here),
            "unowned_adds": rates(true_owner, key_owner, add & here),
            "unowned_adds_list_under_11": rates(true_owner, key_owner, add_cap & here),
        }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
