"""US/India transfer audit of the France pseudo-positive rule.

The rule is applied with labels closed. label.npy is opened only after the
selected pair rows have been written. HOLDOUT-2 is refused. This does not
train anything and does not score SLICE.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from ber.forensic_features import EMPTY, default_vocabulary, forensic_features, s1_multiplicities

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
OUT = Path(os.environ.get("AUDIT_OUT", "/opt/az-ml-new/runs/france_transfer_audit"))
P_BLEND = 0.9
RIVAL_BLEND = 0.5
STREET_JACCARD = 0.5


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def guard() -> None:
    summary = json.loads((ART / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("refusing labels that read HOLDOUT-2")


def pd_factorize(values):
    import pandas as pd
    return pd.factorize(values, sort=False)


def main() -> None:
    guard()
    OUT.mkdir(parents=True, exist_ok=True)
    vocab = default_vocabulary()
    ref = pq.read_table(ART / "v1" / "reference.parquet",
                        columns=["business_name", "business_address", "country"]).to_pandas()
    pool = pq.read_table(ART / "v1" / "pool.parquet",
                         columns=["business_name", "business_address", "country"]).to_pandas()
    log(f"text loaded s1={len(ref)} records={len(pool)}")
    mult = s1_multiplicities(ref.business_name, ref.business_address, ref.country)
    rec_key = np.empty(len(pool), dtype=object)
    for code, sub in pool.groupby("country", sort=False):
        keys = s1_multiplicities(sub.business_name, sub.business_address, code).exact_key.to_numpy()
        rec_key[sub.index.to_numpy()] = keys
    both = np.concatenate([mult.exact_key.to_numpy(), rec_key])
    codes, _ = pd_factorize(both)
    s1_code, rec_code = codes[:len(ref)], codes[len(ref):]
    s1_unique = (mult.key_count.to_numpy() == 1) & (mult.exact_key.to_numpy() != "")
    country = ref.country.to_numpy()
    log("keys ready")

    blend = np.load(ART / "v1" / "probabilities.npy", mmap_mode="r")
    n_pairs = len(blend)
    claims = np.zeros(len(pool), dtype=np.int32)
    cand_row, cand_left, cand_right, cand_blend = [], [], [], []
    pos = 0
    parquet = pq.ParquetFile(ART / "v1" / "pairs.parquet")
    for batch in parquet.iter_batches(batch_size=2_000_000, columns=["left_idx", "right_idx"]):
        left = batch.column("left_idx").to_numpy()
        right = batch.column("right_idx").to_numpy()
        score = np.asarray(blend[pos:pos + len(left)])
        claims += np.bincount(right[score >= RIVAL_BLEND], minlength=len(pool)).astype(np.int32)
        keep = s1_unique[left] & (s1_code[left] == rec_code[right]) & (score >= P_BLEND)
        if keep.any():
            rows = np.flatnonzero(keep)
            cand_row.append(rows + pos)
            cand_left.append(left[rows])
            cand_right.append(right[rows])
            cand_blend.append(score[rows])
        pos += len(left)
    if pos != n_pairs:
        raise SystemExit(f"streamed {pos} pairs, expected {n_pairs}")
    row = np.concatenate(cand_row) if cand_row else np.empty(0, np.int64)
    left = np.concatenate(cand_left) if cand_left else np.empty(0, np.int32)
    right = np.concatenate(cand_right) if cand_right else np.empty(0, np.int32)
    score = np.concatenate(cand_blend) if cand_blend else np.empty(0, np.float32)
    log(f"unique-key blend>=0.9 candidates {len(row)}")

    selected_rows = []
    funnel = {}
    for code in ("US", "India"):
        mask = country[left] == code
        if not mask.any():
            funnel[code] = {"candidates": 0}
            continue
        feats = forensic_features(
            ref.business_name.to_numpy()[left[mask]], ref.business_address.to_numpy()[left[mask]],
            pool.business_name.to_numpy()[right[mask]], pool.business_address.to_numpy()[right[mask]],
            code, vocab)
        street = feats.street_token_jaccard.to_numpy() >= STREET_JACCARD
        nonempty = feats.address_kind.to_numpy() != EMPTY
        own = (score[mask] >= RIVAL_BLEND).astype(np.int32)
        no_rival = (claims[right[mask]] - own) == 0
        keep = street & nonempty & no_rival
        chosen = row[mask][keep]
        selected_rows.append(chosen)
        funnel[code] = {
            "unique_key_blend_ge_0.9": int(mask.sum()),
            "street_and_address_and_no_rival": int(keep.sum()),
        }
        log(f"{code} selected {len(chosen)}")

    chosen = np.concatenate(selected_rows) if selected_rows else np.empty(0, np.int64)
    np.save(OUT / "selected_pair_rows.npy", chosen)
    # Labels stay closed until the selection file exists.
    if not (OUT / "selected_pair_rows.npy").is_file():
        raise SystemExit("selection was not sealed")
    label = np.load(ART / "labels" / "label.npy", mmap_mode="r")
    if len(label) != n_pairs:
        raise SystemExit("label.npy is not aligned to pairs")
    hit = np.asarray(label[chosen], dtype=bool) if len(chosen) else np.empty(0, dtype=bool)
    report = {
        "rule": "unique exact key, street jaccard>=0.5, blend>=0.9, no rival>=0.5, address nonempty",
        "labels_used_for_selection": False,
        "holdout2_rows_read": 0,
        "slice_scored": False,
        "funnel": funnel,
        "selected_pairs": int(len(chosen)),
        "precision": float(hit.mean()) if len(hit) else None,
        "true_pairs": int(hit.sum()) if len(hit) else 0,
        "false_pairs": int((~hit).sum()) if len(hit) else 0,
        "note": "Every selected pair has blend>=0.9, so this rule cannot add links the current threshold missed.",
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps(report))


if __name__ == "__main__":
    main()
