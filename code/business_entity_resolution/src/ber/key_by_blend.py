"""Exact-key adds and moves, split by the model's score.

Unconditional key edits are already rejected. This asks whether the key is
precise once the blend score is high enough. The cutoff is chosen on
partition A only. B_REST is scored once, and only if A has a cutoff at
0.80 precision with at least 500 links. SLICE is not used. HOLDOUT-2 is refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule, one_owner
from ber.forensic_features import EMPTY, address_kinds, exact_keys, s1_multiplicities
from ber.name_only import name_multiplicity, norm_name

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("KEY_OUT", "/opt/az-ml-new/runs/key_by_blend"))
CUTOFFS = (0.0, 0.1, 0.3, 0.5, 0.7)
MIN_PRECISION = 0.80
MIN_COUNT = 500


def rates(correct_mask, mask):
    n = int(mask.sum())
    correct = int((mask & correct_mask).sum())
    return {"n": n, "correct": correct, "precision": float(correct / n) if n else None}


def table(score, correct, mask):
    rows = []
    for low, high in ((-1.0, 0.0), (0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.775), (0.775, 1.01)):
        if low < 0:
            band = mask & np.isnan(score)
        else:
            band = mask & (score >= low) & (score < high)
        entry = rates(correct, band)
        entry["low"] = low
        rows.append(entry)
    return rows


def choose_cutoff(score, correct, mask):
    """Lowest cutoff whose retrieved links are precise enough. None if none qualify."""
    chosen = None
    for cutoff in CUTOFFS:
        band = mask & ~np.isnan(score) & (score >= cutoff)
        stat = rates(correct, band)
        stat["cutoff"] = cutoff
        if stat["n"] >= MIN_COUNT and stat["precision"] is not None and stat["precision"] >= MIN_PRECISION:
            chosen = stat
            break
    return chosen


def slim(result):
    return {key: result[key] for key in ("macro_f05", "precision", "recall", "false_links", "s1", "singleton_accuracy")}


def main() -> None:
    summary = json.loads((ART / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("refusing HOLDOUT-2 labels")
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(ART / "v1" / "reference.parquet",
                        columns=["business_name", "business_address", "country"]).to_pandas()
    pool = pq.read_table(ART / "v1" / "pool.parquet",
                         columns=["business_name", "business_address", "country"]).to_pandas()
    print("building exact keys", flush=True)
    mult = s1_multiplicities(ref.business_name, ref.business_address, ref.country)
    s1_key = np.array([f"{c}|{k}" if k else "" for c, k in zip(ref.country.to_numpy(), mult.exact_key.to_numpy())],
                      dtype=object)
    unique = (mult.key_count.to_numpy() == 1) & (s1_key != "")
    rec_key = np.empty(len(pool), dtype=object)
    for code, sub in pool.groupby("country", sort=False):
        keys = exact_keys(sub.business_name, sub.business_address).astype(object)
        rec_key[sub.index.to_numpy()] = np.array([f"{code}|{k}" if k else "" for k in keys], dtype=object)
    codes, _ = pd.factorize(np.concatenate([s1_key, rec_key]), sort=False)
    s1_code, rec_code = codes[:len(ref)], codes[len(ref):]
    lookup = np.full(int(codes.max()) + 1, -1, np.int32)
    lookup[s1_code[unique]] = np.flatnonzero(unique).astype(np.int32)
    key_owner = lookup[rec_code]
    key_owner[rec_key == ""] = -1
    del s1_key, rec_key, codes

    print("normalizing names for the empty-address baseline", flush=True)
    s1_name = ref.business_name.map(norm_name).to_numpy()
    rec_name = pool.business_name.map(norm_name).to_numpy()
    s1_mult = name_multiplicity(ref.country, s1_name)
    both, _ = pd.factorize(np.concatenate([s1_name, rec_name]), sort=False)
    both = both.astype(np.int32, copy=False)
    name_s1, name_rec = both[:len(ref)], both[len(ref):]
    name_s1[s1_name == ""] = -1
    name_rec[rec_name == ""] = -1
    del s1_name, rec_name, ref
    empty = address_kinds(pool.business_address) == EMPTY
    del pool

    true_owner = np.load(ART / "labels" / "rec_owner.npy")
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    key_owner[(partition[np.maximum(key_owner, 0)] == 2) & (key_owner >= 0)] = -1
    left = np.asarray(np.load(INDEX / "left.npy", mmap_mode="r"))
    right = np.asarray(np.load(INDEX / "right.npy", mmap_mode="r"))
    blend = np.asarray(np.load(ART / "v1" / "probabilities.npy", mmap_mode="r"))
    label = np.asarray(np.load(ART / "labels" / "label.npy", mmap_mode="r"))
    exact_name = (name_s1[left] == name_rec[right]) & (name_s1[left] >= 0)
    base = one_owner(blend >= 0.775, left, right, blend)
    accept, _ = name_only_rule(base, left, right, empty[right], exact_name, s1_mult[left])
    model_owner = np.full(len(key_owner), -1, np.int32)
    kept = np.flatnonzero(accept)
    model_owner[right[kept]] = left[kept]
    list_len = np.bincount(left[kept], minlength=len(partition))

    key_score = np.full(len(key_owner), np.nan, dtype=np.float32)
    retrieved = (key_owner[right] >= 0) & (left == key_owner[right])
    key_score[right[retrieved]] = blend[retrieved]
    correct = true_owner == key_owner
    has_key = key_owner >= 0
    on_a = has_key & (partition[np.maximum(key_owner, 0)] == 0)
    add = on_a & (model_owner < 0) & (list_len[np.maximum(key_owner, 0)] < 11)
    move = on_a & (model_owner >= 0) & (model_owner != key_owner)
    add_choice = choose_cutoff(key_score, correct, add)
    move_choice = choose_cutoff(key_score, correct, move)
    report = {
        "selection_population": "partition A",
        "add_bins_on_A": table(key_score, correct, add),
        "move_bins_on_A": table(key_score, correct, move),
        "chosen_add_cutoff": add_choice,
        "chosen_move_cutoff": move_choice,
        "holdout2_rows_read": 0,
        "slice_used_for_selection": False,
    }
    family = None
    if add_choice and move_choice:
        family = "add" if add_choice["n"] * (add_choice["precision"] - 0.75) >= move_choice["n"] * (move_choice["precision"] - 0.55) else "move"
    elif add_choice:
        family = "add"
    elif move_choice:
        family = "move"
    report["confirmed_family"] = family
    if family is not None:
        cutoff = (add_choice if family == "add" else move_choice)["cutoff"]
        on_b = has_key & (partition[np.maximum(key_owner, 0)] == 1)
        scored = on_b & ~np.isnan(key_score) & (key_score >= cutoff)
        if family == "add":
            scored &= (model_owner < 0) & (list_len[np.maximum(key_owner, 0)] < 11)
            extra = np.zeros(len(left), dtype=bool)
            extra[retrieved & scored[right]] = True
            new = accept | extra
        else:
            scored &= (model_owner >= 0) & (model_owner != key_owner)
            drop = np.zeros(len(left), dtype=bool)
            drop[(model_owner[right] == left) & scored[right]] = True
            extra = np.zeros(len(left), dtype=bool)
            extra[retrieved & scored[right] & (left == key_owner[right])] = True
            new = (accept & ~drop) | extra
        true_count = np.load(ART / "labels" / "true_count.npy")
        components = np.load(ART / "finalize_v1" / "component_ids.npy")
        population = partition == 1
        before = evaluate(accept, left, label, true_count, population=population, return_per_s1=True)
        after = evaluate(new, left, label, true_count, population=population, return_per_s1=True)
        report["baseline_name_only"] = slim(before)
        report["confirmed"] = slim(after)
        report["paired_bootstrap"] = component_bootstrap(
            before["per_s1"], after["per_s1"], components, n=300, seed=0, population=population)
        report["confirmed_precision"] = rates(correct, scored)
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
