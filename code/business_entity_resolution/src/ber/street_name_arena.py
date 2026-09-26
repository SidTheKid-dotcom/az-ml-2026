"""Unique-name adds for unowned records that share street tokens.

The empty-address name-only rule stays as the baseline. This asks whether an
unowned record with the same unique normalized name and street-token Jaccard
at least 0.5 should be added. Scored once on B_REST. SLICE is not used.
HOLDOUT-2 is refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule, one_owner
from ber.forensic_features import EMPTY, address_kinds, parse_addresses
from ber.name_only import name_multiplicity, norm_name

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("NAME_OUT", "/opt/az-ml-new/runs/street_name_arena"))


def _rate(mask, label):
    n = int(np.asarray(mask).sum())
    correct = int((np.asarray(mask) & np.asarray(label)).sum())
    return {"n": n, "true": correct, "precision": float(correct / n) if n else None}


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
    print(f"normalizing {len(ref)} S1 names and {len(pool)} record names", flush=True)
    s1_name = ref.business_name.map(norm_name).to_numpy()
    rec_name = pool.business_name.map(norm_name).to_numpy()
    s1_mult = name_multiplicity(ref.country, s1_name)
    both, _ = pd.factorize(np.concatenate([s1_name, rec_name]), sort=False)
    both = both.astype(np.int32, copy=False)
    s1_code, rec_code = both[:len(ref)], both[len(ref):]
    s1_code[s1_name == ""] = -1
    rec_code[rec_name == ""] = -1
    del s1_name, rec_name
    empty = address_kinds(pool.business_address) == EMPTY

    left = np.asarray(np.load(INDEX / "left.npy", mmap_mode="r"))
    right = np.asarray(np.load(INDEX / "right.npy", mmap_mode="r"))
    blend = np.asarray(np.load(ART / "v1" / "probabilities.npy", mmap_mode="r"))
    label = np.asarray(np.load(ART / "labels" / "label.npy", mmap_mode="r"))
    exact = (s1_code[left] == rec_code[right]) & (s1_code[left] >= 0)
    base = one_owner(blend >= 0.775, left, right, blend)
    owned = np.zeros(len(pool), dtype=bool)
    owned[right[base]] = True
    after_empty, _ = name_only_rule(base, left, right, empty[right], exact, s1_mult[left])
    owned[right[after_empty]] = True
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    candidate = exact & (s1_mult[left] == 1) & ~owned[right] & ~empty[right] & (partition[left] != 2)
    rows = np.flatnonzero(candidate)
    print(f"candidate unowned unique-name pairs {len(rows)}", flush=True)
    s1_rows = np.unique(left[rows])
    rec_rows = np.unique(right[rows])
    s1_street = [frozenset()] * len(ref)
    rec_street = [frozenset()] * len(pool)
    parsed_s1, _ = parse_addresses(ref.business_address.iloc[s1_rows], ref.country.iloc[s1_rows], fuzzy_admin=False)
    parsed_rec, _ = parse_addresses(pool.business_address.iloc[rec_rows], pool.country.iloc[rec_rows], fuzzy_admin=False)
    for index, tokens in zip(s1_rows, parsed_s1):
        s1_street[index] = tokens
    for index, tokens in zip(rec_rows, parsed_rec):
        rec_street[index] = tokens
    del ref, pool, parsed_s1, parsed_rec
    jaccard = np.zeros(len(rows), dtype=np.float32)
    for i, row in enumerate(rows):
        a, b = s1_street[left[row]], rec_street[right[row]]
        if a and b:
            jaccard[i] = len(a & b) / len(a | b)
    keep = jaccard >= 0.5
    added = np.zeros(len(left), dtype=bool)
    added[rows[keep]] = True
    # One owner and the list cap: a record has only one unique-name S1, so the
    # only cap is 11 links on that S1. Keep the highest blend under the cap.
    order = np.argsort(-blend[rows[keep]], kind="stable")
    chosen_rows = rows[keep][order]
    lengths = np.bincount(left[after_empty], minlength=len(partition))
    accept_add = np.zeros(len(chosen_rows), dtype=bool)
    for i, row in enumerate(chosen_rows):
        owner = left[row]
        if lengths[owner] < 11:
            lengths[owner] += 1
            accept_add[i] = True
    new = after_empty.copy()
    new[chosen_rows[accept_add]] = True

    true_count = np.load(ART / "labels" / "true_count.npy")
    components = np.load(ART / "finalize_v1" / "component_ids.npy")
    population = partition == 1
    before = evaluate(after_empty, left, label, true_count, population=population, return_per_s1=True)
    after = evaluate(new, left, label, true_count, population=population, return_per_s1=True)
    paired = component_bootstrap(before["per_s1"], after["per_s1"], components, n=300, seed=0, population=population)
    on_b = population[left[rows]]
    report = {
        "population": "B_REST unique-name street adds on top of empty-address name-only",
        "street_parse": "admin phrases dropped without fuzzy matching",
        "name_only_empty": slim(before),
        "plus_street": slim(after),
        "paired_bootstrap": paired,
        "candidates": int(on_b.sum()),
        "jaccard_ge_0.5": _rate(on_b & (jaccard >= 0.5), label[rows]),
        "jaccard_eq_1": _rate(on_b & (jaccard == 1), label[rows]),
        "added_under_cap": _rate(population[left[chosen_rows[accept_add]]], label[chosen_rows[accept_add]]),
        "holdout2_rows_read": 0,
        "slice_used_for_selection": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
