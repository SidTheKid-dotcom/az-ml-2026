"""Extend the name-only rule from empty addresses to digitless addresses.

The empty-address rule is already kept. This measures only the extra step:
same unique-name add and shared-name drop, applied to addresses that contain
letters but no digits. Scored once on B_REST against the empty-address rule.
SLICE is not read. HOLDOUT-2 is refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule, one_owner
from ber.forensic_features import EMPTY, NO_DIGITS, address_kinds
from ber.name_only import name_multiplicity, norm_name

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("NAME_OUT", "/opt/az-ml-new/runs/name_nodigits_arena"))


def _rate(mask, label):
    n = int(mask.sum())
    correct = int((mask & label).sum())
    return {"n": n, "true": correct, "precision": float(correct / n) if n else None}


def slim(result):
    return {key: result[key] for key in ("macro_f05", "precision", "recall", "false_links", "s1", "singleton_accuracy")}


def main() -> None:
    summary = json.loads((ART / "labels" / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("refusing HOLDOUT-2 labels")
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(ART / "v1" / "reference.parquet", columns=["business_name", "country"]).to_pandas()
    pool = pq.read_table(ART / "v1" / "pool.parquet", columns=["business_name", "business_address"]).to_pandas()
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
    kind = address_kinds(pool.business_address)
    del pool, ref
    empty = kind == EMPTY
    nodigits = kind == NO_DIGITS
    print(f"empty {int(empty.sum())} digitless {int(nodigits.sum())}", flush=True)

    left = np.asarray(np.load(INDEX / "left.npy", mmap_mode="r"))
    right = np.asarray(np.load(INDEX / "right.npy", mmap_mode="r"))
    blend = np.asarray(np.load(ART / "v1" / "probabilities.npy", mmap_mode="r"))
    label = np.asarray(np.load(ART / "labels" / "label.npy", mmap_mode="r"))
    exact = (s1_code[left] == rec_code[right]) & (s1_code[left] >= 0)
    mult = s1_mult[left]
    base = one_owner(blend >= 0.775, left, right, blend)
    after_empty, _ = name_only_rule(base, left, right, empty[right], exact, mult)
    after_both, info = name_only_rule(after_empty, left, right, nodigits[right], exact, mult)

    true_count = np.load(ART / "labels" / "true_count.npy")
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    components = np.load(ART / "finalize_v1" / "component_ids.npy")
    population = partition == 1
    empty_eval = evaluate(after_empty, left, label, true_count, population=population, return_per_s1=True)
    both_eval = evaluate(after_both, left, label, true_count, population=population, return_per_s1=True)
    paired = component_bootstrap(empty_eval["per_s1"], both_eval["per_s1"], components, n=300, seed=0, population=population)
    no_match = population & (true_count == 0)

    def false_merge(accept):
        k = np.bincount(left[accept], minlength=len(true_count))
        hit = no_match & (k > 0)
        return {"no_match_s1": int(no_match.sum()), "false_merges": int(hit.sum()),
                "rate": float(hit.sum() / no_match.sum()) if no_match.any() else None}

    report = {
        "population": "B_REST incremental digitless-name rule on top of empty-address name-only",
        "name_only_empty": slim(empty_eval),
        "plus_digitless": slim(both_eval),
        "paired_bootstrap": paired,
        "edits": {key: _rate(info[key], label) for key in ("added", "dropped", "owner_removed")},
        "conflicted_records": info["conflicted_records"],
        "no_match_false_merges": {"name_only_empty": false_merge(after_empty), "plus_digitless": false_merge(after_both)},
        "holdout2_rows_read": 0,
        "slice_used_for_selection": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
