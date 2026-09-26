"""Score the corrected name-only rule on arena B_REST.

The rule is predeclared: unique empty-address exact names are added, shared
empty-address exact names are dropped, and the exact-name owner wins. SLICE
is not read. HOLDOUT-2 is refused.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule, one_owner
from ber.name_only import is_empty_address, name_multiplicity, norm_name

ART = Path(os.environ.get("ARENA", "/opt/az-ml-new/artifacts/arena"))
INDEX = Path(os.environ.get("WAVE_INDEX", "/opt/az-ml-new/runs/cpu_wave"))
OUT = Path(os.environ.get("NAME_OUT", "/opt/az-ml-new/runs/name_only_arena"))


def _rate(mask, label):
    n = int(mask.sum())
    correct = int((mask & label).sum())
    return {"n": n, "true": correct, "precision": float(correct / n) if n else None}


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
    pool_empty = np.fromiter((is_empty_address(x) for x in pool.business_address), dtype=bool, count=len(pool))
    del pool, ref

    left = np.asarray(np.load(INDEX / "left.npy", mmap_mode="r"))
    right = np.asarray(np.load(INDEX / "right.npy", mmap_mode="r"))
    blend = np.asarray(np.load(ART / "v1" / "probabilities.npy", mmap_mode="r"))
    label = np.asarray(np.load(ART / "labels" / "label.npy", mmap_mode="r"))
    print("applying one-owner then name-only", flush=True)
    base = one_owner(blend >= 0.775, left, right, blend)
    new, info = name_only_rule(
        base, left, right,
        pool_empty[right],
        (s1_code[left] == rec_code[right]) & (s1_code[left] >= 0),
        s1_mult[left],
    )
    true_count = np.load(ART / "labels" / "true_count.npy")
    partition = np.load(ART / "finalize_v1" / "s1_partition.npy")
    components = np.load(ART / "finalize_v1" / "component_ids.npy")
    population = partition == 1
    base_eval = evaluate(base, left, label, true_count, population=population, return_per_s1=True)
    new_eval = evaluate(new, left, label, true_count, population=population, return_per_s1=True)
    paired = component_bootstrap(base_eval["per_s1"], new_eval["per_s1"], components, n=300, seed=0, population=population)
    no_match = population & (true_count == 0)

    def false_merge(accept):
        k = np.bincount(left[accept], minlength=len(true_count))
        hit = no_match & (k > 0)
        return {"no_match_s1": int(no_match.sum()), "false_merges": int(hit.sum()),
                "rate": float(hit.sum() / no_match.sum()) if no_match.any() else None}

    def slim(result):
        return {key: result[key] for key in ("macro_f05", "precision", "recall", "false_links", "s1", "singleton_accuracy") if key in result}

    report = {
        "population": "B_REST, rule predeclared, SLICE unread, holdout2 unread",
        "baseline": slim(base_eval),
        "name_only": slim(new_eval),
        "paired_bootstrap": paired,
        "edits": {key: _rate(info[key], label) for key in ("added", "dropped", "owner_removed")},
        "conflicted_records": info["conflicted_records"],
        "no_match_false_merges": {"baseline": false_merge(base), "name_only": false_merge(new)},
        "holdout2_rows_read": 0,
        "slice_used_for_selection": False,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
