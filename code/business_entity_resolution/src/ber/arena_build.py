"""Arena: every eligible training S1 scored against its dense top-50 of the full pool.

`prepare` writes reference/pool/pairs in exactly the layout the frozen v1
streaming predictor consumed on test (dense-only retrieval columns, compact pool,
pairs sorted by left then right), from P2's completed dense mining. It reads no
labels. HOLDOUT-2 S1s are dropped. `labels` aligns ground truth to the arena
pairs afterwards and never reads HOLDOUT-2 rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.arena_decision import claim_graph_components, evaluate, name_only_rule, one_owner
from ber.name_only import is_empty_address, name_multiplicity, norm_name

RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
ELIGIBLE = "train_tune_eligible"
POOL_ROWS = (5_034_616, 5_285_603)
PARTITION_A = np.uint8(0)
PARTITION_B_REST = np.uint8(1)
PARTITION_SLICE = np.uint8(2)
PARTITION_NAMES = {0: "A", 1: "B_REST", 2: "SLICE"}
FROZEN_V1_PIPELINE_SHA256 = "9ac4050a72c64b05c032a17ec26f2a13b339c1b8c135791bf07b737534371f5e"


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(path: Path, values) -> None:
    """Write a NumPy array beside its destination, then atomically replace it."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_raw(path: Path, chunksize: int = 200_000):
    """Same parser settings as v1's full_dense_candidates pool construction."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                       usecols=RECORD_COLUMNS, chunksize=chunksize)


def prepare(args) -> dict:
    p2, raw, out = Path(args.p2), Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    p2_reference = pd.read_parquet(p2 / "reference.parquet", columns=["entity_id", "population", "raw_source1_row"])
    eligible = (p2_reference.population == ELIGIBLE).to_numpy()
    if int(eligible.sum()) != args.expected_references:
        raise ValueError(f"Expected {args.expected_references} eligible S1s, found {int(eligible.sum())}")
    p2_to_arena = np.full(len(p2_reference), -1, dtype=np.int64)
    p2_to_arena[eligible] = np.arange(int(eligible.sum()))
    keep = p2_reference.loc[eligible].reset_index(drop=True)

    # Reference text re-read from the raw file with v1's parser, matched by row and ID.
    rows = keep.raw_source1_row.to_numpy(dtype=np.int64)
    if np.any(np.diff(rows) <= 0):
        raise ValueError("P2 reference order is not raw Source 1 order")
    pieces, start = [], 0
    for frame in read_raw(raw / "train_source1.tsv"):
        stop = start + len(frame)
        low, high = np.searchsorted(rows, [start, stop])
        if high > low:
            pieces.append(frame.iloc[rows[low:high] - start])
        start = stop
    reference = pd.concat(pieces, ignore_index=True)
    if not np.array_equal(reference.entity_id.to_numpy(), keep.entity_id.to_numpy()):
        raise ValueError("Raw Source 1 rows do not match P2 reference IDs")
    reference[RECORD_COLUMNS].to_parquet(out / "reference.parquet", index=False)
    keep.assign(arena_left_idx=np.arange(len(keep)))[["arena_left_idx", "entity_id", "raw_source1_row"]].to_parquet(
        out / "reference_map.parquet", index=False)
    log(f"reference: {len(reference):,} eligible S1s")

    table = pq.read_table(p2 / "candidate_pairs.parquet",
                          columns=["left_idx", "right_raw_idx", "dense_score", "dense_rank"])
    left = p2_to_arena[table.column("left_idx").to_numpy()]
    mask = left >= 0
    right_raw = table.column("right_raw_idx").to_numpy().astype(np.int64)[mask]
    dense_score = table.column("dense_score").to_numpy()[mask].astype(np.float32)
    dense_rank = table.column("dense_rank").to_numpy()[mask].astype(np.int32)
    left = left[mask]
    del table
    log(f"pairs kept: {len(left):,} (dropped {int((~mask).sum()):,} HOLDOUT-2 pairs)")

    requested = np.unique(right_raw)
    pieces, start = [], 0
    for name, expected in zip(("train_source2.tsv", "train_source3.tsv"), POOL_ROWS):
        seen = 0
        for frame in read_raw(raw / name):
            stop = start + len(frame)
            low, high = np.searchsorted(requested, [start, stop])
            if high > low:
                piece = frame.iloc[requested[low:high] - start].copy()
                piece["raw_row_idx"] = requested[low:high]
                pieces.append(piece)
            start, seen = stop, seen + len(frame)
        if seen != expected:
            raise ValueError(f"{name}: {seen} rows, expected {expected}")
    pool = pd.concat(pieces, ignore_index=True)
    if not np.array_equal(pool.raw_row_idx.to_numpy(), requested) or pool.entity_id.duplicated().any():
        raise ValueError("Compact pool lost or duplicated raw rows")
    pool.to_parquet(out / "pool.parquet", index=False)
    log(f"pool: {len(pool):,} compact records")

    right = np.searchsorted(requested, right_raw)
    order = np.lexsort((right, left))
    left, right = left[order].astype(np.int32), right[order].astype(np.int32)
    right_raw, dense_score, dense_rank = right_raw[order].astype(np.int32), dense_score[order], dense_rank[order]
    if np.any((left[1:] == left[:-1]) & (right[1:] == right[:-1])):
        raise ValueError("Duplicate arena pairs")
    columns = {"left_idx": left, "right_idx": right, "right_raw_idx": right_raw,
               "dense_score": np.clip(dense_score, -1, 1).astype(np.float32), "dense_rank": dense_rank,
               "name_score": np.zeros(len(left), np.float32), "address_score": np.zeros(len(left), np.float32),
               "retrieval_score": np.clip(dense_score, -1, 1).astype(np.float32),
               "retrieval_rank": dense_rank.astype(np.int32), "dense_only": np.ones(len(left), np.float32)}
    pq.write_table(pa.table(columns), out / "pairs.parquet", row_group_size=1_000_000)
    manifest = {"schema": 1, "p2": str(p2), "raw": str(raw), "references": len(reference),
                "pool_records": len(pool), "pairs": int(len(left)), "labels_read": False,
                "holdout2_excluded": True, "population": ELIGIBLE,
                "feature_semantics": {"name_score": "zero", "address_score": "zero",
                                      "retrieval_score": "dense cosine", "dense_only": "one"},
                "sha256": {name: sha256(out / name) for name in ("reference.parquet", "pool.parquet", "pairs.parquet")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(json.dumps({key: manifest[key] for key in ("references", "pool_records", "pairs")}))
    return manifest


def capped_claim_components(s1_idx, rec_idx, blend, n_s1, *, force_a=None, force_b=None, threshold=0.05,
                            max_component_fraction=0.05, threshold_step=0.05, max_threshold=1.0):
    """Raise the claim cut until both the size cap and forced-component isolation pass.

    The documented safe search range is ``[threshold, max_threshold]`` in
    ``threshold_step`` increments. It never weakens below the requested cut or
    silently removes all edges above the maximum.
    """
    if (not 0 <= threshold <= max_threshold <= 1 or not 0 < max_component_fraction <= 1
            or threshold_step <= 0):
        raise ValueError("Invalid component threshold, cap, threshold step, or safe-range maximum")
    forced_a = np.zeros(n_s1, dtype=bool) if force_a is None else np.asarray(force_a, dtype=bool)
    forced_b = np.zeros(n_s1, dtype=bool) if force_b is None else np.asarray(force_b, dtype=bool)
    if forced_a.shape != (n_s1,) or forced_b.shape != (n_s1,) or np.any(forced_a & forced_b):
        raise ValueError("Forced A/B masks must be disjoint and aligned to the S1 population")
    attempts = []
    cut = float(threshold)
    while True:
        component = claim_graph_components(s1_idx, rec_idx, blend, cut, n_s1=n_s1)
        sizes = np.bincount(component, minlength=(int(component.max()) + 1 if component.size else 0))
        largest = int(sizes.max()) if sizes.size else 0
        fraction = largest / max(1, n_s1)
        forced_a_components = np.unique(component[forced_a])
        forced_b_components = np.unique(component[forced_b])
        conflicts = np.intersect1d(forced_a_components, forced_b_components, assume_unique=True)
        attempts.append({"threshold": cut, "components": int(len(sizes)),
                         "largest_component_s1": largest, "largest_component_fraction": fraction,
                         "forced_conflict_components": int(len(conflicts))})
        if fraction <= max_component_fraction and not len(conflicts):
            return component, cut, attempts
        if cut >= max_threshold:
            raise ValueError(
                f"No valid claim graph in safe threshold range [{threshold}, {max_threshold}]; "
                f"largest_fraction={fraction:.6f}, forced_conflict_components={len(conflicts)}")
        cut = min(max_threshold, round(cut + threshold_step, 10))


def _component_order(component_ids, seed):
    """Stable SplitMix64 ordering without Python's process-randomized hash()."""
    with np.errstate(over="ignore"):
        values = np.asarray(component_ids, dtype=np.uint64) + np.uint64(seed)
        values += np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return np.argsort(values, kind="stable")


def make_partitions(component_ids, force_a, force_b, *, a_fraction=0.70, slice_size=100_000,
                    seed=20260926):
    """Assign whole components to A/B and an exact component-disjoint SLICE within B."""
    component = np.asarray(component_ids, dtype=np.int64)
    force_a, force_b = np.asarray(force_a, dtype=bool), np.asarray(force_b, dtype=bool)
    if component.ndim != 1 or force_a.shape != component.shape or force_b.shape != component.shape:
        raise ValueError("Component and forced-assignment arrays must be aligned one-dimensional arrays")
    if not 0 < a_fraction < 1 or slice_size < 0:
        raise ValueError("a_fraction must be in (0,1) and slice_size must be nonnegative")
    if np.any(force_a & force_b):
        raise ValueError("An S1 is forced into both A and B")
    if component.size == 0:
        if slice_size:
            raise ValueError("Cannot take a nonempty SLICE from an empty arena")
        return np.empty(0, dtype=np.uint8), {"A": 0, "B": 0, "B_REST": 0, "SLICE": 0}
    _, compact = np.unique(component, return_inverse=True)
    n_components = int(compact.max()) + 1
    sizes = np.bincount(compact, minlength=n_components)
    forced_a_comp = np.bincount(compact, weights=force_a.astype(np.int8), minlength=n_components) > 0
    forced_b_comp = np.bincount(compact, weights=force_b.astype(np.int8), minlength=n_components) > 0
    conflict = forced_a_comp & forced_b_comp
    if conflict.any():
        examples = np.flatnonzero(conflict)[:10].tolist()
        raise ValueError(f"Forced A/B S1s share {int(conflict.sum())} claim components; examples={examples}")

    assignment = np.full(n_components, -1, dtype=np.int8)
    assignment[forced_a_comp] = 0
    assignment[forced_b_comp] = 1
    free = np.flatnonzero(assignment < 0)
    ordered = free[_component_order(free, seed)]
    target_a = int(round(a_fraction * len(component)))
    need = max(0, target_a - int(sizes[forced_a_comp].sum()))
    cumulative = np.concatenate(([0], np.cumsum(sizes[ordered], dtype=np.int64)))
    split = int(np.argmin(np.abs(cumulative - need)))
    assignment[ordered[:split]] = 0
    assignment[ordered[split:]] = 1

    b_components = np.flatnonzero(assignment == 1)
    b_count = int(sizes[b_components].sum())
    if slice_size > b_count:
        raise ValueError(f"SLICE needs {slice_size} S1s but B contains only {b_count}")
    remaining = int(slice_size)
    selected = np.zeros(n_components, dtype=bool)
    for comp in b_components[_component_order(b_components, seed ^ 0x51CE)]:
        size = int(sizes[comp])
        if size <= remaining:
            selected[comp] = True
            remaining -= size
            if remaining == 0:
                break
    if remaining:
        raise ValueError(f"Could not form an exact component-disjoint SLICE; {remaining} S1s remain")

    partition = np.where(assignment[compact] == 0, PARTITION_A, PARTITION_B_REST).astype(np.uint8)
    partition[selected[compact]] = PARTITION_SLICE
    counts = {name: int((partition == code).sum()) for code, name in PARTITION_NAMES.items()}
    counts["B"] = counts["B_REST"] + counts["SLICE"]
    counts["components"] = n_components
    counts["forced_A_s1"] = int(force_a.sum())
    counts["forced_B_s1"] = int(force_b.sum())
    counts["A_fraction"] = counts["A"] / len(component)
    counts["B_fraction"] = counts["B"] / len(component)
    return partition, counts


def _forced_masks(arena_ids, quick_reference, medium_reference):
    arena_ids = pd.Index(np.asarray(arena_ids, dtype=object))
    audit = {}

    def selected(path, splits, name):
        frame = pd.read_parquet(path, columns=["entity_id", "split"])
        split = frame["split"].str.lower()
        ids = frame.loc[split.isin(splits), "entity_id"]
        if ids.duplicated().any():
            raise ValueError(f"Duplicate forced S1 IDs in {path}")
        positions = arena_ids.get_indexer(ids)
        present = positions >= 0
        audit[name] = {"requested": int(len(ids)), "present": int(present.sum()),
                       "excluded_from_arena": int((~present).sum()), "source": str(path)}
        return positions[present]

    force_a = np.zeros(len(arena_ids), dtype=bool)
    force_b = np.zeros(len(arena_ids), dtype=bool)
    force_a[selected(quick_reference, {"train"}, "quick_train_to_A")] = True
    force_a[selected(medium_reference, {"train"}, "medium_train_to_A")] = True
    force_b[selected(medium_reference, {"tune", "eval", "holdout"}, "medium_development_to_B")] = True
    return force_a, force_b, audit


def _address_inputs(reference, pool, left, right):
    ref_names = np.fromiter((norm_name(x) for x in reference.business_name), dtype=object, count=len(reference))
    multiplicity = name_multiplicity(reference.country.to_numpy(), ref_names)
    pool_empty = np.fromiter((is_empty_address(x) for x in pool.business_address), dtype=bool, count=len(pool))
    empty_names = np.full(len(pool), None, dtype=object)
    empty_names[pool_empty] = [norm_name(x) for x in pool.loc[pool_empty, "business_name"]]
    pair_empty = pool_empty[right]
    exact = pair_empty & (empty_names[right] == ref_names[left])
    return pool_empty, pair_empty, exact, multiplicity[left]


def _contest_stats(raw_accept, accepted, s1, rec, pool_empty, country, label=None):
    rows = np.flatnonzero(raw_accept)
    if rows.size:
        records, counts = np.unique(rec[rows], return_counts=True)
        contested_records = records[counts > 1]
    else:
        contested_records = np.empty(0, dtype=np.int64)
    contested_claim = raw_accept & np.isin(rec, contested_records)
    removed = raw_accept & ~accepted
    empty_contested = int(pool_empty[contested_records].sum())
    empty_removed = int(pool_empty[rec[removed]].sum())
    result = {
        "claimed_records": int(len(np.unique(rec[raw_accept]))),
        "contested_records": int(len(contested_records)),
        "contest_rate_per_claimed_record": float(len(contested_records) / max(1, len(np.unique(rec[raw_accept])))),
        "claims_on_contested_records": int(contested_claim.sum()),
        "s1_with_contested_claim": int(len(np.unique(s1[contested_claim]))),
        "contested_s1_share": float(len(np.unique(s1[contested_claim])) / len(country)),
        "removed_claims": int(removed.sum()),
        "contested_records_by_address_kind": {
            "empty": empty_contested, "nonempty": int(len(contested_records) - empty_contested),
        },
        "contested_record_empty_address_share": float(empty_contested / len(contested_records))
        if len(contested_records) else None,
        "removed_claims_by_address_kind": {
            "empty": empty_removed, "nonempty": int(removed.sum() - empty_removed),
        },
        "removed_claim_empty_address_share": float(empty_removed / removed.sum()) if removed.any() else None,
    }
    country = np.asarray(country, dtype=object)
    groups = {str(code): country == code for code in np.unique(country)}
    us_india = np.isin(country, ["US", "India"])
    if us_india.any():
        groups["US_India"] = us_india
    result["by_country"] = {}
    for name, s1_mask in groups.items():
        claims = raw_accept & s1_mask[s1]
        claimed = np.unique(rec[claims])
        contested = np.intersect1d(claimed, contested_records, assume_unique=True)
        claim_mask = claims & np.isin(rec, contested_records)
        empty = int(pool_empty[contested].sum())
        contested_s1 = int(len(np.unique(s1[claim_mask])))
        result["by_country"][name] = {
            "claimed_records": int(len(claimed)), "contested_records": int(len(contested)),
            "contest_rate_per_claimed_record": float(len(contested) / max(1, len(claimed))),
            "s1_with_contested_claim": contested_s1,
            "contested_s1_share": float(contested_s1 / max(1, int(s1_mask.sum()))),
            "contested_records_by_address_kind": {
                "empty": empty, "nonempty": int(len(contested) - empty),
            },
            "contested_record_empty_address_share": float(empty / len(contested)) if len(contested) else None,
        }
    if label is None:
        result["removed_claim_wrong_rate"] = None
        result["labels_used"] = False
    else:
        labels = np.asarray(label, dtype=bool)
        result["removed_claim_wrong_rate"] = float((~labels[removed]).mean()) if removed.any() else None
        result["removed_true_claims"] = int((labels & removed).sum())
        result["labels_used"] = True
    return result


def _prediction_stats(accept, s1, rec, pool_empty, n_s1, country):
    lengths = np.bincount(s1[accept], minlength=n_s1)

    def group(mask):
        pair_mask = accept & mask[s1]
        selected = lengths[mask]
        hist = np.bincount(selected, minlength=(int(selected.max()) + 1 if selected.size else 1))
        empty_links = int(pool_empty[rec[pair_mask]].sum())
        links = int(pair_mask.sum())
        return {
            "s1": int(mask.sum()), "accepted_links": links,
            "mean_list_length": float(selected.mean()) if selected.size else None,
            "empty_prediction_rate": float((selected == 0).mean()) if selected.size else None,
            "list_length_histogram": {str(i): int(value) for i, value in enumerate(hist) if value},
            "accepted_by_address_kind": {"empty": empty_links, "nonempty": links - empty_links},
            "empty_address_link_share": float(empty_links / links) if links else None,
        }

    country = np.asarray(country, dtype=object)
    result = {"all": group(np.ones(n_s1, dtype=bool))}
    for code in np.unique(country):
        result[str(code)] = group(country == code)
    us_india = np.isin(country, ["US", "India"])
    if us_india.any():
        result["US_India"] = group(us_india)
    return result


def fidelity_profile(left, right, blend, reference, pool, *, label=None, true_count=None, threshold=0.775):
    """Label-free structural profile for v1+one-owner and name-only; labels are arena-only."""
    left, right = np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64)
    blend = np.asarray(blend)
    if not (len(left) == len(right) == len(blend)):
        raise ValueError("Pair and blend arrays are not aligned")
    if label is not None and len(label) != len(left):
        raise ValueError("Arena labels are not aligned to pairs")
    if (label is None) != (true_count is None):
        raise ValueError("Labels and full true_count must be provided together")
    pool_empty, pair_empty, exact, pair_mult = _address_inputs(reference, pool, left, right)
    raw = blend >= threshold
    owned = one_owner(raw, left, right, blend)
    named, info = name_only_rule(owned, left, right, pair_empty, exact, pair_mult, owner_wins=True)
    result = {
        "labels_used": label is not None,
        "v1_pre_ownership_contests": _contest_stats(
            raw, owned, left, right, pool_empty, reference.country.to_numpy(), label),
        "one_owner": _prediction_stats(owned, left, right, pool_empty, len(reference), reference.country.to_numpy()),
        "name_only": _prediction_stats(named, left, right, pool_empty, len(reference), reference.country.to_numpy()),
        "name_only_changes": {"added": int(info["added"].sum()), "dropped": int(info["dropped"].sum()),
                              "owner_removed": int(info["owner_removed"].sum()),
                              "conflicted_records": info["conflicted_records"]},
    }
    if label is not None:
        common = {"s1_idx": left, "label": label, "true_count": true_count,
                  "country": reference.country.to_numpy(), "rec_idx": right,
                  "rec_kind": np.where(pool_empty, "empty", "nonempty")}
        result["arena_truth_metrics"] = {
            "one_owner": evaluate(owned, **common),
            "name_only": evaluate(named, **common),
        }
    return result


def _profile_differences(arena, test):
    fields = ("mean_list_length", "empty_prediction_rate", "empty_address_link_share")
    out = {}
    for variant in ("one_owner", "name_only"):
        groups = set(arena[variant]) & set(test[variant])
        out[variant] = {group: {field: (None if arena[variant][group][field] is None
                                              or test[variant][group][field] is None
                                              else arena[variant][group][field] - test[variant][group][field])
                                for field in fields} for group in sorted(groups)}
    contest_fields = ("contest_rate_per_claimed_record", "contested_s1_share",
                      "contested_record_empty_address_share")
    arena_contest, test_contest = (value["v1_pre_ownership_contests"] for value in (arena, test))
    out["pre_ownership_contests"] = {
        field: (None if arena_contest[field] is None or test_contest[field] is None
                else arena_contest[field] - test_contest[field]) for field in contest_fields}
    out["pre_ownership_contests_by_country"] = {
        group: {field: (None if arena_contest["by_country"][group][field] is None
                              or test_contest["by_country"][group][field] is None
                              else arena_contest["by_country"][group][field]
                              - test_contest["by_country"][group][field])
                for field in contest_fields}
        for group in sorted(set(arena_contest["by_country"]) & set(test_contest["by_country"]))
    }
    return out


def _sha256_inventory(path):
    entries = {}
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split(None, 1)
        if len(fields) != 2 or len(fields[0]) != 64:
            raise ValueError(f"Malformed SHA256 inventory line {number}: {path}")
        name = fields[1].lstrip("*")
        if name in entries:
            raise ValueError(f"Duplicate SHA256 inventory entry: {name}")
        entries[name] = fields[0].lower()
    return entries


def verify_completed_scores(data, pairs_path, blend_path, tree_path, complete_path, hashes_path,
                            *, expected_references, expected_pool, expected_pairs):
    """Verify an atomically completed frozen-v1 score run and its copied-file hashes."""
    data, pairs_path = Path(data), Path(pairs_path)
    blend_path, tree_path = Path(blend_path), Path(tree_path)
    complete_path, hashes_path = Path(complete_path), Path(hashes_path)
    paths = (pairs_path, blend_path, tree_path, complete_path, hashes_path)
    if any(not path.is_file() for path in paths):
        missing = [str(path) for path in paths if not path.is_file()]
        raise ValueError(f"Missing score provenance files: {missing}")
    inventory = _sha256_inventory(hashes_path)
    for path in (pairs_path, blend_path, tree_path):
        expected = inventory.get(path.name)
        if expected is None or sha256(path) != expected:
            raise ValueError(f"Score file is absent from or disagrees with its SHA256 inventory: {path.name}")
        if Path(str(path) + ".tmp").exists():
            raise ValueError(f"Incomplete temporary score artifact remains: {path}.tmp")

    complete = json.loads(complete_path.read_text())
    report = complete.get("report", {})
    report_path = complete_path.with_name("report.json")
    if complete.get("state") != "complete" or not report_path.is_file():
        raise ValueError("Score run lacks its final complete.json/report.json pair")
    if json.loads(report_path.read_text()) != report:
        raise ValueError("Score completion record and report disagree")
    expected_values = {
        "reference_count": expected_references, "full_reference_count": expected_references,
        "pool_count": expected_pool, "candidate_pairs": expected_pairs,
        "scorer": "frozen_tree_ce", "benchmark_only": False,
        "all_selected_references_written": True, "labels_accessed": False,
        "probability_order": "scored_pairs.parquet row order",
        "base_probability_order": "scored_pairs.parquet row order",
        "pipeline_sha256": FROZEN_V1_PIPELINE_SHA256,
    }
    disagreements = {key: (report.get(key), expected) for key, expected in expected_values.items()
                     if report.get(key) != expected}
    pipeline = report.get("pipeline", {})
    pipeline_expected = {"procedure": "medium_dense_bge_ce1800", "alpha": 0.75,
                         "gate": 0.01, "threshold": 0.775}
    disagreements.update({f"pipeline.{key}": (pipeline.get(key), expected)
                          for key, expected in pipeline_expected.items() if pipeline.get(key) != expected})
    if disagreements:
        raise ValueError(f"Score completion provenance is not frozen-v1/full-population: {disagreements}")
    identities = report.get("input_identities", {})
    for name, path in (("reference", data / "reference.parquet"),
                       ("pool", data / "pool.parquet"), ("pairs", pairs_path)):
        if identities.get(name, {}).get("size_bytes") != path.stat().st_size:
            raise ValueError(f"Score report input identity disagrees with {name} size")
    if pq.ParquetFile(pairs_path).metadata.num_rows != expected_pairs:
        raise ValueError("Scored pair parquet row count disagrees with completion report")
    blend = np.load(blend_path, mmap_mode="r", allow_pickle=False)
    tree = np.load(tree_path, mmap_mode="r", allow_pickle=False)
    if blend.shape != (expected_pairs,) or tree.shape != (expected_pairs,):
        raise ValueError("Completed score arrays are not one-dimensional and pair-aligned")
    return {"complete": complete_path, "hashes": hashes_path,
            "complete_sha256": sha256(complete_path), "hashes_sha256": sha256(hashes_path)}


def _load_arena_arrays(data, blend_path, tree_path, labels_dir):
    data, labels_dir = Path(data), Path(labels_dir)
    source_manifest = json.loads((data / "manifest.json").read_text())
    label_summary = json.loads((labels_dir / "labels_summary.json").read_text())
    if source_manifest.get("holdout2_excluded") is not True or source_manifest.get("labels_read") is not False:
        raise ValueError("Arena preparation does not prove label-free HOLDOUT-2 exclusion")
    if label_summary.get("holdout2_rows_read") != 0:
        raise ValueError("Arena label artifact does not prove zero HOLDOUT-2 label rows")
    for name in ("reference.parquet", "pool.parquet", "pairs.parquet"):
        expected = source_manifest.get("sha256", {}).get(name)
        if not expected or sha256(data / name) != expected:
            raise ValueError(f"Arena input hash does not match its preparation manifest: {name}")
    reference = pd.read_parquet(data / "reference.parquet", columns=RECORD_COLUMNS)
    pool = pd.read_parquet(data / "pool.parquet", columns=RECORD_COLUMNS)
    pairs = pq.read_table(data / "pairs.parquet", columns=["left_idx", "right_idx"])
    left = pairs.column("left_idx").to_numpy().astype(np.int64, copy=False)
    right = pairs.column("right_idx").to_numpy().astype(np.int64, copy=False)
    blend = np.load(blend_path, mmap_mode="r")
    tree = np.load(tree_path, mmap_mode="r")
    label = np.load(labels_dir / "label.npy", mmap_mode="r")
    true_count = np.load(labels_dir / "true_count.npy", mmap_mode="r")
    if not (len(left) == len(right) == len(blend) == len(tree) == len(label)):
        raise ValueError("Arena pairs, v1 blend/tree arrays, and labels are not aligned")
    if len(true_count) != len(reference) or int(np.asarray(true_count, dtype=np.int64).sum()) < int(label.sum()):
        raise ValueError("Full true_count is missing or inconsistent with retrieved labels")
    if (not np.isfinite(blend).all() or not np.isfinite(tree).all()
            or np.any((blend < 0) | (blend > 1)) or np.any((tree < 0) | (tree > 1))):
        raise ValueError("v1 score arrays must contain finite probabilities")
    return reference, pool, left, right, blend, tree, label, true_count


def finalize(args) -> dict:
    """Create C1 partitions and the arena-vs-test label-free fidelity report."""
    data, labels_dir, out = Path(args.data), Path(args.labels), Path(args.out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty finalize output: {out}")
    arena_score_pairs = Path(args.blend).with_name("scored_pairs.parquet")
    arena_counts = {name: pq.ParquetFile(data / f"{name}.parquet").metadata.num_rows
                    for name in ("reference", "pool", "pairs")}
    arena_score_provenance = verify_completed_scores(
        data, arena_score_pairs, args.blend, args.tree, args.score_complete, args.score_hashes,
        expected_references=arena_counts["reference"], expected_pool=arena_counts["pool"],
        expected_pairs=arena_counts["pairs"])
    if sha256(arena_score_pairs) != sha256(data / "pairs.parquet"):
        raise ValueError("Completed arena scores are not aligned to the prepared arena pairs")

    test_data = Path(args.test_data)
    test_counts = {"reference": pq.ParquetFile(test_data / "reference.parquet").metadata.num_rows,
                   "pool": pq.ParquetFile(test_data / "pool.parquet").metadata.num_rows,
                   "pairs": pq.ParquetFile(args.test_pairs).metadata.num_rows}
    test_score_provenance = verify_completed_scores(
        test_data, args.test_pairs, args.test_blend, args.test_tree,
        args.test_score_complete, args.test_score_hashes,
        expected_references=test_counts["reference"], expected_pool=test_counts["pool"],
        expected_pairs=test_counts["pairs"])

    reference, pool, left, right, blend, tree, label, true_count = _load_arena_arrays(
        data, args.blend, args.tree, labels_dir)
    force_a, force_b, forced_population = _forced_masks(
        reference.entity_id.to_numpy(), args.quick_reference, args.medium_reference)
    component, cut, attempts = capped_claim_components(
        left, right, blend, len(reference), force_a=force_a, force_b=force_b,
        threshold=args.component_threshold, max_component_fraction=args.max_component_fraction,
        threshold_step=args.threshold_step, max_threshold=args.max_component_threshold)
    partition, counts = make_partitions(component, force_a, force_b, a_fraction=args.a_fraction,
                                        slice_size=args.slice_size, seed=args.seed)
    if np.any(force_a & (partition != PARTITION_A)) or np.any(force_b & (partition == PARTITION_A)):
        raise AssertionError("A forced component was assigned to the wrong partition")

    arena_profile = fidelity_profile(left, right, blend, reference, pool, label=label, true_count=true_count,
                                     threshold=args.accept_threshold)
    test_reference = pd.read_parquet(test_data / "reference.parquet", columns=RECORD_COLUMNS)
    test_pool = pd.read_parquet(test_data / "pool.parquet", columns=RECORD_COLUMNS)
    test_pairs = pq.read_table(args.test_pairs, columns=["left_idx", "right_idx"])
    test_left = test_pairs.column("left_idx").to_numpy().astype(np.int64, copy=False)
    test_right = test_pairs.column("right_idx").to_numpy().astype(np.int64, copy=False)
    test_blend = np.load(args.test_blend, mmap_mode="r")
    test_tree = np.load(args.test_tree, mmap_mode="r")
    if not (len(test_left) == len(test_right) == len(test_blend) == len(test_tree)):
        raise ValueError("Frozen test pairs and v1 blend/tree arrays are not aligned")
    test_profile = fidelity_profile(test_left, test_right, test_blend, test_reference, test_pool,
                                    threshold=args.accept_threshold)
    fidelity = {
        "schema": 1, "scope": "arena_vs_official_test_structural_fidelity",
        "test_labels_read": False, "arena_labels_used_only_for_arena_removed_claim_accuracy": True,
        "arena": arena_profile, "test": test_profile,
        "arena_minus_test": _profile_differences(arena_profile, test_profile),
        "limitations": ["Official test labels do not exist; test accuracy, true-link recall, and French accuracy are not reported.",
                        "Arena truth metrics include every true link through full true_count, including retrieval misses."],
    }
    out.mkdir(parents=True, exist_ok=True)
    atomic_save(out / "component_ids.npy", component)
    atomic_save(out / "s1_partition.npy", partition)
    atomic_json(out / "fidelity.json", fidelity)

    inputs = {"reference": data / "reference.parquet", "pool": data / "pool.parquet",
              "pairs": data / "pairs.parquet", "scored_pairs": arena_score_pairs,
              "blend": Path(args.blend), "tree": Path(args.tree),
              "label": labels_dir / "label.npy", "true_count": labels_dir / "true_count.npy",
              "arena_score_complete": Path(args.score_complete), "arena_score_hashes": Path(args.score_hashes),
              "quick_reference": Path(args.quick_reference), "medium_reference": Path(args.medium_reference),
              "test_reference": test_data / "reference.parquet", "test_pool": test_data / "pool.parquet",
              "test_pairs": Path(args.test_pairs), "test_blend": Path(args.test_blend),
              "test_tree": Path(args.test_tree), "test_score_complete": Path(args.test_score_complete),
              "test_score_hashes": Path(args.test_score_hashes)}
    manifest = {
        "schema": 1, "state": "complete", "references": len(reference), "pairs": len(left),
        "holdout2_excluded": True, "holdout2_labels_read": False,
        "component_rule": {"initial_threshold": args.component_threshold, "selected_threshold": cut,
                           "max_component_fraction": args.max_component_fraction,
                           "threshold_step": args.threshold_step,
                           "safe_threshold_maximum": args.max_component_threshold,
                           "attempts": attempts},
        "partition_codes": {str(code): name for code, name in PARTITION_NAMES.items()},
        "partition_counts": counts, "a_fraction_target": args.a_fraction,
        "forced_population": forced_population,
        "slice_size_target": args.slice_size, "seed": args.seed,
        "full_true_count_sum": int(np.asarray(true_count, dtype=np.int64).sum()),
        "retrieved_positive_pairs": int(label.sum()),
        "score_provenance": {
            "arena": {key: str(value) for key, value in arena_score_provenance.items()},
            "test": {key: str(value) for key, value in test_score_provenance.items()},
        },
        "inputs": {name: {"path": str(path), "sha256": sha256(path)} for name, path in inputs.items()},
        "outputs": {name: {"sha256": sha256(out / name), "bytes": (out / name).stat().st_size}
                    for name in ("component_ids.npy", "s1_partition.npy", "fidelity.json")},
    }
    atomic_json(out / "manifest.json", manifest)
    return manifest


def labels(args) -> dict:
    """Align ground truth to arena pairs; only eligible S1 rows of the truth file are used."""
    data, raw, out = Path(args.data), Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(data / "reference.parquet", columns=["entity_id"])
    pool = pd.read_parquet(data / "pool.parquet", columns=["entity_id"])
    s1_index = pd.Series(np.arange(len(reference)), index=reference.entity_id)
    true_count = np.zeros(len(reference), dtype=np.int16)
    owners = {}
    seen = np.zeros(len(reference), dtype=bool)
    with (raw / "train_ground_truth.tsv").open(encoding="utf-8", newline="") as stream:
        if stream.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise ValueError("Unexpected training ground-truth header")
        for line in stream:
            source_id, tab, payload = line.rstrip("\r\n").partition("\t")
            if not tab:
                raise ValueError("Malformed training ground-truth row")
            position = s1_index.get(source_id, -1)
            if position < 0:
                continue  # In particular, never split or otherwise consume a HOLDOUT-2 label payload.
            if seen[position]:
                raise ValueError(f"Duplicate ground-truth row for {source_id}")
            seen[position] = True
            targets = payload.split(",") if payload else []
            true_count[position] = len(targets)
            for target in targets:
                previous = owners.setdefault(target, int(position))
                if previous != position:
                    raise ValueError(f"Target {target} has multiple owners")
    if not seen.all():
        raise ValueError("Every eligible S1 must have one ground-truth row")
    rec_owner = np.fromiter((owners.get(entity, -1) for entity in pool.entity_id), dtype=np.int32, count=len(pool))
    pairs = pq.read_table(data / "pairs.parquet", columns=["left_idx", "right_idx"])
    left, right = pairs.column("left_idx").to_numpy(), pairs.column("right_idx").to_numpy()
    label = rec_owner[right] == left
    atomic_save(out / "label.npy", label)
    atomic_save(out / "true_count.npy", true_count)
    atomic_save(out / "rec_owner.npy", rec_owner)
    summary = {"pairs": int(len(label)), "positive_pairs": int(label.sum()), "true_links": int(true_count.sum()),
               "link_recall": float(label.sum() / max(1, true_count.sum())), "singletons": int((true_count == 0).sum()),
               "pool_records_owned": int((rec_owner >= 0).sum()), "holdout2_rows_read": 0}
    atomic_json(out / "labels_summary.json", summary)
    log(json.dumps(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--p2", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--expected-references", type=int, default=2_072_821)
    q = commands.add_parser("labels")
    q.add_argument("--data", required=True)
    q.add_argument("--raw", required=True)
    q.add_argument("--out", required=True)
    r = commands.add_parser("finalize")
    for name in ("data", "labels", "blend", "tree", "score-complete", "score-hashes",
                 "quick-reference", "medium-reference", "test-data", "test-pairs", "test-blend",
                 "test-tree", "test-score-complete", "test-score-hashes", "out"):
        r.add_argument(f"--{name}", required=True)
    r.add_argument("--component-threshold", type=float, default=0.05)
    r.add_argument("--max-component-fraction", type=float, default=0.05)
    r.add_argument("--threshold-step", type=float, default=0.05)
    r.add_argument("--max-component-threshold", type=float, default=1.0,
                   help="Inclusive upper bound of the deterministic conflict/size threshold search")
    r.add_argument("--accept-threshold", type=float, default=0.775)
    r.add_argument("--a-fraction", type=float, default=0.70)
    r.add_argument("--slice-size", type=int, default=100_000)
    r.add_argument("--seed", type=int, default=20260926)
    args = parser.parse_args()
    {"prepare": prepare, "labels": labels, "finalize": finalize}[args.command](args)


if __name__ == "__main__":
    main()
