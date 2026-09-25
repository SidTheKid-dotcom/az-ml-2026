"""Controlled 12k medium-TUNE union of full-pool dense, lexical and key retrieval.

The old medium_hard lexical/phonetic and medium dense artifacts were both mined
against all 10,320,219 raw targets. Pairs are aligned by original S1/S2/S3 ID;
new canonicalized keys expand back to every exact original S2/S3 row. Candidate
pairs are persisted *before* TUNE labels are opened. Only then are labels read
for recall/oracle scoring and to retain unretrieved truth **records** in a compact
pool; no missing positive is ever added as a candidate edge.

Tree, CE and relative-context features must be recomputed after this candidate
union, using the complete per-S1 groups. ``score-context`` attaches dense
cosines/ranks to all pairs from existing query/pool embeddings, without rerunning
retrieval or altering a frozen model. TRAIN-only negative compaction is separate
from TUNE scoring and may not remove retrieved positives.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .normalization import normalize_text
from .retrieval_ablation import _script
from .uncertainty import paired_bootstrap
from .v2_lexical_mining import (FullPoolIndex, POOL_ROWS, TUNE_ID_SHA256,
                                atomic_json, file_sha256, ids_digest, text_keys)


RECORD_COLUMNS = ("entity_id", "business_name", "business_address", "country")
BRANCH_FLAGS = ("name_exact_hit", "phonetic_key_hit", "digit_sequence_hit",
                "digit_set_hit", "composite_key_hit")


@dataclass
class SavedMedium:
    reference: pd.DataFrame
    lexical_pool: pd.DataFrame
    dense_pool: pd.DataFrame
    lexical: pd.DataFrame
    dense: pd.DataFrame
    fingerprints: dict


def _valid_pairs(pairs: pd.DataFrame, left_size: int, right_size: int, label: str) -> None:
    if {"left_idx", "right_idx"} - set(pairs):
        raise ValueError(f"{label} needs positional pair columns")
    if pairs.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError(f"{label} contains duplicate candidate pairs")
    if len(pairs) and (pairs.left_idx.min() < 0 or pairs.left_idx.max() >= left_size
                       or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= right_size):
        raise ValueError(f"{label} has an out-of-bounds pair")


def _country_codes(frame: pd.DataFrame) -> np.ndarray:
    mapping = {value: normalize_text(value) for value in frame.country.unique()}
    return frame.country.map(mapping).to_numpy()


def load_saved_medium(lexical_dir: Path, dense_dir: Path) -> SavedMedium:
    """Require the *same* complete original medium TRAIN/TUNE references and pool."""
    lexical_dir, dense_dir = Path(lexical_dir), Path(dense_dir)
    lex_meta = json.loads((lexical_dir / "metadata.json").read_text())
    dense_meta = json.loads((dense_dir / "metadata.json").read_text())
    dense_complete = json.loads((dense_dir / "complete.json").read_text())
    if (lex_meta.get("selected_splits") != ["train", "tune"]
            or dense_complete.get("state") != "complete"
            or lex_meta.get("candidate_pairs") != 2_987_950
            or lex_meta.get("full_pool_records_by_country") != {"india": 4_133_346, "us": 6_186_873}
            or dense_meta.get("mining_full_pool_records") != POOL_ROWS
            or dense_meta.get("positive_candidate_injection") is not False
            or dense_complete["config"]["labels_used_for_mining"] is not False
            or lex_meta.get("reference_id_sha256") != dense_meta.get("reference_id_sha256")):
        raise ValueError("Inputs must be saved, complete medium TRAIN/TUNE full-pool retrieval")
    paths = {"lex_reference": lexical_dir / "reference.parquet", "lex_pool": lexical_dir / "pool.parquet",
             "lex_pairs": lexical_dir / "candidate_pairs.parquet", "dense_reference": dense_dir / "reference.parquet",
             "dense_pool": dense_dir / "pool.parquet", "dense_pairs": dense_dir / "pairs.parquet",
             "lex_metadata": lexical_dir / "metadata.json", "dense_metadata": dense_dir / "metadata.json",
             "dense_complete": dense_dir / "complete.json"}
    lex_ref, dense_ref = (pd.read_parquet(paths[k]) for k in ("lex_reference", "dense_reference"))
    if (len(lex_ref) != 47_999 or set(lex_ref.split) != {"train", "tune"}
            or len(dense_ref) != len(lex_ref) or set(dense_ref.split) != {"train", "tune"}
            or lex_ref.entity_id.duplicated().any() or dense_ref.entity_id.duplicated().any()):
        raise ValueError("Saved references must contain exactly the original TRAIN/TUNE set")
    other = dense_ref.set_index("entity_id").reindex(lex_ref.entity_id)
    if other.isna().any().any() or not np.array_equal(
        lex_ref[list(RECORD_COLUMNS[1:]) + ["split"]].to_numpy(),
        other[list(RECORD_COLUMNS[1:]) + ["split"]].to_numpy()):
        raise ValueError("Lexical/dense S1 text, country or split disagrees")
    tune_positions = np.flatnonzero(lex_ref.split.eq("tune"))
    reference = lex_ref.iloc[tune_positions].reset_index(drop=True)
    if len(reference) != 12_000 or ids_digest(reference.entity_id) != TUNE_ID_SHA256:
        raise ValueError("Saved medium TUNE 12,000-S1 identity disagrees")
    lex_pool, dense_pool = (pd.read_parquet(paths[k]) for k in ("lex_pool", "dense_pool"))
    if (len(lex_pool) != lex_meta["pool_records"] or len(dense_pool) != dense_meta["pool_records"]
            or lex_pool.entity_id.duplicated().any() or dense_pool.entity_id.duplicated().any()
            or "raw_row_idx" not in dense_pool):
        raise ValueError("Saved compact pool/mapping is incomplete or has duplicate target IDs")
    lp, dp = (pd.read_parquet(paths[k]) for k in ("lex_pairs", "dense_pairs"))
    _valid_pairs(lp, len(lex_ref), len(lex_pool), "lexical")
    _valid_pairs(dp, len(dense_ref), len(dense_pool), "dense")
    if len(lp) != lex_meta["candidate_pairs"] or len(dp) != dense_meta["candidate_pairs"]:
        raise ValueError("Saved full-pool pair counts changed")
    lex_to_tune = np.full(len(lex_ref), -1, dtype=np.int32)
    lex_to_tune[tune_positions] = np.arange(len(tune_positions), dtype=np.int32)
    dense_to_tune = np.full(len(dense_ref), -1, dtype=np.int32)
    dense_tune_positions = pd.Index(dense_ref.entity_id).get_indexer(reference.entity_id)
    if np.any(dense_tune_positions < 0):
        raise ValueError("Dense medium reference does not contain all TUNE S1s")
    dense_to_tune[dense_tune_positions] = np.arange(len(reference), dtype=np.int32)
    lp = lp.loc[lex_to_tune[lp.left_idx.to_numpy(dtype=np.int64)] >= 0].copy()
    dp = dp.loc[dense_to_tune[dp.left_idx.to_numpy(dtype=np.int64)] >= 0].copy()
    lleft, lright = lp.left_idx.to_numpy(dtype=np.int64), lp.right_idx.to_numpy(dtype=np.int64)
    dleft, dright = dp.left_idx.to_numpy(dtype=np.int64), dp.right_idx.to_numpy(dtype=np.int64)
    lexical = pd.DataFrame({"left_idx": lex_to_tune[lleft],
                            "target_entity_id": lex_pool.entity_id.to_numpy()[lright],
                            "name_score": lp.name_score.to_numpy(dtype=np.float32),
                            "address_score": lp.address_score.to_numpy(dtype=np.float32),
                            "lexical_retrieval_score": lp.retrieval_score.to_numpy(dtype=np.float32),
                            "lexical_retrieval_rank": lp.retrieval_rank.to_numpy(dtype=np.int32),
                            "legacy_phonetic_hit": lp.phonetic_hit.to_numpy(dtype=bool),
                            "legacy_phonetic_only": lp.phonetic_only.to_numpy(dtype=bool),
                            "phonetic_score": lp.phonetic_score.to_numpy(dtype=np.float32),
                            "phonetic_rank": lp.phonetic_rank.to_numpy(dtype=np.int32),
                            "lexical_scores_missing": lp.lexical_scores_missing.to_numpy(dtype=bool)})
    dense = pd.DataFrame({"left_idx": dense_to_tune[dleft],
                          "target_entity_id": dense_pool.entity_id.to_numpy()[dright],
                          "right_raw_idx": dense_pool.raw_row_idx.to_numpy(dtype=np.int32)[dright],
                          "original_dense_score": dp.dense_score.to_numpy(dtype=np.float32),
                          "dense_rank": dp.dense_rank.to_numpy(dtype=np.int32)})
    for label, pairs, pool, right in (("lexical", lexical, lex_pool, lright),
                                      ("dense", dense, dense_pool, dright)):
        if pairs.duplicated(["left_idx", "target_entity_id"]).any():
            raise ValueError(f"{label} repeated an original ID within a reference")
        if np.any(_country_codes(reference)[pairs.left_idx.to_numpy(dtype=np.int32)] != _country_codes(pool)[right]):
            raise ValueError(f"{label} pair escaped the full-pool country block")
    fingerprints = {key: {"path": str(path.resolve()), "bytes": path.stat().st_size,
                          "sha256": file_sha256(path)} for key, path in paths.items()}
    fingerprints["tune_id_sha256"] = TUNE_ID_SHA256
    fingerprints["raw_target_pool_rows"] = POOL_ROWS
    return SavedMedium(reference, lex_pool, dense_pool, lexical, dense, fingerprints)


def union_by_id(lexical: pd.DataFrame, dense: pd.DataFrame,
                key_pairs: pd.DataFrame | None = None) -> pd.DataFrame:
    """Keep original ID and branch scores separately; refuse duplicate/conflicting maps."""
    keys = ["left_idx", "target_entity_id"]
    for name, table in (("lexical", lexical), ("dense", dense), ("keys", key_pairs)):
        if table is not None and table.duplicated(keys).any():
            raise ValueError(f"{name} has repeated original reference-target IDs")
    merged = lexical.merge(dense, on=keys, how="outer", validate="one_to_one", indicator="_old_merge")
    merged["lexical_hit"] = merged._old_merge.ne("right_only")
    merged["dense_hit"] = merged._old_merge.ne("left_only")
    merged.drop(columns="_old_merge", inplace=True)
    if key_pairs is not None:
        merged = merged.merge(key_pairs, on=keys, how="outer", validate="one_to_one", indicator="_key_merge",
                              suffixes=("", "_key"))
        overlap = merged.right_raw_idx.notna() & merged.right_raw_idx_key.notna()
        if np.any(merged.loc[overlap, "right_raw_idx"] != merged.loc[overlap, "right_raw_idx_key"]):
            raise ValueError("Dense/new retrieval disagree about an original target's raw row")
        merged["right_raw_idx"] = merged.right_raw_idx.fillna(merged.right_raw_idx_key)
        merged.drop(columns=["right_raw_idx_key", "_key_merge"], inplace=True)
    for flag in ("lexical_hit", "dense_hit", "legacy_phonetic_hit", "legacy_phonetic_only", *BRANCH_FLAGS):
        if flag not in merged:
            merged[flag] = False
        merged[flag] = merged[flag].fillna(False).astype(bool)
    for name in ("name_score", "address_score", "phonetic_score", "lexical_retrieval_score", "original_dense_score", "lexical_key_score"):
        if name not in merged:
            merged[name] = np.float32(0)
        merged[name] = merged[name].fillna(0).astype(np.float32)
    for name in ("lexical_retrieval_rank", "dense_rank", "phonetic_rank", "canonical_raw_idx", "canonical_group_size"):
        if name not in merged:
            merged[name] = 0
        merged[name] = merged[name].fillna(0).astype(np.int32)
    missing_scores = (merged["lexical_scores_missing"] if "lexical_scores_missing" in merged
                      else pd.Series(False, index=merged.index))
    merged["lexical_scores_missing"] = missing_scores.fillna(False).astype(bool) | ~merged.lexical_hit
    # The original tree's optional feature order includes both provenance flags.
    # New-key phonetics are "phonetic only" relative to the old lexical branch;
    # preserved legacy flags are not inferred from cosine values or rank ties.
    merged["phonetic_only"] = merged.legacy_phonetic_only | (merged.phonetic_key_hit & ~merged.lexical_hit)
    merged["dense_only"] = (merged.dense_hit & ~merged.lexical_hit).astype(np.float32)
    merged["right_raw_idx"] = merged.right_raw_idx.fillna(-1).astype(np.int32)
    merged.sort_values(keys, kind="stable", inplace=True, ignore_index=True)
    if merged.duplicated(keys).any():
        raise ValueError("Union duplicated original reference-target IDs")
    return merged


def resolve_new_pairs(index: FullPoolIndex, mine_dir: Path, reference: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    path = Path(mine_dir)
    info = json.loads((path / "complete.json").read_text())
    query = pd.read_parquet(path / "reference.parquet")
    if (info.get("state") != "complete" or info.get("references") != len(reference)
            or info.get("full_pool_rows") != index.manifest["rows"]
            or info.get("reference_ids_sha256") != TUNE_ID_SHA256
            or info.get("full_pool_index_manifest_sha256") != file_sha256(index.directory / "complete.json")
            or not np.array_equal(query[list(RECORD_COLUMNS) + ["split"]].to_numpy(),
                                  reference[list(RECORD_COLUMNS) + ["split"]].to_numpy())):
        raise ValueError("New key pairs were not retrieved for these exact TUNE references and full index")
    pairs = pd.read_parquet(path / "candidate_pairs.parquet")
    if (len(pairs) != info["pairs"] or file_sha256(path / "candidate_pairs.parquet") != info["pair_file"]["sha256"]
            or pairs.duplicated(["left_idx", "right_raw_idx"]).any()
            or (len(pairs) and (pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
                                or pairs.right_raw_idx.min() < 0 or pairs.right_raw_idx.max() >= index.manifest["rows"]))):
        raise ValueError("Key retrieval pair identity, uniqueness or original row bounds changed")
    raw = pairs.right_raw_idx.unique().tolist()
    records = index.lookup_rows(raw)
    right = pairs.right_raw_idx.to_numpy(dtype=np.int32)
    countries = _country_codes(reference)
    if any(records[int(raw_row)][-1] != countries[left] for left, raw_row in
           zip(pairs.left_idx.to_numpy(dtype=np.int32), right)):
        raise ValueError("Key candidate crossed a source-country block")
    result = pd.DataFrame({"left_idx": pairs.left_idx.astype(np.int32),
                           "target_entity_id": [records[int(r)][1] for r in right],
                           "right_raw_idx": right,
                           "canonical_raw_idx": pairs.canonical_raw_idx.to_numpy(dtype=np.int32),
                           "canonical_group_size": pairs.canonical_group_size.to_numpy(dtype=np.int32),
                           "lexical_key_score": pairs.lexical_key_score.to_numpy(dtype=np.float32),
                           "name_exact_hit": pairs.name_exact_hit.to_numpy(dtype=bool),
                           "phonetic_key_hit": pairs.phonetic_hit.to_numpy(dtype=bool),
                           "digit_sequence_hit": pairs.digit_sequence_hit.to_numpy(dtype=bool),
                           "digit_set_hit": pairs.digit_set_hit.to_numpy(dtype=bool),
                           "composite_key_hit": pairs.composite_key_hit.to_numpy(dtype=bool)})
    for raw_row, rep, size in zip(right, result.canonical_raw_idx, result.canonical_group_size):
        if int(index.canonical[raw_row]) != rep or int(index.sizes[rep]) != size:
            raise ValueError("A new candidate is not the exact expansion of its canonical group")
    return result, {"new_pairs_file": str((path / "candidate_pairs.parquet").resolve()),
                    "sha256": info["pair_file"]["sha256"], "index_manifest_sha256": info["full_pool_index_manifest_sha256"]}


def align_original_rows(pairs: pd.DataFrame, index: FullPoolIndex) -> pd.DataFrame:
    """Complete and cross-check the original S2/S3 raw-row mapping by ID."""
    missing = pairs.right_raw_idx.lt(0)
    pairs = pairs.copy()
    if missing.any():
        targets = pairs.loc[missing, "target_entity_id"].unique().tolist()
        resolved = index.lookup_ids(targets)
        raw = pairs.right_raw_idx.to_numpy(dtype=np.int32, copy=True)
        raw[missing.to_numpy()] = np.fromiter(
            (resolved[target][0] for target in pairs.loc[missing, "target_entity_id"]),
            dtype=np.int32, count=int(missing.sum()))
        pairs["right_raw_idx"] = raw
    # Existing dense IDs are verified against the new independently built raw map.
    rows = pairs.loc[~missing, ["right_raw_idx", "target_entity_id"]].drop_duplicates("right_raw_idx")
    if rows.target_entity_id.duplicated().any():
        raise ValueError("Different original rows were assigned the same target ID")
    for start in range(0, len(rows), 50_000):
        chunk = rows.iloc[start:start + 50_000]
        mapping = index.lookup_rows(chunk.right_raw_idx.tolist())
        if any(mapping[int(raw)][1] != target for raw, target in chunk.itertuples(index=False, name=None)):
            raise ValueError("Dense/key raw row differs from the original S2/S3 ID")
    if pairs.duplicated(["left_idx", "right_raw_idx"]).any():
        raise ValueError("Two target IDs map to the same original S2/S3 row")
    return pairs


def _tune_truth(lexical_dir: Path, reference: pd.DataFrame) -> tuple[dict[str, set[str]], dict]:
    # This is deliberately called only after candidate_pairs_raw.parquet is saved.
    path = Path(lexical_dir) / "truth.json"
    supplied = json.loads(path.read_text())
    if set(reference.entity_id) - set(supplied):
        raise ValueError("TUNE truth does not cover every fixed medium reference")
    truth = {entity: set(supplied[entity]) for entity in reference.entity_id}
    if any(len(value) != len(supplied[entity]) for entity, value in truth.items()):
        raise ValueError("Duplicate target in ground truth")
    if sum(map(len, truth.values())) != len({target for values in truth.values() for target in values}):
        raise ValueError("Target ground truth must be one-to-one across TUNE S1s")
    return truth, {"path": str(path.resolve()), "sha256": file_sha256(path),
                   "truth_scope": "only 12,000 medium TUNE S1 IDs, loaded after raw pair save"}


def _oracle(reference: pd.DataFrame, pairs: pd.DataFrame, mask: np.ndarray,
            labels: np.ndarray, truth: dict) -> tuple[dict, np.ndarray]:
    left = pairs.left_idx.to_numpy(dtype=np.int32)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(pairs),):
        raise ValueError("Variant must select a Boolean decision per *actual candidate* pair")
    actual = np.asarray([len(truth[x]) for x in reference.entity_id], dtype=np.int32)
    found = np.bincount(left[selected & labels], minlength=len(reference))
    candidate_counts = np.bincount(left[selected], minlength=len(reference))
    denom = 4 * found + actual
    f05 = np.divide(5.0 * found, denom, out=np.zeros(len(reference)), where=denom > 0)
    f05[actual == 0] = 1.0

    def group(included: np.ndarray) -> dict:
        nonempty = included & (actual > 0)
        t, f, p = int(actual[included].sum()), int(found[included].sum()), int(candidate_counts[included].sum())
        return {"references": int(included.sum()), "true_links": t,
                "retrieved_true_links": f, "missed_true_links": t - f,
                "pairs": p, "pairs_per_reference": p / included.sum() if included.any() else None,
                "link_recall": f / t if t else None,
                "complete_match_set_recall": float((found[nonempty] == actual[nonempty]).mean()) if nonempty.any() else None,
                "oracle_macro_f05": float(f05[included].mean()) if included.any() else None,
                "true_singletons": int(np.sum(included & (actual == 0))),
                "oracle_singleton_accuracy": 1.0 if np.any(included & (actual == 0)) else None,
                "oracle_false_links": 0}

    report = group(np.ones(len(reference), dtype=bool))
    report["by_country"] = {str(country): group(reference.country.eq(country).to_numpy())
                            for country in sorted(reference.country.unique())}
    return report, f05


def _script_recall(reference: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame,
                   mask: np.ndarray, labels: np.ndarray, truth: dict) -> list[dict]:
    text = pool.drop_duplicates("entity_id").set_index("entity_id")
    links = pd.DataFrame([(left, entity, reference.country.iat[left])
                          for left, ref in enumerate(reference.entity_id) for entity in truth[ref]],
                         columns=["left_idx", "target_entity_id", "country"])
    if any(entity not in text.index for entity in links.target_entity_id):
        raise ValueError("A TUNE truth record is missing from the preserved compact pool")
    for column in ("business_name", "business_address"):
        links[f"{column}_script"] = text.loc[links.target_entity_id, column].map(_script).to_numpy()
    found = pairs.loc[mask & labels, ["left_idx", "target_entity_id"]].assign(retrieved=True)
    links = links.merge(found, on=["left_idx", "target_entity_id"], how="left", validate="one_to_one")
    links["retrieved"] = links.retrieved.fillna(False).astype(bool)
    return [{"country": str(country), "field": field.removeprefix("business_").removesuffix("_script"),
             "script": script, "true_links": len(group),
             "retrieved_true_links": int(group.retrieved.sum()),
             "link_recall": float(group.retrieved.mean())}
            for field in ("business_name_script", "business_address_script")
            for (country, script), group in links.groupby(["country", field], sort=True)]


def _compact_pool(index: FullPoolIndex, reference: pd.DataFrame, pairs: pd.DataFrame,
                  truth: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Truth affects record retention only; candidate IDs and raw rows are already fixed.
    ids = sorted({target for targets in truth.values() for target in targets})
    truth_rows = index.lookup_ids(ids)
    raw = np.unique(np.r_[pairs.right_raw_idx.to_numpy(dtype=np.int32),
                          np.asarray([row[0] for row in truth_rows.values()], dtype=np.int32)])
    frame = []
    for start in range(0, len(raw), 50_000):
        lookup = index.lookup_rows(raw[start:start + 50_000].tolist())
        frame.extend((row[1], row[3], row[4], row[2], int(row[0])) for row in
                     (lookup[int(value)] for value in raw[start:start + 50_000]))
    pool = pd.DataFrame(frame, columns=[*RECORD_COLUMNS, "raw_row_idx"])
    if pool.entity_id.duplicated().any() or not np.array_equal(pool.raw_row_idx.to_numpy(dtype=np.int32), raw):
        raise ValueError("Compact pool dropped/repeated a full original target")
    right = np.searchsorted(raw, pairs.right_raw_idx.to_numpy(dtype=np.int32))
    if not np.array_equal(raw[right], pairs.right_raw_idx.to_numpy(dtype=np.int32)):
        raise ValueError("Candidate remapping lost an original target")
    pairs = pairs.copy()
    pairs["right_idx"] = right.astype(np.int32)
    if not np.array_equal(pool.entity_id.to_numpy()[right], pairs.target_entity_id.to_numpy()):
        raise ValueError("A candidate's S2/S3 ID changed during compact-pool creation")
    ref_ids = reference.entity_id.to_numpy()
    present_truth = sum(target in truth[ref_ids[int(left)]] for left, target in
                        pairs[["left_idx", "target_entity_id"]].itertuples(index=False, name=None))
    return pool, pairs, {"all_truth_target_records_retained": len(ids),
                         "truth_targets_without_candidate_edge": len(ids) - present_truth,
                         "positive_candidate_injection": False}


def _variant_masks(pairs: pd.DataFrame) -> dict[str, np.ndarray]:
    baseline = pairs.lexical_hit.to_numpy() | pairs.dense_hit.to_numpy()
    exact = pairs.name_exact_hit.to_numpy() | pairs.phonetic_key_hit.to_numpy()
    digits = pairs.digit_sequence_hit.to_numpy()
    sets = pairs.digit_set_hit.to_numpy()
    comp = pairs.composite_key_hit.to_numpy()
    return {"lexical_phonetic_saved": pairs.lexical_hit.to_numpy(),
            "dense_k50_saved": pairs.dense_hit.to_numpy(),
            "saved_union": baseline,
            "saved_plus_exact_name_phonetic": baseline | exact,
            "saved_plus_digit_sequence": baseline | digits,
            "saved_plus_digit_sequence_and_set": baseline | digits | sets,
            "saved_plus_composite": baseline | comp,
            "full_key_union": baseline | exact | digits | sets | comp}


def run_union(lexical_dir: Path, dense_dir: Path, output: Path, *,
              index_dir: Path | None = None, key_dir: Path | None = None,
              bootstrap_replicates: int = 2_000, seed: int = 20260925) -> dict:
    """Baseline-only works before index completion; full mode saves mapped pairs."""
    started = time.monotonic()
    lexical_dir, dense_dir, output = map(Path, (lexical_dir, dense_dir, output))
    if (output / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed union: {output}")
    if (index_dir is None) != (key_dir is None):
        raise ValueError("Supply both a completed full-pool index and new candidates, or neither")
    saved = load_saved_medium(lexical_dir, dense_dir)
    index = FullPoolIndex(index_dir) if index_dir else None
    output.mkdir(parents=True, exist_ok=True)
    try:
        key_pairs, new_identity = resolve_new_pairs(index, key_dir, saved.reference) if index else (None, None)
        pairs = union_by_id(saved.lexical, saved.dense, key_pairs)
        if index:
            pairs = align_original_rows(pairs, index)
        pair_path = output / "candidate_pairs_raw.parquet"
        pairs.to_parquet(str(pair_path) + ".tmp", index=False)
        Path(str(pair_path) + ".tmp").replace(pair_path)
        saved.reference.to_parquet(output / "reference.parquet", index=False)
        # The first truth access happens only after *all* candidate edges are fixed.
        truth, truth_identity = _tune_truth(lexical_dir, saved.reference)
        if index:
            pool, mapped, retention = _compact_pool(index, saved.reference, pairs, truth)
            pool.to_parquet(output / "pool.parquet.tmp", index=False)
            (output / "pool.parquet.tmp").replace(output / "pool.parquet")
            mapped.to_parquet(output / "candidate_pairs.parquet.tmp", index=False)
            (output / "candidate_pairs.parquet.tmp").replace(output / "candidate_pairs.parquet")
            atomic_json(output / "truth.json", {entity: sorted(values) for entity, values in truth.items()})
        else:
            pool, retention = saved.lexical_pool, {"positive_candidate_injection": False,
                                                   "full_pool_raw_mapping_pending": True}
        refids, target_ids = saved.reference.entity_id.to_numpy(), pairs.target_entity_id.to_numpy()
        left = pairs.left_idx.to_numpy(dtype=np.int32)
        labels = np.fromiter((target in truth[refids[pos]] for pos, target in zip(left, target_ids)),
                             dtype=bool, count=len(pairs))
        masks, measures, per_entity = _variant_masks(pairs), {}, {}
        for name, mask in masks.items():
            measures[name], per_entity[name] = _oracle(saved.reference, pairs, mask, labels, truth)
        saved_dense_measure = json.loads((dense_dir / "metadata.json").read_text())["retrieval"]["tune"]
        if (measures["dense_k50_saved"]["pairs"] != saved_dense_measure["pairs"]
                or not np.isclose(measures["dense_k50_saved"]["link_recall"],
                                  saved_dense_measure["link_recall"], atol=1e-12, rtol=0)
                or not np.isclose(measures["dense_k50_saved"]["oracle_macro_f05"],
                                  saved_dense_measure["oracle_macro_f05"], atol=1e-12, rtol=0)):
            raise ValueError("Saved dense TUNE recall/oracle does not replay on the aligned full pool")
        if index and retention["truth_targets_without_candidate_edge"] != measures["full_key_union"]["missed_true_links"]:
            raise ValueError("Unretrieved truth record accounting differs from the actual candidate misses")
        comparison = {}
        for name in ("saved_union", "full_key_union") if index else ("saved_union",):
            if name == "saved_union":
                prior = "dense_k50_saved"
            else:
                prior = "saved_union"
            newer, older = per_entity[name], per_entity[prior]
            comparison[f"{name}_minus_{prior}"] = {
                "paired_entity_delta_macro_f05": float((newer - older).mean()),
                "newly_retrieved_true_links": int(measures[name]["retrieved_true_links"] - measures[prior]["retrieved_true_links"]),
                "new_pairs": int(measures[name]["pairs"] - measures[prior]["pairs"]),
                "bootstrap": paired_bootstrap(newer, older, n_bootstrap=bootstrap_replicates, seed=seed)}
        for name in ("dense_k50_saved", "lexical_phonetic_saved", "saved_union", "full_key_union" if index else "saved_union"):
            measures[name]["target_name_script"] = _script_recall(saved.reference, pool, pairs,
                                                                     masks[name], labels, truth)
        report = {"state": "complete", "schema": 1,
                  "population": "12,000 fixed medium TUNE S1s against all 10,320,219 original training S2/S3 targets",
                  "reference_ids_sha256": TUNE_ID_SHA256, "references": len(saved.reference),
                  "full_target_pool_rows": POOL_ROWS, "seed": seed,
                  "saved_input_fingerprints": saved.fingerprints,
                  "new_branch_identity": new_identity,
                  "raw_pairs": {"path": str(pair_path.resolve()), "rows": len(pairs),
                                "bytes": pair_path.stat().st_size, "sha256": file_sha256(pair_path)},
                  "pairs_by_variant": {name: entry["pairs"] for name, entry in measures.items()},
                  "retrieval": measures, "comparisons": comparison,
                  "retention": retention, "truth_identity": truth_identity,
                  "labels_read_after_candidate_persistence": True,
                  "truth_used_in_retrieval": False, "positive_candidate_injection": False,
                  "model_f05_reported": False,
                  "feature_contract": {"unmodified": "original lexical and dense top-K scores/ranks remain distinct with hit/missing flags",
                                       "recompute": "full-union dense cosine/rank for every pair, then all per-candidate-group gaps/counts and matcher features before training/inference; do not reuse saved v1 tree probabilities or CE scores",
                                       "candidate_set": "all complete per-S1 groups; do not insert unretrieved truth links"},
                  "elapsed_seconds": time.monotonic() - started,
                  "limitations": ["TUNE development retrieval/oracle ceiling is not a learned-model or entire-test score.",
                                  "Only medium TRAIN/TUNE truth files are read; old eval and HOLDOUT-2 are not inputs.",
                                  "Existing medium_hard char-TF-IDF and dense branches are reused without rebuilding their full-pool indexes.",
                                  "Large digit/name postings are bounded around the query's core name; capped-query counts are in the miner manifest."]}
        atomic_json(output / "complete.json", report)
        print(json.dumps({"references": len(saved.reference), "saved_union": measures["saved_union"],
                          "full_key_union": measures["full_key_union"] if index else None,
                          "pairs": len(pairs), "elapsed_seconds": report["elapsed_seconds"]},
                         allow_nan=False), flush=True)
        return report
    finally:
        if index:
            index.close()


def compact_train_negatives(reference: pd.DataFrame, pool: pd.DataFrame, pairs: pd.DataFrame,
                            truth: dict[str, set[str]], *, max_negatives: int = 40) -> tuple[pd.DataFrame, dict]:
    """TRAIN-only hard negatives; keep every *retrieved* positive without injection.

    Same-name/same-digit-set nonmatches precede high-score and near-name
    alternatives. The output is a training selection, not a retrieval artifact;
    leave TUNE and held-out candidate groups complete for scoring.
    """
    if (max_negatives < 0 or set(reference.split) != {"train"}
            or reference.entity_id.duplicated().any() or set(truth) != set(reference.entity_id)
            or pairs.duplicated(["left_idx", "right_idx"]).any()):
        raise ValueError("Compaction requires unique complete TRAIN-only S1/pairs and nonnegative cap")
    if len(pairs) and (pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
                       or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool)):
        raise ValueError("Compaction received an out-of-range pool/reference position")
    ref_id = reference.entity_id.to_numpy()
    pool_id = pool.entity_id.to_numpy()
    keep = np.zeros(len(pairs), dtype=bool)
    labels = np.zeros(len(pairs), dtype=bool)
    priority = np.full(len(pairs), 9, dtype=np.int8)
    for left, rows in pairs.groupby("left_idx", sort=False).indices.items():
        query = text_keys(reference.business_name.iat[left], reference.business_address.iat[left], reference.country.iat[left])
        negatives = []
        for row in rows:
            right = int(pairs.right_idx.iat[row])
            target = pool_id[right]
            is_true = target in truth[ref_id[left]]
            labels[row] = is_true
            if is_true:
                keep[row] = True
                priority[row] = -1
                continue
            other = text_keys(pool.business_name.iat[right], pool.business_address.iat[right], pool.country.iat[right])
            same_name = bool(query.core and query.core == other.core)
            same_set = bool(query.digit_set and query.digit_set == other.digit_set)
            some_digits = bool(query.digit_set and other.digit_set and
                               set(query.digit_set.split("|")) & set(other.digit_set.split("|")))
            category = (0 if same_name and same_set else 1 if same_name and some_digits else
                        2 if same_name else 3 if same_set else 4)
            priority[row] = category
            score = float(pairs.original_dense_score.iat[row]) if "original_dense_score" in pairs else 0.
            fallback = float(pairs.lexical_key_score.iat[row]) if "lexical_key_score" in pairs else 0.
            negatives.append((category, -score, -fallback, str(target), row))
        negatives.sort()
        for *_, row in negatives[:max_negatives]:
            keep[row] = True
    compact = pairs.loc[keep].copy()
    compact["training_label"] = labels[keep]
    compact["hard_negative_priority"] = priority[keep]
    if not np.array_equal(labels & keep, labels):
        raise ValueError("A retrieved TRAIN positive was removed by negative compaction")
    count = np.bincount(compact.loc[~compact.training_label, "left_idx"], minlength=len(reference))
    if np.any(count > max_negatives):
        raise ValueError("TRAIN negative cap was exceeded")
    return compact, {"train_references": len(reference), "original_candidate_pairs": len(pairs),
                     "retrieved_positive_pairs": int(labels.sum()),
                     "missed_true_links": sum(map(len, truth.values())) - int(labels.sum()),
                     "negative_pairs_retained": int((~compact.training_label).sum()),
                     "max_negative_pairs_per_s1": max_negatives,
                     "same_name_same_digit_set_negatives": int((compact.hard_negative_priority.eq(0)).sum()),
                     "truth_edge_injection": False, "tune_compacted": False}


def probe_train_compaction(lexical_dir: Path, dense_dir: Path, output: Path, *,
                           references: int = 1_000, max_negatives: int = 40,
                           seed: int = 20260925) -> dict:
    """Bounded real TRAIN screen; saved full-pool pairs, no TUNE/eval truth."""
    lexical_dir, dense_dir, output = map(Path, (lexical_dir, dense_dir, output))
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace a prior TRAIN compaction screen: {output}")
    lex_info = json.loads((lexical_dir / "metadata.json").read_text())
    dense_info = json.loads((dense_dir / "metadata.json").read_text())
    if (lex_info.get("selected_splits") != ["train", "tune"]
            or dense_info.get("mining_full_pool_records") != POOL_ROWS
            or lex_info.get("reference_id_sha256") != dense_info.get("reference_id_sha256")):
        raise ValueError("TRAIN probe needs the same saved complete full-pool medium population")
    lex_ref = pd.read_parquet(lexical_dir / "reference.parquet")
    dense_ref = pd.read_parquet(dense_dir / "reference.parquet")
    if (len(lex_ref) != 47_999 or lex_ref.entity_id.duplicated().any()
            or not lex_ref[list(RECORD_COLUMNS) + ["split"]].equals(dense_ref[list(RECORD_COLUMNS) + ["split"]])):
        raise ValueError("Original medium TRAIN/TUNE source order, text or splits disagree")
    eligible = np.flatnonzero(lex_ref.split.eq("train"))
    if (len(eligible) != 35_999 or ids_digest(lex_ref.entity_id.iloc[eligible])
            != "47ec6f4736aaa6069c9dbddde67354637b3046d414165bea0c54ab0dcd5bc0dc"
            or references < 1 or references > len(eligible)):
        raise ValueError("TRAIN probe reference count/digest is outside the saved medium split")
    selected = np.sort(np.random.default_rng(seed).choice(eligible, references, replace=False))
    ref = lex_ref.iloc[selected].reset_index(drop=True)
    select_map = np.full(len(lex_ref), -1, dtype=np.int32)
    select_map[selected] = np.arange(len(ref), dtype=np.int32)
    lex_pool = pd.read_parquet(lexical_dir / "pool.parquet")
    dense_pool = pd.read_parquet(dense_dir / "pool.parquet")
    lex_pairs = pd.read_parquet(lexical_dir / "candidate_pairs.parquet")
    dense_pairs = pd.read_parquet(dense_dir / "pairs.parquet")
    _valid_pairs(lex_pairs, len(lex_ref), len(lex_pool), "TRAIN probe lexical")
    _valid_pairs(dense_pairs, len(dense_ref), len(dense_pool), "TRAIN probe dense")
    lex_pairs = lex_pairs.loc[select_map[lex_pairs.left_idx.to_numpy(dtype=np.int64)] >= 0]
    dense_pairs = dense_pairs.loc[select_map[dense_pairs.left_idx.to_numpy(dtype=np.int64)] >= 0]
    lex = pd.DataFrame({"left_idx": select_map[lex_pairs.left_idx.to_numpy(dtype=np.int64)],
                        "target_entity_id": lex_pool.entity_id.to_numpy()[lex_pairs.right_idx.to_numpy(dtype=np.int64)],
                        "lexical_key_score": lex_pairs.retrieval_score.to_numpy(dtype=np.float32)})
    dense = pd.DataFrame({"left_idx": select_map[dense_pairs.left_idx.to_numpy(dtype=np.int64)],
                          "target_entity_id": dense_pool.entity_id.to_numpy()[dense_pairs.right_idx.to_numpy(dtype=np.int64)],
                          "right_raw_idx": dense_pool.raw_row_idx.to_numpy(dtype=np.int32)[dense_pairs.right_idx.to_numpy(dtype=np.int64)],
                          "original_dense_score": dense_pairs.dense_score.to_numpy(dtype=np.float32)})
    union = union_by_id(lex, dense)
    output.mkdir(parents=True, exist_ok=True)
    ref.to_parquet(output / "reference.parquet", index=False)
    union.to_parquet(output / "raw_candidate_pairs.parquet", index=False)
    # Only now access labels, and only these sampled TRAIN IDs. Candidate edges
    # above have already been durably fixed; truth only retains missed records.
    supplied = json.loads((lexical_dir / "truth.json").read_text())
    truth = {entity: set(supplied[entity]) for entity in ref.entity_id}
    ids = set(union.target_entity_id) | {target for values in truth.values() for target in values}
    lp = lex_pool.loc[lex_pool.entity_id.isin(ids), list(RECORD_COLUMNS)]
    dp = dense_pool.loc[dense_pool.entity_id.isin(ids - set(lp.entity_id)), list(RECORD_COLUMNS)]
    pool = pd.concat((lp, dp), ignore_index=True)
    if pool.entity_id.duplicated().any() or ids - set(pool.entity_id):
        raise ValueError("TRAIN compact pool lost a candidate or an unretrieved truth record")
    right = pd.Index(pool.entity_id).get_indexer(union.target_entity_id)
    if np.any(right < 0):
        raise ValueError("A saved full-pool candidate ID is absent from the TRAIN compact records")
    union["right_idx"] = right.astype(np.int32)
    compact, ranking = compact_train_negatives(ref, pool, union, truth, max_negatives=max_negatives)
    if ranking["retrieved_positive_pairs"] + ranking["missed_true_links"] != sum(map(len, truth.values())):
        raise ValueError("Compaction truth accounting lost an original TRAIN positive")
    pool.to_parquet(output / "pool.parquet", index=False)
    compact.to_parquet(output / "compacted_train_pairs.parquet", index=False)
    atomic_json(output / "truth.json", {key: sorted(values) for key, values in truth.items()})
    paths = ("reference.parquet", "raw_candidate_pairs.parquet", "pool.parquet",
             "compacted_train_pairs.parquet", "truth.json")
    report = {"state": "complete", "schema": 1,
              "scope": "TRAIN-only bounded compatibility screen on saved lexical/phonetic + dense top-50; all 10,320,219 targets compete",
              "sample_references": len(ref), "sampling_seed": seed,
              "sample_id_sha256": ids_digest(ref.entity_id), "full_pool_records": POOL_ROWS,
              "ranking": ranking, "full_pool_candidate_pairs": len(union),
              "files": {name: {"bytes": (output / name).stat().st_size, "sha256": file_sha256(output / name)}
                        for name in paths},
              "truth_read_after_raw_candidates_saved": True,
              "positive_candidate_injection": False,
              "limitations": "TRAIN probe excludes the new key branch, future full-fit population and any TUNE/eval/HOLDOUT-2 labels; its compaction frequencies do not estimate final trained-model precision."}
    atomic_json(output / "complete.json", report)
    return report


def dense_scores_pool_tiled(queries: np.ndarray, pool: np.ndarray,
                            query_rows: np.ndarray, raw_rows: np.ndarray, *,
                            batch_size: int = 20_000, pool_tile: int = 50_000) -> np.ndarray:
    """Score in original pair order while reading object-backed vectors sequentially.

    Reference-order fancy indexing against the remote 7.93-GB pool causes millions
    of tiny FUSE requests. Sort *indices*, not pairs, read each pool tile once,
    then scatter scores back into the immutable candidate-pair order.
    """
    query_rows, raw_rows = (np.asarray(values, dtype=np.int64) for values in (query_rows, raw_rows))
    if (query_rows.shape != raw_rows.shape or query_rows.ndim != 1 or queries.ndim != 2
            or pool.ndim != 2 or queries.shape[1] != pool.shape[1]
            or not 0 < batch_size <= 50_000 or not 0 < pool_tile <= 50_000
            or (len(query_rows) and (query_rows.min() < 0 or query_rows.max() >= len(queries)
                                    or raw_rows.min() < 0 or raw_rows.max() >= len(pool)))):
        raise ValueError("Pool-tiled dense scoring requires aligned, bounded original rows")
    values = np.empty(len(raw_rows), dtype=np.float32)
    order = np.argsort(raw_rows, kind="stable")
    sorted_rows = raw_rows[order]
    for pool_start in range(0, len(pool), pool_tile):
        pool_stop = min(pool_start + pool_tile, len(pool))
        first, last = np.searchsorted(sorted_rows, [pool_start, pool_stop])
        if first == last:
            continue
        # One contiguous remote read, at most ~77 MiB in float32; all following
        # fancy-index access stays in ordinary local RAM.
        tile = np.asarray(pool[pool_start:pool_stop], dtype=np.float32)
        for start in range(first, last, batch_size):
            pair_rows = order[start:min(start + batch_size, last)]
            left = np.asarray(queries[query_rows[pair_rows]], dtype=np.float32)
            right = tile[raw_rows[pair_rows] - pool_start]
            values[pair_rows] = np.einsum("ij,ij->i", left, right)
        if len(pool) > 1_000_000 and (pool_start // pool_tile) % 40 == 0:
            print(f"dense context: pool rows {pool_stop:,}/{len(pool):,}", flush=True)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite dense score after pool-tiled replay")
    return values


def score_context(pairs_file: Path, reference_file: Path, query_reference_file: Path,
                  query_vectors_file: Path, pool_vectors_file: Path, output: Path, *, batch_size: int = 20_000) -> dict:
    """Recompute full-union dense context from the original saved embeddings."""
    if not 0 < batch_size <= 50_000:
        raise ValueError("Context batch size must be 1..50,000")
    pairs_file, output = Path(pairs_file), Path(output)
    query_vectors_file, pool_vectors_file = Path(query_vectors_file), Path(pool_vectors_file)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior scored union: {output}")
    pairs = pd.read_parquet(pairs_file)
    ref, query_ref = pd.read_parquet(reference_file), pd.read_parquet(query_reference_file)
    if pairs.duplicated(["left_idx", "right_raw_idx"]).any() or len(ref) != 12_000 or ids_digest(ref.entity_id) != TUNE_ID_SHA256:
        raise ValueError("Context scoring requires the complete unique medium-TUNE raw union")
    required = {"phonetic_score", "phonetic_only", "dense_only", "lexical_scores_missing",
                "lexical_retrieval_score", "name_score", "address_score", "right_idx", "dense_hit"}
    if required - set(pairs):
        raise ValueError(f"Union omitted matcher provenance columns: {sorted(required - set(pairs))}")
    indices = pd.Index(query_ref.entity_id).get_indexer(ref.entity_id)
    if np.any(indices < 0) or not np.array_equal(ref[list(RECORD_COLUMNS)].to_numpy(),
                                                  query_ref.iloc[indices][list(RECORD_COLUMNS)].to_numpy()):
        raise ValueError("Saved dense query embeddings are not aligned with exact S1 text")
    from .hf_experiment import frame_digest

    query_identity = json.loads((query_vectors_file.parent / "query_manifest.json").read_text())
    pool_complete = json.loads((pool_vectors_file.parent / "complete.json").read_text())
    preservation = json.loads((pool_vectors_file.parent / "preservation.json").read_text())
    pool_identity = pool_complete["identity"]
    if (query_identity["reference_digest"] != frame_digest(query_ref, list(RECORD_COLUMNS))
            or query_identity["checkpoint_sha256"] != pool_identity["checkpoint"]["sha256"]
            or pool_complete.get("state") != "complete" or pool_complete.get("full_dataset_complete") is not True
            or pool_complete.get("verification", {}).get("all_finite") is not True
            or preservation.get("copy_verified") is not True
            or preservation["bytes"] != pool_vectors_file.stat().st_size
            or pool_identity["target_rows"] != POOL_ROWS):
        raise ValueError("Saved query/pool checkpoint, text or preserved full-pool identity differs")
    q = np.load(query_vectors_file, mmap_mode="r")
    p = np.load(pool_vectors_file, mmap_mode="r")
    if (q.shape[0] != len(query_ref) or p.shape[0] != POOL_ROWS or q.shape[1] != p.shape[1]
            or np.any(pairs.right_raw_idx.to_numpy(dtype=np.int64) >= POOL_ROWS)
            or np.any(pairs.right_raw_idx.to_numpy(dtype=np.int64) < 0)):
        raise ValueError("Original pool/query embeddings or full target mapping are incomplete")
    left = indices[pairs.left_idx.to_numpy(dtype=np.int32)]
    right = pairs.right_raw_idx.to_numpy(dtype=np.int64)
    values = dense_scores_pool_tiled(q, p, left, right, batch_size=batch_size)
    previously_dense = pairs.dense_hit.to_numpy(dtype=bool)
    if previously_dense.any() and np.max(np.abs(
            values[previously_dense] - pairs.original_dense_score.to_numpy(dtype=np.float32)[previously_dense])) > 2e-4:
        raise ValueError("Original dense top-K scores do not replay from aligned E5 embeddings")
    pairs["dense_score"] = np.clip(values, -1, 1)
    pairs["retrieval_score"] = pairs.dense_score.to_numpy(dtype=np.float32)
    pairs["retrieval_rank"] = pairs.groupby("left_idx", sort=False).dense_score.rank(
        method="min", ascending=False).astype(np.int32)
    output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(str(output) + ".tmp", index=False)
    Path(str(output) + ".tmp").replace(output)
    manifest = {"state": "complete", "pair_file": {"path": str(output.resolve()), "rows": len(pairs),
                 "bytes": output.stat().st_size, "sha256": file_sha256(output)},
                "inputs": {name: {"path": str(Path(path).resolve()), "bytes": Path(path).stat().st_size,
                                    "sha256": file_sha256(Path(path)) if name != "pool_embeddings" else None}
                           for name, path in (("pairs", pairs_file), ("reference", reference_file),
                                              ("query_reference", query_reference_file), ("query_vectors", query_vectors_file),
                                              ("pool_embeddings", pool_vectors_file))},
                "full_pool_embedding_hash": preservation["sha256"],
                "full_pool_preservation_manifest_sha256": file_sha256(pool_vectors_file.parent / "preservation.json"),
                "query_manifest_sha256": file_sha256(query_vectors_file.parent / "query_manifest.json"),
                "score_execution": "pool-row ascending, 50,000-row sequential tiles; scatter into unchanged pair order",
                "candidate_dependent_features_recomputed": ["dense_score", "retrieval_score", "retrieval_rank"],
                "remaining_before_matching": "Call ber.features.build_features and ber.numeric_features.build_numeric_features on complete union per-S1 groups; re-fit matcher for changed candidate set.",
                "positive_candidate_injection": False}
    atomic_json(output.with_suffix(".context.json"), manifest)
    return manifest


def replay_paired_retrieval(directory: Path, *, replicates: int = 10_000,
                            seed: int = 20260925) -> dict:
    """10k paired S1 bootstrap from frozen TUNE pairs; no retrieval or refitting."""
    directory = Path(directory)
    output = directory / "paired_retrieval_10k.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prior paired analysis: {output}")
    report = json.loads((directory / "complete.json").read_text())
    pair_file = directory / "candidate_pairs_raw.parquet"
    if (report.get("state") != "complete" or report.get("new_branch_identity") is None
            or report["raw_pairs"]["sha256"] != file_sha256(pair_file)):
        raise ValueError("Require complete untouched raw union pairs before paired analysis")
    reference = pd.read_parquet(directory / "reference.parquet")
    pairs = pd.read_parquet(pair_file)
    truth = {key: set(values) for key, values in json.loads((directory / "truth.json").read_text()).items()}
    if (len(pairs) != report["raw_pairs"]["rows"] or len(reference) != 12_000
            or ids_digest(reference.entity_id) != TUNE_ID_SHA256
            or set(truth) != set(reference.entity_id)
            or pairs.duplicated(["left_idx", "target_entity_id"]).any()):
        raise ValueError("Aligned original TUNE IDs/truth/pairs differ from the completed screen")
    refids = reference.entity_id.to_numpy()
    left = pairs.left_idx.to_numpy(dtype=np.int32)
    targets = pairs.target_entity_id.to_numpy()
    labels = np.fromiter((target in truth[refids[pos]] for pos, target in zip(left, targets)),
                         dtype=bool, count=len(pairs))
    masks = _variant_masks(pairs)
    saved, saved_values = _oracle(reference, pairs, masks["saved_union"], labels, truth)
    extended, extended_values = _oracle(reference, pairs, masks["full_key_union"], labels, truth)
    for name, value in (("saved_union", saved), ("full_key_union", extended)):
        for key in ("pairs", "retrieved_true_links", "oracle_macro_f05"):
            if not np.isclose(value[key], report["retrieval"][name][key], rtol=0, atol=1e-12):
                raise ValueError(f"Candidate or oracle score changed since {name} finished")
    result = {"scope": "same 12,000 medium TUNE S1s, complete full-pool candidate sets",
              "variant": "full_key_union_minus_saved_union",
              "pair_sha256": report["raw_pairs"]["sha256"],
              "reference_ids_sha256": TUNE_ID_SHA256,
              "newly_retrieved_true_links": extended["retrieved_true_links"] - saved["retrieved_true_links"],
              "newly_retrieved_pairs": extended["pairs"] - saved["pairs"],
              "bootstrap": paired_bootstrap(extended_values, saved_values,
                                            n_bootstrap=replicates, seed=seed),
              "limitation": "Conditional on reused TUNE and fixed retrieval; not a learned-model or holdout result."}
    atomic_json(output, result)
    return result


def inventory_union(directory: Path, index_dir: Path, key_dir: Path) -> dict:
    """Hash all consumable TUNE files, not just the label-free raw edge file."""
    directory, index_dir, key_dir = map(Path, (directory, index_dir, key_dir))
    manifest_path = directory / "artifact_inventory.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace prior artifact inventory: {manifest_path}")
    complete = json.loads((directory / "complete.json").read_text())
    keys = json.loads((key_dir / "complete.json").read_text())
    index = json.loads((index_dir / "complete.json").read_text())
    if (any(value.get("state") != "complete" for value in (complete, keys, index))
            or complete["new_branch_identity"]["index_manifest_sha256"] != file_sha256(index_dir / "complete.json")
            or complete["new_branch_identity"]["sha256"] != keys["pair_file"]["sha256"]
            or complete["reference_ids_sha256"] != keys["reference_ids_sha256"]):
        raise ValueError("Full-pool index, key pairs and controlled union do not share identities")
    files = {}
    for name in ("reference.parquet", "pool.parquet", "candidate_pairs.parquet",
                 "candidate_pairs_raw.parquet", "truth.json", "complete.json", "paired_retrieval_10k.json"):
        path = directory / name
        files[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    if files["candidate_pairs_raw.parquet"]["sha256"] != complete["raw_pairs"]["sha256"]:
        raise ValueError("Fixed pre-truth raw candidate pairs changed")
    rows = pq.ParquetFile(directory / "candidate_pairs.parquet").metadata.num_rows
    if rows != complete["raw_pairs"]["rows"]:
        raise ValueError("Compact pair row count differs from label-free raw retrieval")
    report = {"schema": 1, "state": "complete", "scope": "12k medium TUNE; all 10,320,219 original targets",
              "reference_ids_sha256": complete["reference_ids_sha256"],
              "candidate_pairs": rows,
              "full_index_manifest_sha256": file_sha256(index_dir / "complete.json"),
              "full_index_sqlite_sha256": index["files"]["pool.sqlite"]["sha256"],
              "key_candidate_pairs_sha256": keys["pair_file"]["sha256"],
              "files": files, "unretrieved_truth_records_in_pool": complete["retention"]["truth_targets_without_candidate_edge"],
              "positive_candidate_injection": False,
              "candidate_context_ready_for_matcher": False,
              "next_step": "Recompute dense cosine/rank on all union pairs, and every candidate-dependent matcher feature on complete per-S1 groups."}
    atomic_json(manifest_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    merge = modes.add_parser("compare", help="Controlled 12k TUNE recall/oracle; optional new index and key pairs")
    merge.add_argument("--lexical", required=True, type=Path)
    merge.add_argument("--dense", required=True, type=Path)
    merge.add_argument("--index", type=Path)
    merge.add_argument("--keys", type=Path)
    merge.add_argument("--output", required=True, type=Path)
    merge.add_argument("--bootstrap-replicates", type=int, default=2_000)
    context = modes.add_parser("score-context", help="Attach all full-union dense scores/ranks from saved embeddings")
    context.add_argument("--pairs", required=True, type=Path)
    context.add_argument("--reference", required=True, type=Path)
    context.add_argument("--query-reference", required=True, type=Path)
    context.add_argument("--query-vectors", required=True, type=Path)
    context.add_argument("--pool-vectors", required=True, type=Path)
    context.add_argument("--output", required=True, type=Path)
    replay = modes.add_parser("replay-paired", help="10k complete-entity bootstrap from saved TUNE candidate IDs")
    replay.add_argument("--union", required=True, type=Path)
    replay.add_argument("--replicates", type=int, default=10_000)
    replay.add_argument("--seed", type=int, default=20260925)
    inventory = modes.add_parser("inventory", help="SHA256 and coverage manifest for the completed full-pool TUNE union")
    inventory.add_argument("--union", required=True, type=Path)
    inventory.add_argument("--index", required=True, type=Path)
    inventory.add_argument("--keys", required=True, type=Path)
    train_probe = modes.add_parser("train-probe", help="TRAIN-only hard-negative compaction on bounded saved full-pool S1s")
    train_probe.add_argument("--lexical", required=True, type=Path)
    train_probe.add_argument("--dense", required=True, type=Path)
    train_probe.add_argument("--output", required=True, type=Path)
    train_probe.add_argument("--references", type=int, default=1_000)
    train_probe.add_argument("--max-negatives", type=int, default=40)
    train_probe.add_argument("--seed", type=int, default=20260925)
    args = parser.parse_args()
    if args.mode == "compare":
        result = run_union(args.lexical, args.dense, args.output, index_dir=args.index,
                           key_dir=args.keys, bootstrap_replicates=args.bootstrap_replicates)
        print(json.dumps({"report": str((args.output / "complete.json").resolve()),
                          "link_recall": result["retrieval"]["full_key_union" if args.index else "saved_union"]["link_recall"],
                          "oracle_macro_f05": result["retrieval"]["full_key_union" if args.index else "saved_union"]["oracle_macro_f05"]}))
    elif args.mode == "score-context":
        print(json.dumps(score_context(args.pairs, args.reference, args.query_reference,
                                       args.query_vectors, args.pool_vectors, args.output)["pair_file"]))
    elif args.mode == "replay-paired":
        result = replay_paired_retrieval(args.union, replicates=args.replicates, seed=args.seed)
        print(json.dumps({"newly_retrieved_true_links": result["newly_retrieved_true_links"],
                          "new_pairs": result["newly_retrieved_pairs"],
                          "paired_delta": result["bootstrap"]["estimates"]["left_minus_right"],
                          "ci95": result["bootstrap"]["confidence_intervals"]["left_minus_right"]}))
    elif args.mode == "inventory":
        value = inventory_union(args.union, args.index, args.keys)
        print(json.dumps({"candidate_pairs": value["candidate_pairs"],
                          "pairs_sha256": value["files"]["candidate_pairs.parquet"]["sha256"],
                          "unretrieved_truth_records": value["unretrieved_truth_records_in_pool"]}))
    else:
        result = probe_train_compaction(args.lexical, args.dense, args.output,
                                        references=args.references, max_negatives=args.max_negatives, seed=args.seed)
        print(json.dumps({"sample_references": result["sample_references"],
                          "candidate_pairs": result["full_pool_candidate_pairs"],
                          "ranking": result["ranking"]}))


if __name__ == "__main__":
    main()
