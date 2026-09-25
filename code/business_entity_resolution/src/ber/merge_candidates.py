"""Merge fixed full-pool lexical/phonetic and dense candidates by record ID.

No retrieval index is rebuilt and no truth edge is inserted. All original lexical
scores are retained. The common retrieval score/rank use dense cosine for every
union pair, while lexical score/rank keep distinct explicit column names.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .full_dense_candidates import _atomic_parquet, read_embedding_manifest
from .full_embeddings import Reporter, atomic_json
from .hf_experiment import frame_digest
from .metrics import candidate_report


RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]


def _load_records(directory: Path):
    reference = pd.read_parquet(directory / "reference.parquet").fillna("")
    pool = pd.read_parquet(directory / "pool.parquet").fillna("")
    if reference.entity_id.duplicated().any() or pool.entity_id.duplicated().any():
        raise ValueError(f"Duplicate entity IDs in {directory}")
    if not reference.split.isin(["train", "tune"]).all():
        raise ValueError("Union utility accepts existing train/tune references only")
    truth = {key: set(values) for key, values in json.loads((directory / "truth.json").read_text()).items()}
    if set(truth) != set(reference.entity_id):
        raise ValueError(f"Truth must cover exactly the existing reference records in {directory}")
    return reference, pool, truth


def _pairs_path(directory: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    for name in ("candidate_pairs.parquet", "pairs.parquet"):
        if (directory / name).exists():
            return directory / name
    raise FileNotFoundError(f"No candidate pair artifact in {directory}")


def _check_pairs(pairs: pd.DataFrame, reference: pd.DataFrame, pool: pd.DataFrame) -> None:
    if pairs.duplicated(["left_idx", "right_idx"]).any():
        raise ValueError("Input candidate artifact contains duplicate pairs")
    if len(pairs):
        if pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference):
            raise ValueError("Input candidate reference positions are invalid")
        if pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool):
            raise ValueError("Input candidate pool positions are invalid")


def align_records(lex_reference, lex_pool, lex_truth, dense_reference, dense_pool, dense_truth):
    """Keep the lexical reference/split order and append only new pool IDs."""
    if set(lex_reference.entity_id) != set(dense_reference.entity_id):
        raise ValueError("Controlled union comparison requires identical reference ID sets")
    reference_index = pd.Index(lex_reference.entity_id)
    dense_left = reference_index.get_indexer(dense_reference.entity_id)
    columns = RECORD_COLUMNS + ["split"]
    if not np.array_equal(lex_reference.iloc[dense_left][columns].to_numpy(), dense_reference[columns].to_numpy()):
        raise ValueError("Reference text or fixed splits disagree between lexical and dense artifacts")
    if lex_truth != dense_truth:
        raise ValueError("Provided ground truth differs between candidate artifacts")
    lex_index = pd.Index(lex_pool.entity_id)
    shared = lex_index.get_indexer(dense_pool.entity_id)
    exists = shared >= 0
    if not np.array_equal(lex_pool.iloc[shared[exists]][RECORD_COLUMNS].to_numpy(), dense_pool.loc[exists, RECORD_COLUMNS].to_numpy()):
        raise ValueError("Shared pool IDs have conflicting source text or country")
    pool = pd.concat([lex_pool[RECORD_COLUMNS], dense_pool.loc[~exists, RECORD_COLUMNS]], ignore_index=True)
    pool_index = pd.Index(pool.entity_id)
    dense_right = pool_index.get_indexer(dense_pool.entity_id)
    raw_rows = np.full(len(pool), -1, dtype=np.int64)
    if "raw_row_idx" in lex_pool:
        values = lex_pool.raw_row_idx.to_numpy(dtype=np.int64)
        raw_rows[:len(lex_pool)] = values
    if "raw_row_idx" not in dense_pool:
        raise ValueError("Dense compact pool must retain the original raw_row_idx mapping")
    known = dense_pool.raw_row_idx.to_numpy(dtype=np.int64)
    conflict = (raw_rows[dense_right] >= 0) & (raw_rows[dense_right] != known)
    if conflict.any():
        raise ValueError("Pool raw row mappings disagree for shared IDs")
    raw_rows[dense_right] = known
    pool["raw_row_idx"] = raw_rows
    return lex_reference.copy(), pool, dense_left, dense_right


def fill_raw_rows(pool: pd.DataFrame, identity: dict, reporter: Reporter) -> None:
    """Resolve only unknown compact-pool IDs by streaming the supplied raw IDs."""
    raw_rows = pool.raw_row_idx.to_numpy(dtype=np.int64, copy=True)
    unknown = int((raw_rows < 0).sum())
    reporter.write(f"Resolving {unknown:,} lexical-pool IDs absent from the dense compact mapping")
    if unknown:
        lookup = pd.Index(pool.entity_id)
        offset = 0
        for item in identity["inputs_in_order"]:
            source_count = 0
            for frame in pd.read_csv(item["path"], sep="\t", dtype=str, keep_default_na=False,
                                     usecols=["entity_id"], chunksize=50000):
                positions = lookup.get_indexer(frame.entity_id)
                retained = positions >= 0
                compact = positions[retained]
                original = offset + np.flatnonzero(retained)
                already = raw_rows[compact] >= 0
                if np.any(raw_rows[compact[already]] != original[already]):
                    raise ValueError("A known raw-row mapping does not match the provided source order")
                raw_rows[compact] = original
                offset += len(frame)
                source_count += len(frame)
            if source_count != item["expected_rows"]:
                raise ValueError("Raw source row counts changed while resolving IDs")
        if np.any(raw_rows < 0):
            raise ValueError(f"{int((raw_rows < 0).sum())} compact-pool IDs do not exist in the full raw inputs")
    if np.any(raw_rows >= identity["target_rows"]) or len(np.unique(raw_rows)) != len(raw_rows):
        raise ValueError("Invalid or duplicate global raw-row mapping")
    pool["raw_row_idx"] = raw_rows


def merge_edges(lex_pairs: pd.DataFrame, dense_pairs: pd.DataFrame,
                 dense_left: np.ndarray, dense_right: np.ndarray) -> pd.DataFrame:
    required = {"name_score", "address_score", "retrieval_score", "retrieval_rank"}
    if required - set(lex_pairs):
        raise ValueError(f"Lexical artifact lacks scores: {sorted(required - set(lex_pairs))}")
    lexical = lex_pairs.copy()
    if "dense_only" in lexical or "dense_score" in lexical:
        raise ValueError("Lexical input is already a dense union; supply the original lexical/phonetic artifact")
    lexical.rename(columns={"retrieval_score": "lexical_retrieval_score",
                            "retrieval_rank": "lexical_retrieval_rank"}, inplace=True)
    dense = pd.DataFrame({"left_idx": dense_left[dense_pairs.left_idx.to_numpy(dtype=np.int64)],
                          "right_idx": dense_right[dense_pairs.right_idx.to_numpy(dtype=np.int64)],
                          "dense_rank": dense_pairs.dense_rank.to_numpy(dtype=np.int32)})
    union = lexical.merge(dense, on=["left_idx", "right_idx"], how="outer", validate="one_to_one", indicator=True)
    dense_only = union["_merge"].eq("right_only")
    union["dense_only"] = dense_only.astype(np.float32)
    union["dense_hit"] = union["_merge"].ne("left_only")
    union["lexical_hit"] = union["_merge"].ne("right_only")
    if "lexical_scores_missing" in union:
        union["lexical_scores_missing"] = union.lexical_scores_missing.fillna(False).astype(bool) | dense_only
    else:
        union["lexical_scores_missing"] = dense_only
    for name in ("name_score", "address_score", "lexical_retrieval_score", "phonetic_score"):
        if name in union:
            union[name] = union[name].fillna(0).astype(np.float32)
    for name in ("lexical_retrieval_rank", "phonetic_rank", "dense_rank"):
        if name in union:
            union[name] = union[name].fillna(0).astype(np.int32)
    for name in ("phonetic_hit", "phonetic_only"):
        if name in union:
            union[name] = union[name].fillna(False).astype(bool)
    union.drop(columns="_merge", inplace=True)
    return union


def attach_dense_scores(union: pd.DataFrame, pool: pd.DataFrame,
                        queries: np.ndarray, query_rows: np.ndarray,
                        embeddings: np.ndarray, batch_size: int) -> None:
    left = query_rows[union.left_idx.to_numpy(dtype=np.int64)]
    raw_rows = pool.raw_row_idx.to_numpy(dtype=np.int64)[union.right_idx.to_numpy(dtype=np.int64)]
    scores = np.empty(len(union), dtype=np.float32)
    for start in range(0, len(union), batch_size):
        stop = start + batch_size
        q = np.asarray(queries[left[start:stop]], dtype=np.float32)
        p = np.asarray(embeddings[raw_rows[start:stop]], dtype=np.float32)
        scores[start:stop] = np.einsum("ij,ij->i", q, p)
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite dense cosine in merged candidates")
    union["dense_score"] = np.clip(scores, -1, 1)
    union["retrieval_score"] = union.dense_score.to_numpy(dtype=np.float32)
    # One comparable score family across the union. Ties share a rank, avoiding
    # arbitrary identifier/row-order effects in the ranking feature.
    union["retrieval_rank"] = union.groupby("left_idx", sort=False).dense_score.rank(
        method="min", ascending=False).astype(np.int32)
    union["right_raw_idx"] = raw_rows
    union.sort_values(["left_idx", "retrieval_rank"], kind="stable", inplace=True, ignore_index=True)


def run(args):
    started = time.time()
    lexical_dir, dense_dir, output = map(Path, (args.lexical_data, args.dense_data, args.output))
    reporter = Reporter(output, Path(args.durable_output) if args.durable_output else None)
    lock = (output / "merge.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another candidate merge owns this output directory") from error
    identity = read_embedding_manifest(Path(args.embeddings))
    lex_ref, lex_pool, lex_truth = _load_records(lexical_dir)
    dense_ref, dense_pool, dense_truth = _load_records(dense_dir)
    lexical_path = _pairs_path(lexical_dir, args.lexical_pairs)
    dense_path = _pairs_path(dense_dir, args.dense_pairs)
    lex_pairs, dense_pairs = pd.read_parquet(lexical_path), pd.read_parquet(dense_path)
    _check_pairs(lex_pairs, lex_ref, lex_pool)
    _check_pairs(dense_pairs, dense_ref, dense_pool)
    reference, pool, dense_left, dense_right = align_records(
        lex_ref, lex_pool, lex_truth, dense_ref, dense_pool, dense_truth)
    fill_raw_rows(pool, identity, reporter)
    truth_targets = {value for values in lex_truth.values() for value in values}
    if truth_targets - set(pool.entity_id):
        raise ValueError("A truth target was lost from the merged compact pool")
    query_manifest = json.loads((dense_dir / "query_manifest.json").read_text())
    if query_manifest["checkpoint_sha256"] != identity["checkpoint"]["sha256"]:
        raise ValueError("Dense query checkpoint differs from the full-pool embedding checkpoint")
    expected_query_digest = frame_digest(dense_ref, RECORD_COLUMNS)
    if query_manifest["reference_digest"] != expected_query_digest:
        raise ValueError("Dense query embeddings are not aligned with their reference artifact")
    queries = np.load(dense_dir / "query_embeddings.npy", mmap_mode="r")
    if queries.shape != (len(dense_ref), identity["dimension"]):
        raise ValueError("Dense query embedding shape is invalid")
    query_rows = pd.Index(dense_ref.entity_id).get_indexer(reference.entity_id)
    union = merge_edges(lex_pairs, dense_pairs, dense_left, dense_right)
    reporter.write(f"Scoring {len(union):,} union pairs from saved query/pool embeddings, with no TF-IDF rebuild")
    embeddings = np.load(args.embeddings, mmap_mode="r")
    attach_dense_scores(union, pool, queries, query_rows, embeddings, args.score_batch_size)
    # Verify all original lexical scores survived ID remapping exactly.
    check = lex_pairs.merge(union, on=["left_idx", "right_idx"], suffixes=("_before", "_after"), validate="one_to_one")
    if len(check) != len(lex_pairs):
        raise ValueError("Original lexical candidate edges were lost from the union")
    for name in ("name_score", "address_score", "phonetic_score"):
        if name in lex_pairs and not np.array_equal(check[name + "_before"].to_numpy(dtype=np.float32),
                                                    check[name + "_after"].to_numpy(dtype=np.float32)):
            raise ValueError(f"Original {name} was altered by the candidate merge")
    if not np.array_equal(check.retrieval_score_before.to_numpy(dtype=np.float32),
                          check.lexical_retrieval_score.to_numpy(dtype=np.float32)):
        raise ValueError("Original lexical retrieval scores were altered")
    if not np.array_equal(check.retrieval_rank_before.to_numpy(dtype=np.int32),
                          check.lexical_retrieval_rank.to_numpy(dtype=np.int32)):
        raise ValueError("Original lexical retrieval ranks were altered")
    del check
    dense_aligned = pd.DataFrame({"left_idx": dense_left[dense_pairs.left_idx.to_numpy(dtype=np.int64)],
                                  "right_idx": dense_right[dense_pairs.right_idx.to_numpy(dtype=np.int64)]})
    splits = sorted(reference.split.unique())
    retrieval = {name: {split: candidate_report(reference, pool, pairs, lex_truth, reference.split.eq(split))
                        for split in splits}
                 for name, pairs in (("lexical", lex_pairs), ("dense", dense_aligned), ("union", union))}
    _atomic_parquet(reference, output / "reference.parquet")
    _atomic_parquet(pool, output / "pool.parquet")
    _atomic_parquet(union, output / "candidate_pairs.parquet")
    atomic_json(output / "truth.json", {key: sorted(values) for key, values in lex_truth.items()})
    metadata = {"dataset_kind": "full_pool_lexical_phonetic_dense_union", "references": len(reference),
                "pool_records": len(pool), "candidate_pairs": len(union),
                "candidate_pairs_file": "candidate_pairs.parquet", "true_links": sum(map(len, lex_truth.values())),
                "reference_id_sha256": hashlib.sha256("\n".join(sorted(reference.entity_id)).encode()).hexdigest(),
                "split_counts": {f"{split}/{country}": int(count) for (split, country), count in
                                 reference.groupby(["split", "country"]).size().items()},
                "parent_lexical": str(lexical_dir.resolve()), "parent_dense": str(dense_dir.resolve()),
                "full_pool_records": identity["target_rows"], "checkpoint_sha256": identity["checkpoint"]["sha256"],
                "retrieval": retrieval, "dense_only_pairs": int(union.dense_only.sum()),
                "lexical_scores_missing_pairs": int(union.lexical_scores_missing.sum()),
                "positive_candidate_injection": False, "original_splits_preserved": True,
                "feature_semantics": {"name_score/address_score/phonetic_score": "original full-pool scores, unchanged; absent values zero with flags",
                                      "lexical_retrieval_score/rank": "original lexical/phonetic branch values, unchanged; zero when absent",
                                      "dense_score": "float32 dot from saved query/full-pool embeddings for every union edge",
                                      "dense_rank": "original full-pool dense top-K rank; zero when absent",
                                      "retrieval_score": "dense_score for every union edge",
                                      "retrieval_rank": "rank by dense_score within each reference union, equal scores share rank",
                                      "lexical_scores_missing": "true if the pair has no computed lexical channel scores"},
                "required_optional_model_features": ["dense_score", "dense_only", "lexical_scores_missing"],
                "limitations": ["Union quality is measured only on these shared training/tuning references.",
                                "Dense-only medium-set scores do not validate the union; equivalent full-pool lexical/phonetic branches must first be mined for those references."],
                "elapsed_seconds": time.time() - started}
    reporter.manifest(metadata, "metadata.json")
    reporter.manifest({"state": "complete", "metadata": metadata}, "complete.json")
    reporter.write(f"COMPLETE union tuning retrieval={retrieval['union'].get('tune')}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lexical-data", default="data/hard")
    parser.add_argument("--dense-data", default="artifacts/full_dense_quick")
    parser.add_argument("--lexical-pairs")
    parser.add_argument("--dense-pairs")
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--output", default="data/hard_union")
    parser.add_argument("--durable-output")
    parser.add_argument("--score-batch-size", type=int, default=20000)
    args = parser.parse_args()
    if not 0 < args.score_batch_size <= 50000:
        parser.error("Score batch size must be between 1 and 50,000")
    run(args)


if __name__ == "__main__":
    main()
