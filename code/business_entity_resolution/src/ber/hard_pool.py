"""Mine full-release hard candidates for fixed quick train/tune references.

Mining uses only source text. After each country's retrieval completes, labels are
used to retain all true target records in the compact output pool. A positive missed
by retrieval is retained as a record, but no artificial candidate edge is added.
The resulting candidate_pairs.parquet therefore preserves the full-pool retrieval
problem and can be scored/trained directly without rerunning retrieval on a reduced
pool. Evaluation references never enter mining or the resulting prepared dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .candidates import PoolCandidateIndex, _country_values
from .normalization import normalize_text
from .phonetic_ablation import augment_phonetic_candidates
from .pruned_retrieval import generate_pruned_candidates
from .retrieval_ablation import _log

_RECORD_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
_PAIR_DTYPES = {
    "left_idx": "int64", "right_idx": "int64", "name_score": "float32",
    "address_score": "float32", "retrieval_score": "float32", "retrieval_rank": "int32",
    "phonetic_score": "float32", "phonetic_rank": "int32", "phonetic_hit": "bool",
    "phonetic_only": "bool", "lexical_scores_missing": "bool",
}


def _read_country(source_paths: list[Path], country: str, chunksize: int) -> pd.DataFrame:
    """Read a single dynamic country in bounded TSV chunks; never load all countries."""
    pieces = []
    for path in source_paths:
        retained = 0
        for chunk in pd.read_csv(path, sep="\t", usecols=_RECORD_COLUMNS,
                                 dtype=str, keep_default_na=False, chunksize=chunksize):
            labels = chunk.country
            normalized = {label: normalize_text(label) for label in labels.unique()}
            selected = chunk.loc[labels.map(normalized).eq(country), _RECORD_COLUMNS]
            if len(selected):
                pieces.append(selected.copy())
                retained += len(selected)
        _log(f"Read {path.name}: {retained:,} rows for {country}")
    if not pieces:
        return pd.DataFrame({column: pd.Series(dtype=str) for column in _RECORD_COLUMNS})
    result = pd.concat(pieces, ignore_index=True)
    if result.entity_id.duplicated().any():
        raise ValueError(f"Duplicate entity IDs in raw {country} pool")
    return result


def _load_retained_truth(path: Path, reference_ids: list[str]) -> dict[str, list[str]]:
    """Called after retrieval; unrelated labels are never returned or inspected."""
    supplied = json.loads(path.read_text(encoding="utf-8"))
    missing = set(reference_ids) - supplied.keys()
    if missing:
        raise ValueError(f"Missing truth for {len(missing)} retained reference IDs")
    return {entity: sorted(set(supplied[entity])) for entity in reference_ids}


def _coerce_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for column, dtype in _PAIR_DTYPES.items():
        if column in frame:
            frame[column] = frame[column].astype(dtype)
    return frame


def _write_table(writer, frame: pd.DataFrame, path: Path):
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(path, table.schema, compression="snappy")
    elif table.schema != writer.schema:
        table = table.cast(writer.schema)
    writer.write_table(table)
    return writer


def build_hard_pool(
    data_dir: str | Path, dataset_dir: str | Path, output_dir: str | Path, *,
    splits: tuple[str, ...] = ("train", "tune"),
    query_terms: int = 32, approximate_top_k: int = 300, top_k: int = 30,
    phonetic_top_k: int = 20, exact_name_cap: int = 100,
    min_similarity: float = .08, max_features: int = 100_000,
    chunksize: int = 200_000, batch_size: int = 256, n_jobs: int = 4,
) -> dict:
    """Build a prepared train/tune dataset plus original full-pool candidate edges.

    Raw input is dataset/train/train_source{2,3}.tsv, or the train directory itself.
    ``splits`` may select train, tune, or both; evaluation is always prohibited.
    No raw Source 1 or evaluation labels are needed. Returned metadata explains why
    callers should reuse candidate_pairs.parquet instead of rebuilding candidates.
    """
    started = time.time()
    data_dir, dataset_dir, output_dir = map(Path, (data_dir, dataset_dir, output_dir))
    if isinstance(splits, str):
        splits = tuple(value.strip() for value in splits.split(","))
    splits = tuple(dict.fromkeys(splits))
    if not splits or set(splits) - {"train", "tune"}:
        raise ValueError("splits must select train and/or tune; evaluation mining is prohibited")
    if top_k <= 0 or query_terms <= 0 or approximate_top_k < top_k:
        raise ValueError("Require query_terms>0 and approximate_top_k>=top_k>0")
    if min(chunksize, batch_size, n_jobs) <= 0 or phonetic_top_k < 0:
        raise ValueError("Chunk/batch/thread counts must be positive; phonetic_top_k nonnegative")
    train_dir = dataset_dir / "train" if (dataset_dir / "train").is_dir() else dataset_dir
    source_paths = [train_dir / f"train_source{source}.tsv" for source in (2, 3)]
    for path in source_paths + [data_dir / "reference.parquet", data_dir / "truth.json"]:
        if not path.is_file():
            raise FileNotFoundError(path)
    original = pd.read_parquet(data_dir / "reference.parquet").fillna("")
    if "split" not in original:
        raise ValueError("Prepared reference requires fixed split labels")
    reference = original.loc[original.split.isin(splits)].reset_index(drop=True)
    del original
    if not len(reference) or reference.entity_id.duplicated().any():
        raise ValueError("Retained train/tune reference set must be nonempty with unique IDs")
    if any(not reference.split.eq(split).any() for split in splits):
        raise ValueError("Every explicitly selected development partition must be nonempty")
    reference_keys = _country_values(reference)
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary = {name: output_dir / f".{name}.tmp" for name in
                 ("reference.parquet", "pool.parquet", "candidate_pairs.parquet", "truth.json", "metadata.json")}
    reference.to_parquet(temporary["reference.parquet"], index=False)
    pool_writer = candidate_writer = None
    pool_offset = candidate_count = true_record_count = 0
    full_counts, country_reports, retained_truth = {}, [], {}
    try:
        for country in sorted(set(reference_keys)):
            country_started = time.time()
            left_positions = np.flatnonzero(reference_keys == country)
            country_reference = reference.iloc[left_positions].reset_index(drop=True)
            country_pool = _read_country(source_paths, country, chunksize)
            full_counts[country] = len(country_pool)
            _log(f"Mining {country}: {len(country_reference):,} train/tune references against {len(country_pool):,} full-pool targets")
            index = PoolCandidateIndex(country_pool, max_features=max_features, ngram_range=(3, 4))
            # No truth is loaded or supplied to either retrieval branch.
            pairs = generate_pruned_candidates(
                country_reference, country_pool, query_terms=query_terms,
                approximate_top_k=approximate_top_k, top_k=top_k,
                exact_name_cap=exact_name_cap, min_similarity=min_similarity,
                n_jobs=n_jobs, batch_size=batch_size, pool_index=index,
            )
            if phonetic_top_k:
                pairs = augment_phonetic_candidates(
                    country_reference, country_pool, pairs, top_k=phonetic_top_k,
                    native_only=True, n_jobs=n_jobs, batch_size=batch_size,
                    max_features=max_features, min_similarity=min_similarity,
                    lexical_index=index,
                )
            pairs = _coerce_pairs(pairs)
            if pairs.duplicated(["left_idx", "right_idx"]).any():
                raise ValueError("Retriever produced duplicate candidate edges")

            # Labels affect retention only, after the full-pool candidate set is fixed.
            country_truth = _load_retained_truth(data_dir / "truth.json", country_reference.entity_id.tolist())
            true_ids = sorted({target for values in country_truth.values() for target in values})
            true_positions = pd.Index(country_pool.entity_id).get_indexer(true_ids)
            if (true_positions < 0).any():
                raise ValueError(f"{int((true_positions < 0).sum())} positive IDs are absent from raw country {country!r}; check source release and country agreement")
            candidate_positions = pairs.right_idx.to_numpy(dtype=np.int64)
            keep = np.unique(np.concatenate((candidate_positions, true_positions))).astype(np.int64)
            compact_pool = country_pool.iloc[keep][_RECORD_COLUMNS].reset_index(drop=True)
            # Missing positives remain pool records without manufactured pair edges.
            mapped = pairs.copy()
            mapped["left_idx"] = left_positions[pairs.left_idx.to_numpy(dtype=np.int64)]
            mapped["right_idx"] = np.searchsorted(keep, candidate_positions).astype(np.int64) + pool_offset
            if len(keep) and not np.array_equal(keep[np.searchsorted(keep, candidate_positions)], candidate_positions):
                raise ValueError("Candidate remapping lost a retained target")
            if len(compact_pool):
                pool_writer = _write_table(pool_writer, compact_pool, temporary["pool.parquet"])
            if len(mapped):
                candidate_writer = _write_table(candidate_writer, mapped, temporary["candidate_pairs.parquet"])
            retained_truth.update(country_truth)
            report = {"country": country, "references": len(country_reference), "full_pool_records": len(country_pool),
                      "retained_pool_records": len(compact_pool), "candidate_pairs": len(mapped),
                      "retained_true_target_records": len(true_ids), "elapsed_seconds": time.time() - country_started}
            country_reports.append(report)
            pool_offset += len(compact_pool)
            candidate_count += len(mapped)
            true_record_count += len(true_ids)
            _log(f"Completed {country}: {len(mapped):,} full-pool candidate edges; retained {len(compact_pool):,} target records")
            index.clear()
            del index, country_pool, compact_pool, pairs, mapped
    finally:
        if pool_writer is not None:
            pool_writer.close()
        if candidate_writer is not None:
            candidate_writer.close()
    if pool_writer is None:
        pd.DataFrame({column: pd.Series(dtype=str) for column in _RECORD_COLUMNS}).to_parquet(temporary["pool.parquet"], index=False)
    if candidate_writer is None:
        pd.DataFrame({column: pd.Series(dtype=dtype) for column, dtype in _PAIR_DTYPES.items()
                      if phonetic_top_k or not column.startswith("phonetic") and column != "lexical_scores_missing"}).to_parquet(temporary["candidate_pairs.parquet"], index=False)
    if set(retained_truth) != set(reference.entity_id):
        raise ValueError("Output truth does not cover all retained references")
    metadata = {
        "dataset_kind": "full_pool_hard_candidates", "references": len(reference), "pool_records": pool_offset,
        "selected_splits": list(splits),
        "true_links": sum(map(len, retained_truth.values())), "true_target_records": true_record_count,
        "candidate_pairs": candidate_count, "candidate_pairs_file": "candidate_pairs.parquet",
        "full_pool_records_by_country": full_counts,
        "reference_id_sha256": hashlib.sha256("\n".join(sorted(reference.entity_id)).encode()).hexdigest(),
        "split_counts": {f"{split}/{country}": int(count) for (split, country), count in reference.groupby(["split", "country"]).size().items()},
        "retrieval_config": {"query_terms": query_terms, "approximate_top_k": approximate_top_k, "top_k_per_channel": top_k,
                             "phonetic_top_k": phonetic_top_k, "exact_name_cap": exact_name_cap,
                             "min_similarity": min_similarity, "max_features": max_features,
                             "ngram_range": [3, 4], "phonetic_ngram_range": [2, 3]},
        "raw_sources": [{"path": str(path.resolve()), "bytes": path.stat().st_size} for path in source_paths],
        "parent_data": str(data_dir.resolve()), "countries": country_reports,
        "label_usage": "Only train/tune labels retain all true target records after unlabeled retrieval; no positive candidate injection.",
        "limitations": [
            "Use candidate_pairs.parquet for matching experiments; regenerating candidates on the compact pool changes the difficulty and TF-IDF scores.",
            "All true targets are retained as records, including retrieval misses. Candidate edges contain only actual full-pool retrieval results.",
            "Existing train/tune partitions are preserved. Evaluation references and their labels are excluded.",
            "The fixed tuning references remain a development set; an independent full-pool evaluation is still required.",
            "Countries are processed sequentially, but one complete country's text and sparse indexes must fit in RAM.",
        ],
        "elapsed_seconds": time.time() - started,
    }
    temporary["truth.json"].write_text(json.dumps(retained_truth), encoding="utf-8")
    temporary["metadata.json"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Metadata is committed last, so a failed mining run cannot masquerade as complete.
    for name in ("reference.parquet", "pool.parquet", "candidate_pairs.parquet", "truth.json", "metadata.json"):
        temporary[name].replace(output_dir / name)
    _log(f"Prepared {output_dir}: {len(reference):,} references, {pool_offset:,} records, {candidate_count:,} candidate edges")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/quick")
    parser.add_argument("--dataset", default="data/raw/dataset")
    parser.add_argument("--output", default="data/hard")
    parser.add_argument("--splits", default="train,tune", help="Comma-separated train/tune selection; eval is prohibited")
    parser.add_argument("--query-terms", type=int, default=32)
    parser.add_argument("--approximate-top-k", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--phonetic-top-k", type=int, default=20)
    parser.add_argument("--exact-name-cap", type=int, default=100)
    parser.add_argument("--min-similarity", type=float, default=.08)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    build_hard_pool(args.data, args.dataset, args.output, splits=args.splits, query_terms=args.query_terms,
        approximate_top_k=args.approximate_top_k, top_k=args.top_k, phonetic_top_k=args.phonetic_top_k,
        exact_name_cap=args.exact_name_cap, min_similarity=args.min_similarity, max_features=args.max_features,
        chunksize=args.chunksize, batch_size=args.batch_size, n_jobs=args.threads)


if __name__ == "__main__":
    main()
