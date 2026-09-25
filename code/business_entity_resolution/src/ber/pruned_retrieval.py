"""Experimental rare-gram retrieval followed by exact full-cosine reranking.

Keep only the strongest TF-IDF terms of each query, retrieve a larger envelope,
recompute unpruned cosine on that envelope, and retain the ordinary per-field K.
Pool vectors, exact-name expansion, and country blocking are unchanged. The CLI
compares tuning references only and shares each country index across all methods.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .candidates import PoolCandidateIndex, _country_values, _pair_cosines, _sparse_retrieve
from .normalization import normalize_text
from .retrieval_ablation import _log, measure_variant, truth_links


def prune_query_terms(matrix: sparse.csr_matrix, keep: int) -> sparse.csr_matrix:
    """Retain strongest original TF-IDF weights without renormalizing the row."""
    if keep <= 0:
        raise ValueError("keep must be positive")
    result = matrix.copy()
    for row in range(result.shape[0]):
        start, stop = result.indptr[row:row + 2]
        if stop - start > keep:
            weights, indices = result.data[start:stop], result.indices[start:stop]
            order = np.lexsort((indices, -weights))
            weights[order[keep:]] = 0
    result.eliminate_zeros()
    return result


def _query_matrices(reference: pd.DataFrame, country_index) -> dict:
    result = {}
    for channel, column in (("name", "business_name"), ("address", "business_address")):
        field = country_index.fields[channel]
        result[channel] = (
            field.vectorizer.transform(reference[column].map(normalize_text)).tocsr()
            if field.vectorizer is not None
            else sparse.csr_matrix((len(reference), 0), dtype=np.float32)
        )
    return result


def _exact_pairs(reference: pd.DataFrame, country_index, cap: int) -> np.ndarray:
    pairs = []
    if cap:
        names = reference.business_name.map(normalize_text).to_numpy()
        lows = np.searchsorted(country_index.sorted_names, names, side="left")
        highs = np.searchsorted(country_index.sorted_names, names, side="right")
        for left, (name, lo, hi) in enumerate(zip(names, lows, highs)):
            if name:
                pairs.extend((left, int(right)) for right in country_index.name_order[lo:min(hi, lo + cap)])
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def _country_candidates(
    queries: dict, country_index, exact_pairs: np.ndarray, *, query_terms: int | None,
    approximate_top_k: int, top_k: int, min_similarity: float, n_jobs: int, batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    statistics = {"retrieve_seconds": 0., "rerank_seconds": 0., "full_pair_score_seconds": 0.,
                  "query_nonzeros": 0, "posting_list_visits": 0, "envelope_pairs_before_rerank": 0}
    selected_pairs = [exact_pairs] if len(exact_pairs) else []
    for channel in ("name", "address"):
        field, full_query = country_index.fields[channel], queries[channel]
        if not full_query.nnz:
            continue
        started = time.perf_counter()
        retrieval_query = prune_query_terms(full_query, query_terms) if query_terms else full_query
        statistics["query_nonzeros"] += int(retrieval_query.nnz)
        lengths = np.diff(field.inverted.indptr)
        statistics["posting_list_visits"] += int(lengths[retrieval_query.indices].sum(dtype=np.int64))
        blocks = []
        for start in range(0, full_query.shape[0], batch_size):
            found = _sparse_retrieve(
                retrieval_query[start:start + batch_size], field.inverted,
                approximate_top_k if query_terms else top_k,
                0. if query_terms else min_similarity, n_jobs, 20_000,
            ).tocoo()
            if found.nnz:
                blocks.append(pd.DataFrame({"left_idx": found.row.astype(np.int64) + start,
                                            "right_idx": found.col.astype(np.int64), "score": found.data}))
        statistics["retrieve_seconds"] += time.perf_counter() - started
        if not blocks:
            continue
        candidates = pd.concat(blocks, ignore_index=True)
        statistics["envelope_pairs_before_rerank"] += len(candidates)
        if query_terms:
            started = time.perf_counter()
            candidates["score"] = _pair_cosines(full_query, field.matrix,
                candidates.left_idx.to_numpy(dtype=np.int64), candidates.right_idx.to_numpy(dtype=np.int64))
            candidates = candidates.loc[candidates.score >= min_similarity]
            candidates = candidates.sort_values(["left_idx", "score", "right_idx"],
                                                  ascending=[True, False, True], kind="stable")
            candidates = candidates.groupby("left_idx", sort=False).head(top_k)
            statistics["rerank_seconds"] += time.perf_counter() - started
        selected_pairs.append(candidates[["left_idx", "right_idx"]].to_numpy(dtype=np.int64))
    if not selected_pairs:
        return pd.DataFrame(columns=["left_idx", "right_idx", "name_score", "address_score", "retrieval_score", "retrieval_rank"]), statistics
    pairs = np.unique(np.concatenate(selected_pairs), axis=0)
    result = pd.DataFrame(pairs, columns=["left_idx", "right_idx"])
    started = time.perf_counter()
    for channel in ("name", "address"):
        result[f"{channel}_score"] = _pair_cosines(queries[channel], country_index.fields[channel].matrix, pairs[:, 0], pairs[:, 1])
    result["retrieval_score"] = result[["name_score", "address_score"]].max(axis=1).astype(np.float32)
    result.sort_values(["left_idx", "retrieval_score", "right_idx"], ascending=[True, False, True], kind="stable", inplace=True)
    result["retrieval_rank"] = (result.groupby("left_idx", sort=False).cumcount() + 1).astype(np.int32)
    statistics["full_pair_score_seconds"] += time.perf_counter() - started
    return result.reset_index(drop=True), statistics


def generate_pruned_candidates(
    reference: pd.DataFrame, pool: pd.DataFrame, *, query_terms: int = 12,
    approximate_top_k: int = 100, top_k: int = 20, exact_name_cap: int = 100,
    min_similarity: float = .08, n_jobs: int = 4, batch_size: int = 256,
    pool_index: PoolCandidateIndex | None = None, **index_kwargs,
) -> pd.DataFrame:
    """Experimental drop-in generator with complete lexical cosine score columns."""
    if approximate_top_k < top_k:
        raise ValueError("approximate_top_k must be at least top_k")
    index = pool_index or PoolCandidateIndex(pool, **index_kwargs)
    if index.pool is not pool or set(index.channels) != {"name", "address"}:
        raise ValueError("pool_index must use this pool and both name/address channels")
    keys, frames = _country_values(reference), []
    for country in sorted(set(keys) & set(index.country_keys)):
        country_index = index.country(country)
        positions = np.flatnonzero(keys == country)
        country_reference = reference.iloc[positions]
        frame, _ = _country_candidates(
            _query_matrices(country_reference, country_index), country_index,
            _exact_pairs(country_reference, country_index, exact_name_cap),
            query_terms=query_terms, approximate_top_k=approximate_top_k, top_k=top_k,
            min_similarity=min_similarity, n_jobs=n_jobs, batch_size=batch_size,
        )
        if len(frame):
            frame["left_idx"] = positions[frame.left_idx.to_numpy(dtype=np.int64)]
            frame["right_idx"] = country_index.positions[frame.right_idx.to_numpy(dtype=np.int64)]
            frames.append(frame)
        del country_index
    return pd.concat(frames, ignore_index=True).sort_values(["left_idx", "retrieval_rank"], ignore_index=True) if frames else pd.DataFrame(columns=["left_idx", "right_idx", "name_score", "address_score", "retrieval_score", "retrieval_rank"])


def run(args) -> dict:
    started = time.time()
    data, output = Path(args.data), Path(args.output)
    all_reference = pd.read_parquet(data / "reference.parquet").fillna("")
    positions = np.flatnonzero(all_reference.split.eq("tune").to_numpy())
    if len(positions) > args.max_references:
        positions = np.sort(np.random.default_rng(args.seed).choice(positions, args.max_references, replace=False))
    reference = all_reference.iloc[positions].reset_index(drop=True)
    if not len(reference):
        raise ValueError("A nonempty tuning split is required")
    pool = pd.read_parquet(data / "pool.parquet").fillna("")
    raw_truth = json.loads((data / "truth.json").read_text())
    truth = {entity: raw_truth[entity] for entity in reference.entity_id}
    del raw_truth
    links = truth_links(reference, pool, truth)
    terms = [None] + sorted(set(int(value) for value in args.query_terms.split(",")))
    results = {term: [] for term in terms}
    timings = {term: {} for term in terms}
    index = PoolCandidateIndex(pool, max_features=args.max_features, ngram_range=(3, 4))
    keys = _country_values(reference)
    index_seconds = 0.
    for country in sorted(set(keys) & set(index.country_keys)):
        _log(f"Building shared {country} indexes")
        index_started = time.perf_counter()
        country_index = index.country(country)
        local_positions = np.flatnonzero(keys == country)
        country_reference = reference.iloc[local_positions]
        queries = _query_matrices(country_reference, country_index)
        exact = _exact_pairs(country_reference, country_index, args.exact_name_cap)
        index_seconds += time.perf_counter() - index_started
        for term in terms:
            label = "full" if term is None else str(term)
            _log(f"Searching {country}: query grams={label}, finalK={args.top_k}")
            frame, stats = _country_candidates(queries, country_index, exact,
                query_terms=term, approximate_top_k=args.approximate_top_k, top_k=args.top_k,
                min_similarity=args.min_similarity, n_jobs=args.threads, batch_size=args.batch_size)
            if len(frame):
                frame["left_idx"] = local_positions[frame.left_idx.to_numpy(dtype=np.int64)]
                frame["right_idx"] = country_index.positions[frame.right_idx.to_numpy(dtype=np.int64)]
                results[term].append(frame)
            for key, value in stats.items():
                timings[term][key] = timings[term].get(key, 0) + value
        del country_index
    output.mkdir(parents=True, exist_ok=True)
    records, full_found = [], None
    for term in terms:
        pairs = pd.concat(results[term], ignore_index=True) if results[term] else pd.DataFrame(columns=["left_idx", "right_idx"])
        report, found = measure_variant(reference, len(pool), pairs, links)
        if term is None:
            full_found = found
        timing = timings[term]
        search_seconds = sum(timing[key] for key in ("retrieve_seconds", "rerank_seconds", "full_pair_score_seconds"))
        report.update({"query_terms": term, "timing": timing, "total_search_seconds": search_seconds,
                       "lost_true_links_vs_full": int((full_found & ~found).sum()),
                       "gained_true_links_vs_full": int((found & ~full_found).sum()),
                       "posting_visit_fraction": timing["posting_list_visits"] / timings[None]["posting_list_visits"],
                       "approximate_top_k": args.approximate_top_k if term else args.top_k})
        records.append(report)
        label = "full" if term is None else str(term)
        pairs.to_parquet(output / f"query_terms_{label}_pairs.parquet", index=False)
        _log(f"grams={label:4s}: recall={report['link_recall']:.6f} oracle={report['oracle_macro_f05']:.6f} search={search_seconds:.2f}s posting_fraction={report['posting_visit_fraction']:.3f} lost={report['lost_true_links_vs_full']} gained={report['gained_true_links_vs_full']}")
    report = {"reference_count": len(reference), "pool_count": len(pool),
              "reference_id_sha256": __import__('hashlib').sha256("\n".join(reference.entity_id).encode()).hexdigest(),
              "shared_index_seconds": index_seconds, "results": records, "elapsed_seconds": time.time() - started,
              "limitations": ["Only sampled tuning references are measured; no evaluation scoring or model fitting.",
                "Search timing excludes shared index construction and includes full-cosine reranking/scoring.",
                "Posting-list visits are a scaling proxy, not a full-pool runtime guarantee.",
                "Queries retain original TF-IDF weights; approximate candidates use threshold0 before exact reranking.",
                "Recomputing cosine and changing candidate envelopes can change cutoff tie selection."]}
    (output / "pruned_retrieval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in record.items() if key not in {"country", "target_script_link_recall", "timing"}}
                  for record in records]).to_csv(output / "comparison.tsv", sep="\t", index=False)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/quick")
    parser.add_argument("--output", default="artifacts/pruned_retrieval")
    parser.add_argument("--max-references", type=int, default=1000)
    parser.add_argument("--query-terms", default="8,12,20")
    parser.add_argument("--approximate-top-k", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--exact-name-cap", type=int, default=100)
    parser.add_argument("--min-similarity", type=float, default=.08)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260925)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
