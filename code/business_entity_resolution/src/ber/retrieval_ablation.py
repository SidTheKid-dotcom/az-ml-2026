"""Measure tuning-only candidate recall by channel, country, and target script.

Fresh mode builds one reusable country index, retrieves the largest requested K
once per channel, and preserves retrieval provenance. Smaller K values are prefixes
of that ranked envelope; cosine ties use pool position within the envelope. Exact
name expansion is reported both enabled and disabled. No matching model is fitted.

Cached mode is explicitly an approximate diagnostic: existing candidate caches have
scores but lack channel provenance, so ranks must be reconstructed within the cache.
Never interpret larger K values than the source cache contains as a real ablation.

Example:
    python -m ber.retrieval_ablation --data data/quick \
        --output artifacts/retrieval_ablation --top-k 10,20,50 --threads 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .candidates import (
    PoolCandidateIndex, _country_values, _pair_cosines, _sparse_retrieve,
)
from .normalization import normalize_text

_SCRIPT_RANGES = (
    (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Odia"), (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"),
)
_INDIC = re.compile("[\\u0900-\\u0d7f]")


def _log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def _script(text: str) -> str:
    """Return an observed Indic script, or Latin/other; no language inference."""
    match = _INDIC.search(str(text))
    if match:
        codepoint = ord(match.group())
        for lo, hi, name in _SCRIPT_RANGES:
            if lo <= codepoint <= hi:
                return name
    return "Latin_or_other"


def _rank(frame: pd.DataFrame, channel: str) -> pd.DataFrame:
    frame = frame.sort_values(
        ["left_idx", f"{channel}_score", "right_idx"],
        ascending=[True, False, True], kind="stable",
    ).copy()
    frame[f"{channel}_rank"] = (frame.groupby("left_idx", sort=False).cumcount() + 1).astype(np.int32)
    return frame


def build_envelope(
    reference: pd.DataFrame, pool: pd.DataFrame, *, max_k: int,
    exact_name_cap: int = 100, batch_size: int = 256, min_similarity: float = .08,
    threads: int = 4, max_features: int = 100_000,
    ngram_range: tuple[int, int] = (3, 4),
) -> pd.DataFrame:
    """Return real per-channel ranks from one max-K search over the full pool."""
    index = PoolCandidateIndex(pool, max_features=max_features, ngram_range=ngram_range)
    keys = _country_values(reference)
    country_results = []
    for country in sorted(set(keys) & set(index.country_keys)):
        _log(f"Indexing/searching {country}: max K={max_k}")
        country_index = index.country(country)
        global_left = np.flatnonzero(keys == country)
        queries = reference.iloc[global_left]
        query_matrices, channel_frames = {}, []
        for channel, column in (("name", "business_name"), ("address", "business_address")):
            field = country_index.fields[channel]
            if field.vectorizer is None:
                from scipy import sparse
                query_matrices[channel] = sparse.csr_matrix((len(queries), 0), dtype=np.float32)
                continue
            query_matrix = field.vectorizer.transform(queries[column].map(normalize_text)).tocsr()
            query_matrices[channel] = query_matrix
            blocks = []
            for start in range(0, len(queries), batch_size):
                found = _sparse_retrieve(
                    query_matrix[start:start + batch_size], field.inverted,
                    max_k, min_similarity, threads, 20_000,
                ).tocoo()
                if found.nnz:
                    blocks.append(pd.DataFrame({
                        "left_idx": found.row.astype(np.int64) + start,
                        "right_idx": found.col.astype(np.int64),
                        f"{channel}_score": found.data,
                    }))
            if blocks:
                channel_frames.append(_rank(pd.concat(blocks, ignore_index=True), channel))
        exact_left, exact_right = [], []
        if exact_name_cap:
            names = queries.business_name.map(normalize_text).to_numpy()
            lows = np.searchsorted(country_index.sorted_names, names, side="left")
            highs = np.searchsorted(country_index.sorted_names, names, side="right")
            for row, (name, lo, hi) in enumerate(zip(names, lows, highs)):
                if name:
                    matches = country_index.name_order[lo:min(hi, lo + exact_name_cap)]
                    exact_left.extend([row] * len(matches))
                    exact_right.extend(matches.tolist())
        if exact_left:
            channel_frames.append(pd.DataFrame({
                "left_idx": np.asarray(exact_left, dtype=np.int64),
                "right_idx": np.asarray(exact_right, dtype=np.int64), "exact_name": True,
            }))
        if channel_frames:
            joined = channel_frames[0]
            for frame in channel_frames[1:]:
                joined = joined.merge(frame, on=["left_idx", "right_idx"], how="outer", validate="one_to_one")
            for channel in ("name", "address"):
                rank_column = f"{channel}_rank"
                joined[rank_column] = joined.get(rank_column, pd.Series(0, index=joined.index)).fillna(0).astype(np.int32)
                # Every score is the true pair cosine, independently of which
                # branch retrieved it. Membership comes from the rank columns.
                joined[f"{channel}_score"] = _pair_cosines(
                    query_matrices[channel], country_index.fields[channel].matrix,
                    joined.left_idx.to_numpy(dtype=np.int64), joined.right_idx.to_numpy(dtype=np.int64),
                )
            joined["exact_name"] = joined.get("exact_name", pd.Series(False, index=joined.index)).fillna(False).astype(bool)
            joined["left_idx"] = global_left[joined.left_idx.to_numpy(dtype=np.int64)]
            joined["right_idx"] = country_index.positions[joined.right_idx.to_numpy(dtype=np.int64)]
            country_results.append(joined)
        del country_index, field
    if not country_results:
        return pd.DataFrame(columns=["left_idx", "right_idx", "name_score", "address_score", "name_rank", "address_rank", "exact_name"])
    return pd.concat(country_results, ignore_index=True).sort_values(["left_idx", "right_idx"], ignore_index=True)


def cached_envelope(
    cached: pd.DataFrame, original_positions: np.ndarray, full_reference_size: int,
    reference: pd.DataFrame, pool: pd.DataFrame, min_similarity: float,
) -> pd.DataFrame:
    remap = np.full(full_reference_size, -1, dtype=np.int64)
    remap[original_positions] = np.arange(len(original_positions))
    cached = cached.loc[remap[cached.left_idx.to_numpy(dtype=np.int64)] >= 0].copy()
    cached["left_idx"] = remap[cached.left_idx.to_numpy(dtype=np.int64)]
    cached = cached.drop_duplicates(["left_idx", "right_idx"]).reset_index(drop=True)
    for channel in ("name", "address"):
        ranked = _rank(cached.loc[cached[f"{channel}_score"] >= min_similarity], channel)
        cached[f"{channel}_rank"] = ranked[f"{channel}_rank"].reindex(cached.index).fillna(0).astype(np.int32)
    names_left = reference.business_name.map(normalize_text).to_numpy()
    names_right = pool.business_name.map(normalize_text).to_numpy()
    left_names = names_left[cached.left_idx.to_numpy(dtype=np.int64)]
    cached["exact_name"] = (left_names != "") & (left_names == names_right[cached.right_idx.to_numpy(dtype=np.int64)])
    return cached


def truth_links(reference: pd.DataFrame, pool: pd.DataFrame, truth: dict) -> pd.DataFrame:
    left, ids = [], []
    for pos, entity in enumerate(reference.entity_id):
        values = sorted(set(truth[entity]))
        left.extend([pos] * len(values))
        ids.extend(values)
    right = pd.Index(pool.entity_id).get_indexer(ids)
    links = pd.DataFrame({"left_idx": np.asarray(left, dtype=np.int64), "right_idx": right, "target_entity_id": ids})
    links["country"] = reference.country.to_numpy()[links.left_idx.to_numpy(dtype=np.int64)]
    for field, column in (("name", "business_name"), ("address", "business_address")):
        values = np.full(len(links), "Missing_from_pool", dtype=object)
        valid = right >= 0
        values[valid] = pool[column].iloc[right[valid]].map(_script).to_numpy()
        links[f"{field}_script"] = values
    return links


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def measure_variant(reference: pd.DataFrame, pool_size: int, pairs: pd.DataFrame, links: pd.DataFrame) -> tuple[dict, np.ndarray]:
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    true_left = links.left_idx.to_numpy(dtype=np.int64)
    true_right = links.right_idx.to_numpy(dtype=np.int64)
    found = (true_right >= 0) & np.isin(true_left * pool_size + true_right, left * pool_size + right)
    true_counts = np.bincount(true_left, minlength=len(reference))
    found_counts = np.bincount(true_left[found], minlength=len(reference))
    pair_counts = np.bincount(left, minlength=len(reference))
    denominator = 4 * found_counts + true_counts
    oracle = np.divide(5.0 * found_counts, denominator, out=np.ones(len(reference)), where=denominator > 0)

    def aggregate(mask):
        nonempty = mask & (true_counts > 0)
        return {
            "references": int(mask.sum()), "true_links": int(true_counts[mask].sum()),
            "retrieved_true_links": int(found_counts[mask].sum()), "pairs": int(pair_counts[mask].sum()),
            "pairs_per_reference": _fraction(pair_counts[mask].sum(), mask.sum()),
            "link_recall": _fraction(found_counts[mask].sum(), true_counts[mask].sum()),
            "complete_match_set_recall": float((found_counts[nonempty] == true_counts[nonempty]).mean()) if nonempty.any() else None,
            "oracle_macro_f05": float(oracle[mask].mean()) if mask.any() else None,
        }

    result = aggregate(np.ones(len(reference), dtype=bool))
    result["country"] = {
        str(country): aggregate(reference.country.eq(country).to_numpy())
        for country in sorted(reference.country.unique())
    }
    script_results = []
    for field in ("name", "address"):
        for (country, script), rows in links.groupby(["country", f"{field}_script"], sort=True).groups.items():
            positions = np.asarray(rows, dtype=np.int64)
            script_results.append({
                "country": str(country), "field": field, "script": script,
                "true_links": len(positions), "retrieved_true_links": int(found[positions].sum()),
                "link_recall": float(found[positions].mean()),
            })
    result["target_script_link_recall"] = script_results
    return result, found


def run(args) -> dict:
    started = time.time()
    data, output = Path(args.data), Path(args.output)
    pool_path = data / "pool.parquet"
    pool_rows = pq.ParquetFile(pool_path).metadata.num_rows
    if pool_rows > args.max_pool_records:
        raise ValueError(f"Pool has {pool_rows:,} rows, above --max-pool-records={args.max_pool_records:,}. Raise this explicitly on a machine with enough memory.")
    ks = sorted(set(int(k) for k in args.top_k.split(",")))
    if not ks or min(ks) <= 0:
        raise ValueError("--top-k must contain positive integers")
    if args.cached_pairs and max(ks) > args.cached_top_k:
        raise ValueError("Cached mode cannot measure K larger than --cached-top-k; request only supported K values.")
    full_reference = pd.read_parquet(data / "reference.parquet").fillna("")
    if "split" not in full_reference:
        raise ValueError("reference.parquet must contain a fixed train/tune/eval split column")
    original_positions = np.flatnonzero(full_reference.split.eq("tune").to_numpy())
    if args.max_references and len(original_positions) > args.max_references:
        original_positions = np.sort(np.random.default_rng(args.seed).choice(original_positions, args.max_references, replace=False))
    if not len(original_positions):
        raise ValueError("The tuning split is empty")
    reference = full_reference.iloc[original_positions].reset_index(drop=True)
    pool = pd.read_parquet(pool_path).fillna("")
    raw_truth = json.loads((data / "truth.json").read_text())
    truth = {entity: raw_truth[entity] for entity in reference.entity_id}
    del raw_truth
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "top_k": ks, "exact_name_cap": args.exact_name_cap, "min_similarity": args.min_similarity,
        "max_features": args.max_features, "ngram_range": [args.ngram_min, args.ngram_max],
        "tuning_reference_sha256": hashlib.sha256("\n".join(reference.entity_id).encode()).hexdigest(),
        "input_files": {name: {"size": (data / name).stat().st_size, "mtime_ns": (data / name).stat().st_mtime_ns}
                        for name in ("reference.parquet", "pool.parquet", "truth.json")},
        "code_sha256": hashlib.sha256(b"".join((Path(__file__).parent / name).read_bytes()
                                               for name in ("retrieval_ablation.py", "candidates.py", "normalization.py"))).hexdigest(),
    }
    mode = "approximate_cached_diagnostic" if args.cached_pairs else "fresh_channel_provenance"
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    envelope_path = output / f"envelope_{fingerprint}.parquet"
    if args.cached_pairs:
        envelope = cached_envelope(pd.read_parquet(args.cached_pairs), original_positions, len(full_reference), reference, pool, args.min_similarity)
    elif envelope_path.exists():
        _log(f"Reusing {envelope_path}")
        envelope = pd.read_parquet(envelope_path)
    else:
        envelope = build_envelope(
            reference, pool, max_k=max(ks), exact_name_cap=args.exact_name_cap,
            batch_size=args.batch_size, min_similarity=args.min_similarity, threads=args.threads,
            max_features=args.max_features, ngram_range=(args.ngram_min, args.ngram_max),
        )
        envelope.to_parquet(envelope_path, index=False)
    links = truth_links(reference, pool, truth)
    records, best_found = [], None
    exact_options = (False, True) if args.exact_name_cap else (False,)
    for k in ks:
        for channel in ("name", "address", "union"):
            base = np.zeros(len(envelope), dtype=bool)
            for field in (("name", "address") if channel == "union" else (channel,)):
                rank = envelope[f"{field}_rank"].to_numpy()
                base |= (rank > 0) & (rank <= k)
            for use_exact in exact_options:
                mask = base | envelope.exact_name.to_numpy(dtype=bool) if use_exact else base
                report, found = measure_variant(reference, len(pool), envelope.loc[mask], links)
                report.update({"channel": channel, "top_k": k, "exact_name": use_exact})
                records.append(report)
                if channel == "union" and k == max(ks) and use_exact == exact_options[-1]:
                    best_found = found
                _log(f"{channel:7s} K={k:2d} exact={use_exact!s:5s}: recall={report['link_recall']:.6f} oracle={report['oracle_macro_f05']:.6f} pairs/ref={report['pairs_per_reference']:.2f}")
    limitations = [
        "Only the tuning split is inspected; evaluation labels are not used.",
        "Recall measures this provided candidate pool; sampled pools are easier than the complete release.",
        "No French reference labels are available unless present in this prepared tuning data.",
        "Smaller K values are prefixes of max-K retrieval; equal-cosine ties are ordered by pool position inside the envelope.",
        "Script slices inspect target text Unicode blocks, not inferred language or business identity.",
    ]
    if args.cached_pairs:
        limitations.append("APPROXIMATE: old cache lacks channel provenance. Ranks are reconstructed inside its union, and exact-name flags identify equality among cached candidates. Missing candidates cannot be recovered; exact-name caps and cosine ties may differ.")
    result = {
        "mode": mode, "config": config, "source_cached_top_k": args.cached_top_k if args.cached_pairs else None,
        "reference_count": len(reference), "pool_count": len(pool), "envelope_pairs": len(envelope),
        "true_links_missing_from_pool": int((links.right_idx < 0).sum()),
        "results": records, "limitations": limitations, "elapsed_seconds": time.time() - started,
    }
    (output / "retrieval_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([{k: v for k, v in record.items() if k not in {"country", "target_script_link_recall"}} for record in records]).to_csv(output / "comparison.tsv", sep="\t", index=False)
    if best_found is not None:
        misses = links.loc[~best_found].copy()
        misses["source1_entity_id"] = reference.entity_id.to_numpy()[misses.left_idx.to_numpy(dtype=np.int64)]
        misses.to_csv(output / "max_union_missed_links.tsv", sep="\t", index=False)
    _log(f"Saved tuning-only report to {output}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/quick")
    parser.add_argument("--output", default="artifacts/retrieval_ablation")
    parser.add_argument("--top-k", default="10,20,50")
    parser.add_argument("--cached-pairs", type=Path)
    parser.add_argument("--cached-top-k", type=int, default=20)
    parser.add_argument("--exact-name-cap", type=int, default=100)
    parser.add_argument("--min-similarity", type=float, default=.08)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--ngram-min", type=int, default=3)
    parser.add_argument("--ngram-max", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-pool-records", type=int, default=1_000_000)
    parser.add_argument("--max-references", type=int)
    parser.add_argument("--seed", type=int, default=20260925)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
