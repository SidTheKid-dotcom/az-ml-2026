"""Tuning-only test of a supplementary phonetic business-name retrieval channel.

This experimental branch leaves the production normalization and lexical caches
unchanged. It fits no matching model and never scores evaluation references.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .candidates import PoolCandidateIndex, _country_values, _pair_cosines, generate_candidates
from .normalization import normalize_text
from .retrieval_ablation import _log, _script, measure_variant, truth_links

_VOWELS = re.compile("[aeiouy]")
_REPEATS = re.compile(r"(.)\1+")


def _skeleton(token: str, harmonize: bool = False) -> str:
    if harmonize:
        # Generic sound correspondences, not a table of entities or training IDs.
        token = token.replace("ph", "f").replace("sh", "s").replace("ch", "c")
        token = re.sub(r"([bkdgt])h", r"\1", token)
        token = re.sub(r"c(?=[eiy])", "s", token).replace("c", "k").replace("q", "k")
        token = token.replace("v", "b")
    return _REPEATS.sub(r"\1", _VOWELS.sub("", token))


_LEGAL = {
    _skeleton(token, True)
    for token in "private pvt limited ltd incorporated inc corporation corp company co llc llp plc".split()
}


def phonetic_text(text: str, variant: str = "basic") -> str:
    """Remove vowels and repeated consonants after conservative transliteration."""
    harmonize = variant == "harmonized_core"
    tokens = [_skeleton(token, harmonize) for token in normalize_text(text).split()]
    if harmonize:
        tokens = [token for token in tokens if token not in _LEGAL]
    return " ".join(token for token in tokens if token)


def augment_phonetic_candidates(
    reference: pd.DataFrame, pool: pd.DataFrame, base: pd.DataFrame,
    top_k: int = 20, native_only: bool = True, n_jobs: int = 4, *,
    variant: str = "harmonized_core", batch_size: int = 256,
    max_features: int = 100_000, min_similarity: float = .08,
    lexical_index: PoolCandidateIndex | None = None,
    lexical_ngram_range: tuple[int, int] = (3, 4),
    compute_lexical_scores: bool = True,
) -> pd.DataFrame:
    """Add phonetic candidates without changing the base pairs' lexical scores.

    All indexes remain positional in the original input frames. The phonetic branch
    searches the complete same-country pool, then optionally retains only targets
    whose raw names contain Indic script. Countries with no such targets are skipped
    when ``native_only=True`` because their retained result is necessarily empty.

    New pairs receive actual lexical TF-IDF cosine scores by default, from a reused
    ``lexical_index`` or a matching pool-only index. If that is disabled, new lexical
    scores are zero and ``lexical_scores_missing`` explicitly marks them. Base lexical
    scores are preserved. Union ranks are recomputed. Additional columns identify
    phonetic hits, phonetic-only pairs, and their retrieval score/rank. No labels are
    accepted or inspected, so this function can process train and inference frames.
    """
    if top_k < 0:
        raise ValueError("top_k must be nonnegative")
    if variant not in {"basic", "harmonized_core"}:
        raise ValueError("Unsupported phonetic variant")
    if lexical_index is not None and (lexical_index.pool is not pool or set(lexical_index.channels) != {"name", "address"}):
        raise ValueError("lexical_index must use this pool object and both lexical channels")
    required = {"left_idx", "right_idx", "name_score", "address_score", "retrieval_score", "retrieval_rank"}
    if required - set(base.columns):
        raise ValueError(f"base is missing columns: {sorted(required - set(base.columns))}")
    result = base.drop_duplicates(["left_idx", "right_idx"]).copy()
    # Repeated application is defined against the provided candidate set; provenance
    # in this result refers to this invocation's new branch.
    for column in ("phonetic_score", "phonetic_rank", "phonetic_hit", "phonetic_only", "lexical_scores_missing"):
        if column in result:
            result = result.drop(columns=column)
    native_targets = pool.business_name.map(_script).ne("Latin_or_other").to_numpy() if native_only else np.ones(len(pool), dtype=bool)
    pool_countries, reference_countries = _country_values(pool), _country_values(reference)
    searchable = set(pool_countries[native_targets])
    reference_positions = np.flatnonzero(np.isin(reference_countries, list(searchable)))
    if not top_k or not len(reference_positions):
        result["phonetic_score"] = np.float32(0)
        result["phonetic_rank"] = np.int32(0)
        result["phonetic_hit"] = False
        result["phonetic_only"] = False
        result["lexical_scores_missing"] = False
        return result

    transformed_reference = reference.iloc[reference_positions].copy(deep=False)
    transformed_pool = pool.copy(deep=False)
    transformed_reference["business_name"] = transformed_reference.business_name.map(lambda text: phonetic_text(text, variant))
    transformed_pool["business_name"] = pool.business_name.map(lambda text: phonetic_text(text, variant))
    _log(f"Retrieving supplemental phonetic top{top_k} for {len(reference_positions):,} references")
    phonetic = generate_candidates(
        transformed_reference, transformed_pool, top_k=top_k, channels=("name",),
        ngram_range=(2, 3), max_features=max_features, exact_name_cap=0,
        min_similarity=min_similarity, n_jobs=n_jobs, batch_size=batch_size,
    )
    phonetic["left_idx"] = reference_positions[phonetic.left_idx.to_numpy(dtype=np.int64)]
    phonetic = phonetic.loc[native_targets[phonetic.right_idx.to_numpy(dtype=np.int64)]]
    phonetic = phonetic.rename(columns={"name_score": "phonetic_score", "retrieval_rank": "phonetic_rank"})
    phonetic = phonetic[["left_idx", "right_idx", "phonetic_score", "phonetic_rank"]]
    del transformed_reference, transformed_pool
    result = result.merge(phonetic, on=["left_idx", "right_idx"], how="outer", validate="one_to_one", indicator=True)
    result["phonetic_hit"] = result.phonetic_rank.notna()
    result["phonetic_only"] = result["_merge"].eq("right_only")
    result["lexical_scores_missing"] = result.phonetic_only & (not compute_lexical_scores)
    result = result.drop(columns="_merge")
    result["phonetic_score"] = result.phonetic_score.fillna(0).astype(np.float32)
    result["phonetic_rank"] = result.phonetic_rank.fillna(0).astype(np.int32)
    added = result.phonetic_only.to_numpy()
    for column in ("name_score", "address_score", "retrieval_score"):
        result[column] = result[column].fillna(0).astype(np.float32)
    if compute_lexical_scores and added.any():
        index = lexical_index or PoolCandidateIndex(
            pool, max_features=max_features, ngram_range=lexical_ngram_range,
        )
        pair_countries = reference_countries[result.left_idx.to_numpy(dtype=np.int64)]
        _log(f"Computing true lexical cosine scores for {int(added.sum()):,} new pairs")
        for country in sorted(set(pair_countries[added])):
            country_index = index.country(country)
            rows = np.flatnonzero(added & (pair_countries == country))
            left = result.left_idx.to_numpy(dtype=np.int64)[rows]
            right = result.right_idx.to_numpy(dtype=np.int64)[rows]
            query_positions = np.unique(left)
            local_left = np.searchsorted(query_positions, left)
            local_right = np.searchsorted(country_index.positions, right)
            if not np.array_equal(country_index.positions[local_right], right):
                raise ValueError("Supplemental candidates escaped their country block")
            for channel, column in (("name", "business_name"), ("address", "business_address")):
                field = country_index.fields[channel]
                if field.vectorizer is not None:
                    query = field.vectorizer.transform(reference[column].iloc[query_positions].map(normalize_text)).tocsr()
                    result.loc[rows, f"{channel}_score"] = _pair_cosines(query, field.matrix, local_left, local_right)
            del country_index, field
    result.loc[added, "retrieval_score"] = result.loc[added, ["name_score", "address_score"]].max(axis=1)
    result.sort_values(["left_idx", "retrieval_score", "right_idx"], ascending=[True, False, True], kind="stable", inplace=True)
    result["retrieval_rank"] = (result.groupby("left_idx", sort=False).cumcount() + 1).astype(np.int32)
    result = result.reset_index(drop=True)
    result.attrs["phonetic_config"] = {
        "variant": variant, "top_k": top_k, "native_only": native_only,
        "ngram_range": (2, 3), "compute_lexical_scores": compute_lexical_scores,
    }
    return result


def run(args) -> dict:
    started = time.time()
    data, output = Path(args.data), Path(args.output)
    full_reference = pd.read_parquet(data / "reference.parquet").fillna("")
    original_positions = np.flatnonzero(full_reference.split.eq("tune").to_numpy())
    if not len(original_positions):
        raise ValueError("A nonempty fixed tuning split is required")
    reference = full_reference.iloc[original_positions].reset_index(drop=True)
    pool = pd.read_parquet(data / "pool.parquet").fillna("")
    if len(pool) > args.max_pool_records:
        raise ValueError("Pool exceeds --max-pool-records; use a machine sized for the pool")
    raw_truth = json.loads((data / "truth.json").read_text())
    truth = {entity: raw_truth[entity] for entity in reference.entity_id}
    del raw_truth
    baseline = pd.read_parquet(args.baseline_pairs)
    remap = np.full(len(full_reference), -1, dtype=np.int64)
    remap[original_positions] = np.arange(len(reference))
    baseline = baseline.loc[remap[baseline.left_idx.to_numpy(dtype=np.int64)] >= 0].copy()
    baseline["left_idx"] = remap[baseline.left_idx.to_numpy(dtype=np.int64)]
    baseline = baseline.drop_duplicates(["left_idx", "right_idx"])
    links = truth_links(reference, pool, truth)
    baseline_report, base_found = measure_variant(reference, len(pool), baseline, links)
    target_native = pool.business_name.map(_script).ne("Latin_or_other").to_numpy()
    truth_native = links.name_script.ne("Latin_or_other").to_numpy()
    ks = sorted(set(int(value) for value in args.top_k.split(",")))
    variants = args.variants.split(",")
    if any(value not in {"basic", "harmonized_core"} for value in variants):
        raise ValueError("Supported variants: basic,harmonized_core")
    output.mkdir(parents=True, exist_ok=True)
    records = []
    _log(f"Baseline recall={baseline_report['link_recall']:.6f}, pairs/reference={baseline_report['pairs_per_reference']:.2f}")
    for variant in variants:
        variant_started = time.time()
        transformed_reference = reference.copy(deep=False)
        transformed_pool = pool.copy(deep=False)
        transformed_reference["business_name"] = reference.business_name.map(lambda value: phonetic_text(value, variant))
        transformed_pool["business_name"] = pool.business_name.map(lambda value: phonetic_text(value, variant))
        _log(f"Retrieving {variant} phonetic names; K={max(ks)}")
        additional = generate_candidates(
            transformed_reference, transformed_pool, top_k=max(ks), channels=("name",),
            ngram_range=(2, 3), max_features=args.max_features,
            exact_name_cap=0, min_similarity=args.min_similarity,
            n_jobs=args.threads, batch_size=args.batch_size,
        )
        additional = additional.rename(columns={"name_score": "phonetic_score", "retrieval_rank": "phonetic_rank"})
        additional = additional[["left_idx", "right_idx", "phonetic_score", "phonetic_rank"]]
        additional.to_parquet(output / f"{variant}_candidates.parquet", index=False)
        elapsed = time.time() - variant_started
        del transformed_reference, transformed_pool
        for k in ks:
            prefix = additional.loc[additional.phonetic_rank <= k]
            for native_only in (False, True):
                extra = prefix.loc[target_native[prefix.right_idx.to_numpy(dtype=np.int64)]] if native_only else prefix
                # Metrics need pair identities only. Existing lexical scores are
                # neither overwritten nor misrepresented as phonetic similarities.
                combined = pd.concat([baseline[["left_idx", "right_idx"]], extra[["left_idx", "right_idx"]]], ignore_index=True).drop_duplicates()
                report, found = measure_variant(reference, len(pool), combined, links)
                report.update({
                    "variant": variant, "top_k": k, "native_targets_only": native_only,
                    "added_pairs": len(combined) - len(baseline),
                    "added_pairs_per_reference": (len(combined) - len(baseline)) / len(reference),
                    "added_true_links": int((found & ~base_found).sum()),
                    "native_true_links": int(truth_native.sum()),
                    "native_retrieved_links": int(found[truth_native].sum()),
                    "native_added_true_links": int((found & ~base_found & truth_native).sum()),
                    "retrieval_elapsed_seconds": elapsed,
                })
                records.append(report)
                recovered = links.loc[found & ~base_found].copy()
                recovered["source1_entity_id"] = reference.entity_id.to_numpy()[recovered.left_idx.to_numpy(dtype=np.int64)]
                recovered.to_csv(output / f"{variant}_k{k}_{'native' if native_only else 'all'}_recovered.tsv", sep="\t", index=False)
                _log(f"{variant:15s} K={k:2d} native_only={native_only!s:5s}: recall={report['link_recall']:.6f} oracle={report['oracle_macro_f05']:.6f}; +{report['added_true_links']} true links, +{report['added_pairs_per_reference']:.2f} pairs/ref, native={report['native_retrieved_links']}/{report['native_true_links']}")
    result = {
        "baseline": baseline_report, "baseline_pairs": str(Path(args.baseline_pairs).resolve()),
        "baseline_native_retrieved_links": int(base_found[truth_native].sum()),
        "reference_count": len(reference), "pool_count": len(pool), "results": records,
        "elapsed_seconds": time.time() - started,
        "limitations": [
            "Tuning references and their labels only; no evaluation measurements or model fitting.",
            "Results use a sampled pool and do not establish recall or precision against the full release.",
            "Phonetic retrieval is evaluated as an additional candidate branch; a matching model must reject its false positives.",
            "Lower K results are prefixes of the max-K ranked envelope; cutoff ties can differ from independent searches.",
            "native_targets_only filters the retrieved envelope after searching the complete pool; it is not top K within a native-only index.",
            "Production lexical normalization and existing candidate scores remain unchanged.",
        ],
    }
    (output / "phonetic_ablation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame([{key: value for key, value in record.items() if key not in {"country", "target_script_link_recall"}} for record in records]).to_csv(output / "comparison.tsv", sep="\t", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/quick")
    parser.add_argument("--baseline-pairs", default="artifacts/cache/7fa3a032c1070bc6/pairs.parquet")
    parser.add_argument("--output", default="artifacts/phonetic_ablation")
    parser.add_argument("--top-k", default="20,50")
    parser.add_argument("--variants", default="basic,harmonized_core")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-features", type=int, default=100_000)
    parser.add_argument("--min-similarity", type=float, default=.08)
    parser.add_argument("--max-pool-records", type=int, default=1_000_000)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
