"""Country-blocked character TF-IDF retrieval with bounded exact-name expansion.

``top_k`` is a per-channel limit: name/address union can return 2 * top_k plus
exact-name candidates. Scores are actual cosine similarities for every enabled
channel, including candidates introduced by a different channel. No country list
is hard-coded, and all returned indexes are positional, independent of DF indexes.

For small experiments use ``generate_candidates``. For full inference consume
``iter_candidate_batches`` so the candidate output need not fit in RAM. Reuse a
``PoolCandidateIndex`` for repeated query batches against the same pool. It keeps
one country's two sparse indexes by default and never makes a dense all-pairs
matrix. Vocabulary/IDF fitting uses a reproducible sample for very large countries.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, Sequence
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalization import normalize_text

try:
    from sparse_dot_topn import sp_matmul_topn
except ImportError:  # bounded fallback permits portable small-data smoke tests
    sp_matmul_topn = None

_FIELDS = {"name": "business_name", "address": "business_address"}
_DTYPES = {
    "left_idx": "int64", "right_idx": "int64", "name_score": "float32",
    "address_score": "float32", "retrieval_score": "float32",
    "retrieval_rank": "int32",
}


def _empty_candidates() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in _DTYPES.items()})


def _validate_frame(frame: pd.DataFrame, label: str) -> None:
    missing = {"entity_id", "business_name", "business_address", "country"} - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")


def _validate_channels(channels: Sequence[str]) -> tuple[str, ...]:
    channels = tuple(dict.fromkeys(channels))
    if not channels or any(channel not in _FIELDS for channel in channels):
        raise ValueError("channels must be a nonempty subset of ('name', 'address')")
    return channels


def _country_values(frame: pd.DataFrame) -> np.ndarray:
    # Country labels repeat millions of times; normalize each distinct value
    # once so the object array reuses a handful of strings instead of millions.
    labels = frame["country"]
    normalized = {value: normalize_text(value) for value in labels.unique()}
    return labels.map(normalized).fillna("").to_numpy()


def _topk_csr(matrix: sparse.csr_matrix, top_k: int, threshold: float) -> sparse.csr_matrix:
    """Prune a bounded sparse multiplication, with deterministic tie ordering."""
    rows, cols, values = [], [], []
    for row in range(matrix.shape[0]):
        start, end = matrix.indptr[row : row + 2]
        data, indices = matrix.data[start:end], matrix.indices[start:end]
        eligible = np.flatnonzero((data >= threshold) & (data > 0))
        if len(eligible) > top_k:
            cutoff = np.partition(data[eligible], -top_k)[-top_k]
            better = eligible[data[eligible] > cutoff]
            tied = eligible[data[eligible] == cutoff]
            tied = tied[np.argsort(indices[tied], kind="stable")[:top_k - len(better)]]
            eligible = np.concatenate((better, tied))
        eligible = eligible[np.lexsort((indices[eligible], -data[eligible]))]
        rows.extend([row] * len(eligible))
        cols.extend(indices[eligible].tolist())
        values.extend(data[eligible].tolist())
    return sparse.csr_matrix(
        (np.asarray(values, dtype=np.float32), (rows, cols)), shape=matrix.shape,
    )


def _sparse_retrieve(
    query: sparse.csr_matrix, inverted: sparse.csr_matrix, top_k: int,
    min_similarity: float, n_jobs: int, fallback_pool_batch_size: int,
) -> sparse.csr_matrix:
    top_k = min(top_k, inverted.shape[1])
    if top_k <= 0 or query.nnz == 0:
        return sparse.csr_matrix((query.shape[0], inverted.shape[1]), dtype=np.float32)
    if sp_matmul_topn is not None:
        return sp_matmul_topn(
            query, inverted, top_n=top_k, threshold=min_similarity,
            sort=False, n_threads=n_jobs,
        )

    # Bound both dimensions even if very common character n-grams produce a
    # nearly full intermediate sparse product. This path is for small experiments.
    chunks = []
    for query_start in range(0, query.shape[0], 128):
        query_chunk = query[query_start:query_start + 128]
        best = sparse.csr_matrix((query_chunk.shape[0], inverted.shape[1]), dtype=np.float32)
        for pool_start in range(0, inverted.shape[1], fallback_pool_batch_size):
            scores = (query_chunk @ inverted[:, pool_start:pool_start + fallback_pool_batch_size]).tocsr()
            scores = _topk_csr(scores, top_k, min_similarity).tocoo()
            global_scores = sparse.csr_matrix(
                (scores.data, (scores.row, scores.col + pool_start)), shape=best.shape,
            )
            best = _topk_csr(best + global_scores, top_k, min_similarity)
        chunks.append(best)
    return sparse.vstack(chunks, format="csr")


@dataclass
class _FieldIndex:
    vectorizer: TfidfVectorizer | None
    matrix: sparse.csr_matrix
    inverted: sparse.csr_matrix


@dataclass
class _CountryIndex:
    positions: np.ndarray
    fields: dict[str, _FieldIndex]
    names: np.ndarray
    name_order: np.ndarray
    sorted_names: np.ndarray


class PoolCandidateIndex:
    """Reusable pool-only retrieval index; no labels or Source 1 text are fitted.

    ``fit_sample_size=None`` fits vocabulary/IDF on the full country. At scale the
    default fits on at most 250k deterministic random rows, then transforms every
    pool row in chunks. Only this fitting stage is sampled; retrieval sees the
    complete pool. Do not mutate ``pool`` while this index is in use.
    """

    def __init__(
        self, pool: pd.DataFrame, *, channels: tuple[str, ...] = ("name", "address"),
        ngram_range: tuple[int, int] = (3, 4), max_features: int | None = 100_000,
        min_df: int | float = 1, sublinear_tf: bool = True, analyzer: str = "char_wb",
        fit_sample_size: int | None = 250_000, transform_batch_size: int = 100_000,
        cache_countries: int = 1, random_state: int = 2026,
    ) -> None:
        _validate_frame(pool, "pool")
        self.pool = pool
        self.channels = _validate_channels(channels)
        if transform_batch_size <= 0 or cache_countries <= 0:
            raise ValueError("transform_batch_size and cache_countries must be positive")
        if fit_sample_size is not None and fit_sample_size <= 0:
            raise ValueError("fit_sample_size must be positive or None")
        self.ngram_range, self.max_features = ngram_range, max_features
        self.min_df, self.sublinear_tf, self.analyzer = min_df, sublinear_tf, analyzer
        self.fit_sample_size, self.transform_batch_size = fit_sample_size, transform_batch_size
        self.cache_countries, self.random_state = cache_countries, random_state
        self.country_keys = _country_values(pool)
        self._cache: OrderedDict[str, _CountryIndex] = OrderedDict()

    def clear(self) -> None:
        self._cache.clear()

    def _build_field(self, texts: np.ndarray) -> _FieldIndex:
        vectorizer = TfidfVectorizer(
            analyzer=self.analyzer, ngram_range=self.ngram_range, max_features=self.max_features,
            min_df=self.min_df, sublinear_tf=self.sublinear_tf, dtype=np.float32,
            lowercase=False, norm="l2",
        )
        fit_texts = texts
        if self.fit_sample_size is not None and len(texts) > self.fit_sample_size:
            positions = np.random.default_rng(self.random_state).choice(
                len(texts), self.fit_sample_size, replace=False,
            )
            fit_texts = texts[np.sort(positions)]
        try:
            vectorizer.fit(fit_texts)
        except ValueError as error:
            if "empty vocabulary" not in str(error) and "no terms remain" not in str(error).lower():
                raise
            empty = sparse.csr_matrix((len(texts), 0), dtype=np.float32)
            return _FieldIndex(None, empty, empty.T.tocsr())
        pieces = [
            vectorizer.transform(texts[start:start + self.transform_batch_size]).tocsr()
            for start in range(0, len(texts), self.transform_batch_size)
        ]
        matrix = pieces[0] if len(pieces) == 1 else sparse.vstack(pieces, format="csr")
        del pieces
        matrix.sort_indices()
        inverted = matrix.T.tocsr()
        return _FieldIndex(vectorizer, matrix, inverted)

    def country(self, key: str) -> _CountryIndex:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        # Evict before building, to keep peak memory limited to one country.
        while len(self._cache) >= self.cache_countries:
            self._cache.popitem(last=False)
        positions = np.flatnonzero(self.country_keys == key)
        names = self.pool["business_name"].iloc[positions].map(normalize_text).to_numpy()
        name_order = np.argsort(names, kind="stable")
        fields = {}
        for channel in self.channels:
            texts = names if channel == "name" else self.pool[_FIELDS[channel]].iloc[positions].map(normalize_text).to_numpy()
            fields[channel] = self._build_field(texts)
        result = _CountryIndex(positions, fields, names, name_order, names[name_order])
        self._cache[key] = result
        return result


def _pair_cosines(
    query: sparse.csr_matrix, pool: sparse.csr_matrix,
    left: np.ndarray, right: np.ndarray,
) -> np.ndarray:
    scores = np.empty(len(left), dtype=np.float32)
    for start in range(0, len(left), 25_000):
        stop = min(start + 25_000, len(left))
        scores[start:stop] = np.asarray(
            query[left[start:stop]].multiply(pool[right[start:stop]]).sum(axis=1)
        ).ravel()
    return np.clip(scores, 0, 1)


def iter_candidate_batches(
    reference: pd.DataFrame, pool: pd.DataFrame, *, top_k: int = 30,
    channels: tuple[str, ...] = ("name", "address"), exact_name_cap: int = 100,
    batch_size: int = 1024, min_similarity: float = 0.08, n_jobs: int = 1,
    fallback_pool_batch_size: int = 20_000,
    pool_index: PoolCandidateIndex | None = None, **index_kwargs,
) -> Iterator[pd.DataFrame]:
    """Yield complete candidate unions for country-blocked batches of Source 1.

    Additional keyword arguments configure ``PoolCandidateIndex``. ``top_k`` is
    per channel, and ``exact_name_cap=0`` disables the exact-name branch. A reused
    ``pool_index`` must have been built over the same pool object and channels.
    Empty-name matches are always excluded. No candidate is forced for empty text.
    """
    _validate_frame(reference, "reference")
    _validate_frame(pool, "pool")
    channels = _validate_channels(channels)
    if top_k < 0 or exact_name_cap < 0 or batch_size <= 0 or n_jobs <= 0:
        raise ValueError("top_k/exact_name_cap must be nonnegative; batch_size/n_jobs positive")
    if not 0 <= min_similarity <= 1 or fallback_pool_batch_size <= 0:
        raise ValueError("min_similarity must be in [0,1]; fallback_pool_batch_size positive")
    if not len(reference) or not len(pool) or (top_k == 0 and exact_name_cap == 0):
        return
    if pool_index is not None:
        if pool_index.pool is not pool or pool_index.channels != channels:
            raise ValueError("pool_index must use the same pool object and channels")
        if index_kwargs:
            raise ValueError("configure index parameters on the supplied pool_index")
    else:
        pool_index = PoolCandidateIndex(pool, channels=channels, **index_kwargs)
    if sp_matmul_topn is None:
        warnings.warn("Install sparse-dot-topn for fast retrieval; using a bounded sparse fallback.", stacklevel=2)

    reference_keys = _country_values(reference)
    pool_countries = set(pool_index.country_keys)
    for country in sorted(set(reference_keys) & pool_countries):
        country_index = pool_index.country(country)
        left_positions = np.flatnonzero(reference_keys == country)
        for start in range(0, len(left_positions), batch_size):
            global_left = left_positions[start:start + batch_size]
            query_frame = reference.iloc[global_left]
            query_matrices, pair_arrays = {}, []
            for channel in channels:
                field = country_index.fields[channel]
                if field.vectorizer is None:
                    query_matrices[channel] = sparse.csr_matrix((len(global_left), 0), dtype=np.float32)
                    continue
                query_matrix = field.vectorizer.transform(query_frame[_FIELDS[channel]].map(normalize_text)).tocsr()
                query_matrices[channel] = query_matrix
                retrieved = _sparse_retrieve(
                    query_matrix, field.inverted, top_k, min_similarity, n_jobs,
                    fallback_pool_batch_size,
                ).tocoo()
                if retrieved.nnz:
                    pair_arrays.append(np.column_stack((retrieved.row, retrieved.col)))
            if exact_name_cap:
                query_names = query_frame["business_name"].map(normalize_text).to_numpy()
                lo = np.searchsorted(country_index.sorted_names, query_names, side="left")
                hi = np.searchsorted(country_index.sorted_names, query_names, side="right")
                exact_left, exact_right = [], []
                for row, (name, lower, upper) in enumerate(zip(query_names, lo, hi)):
                    if not name:
                        continue
                    matches = country_index.name_order[lower:min(upper, lower + exact_name_cap)]
                    exact_left.extend([row] * len(matches))
                    exact_right.extend(matches.tolist())
                if exact_left:
                    pair_arrays.append(np.column_stack((exact_left, exact_right)))
            if not pair_arrays:
                continue
            pairs = np.unique(np.concatenate(pair_arrays).astype(np.int64, copy=False), axis=0)
            left, right = pairs[:, 0], pairs[:, 1]
            result = pd.DataFrame({"left_idx": global_left[left], "right_idx": country_index.positions[right]})
            for channel in _FIELDS:
                result[f"{channel}_score"] = (
                    _pair_cosines(query_matrices[channel], country_index.fields[channel].matrix, left, right)
                    if channel in channels else np.zeros(len(pairs), dtype=np.float32)
                )
            result["retrieval_score"] = result[["name_score", "address_score"]].max(axis=1).astype(np.float32)
            result.sort_values(["left_idx", "retrieval_score", "right_idx"], ascending=[True, False, True], kind="stable", inplace=True)
            result["retrieval_rank"] = (result.groupby("left_idx", sort=False).cumcount() + 1).astype(np.int32)
            yield result.reset_index(drop=True)
        # Drop the local reference before the next country index is allocated.
        del country_index, field


def generate_candidates(
    reference: pd.DataFrame, pool: pd.DataFrame, *, top_k: int = 30,
    channels: tuple[str, ...] = ("name", "address"), **kwargs,
) -> pd.DataFrame:
    """Return unique candidate pairs; use the iterator to stream full-scale output."""
    pieces = list(iter_candidate_batches(reference, pool, top_k=top_k, channels=channels, **kwargs))
    result = pd.concat(pieces, ignore_index=True) if pieces else _empty_candidates()
    if len(result):
        result.sort_values(["left_idx", "retrieval_rank"], kind="stable", inplace=True, ignore_index=True)
    result.attrs["retrieval_config"] = {
        "top_k_per_channel": top_k, "channels": tuple(channels),
        "exact_name_cap": kwargs.get("exact_name_cap", 100),
        "min_similarity": kwargs.get("min_similarity", 0.08),
        "country_blocked": True,
    }
    return result
