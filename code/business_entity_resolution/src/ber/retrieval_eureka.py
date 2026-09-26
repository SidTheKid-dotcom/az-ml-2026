"""Extra retrieval branches over dense top-50 ∪ union-k3 (Operation EUREKA, track R).

The B_REST audit found three dense blind spots: invented-brand aliases at the
S1's own address (name shares nothing), Indian native-script names, and
empty-address records with typo'd names. Each branch is a per-country sparse
TF-IDF nearest-neighbour search (hashed features, document-frequency pruning,
top-k by cosine):

* ``address``: word uni+bigrams of the normalized address (digits without
  leading zeros, common street abbreviations expanded).
* ``native``: pool records whose name contains a non-Latin letter; names are
  transliterated with ``anyascii`` (ISC) and reduced to a consonant skeleton on
  both sides, char 3-grams.
* ``empty``: pool records with an empty address; folded core names (legal forms
  removed), char 3-grams.

`arena`: B_REST S1s against the arena pool. Labels are read only to score the
retrieval (link recall, truth rate, oracle F0.5); SLICE and HOLDOUT-2 are not
queried or scored. `test`: all test S1s against the raw test pool (including
records the compact dense pool never kept); writes a union-style set dir
(reference/pool parquet, left/right, tree = 1) per country-mixed branch union.
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import get_context
import os
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

ARENA = Path("/workspace/az-ml-new/arena")
TEST_INPUTS = Path("/opt/az-ml-new/test_inputs")
UNION = Path("/workspace/az-ml-new/union")
THREADS = int(os.environ.get("EUREKA_THREADS", "6"))
N_FEATURES = 2 ** 26
LEGAL = frozenset("the inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl "
                  "sas sasu sa sci eurl ei snc gmbh opc lp pllc pc".split())
ABBREV = {"st": "street", "str": "street", "ave": "avenue", "av": "avenue", "rd": "road", "ln": "lane",
          "dr": "drive", "blvd": "boulevard", "bd": "boulevard", "ct": "court", "pl": "place", "cir": "circle",
          "hwy": "highway", "pkwy": "parkway", "r": "rue", "all": "allee", "q": "quai", "imp": "impasse",
          "crs": "cours", "ter": "terrace", "trl": "trail", "sq": "square", "flr": "floor", "fl": "floor"}
DROP = frozenset({"null", "no", "nr", "unit", "apt", "apartment", "suite", "ste", "bis"})
BRANCHES = {  # max document frequency, top-k, minimum cosine
    "address": (5000, 10, 0.25),
    "empty": (5000, 5, 0.30),
}
# Kept for scoring (B_REST new-pair truth: address >= 0.6 is 3,796/444,377; empty >= 0.8 at rank <= 2 is
# 1,101/245,097). The native-script branch (transliterated consonant skeletons) found 3 new true links on
# B_REST and is off; the address branch already reaches those records.
KEEP = {"address": (0.6, 99), "empty": (0.8, 2)}  # minimum cosine, maximum rank
_TOK = re.compile(r"[a-z0-9]+")
_WORD = re.compile(r"[a-z]+")
SKELETON_STOP = frozenset({"prvt", "prv", "pvt", "lmtd", "lmt", "ltd", "lmtt", "kmpn", "krp", "ink", "llk"})
_SKELETON = [("ph", "f"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"),
             ("ch", "c"), ("x", "ks"), ("q", "k"), ("c", "k"), ("w", "v"), ("z", "j")]


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def fold(text: str) -> str:
    return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()


def norm_address(text: str) -> str:
    out = []
    for token in _TOK.findall(fold(text)):
        if token in DROP:
            continue
        if token.isdigit():
            token = token.lstrip("0") or "0"
        out.append(ABBREV.get(token, token))
    return " ".join(out)


def core_name(text: str) -> str:
    return " ".join(t for t in _TOK.findall(fold(text)) if t not in LEGAL)


def is_native(text: str) -> bool:
    return any(ord(ch) > 0x24F and unicodedata.category(ch).startswith("L") for ch in text or "")


def skeleton(text: str, transliterate: bool) -> str:
    if transliterate:
        from anyascii import anyascii
        text = anyascii(text or "").lower()
    else:
        text = fold(text)
    words = [w for w in _WORD.findall(text) if w not in LEGAL]
    out = []
    for word in words:
        for a, b in _SKELETON:
            word = word.replace(a, b)
        word = word[0] + re.sub(r"[aeiouy]", "", word[1:]) if word else word
        word = re.sub(r"(.)\1+", r"\1", word)
        if word and word not in SKELETON_STOP:
            out.append(word)
    return " ".join(out)


def empty_address(text: str) -> bool:
    return not _TOK.findall(fold(text).replace("null", ""))


# ----------------------------------------------------------------------------- vectorizing

def _vectorizer(kind: str) -> HashingVectorizer:
    if kind == "address":
        return HashingVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), n_features=N_FEATURES,
                                 alternate_sign=False, norm=None, lowercase=False, binary=True, dtype=np.float32)
    return HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=N_FEATURES,
                             alternate_sign=False, norm=None, lowercase=False, binary=True, dtype=np.float32)


def _prep(kind: str, texts: list[str], transliterate: bool) -> list[str]:
    if kind == "address":
        return [norm_address(t) for t in texts]
    if kind == "native":
        return [skeleton(t, transliterate) for t in texts]
    return [core_name(t) for t in texts]


def _transform_chunk(job):
    kind, texts, transliterate = job
    return _vectorizer(kind).transform(_prep(kind, texts, transliterate)).tocsr()


def vectorize(kind: str, texts, transliterate: bool = False) -> sp.csr_matrix:
    texts = list(texts)
    if not texts:
        return sp.csr_matrix((0, N_FEATURES), dtype=np.float32)
    step = max(1, -(-len(texts) // (THREADS * 4)))
    jobs = [(kind, texts[i:i + step], transliterate) for i in range(0, len(texts), step)]
    with get_context("fork").Pool(THREADS) as pool:
        parts = pool.map(_transform_chunk, jobs)
    return sp.vstack(parts).tocsr()


def tfidf_pair(pool_x: sp.csr_matrix, query_x: sp.csr_matrix, max_df: int):
    """Pool-fitted smoothed IDF with features above ``max_df`` zeroed; both sides L2-normalized."""
    df = np.bincount(pool_x.indices, minlength=N_FEATURES)
    idf = (np.log((1 + pool_x.shape[0]) / (1 + df)) + 1).astype(np.float32)
    idf[(df > max_df) | (df == 0)] = 0
    out = []
    for x in (pool_x, query_x):
        x = x.copy()
        x.data *= idf[x.indices]
        x.eliminate_zeros()
        out.append(normalize(x, copy=False))
    return out


_Q = _PT = None
_K = _MIN = 0


def _topk_chunk(bounds):
    start, stop = bounds
    s = (_Q[start:stop] @ _PT).tocsr()
    rows, cols, vals = [], [], []
    for i in range(stop - start):
        a, b = s.indptr[i], s.indptr[i + 1]
        if a == b:
            continue
        d, c = s.data[a:b], s.indices[a:b]
        keep = d >= _MIN
        d, c = d[keep], c[keep]
        if len(d) > _K:
            sel = np.argpartition(-d, _K)[:_K]
            d, c = d[sel], c[sel]
        rows.append(np.full(len(d), start + i, dtype=np.int64))
        cols.append(c.astype(np.int64))
        vals.append(d.astype(np.float32))
    if not rows:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.float32)
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def topk(query: sp.csr_matrix, pool: sp.csr_matrix, k: int, min_score: float, chunk: int = 1000):
    global _Q, _PT, _K, _MIN
    _Q, _PT, _K, _MIN = query, pool.T.tocsr(), k, min_score
    bounds = [(i, min(i + chunk, query.shape[0])) for i in range(0, query.shape[0], chunk)]
    with get_context("fork").Pool(THREADS) as workers:
        parts = workers.map(_topk_chunk, bounds, chunksize=4)
    _Q = _PT = None
    rows = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, np.int64)
    cols = np.concatenate([p[1] for p in parts]) if parts else np.empty(0, np.int64)
    vals = np.concatenate([p[2] for p in parts]) if parts else np.empty(0, np.float32)
    return rows, cols, vals


def run_branches(ref: pd.DataFrame, pool: pd.DataFrame, query_rows: np.ndarray) -> dict[str, pd.DataFrame]:
    """All branches, per country. Returns {branch: DataFrame(s1, rec, score, rank)} in global indices."""
    out = {b: [] for b in BRANCHES}
    ref_country = ref.country.to_numpy()
    pool_country = pool.country.to_numpy()
    names, addresses = pool.business_name.fillna("").to_numpy(), pool.business_address.fillna("").to_numpy()
    native_mask = np.fromiter((is_native(t) for t in names), dtype=bool, count=len(names))
    empty_mask = np.fromiter((empty_address(t) for t in addresses), dtype=bool, count=len(addresses))
    log(f"pool {len(pool)}: native-script names {int(native_mask.sum())}, empty addresses {int(empty_mask.sum())}")
    for country in sorted(set(ref_country[query_rows])):
        q = query_rows[ref_country[query_rows] == country]
        p_all = np.flatnonzero(pool_country == country)
        for branch, (max_df, k, min_score) in BRANCHES.items():
            t0 = time.time()
            if branch == "address":
                p = p_all
                px = vectorize("address", addresses[p])
                qx = vectorize("address", ref.business_address.fillna("").to_numpy()[q])
            elif branch == "native":  # off by default (see KEEP)
                p = p_all[native_mask[p_all]]
                if len(p) == 0:
                    continue
                px = vectorize("native", names[p], transliterate=True)
                qx = vectorize("native", ref.business_name.fillna("").to_numpy()[q])
            else:
                p = p_all[empty_mask[p_all]]
                px = vectorize("empty", names[p])
                qx = vectorize("empty", ref.business_name.fillna("").to_numpy()[q])
            px, qx = tfidf_pair(px, qx, max_df)
            r, c, v = topk(qx, px, k, min_score)
            frame = pd.DataFrame({"s1": q[r], "rec": p[c], "score": v})
            frame["rank"] = frame.groupby("s1").score.rank(ascending=False, method="first").astype(np.int16) - 1
            out[branch].append(frame)
            log(f"{country} {branch}: pool {len(p)}, queries {len(q)}, pairs {len(frame)}, {time.time() - t0:.0f}s")
    return {b: pd.concat(v, ignore_index=True) if v else pd.DataFrame(columns=["s1", "rec", "score", "rank"])
            for b, v in out.items()}


# ----------------------------------------------------------------------------- arena

def arena_cmd(args) -> dict:
    t_start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(ARENA / "v1/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
    pool = pq.read_table(ARENA / "v1/pool.parquet",
                         columns=["business_name", "business_address", "country", "raw_row_idx"]).to_pandas()
    part = np.load(ARENA / "finalize_v1/s1_partition.npy")
    brest = np.flatnonzero(part == 1)
    n_pool = len(pool)
    # base candidate keys: dense top-50 and union k3 (B_REST only)
    t = pq.read_table(ARENA / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
    left = t.column("left_idx").to_numpy().astype(np.int64)
    right = t.column("right_idx").to_numpy().astype(np.int64)
    del t
    m = part[left] == 1
    dense_keys = np.unique(left[m] * n_pool + right[m])
    del left, right, m
    uref = pq.read_table(UNION / "set_b_rest_k3/reference.parquet", columns=["arena_idx"]).column("arena_idx").to_numpy()
    upool = pq.read_table(UNION / "set_b_rest_k3/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    raw_to_pool = np.full(int(max(pool.raw_row_idx.max(), upool.max())) + 1, -1, dtype=np.int64)
    raw_to_pool[pool.raw_row_idx.to_numpy()] = np.arange(n_pool)
    ul = uref[np.load(UNION / "set_b_rest_k3/left.npy")].astype(np.int64)
    ur = raw_to_pool[upool[np.load(UNION / "set_b_rest_k3/right.npy")]]
    union_keys = np.unique(ul[ur >= 0] * n_pool + ur[ur >= 0])
    base_keys = np.union1d(dense_keys, union_keys)
    log(f"B_REST S1 {len(brest)}; dense pairs {len(dense_keys)}, union k3 in pool {len(union_keys)} (of {len(ul)}), base {len(base_keys)}")

    branches = run_branches(ref, pool, brest)

    # scoring (labels read only here, B_REST owners only)
    owner = np.load(ARENA / "labels/rec_owner.npy").astype(np.int64)
    true_count = np.load(ARENA / "labels/true_count.npy").astype(np.int64)
    n_s1 = len(true_count)
    owner_b = np.where((owner >= 0) & (part[np.maximum(owner, 0)] == 1), owner, -1)
    true_keys = np.unique(owner_b[owner_b >= 0] * n_pool + np.flatnonzero(owner_b >= 0))
    total_true = int(true_count[brest].sum())

    def found(keys):
        return int(np.isin(true_keys, keys, assume_unique=True).sum())

    def oracle(keys):
        hit = true_keys[np.isin(true_keys, keys, assume_unique=True)]
        tp = np.bincount(hit // n_pool, minlength=n_s1)
        tc = true_count
        f = np.where(tc == 0, 1.0, np.where(tp == 0, 0.0, 5 * tp / np.maximum(tc + 4 * tp, 1)))
        return float(f[brest].mean())

    report = {"scope": "arena B_REST (SLICE, HOLDOUT-2 not queried or scored)", "b_rest_s1": int(len(brest)),
              "true_links": total_true, "branches_config": BRANCHES,
              "dense": {"pairs": int(len(dense_keys)), "true": found(dense_keys),
                        "recall": found(dense_keys) / total_true, "oracle_f05": oracle(dense_keys)},
              "dense_union": {"pairs": int(len(base_keys)), "true": found(base_keys),
                              "recall": found(base_keys) / total_true, "oracle_f05": oracle(base_keys)}}
    cumulative = base_keys
    for branch, frame in branches.items():
        keys = frame.s1.to_numpy(np.int64) * n_pool + frame.rec.to_numpy(np.int64)
        new = ~np.isin(keys, base_keys)
        is_true = owner_b[frame.rec.to_numpy()] == frame.s1.to_numpy()
        new_keys = np.unique(keys[new])
        before = found(cumulative)
        cumulative = np.union1d(cumulative, new_keys)
        by_rank = {int(r): [int(((frame["rank"] == r) & new).sum()), int(((frame["rank"] == r) & new & is_true).sum())]
                   for r in sorted(frame["rank"].unique())}
        report[branch] = {"pairs": int(len(frame)), "new_pairs": int(len(new_keys)),
                          "new_pairs_per_s1": len(new_keys) / len(brest),
                          "new_true_vs_base": int((new & is_true).sum()),
                          "new_true_rate": float(is_true[new].mean()) if new.any() else None,
                          "incremental_true_cumulative": found(cumulative) - before,
                          "cumulative_recall": found(cumulative) / total_true,
                          "cumulative_oracle_f05": oracle(cumulative),
                          "new_by_rank_[pairs,true]": by_rank}
        frame.assign(new=new, true=is_true).to_parquet(out / f"b_rest_{branch}.parquet", index=False)
        log(f"{branch}: {json.dumps(report[branch], default=float)[:400]}")
    report["cumulative"] = {"pairs": int(len(cumulative)), "true": found(cumulative),
                            "added_true_over_dense_union": found(cumulative) - found(base_keys),
                            "recall": found(cumulative) / total_true, "oracle_f05": oracle(cumulative)}
    kept = [np.empty(0, np.int64)]
    for branch, frame in branches.items():
        lo, max_rank = KEEP[branch]
        f = frame[(frame.score >= lo) & (frame["rank"] <= max_rank)]
        kept.append(f.s1.to_numpy(np.int64) * n_pool + f.rec.to_numpy(np.int64))
    new_all = np.setdiff1d(np.unique(np.concatenate(kept)), base_keys, assume_unique=True)
    trimmed = np.union1d(base_keys, new_all)
    report["kept_for_scoring"] = {"rule": {k: {"min_cosine": v[0], "max_rank": v[1]} for k, v in KEEP.items()},
                                  "new_pairs": int(len(new_all)), "new_true": found(trimmed) - found(base_keys),
                                  "recall": found(trimmed) / total_true, "oracle_f05": oracle(trimmed)}
    np.save(out / "left.npy", new_all // n_pool)
    np.save(out / "right.npy", new_all % n_pool)
    np.save(out / "tree.npy", np.ones(len(new_all), dtype=np.float32))
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps(report["cumulative"]))
    return report


# ----------------------------------------------------------------------------- test

def read_raw_pool(raw_dir: Path) -> pd.DataFrame:
    frames = [pd.read_csv(raw_dir / f"test_source{i}.tsv", sep="\t", dtype=str, keep_default_na=False, quoting=3,
                          usecols=["entity_id", "business_name", "business_address", "country"]) for i in (2, 3)]
    pool = pd.concat(frames, ignore_index=True)
    pool["raw_row_idx"] = np.arange(len(pool), dtype=np.int64)
    return pool


def test_cmd(args) -> dict:
    t_start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(TEST_INPUTS / "dense_test/reference.parquet",
                        columns=["entity_id", "business_name", "business_address", "country"]).to_pandas()
    pool = read_raw_pool(Path(args.raw))
    dense_pool = pq.read_table(TEST_INPUTS / "dense_test/pool.parquet", columns=["entity_id", "raw_row_idx"]).to_pandas()
    if not np.array_equal(pool.entity_id.to_numpy()[dense_pool.raw_row_idx.to_numpy()], dense_pool.entity_id.to_numpy()):
        raise SystemExit("raw pool rows do not match the dense pool raw_row_idx mapping")
    n_raw = len(pool)
    left = np.asarray(np.load(TEST_INPUTS / "index/left.npy", mmap_mode="r")).astype(np.int64)
    right = dense_pool.raw_row_idx.to_numpy()[np.asarray(np.load(TEST_INPUTS / "index/right.npy", mmap_mode="r"))]
    dense_keys = np.unique(left * n_raw + right)
    del left, right
    uref_ids = pq.read_table(UNION / "set_test_k3/reference.parquet", columns=["entity_id"]).column("entity_id").to_numpy()
    ref_pos = pd.Series(np.arange(len(ref)), index=ref.entity_id.to_numpy())
    uref = ref_pos.reindex(uref_ids).to_numpy()
    upool = pq.read_table(UNION / "set_test_k3/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    ul = uref[np.load(UNION / "set_test_k3/left.npy")].astype(np.int64)
    ur = upool[np.load(UNION / "set_test_k3/right.npy")].astype(np.int64)
    base_keys = np.union1d(dense_keys, np.unique(ul * n_raw + ur))
    log(f"test S1 {len(ref)}, raw pool {n_raw}, dense pairs {len(dense_keys)}, base {len(base_keys)}")
    branches = run_branches(ref, pool, np.arange(len(ref)))
    report = {"scope": "test (no labels exist)", "raw_pool": n_raw, "dense_pairs": int(len(dense_keys)),
              "base_pairs": int(len(base_keys)), "branches_config": BRANCHES}
    in_dense_pool = np.zeros(n_raw, bool)
    in_dense_pool[dense_pool.raw_row_idx.to_numpy()] = True
    cumulative = [np.empty(0, np.int64)]
    country = ref.country.to_numpy()
    for branch, frame in branches.items():
        lo, max_rank = KEEP[branch]
        frame = frame[(frame.score >= lo) & (frame["rank"] <= max_rank)].reset_index(drop=True)
        keys = frame.s1.to_numpy(np.int64) * n_raw + frame.rec.to_numpy(np.int64)
        new = ~np.isin(keys, base_keys)
        nk = np.unique(keys[new])
        cumulative.append(nk)
        report[branch] = {"pairs": int(len(frame)), "new_pairs": int(len(nk)),
                          "new_by_country": {str(c): int((country[nk // n_raw] == c).sum()) for c in np.unique(country)},
                          "new_pairs_to_records_outside_dense_pool": int((~in_dense_pool[nk % n_raw]).sum())}
        frame.assign(new=new).to_parquet(out / f"test_{branch}.parquet", index=False)
        log(f"{branch}: {json.dumps(report[branch])}")
    new_all = np.unique(np.concatenate(cumulative))
    s1, raw = new_all // n_raw, new_all % n_raw
    recs = np.unique(raw)
    pool_out = pool.iloc[recs].reset_index(drop=True)
    local = np.searchsorted(recs, raw)
    ref.to_parquet(out / "reference.parquet", index=False)
    pool_out.to_parquet(out / "pool.parquet", index=False)
    np.save(out / "left.npy", s1.astype(np.int64))
    np.save(out / "right.npy", local.astype(np.int64))
    np.save(out / "tree.npy", np.ones(len(s1), dtype=np.float32))
    report["union_new_pairs"] = int(len(new_all))
    report["union_new_by_country"] = {str(c): int((country[s1] == c).sum()) for c in np.unique(country)}
    report["union_new_records_outside_dense_pool"] = int((~in_dense_pool[raw]).sum())
    report["set_dir"] = {"reference": "reference.parquet (test order)", "pool": "pool.parquet (raw_row_idx kept)",
                         "left": "left.npy -> reference row", "right": "right.npy -> pool.parquet row",
                         "tree": "tree.npy all ones (gate bypass)"}
    report["runtime_seconds"] = round(time.time() - t_start, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("arena")
    a.add_argument("--output", default="/workspace/az-ml-new/eureka/retrieval/b_rest")
    t = sub.add_parser("test")
    t.add_argument("--raw", default=str(UNION / "raw"))
    t.add_argument("--output", default="/workspace/az-ml-new/eureka/retrieval/test")
    args = parser.parse_args()
    (arena_cmd if args.command == "arena" else test_cmd)(args)


if __name__ == "__main__":
    main()
