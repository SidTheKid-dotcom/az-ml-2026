"""India native-script recall (27 Sep, Agent U): a learned Indic->Latin word dictionary + transliterated-name retrieval.

    python nat_dict.py learn      # dictionary from partition-A true pairs only (option A: TRAIN labels only)
    python nat_dict.py arena      # B_REST: retrieve, drop pairs already in dense / routes / R-reverse, measure
    python nat_dict.py test       # test: same retrieval (label-free), new pairs vs test dense / routes / reverse

Dictionary: for every A-owned native-script record (owner = A S1 with a Latin name), native tokens are aligned
monotonically to the S1's Latin tokens (DP on RapidFuzz ratio of the anyascii romanization; equal token counts align
by position). A native token maps to its most frequent Latin partner if (count >= 2 and share >= 0.5) or (count == 1
and ratio >= 60). Transliteration = dictionary token, else anyascii (fallback), lowercased.
Retrieval: per record (native-script, India), top-K India S1s by char-3gram TF-IDF cosine of the transliterated name
vs the folded S1 name (sparse_dot_topn). B_REST labels (B_REST S1s only) are read only in `arena` to measure;
HOLDOUT-2 / SLICE ground-truth payloads are never parsed.
"""
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from anyascii import anyascii
from rapidfuzz import fuzz

ROOT = Path("/home/ubuntu/run27/U")
N = ROOT / "nat"
INDIC = re.compile("[ऀ-෿]")
SPLIT = re.compile(r"[\s,.;:/\\()\[\]&'\"!?\-+*#@|]+")
LAT = re.compile(r"[^a-z0-9]+")
K = int(os.environ.get("NAT_K", "10"))
THREADS = int(os.environ.get("NAT_THREADS", "26"))
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def is_native(s):
    return bool(INDIC.search(s or ""))


def lat_tokens(s):
    return [t for t in LAT.split(anyascii(s or "").lower()) if t]


def nat_tokens(s):
    return [t for t in SPLIT.split(s or "") if t]


def romanize(tok):
    return LAT.sub("", anyascii(tok).lower())


def align(nt, lt):
    """Monotone alignment native tokens -> Latin tokens; returns [(native, latin, ratio)]."""
    if not nt or not lt:
        return []
    ra = [romanize(t) for t in nt]
    if len(nt) == len(lt):
        return [(n, l, fuzz.ratio(r, l)) for n, r, l in zip(nt, ra, lt)]
    m, n = len(nt), len(lt)
    sim = [[fuzz.ratio(ra[i], lt[j]) for j in range(n)] for i in range(m)]
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    bt = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            best, arg = dp[i - 1][j], 1
            if dp[i][j - 1] > best:
                best, arg = dp[i][j - 1], 2
            v = dp[i - 1][j - 1] + (sim[i - 1][j - 1] if sim[i - 1][j - 1] >= 40 else -1e9)
            if v > best:
                best, arg = v, 3
            dp[i][j], bt[i][j] = best, arg
    out, i, j = [], m, n
    while i > 0 and j > 0:
        a = bt[i][j]
        if a == 3:
            out.append((nt[i - 1], lt[j - 1], sim[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif a == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def truth(allowed_s1):
    """{s1_id: [record ids]} for the given S1 ids only (other payloads are never split)."""
    out = {}
    with open(N / "gt/train_ground_truth.tsv", encoding="utf-8") as fh:
        if fh.readline().rstrip("\r\n") != "source1_entity_id\tmatched_entity_ids":
            raise SystemExit("unexpected ground-truth header")
        for line in fh:
            s1, _, payload = line.rstrip("\r\n").partition("\t")
            if s1 in allowed_s1:
                out[s1] = [v for v in payload.split(",") if v]
    return out


def texts(name):
    t = pq.read_table(N / "texts" / f"{name}.parquet")
    ids = t.column("entity_id").to_pylist()
    tx = t.column("text").to_pylist()
    cc = t.column("cc").to_numpy()
    names, addrs = [], []
    for s in tx:
        s = s.split(": ", 1)[1] if ": " in s[:10] else s
        n, _, a = s.partition(" | ")
        names.append(n)
        addrs.append(a)
    return ids, names, addrs, cc


def learn():
    ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["entity_id", "business_name"]).to_pandas()
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    a_ids = set(ref.entity_id[part == 0])
    gt = truth(a_ids)
    s1_name = dict(zip(ref.entity_id, ref.business_name))
    ids, names, _, _ = texts("train_pool")
    rec_name = {i: n for i, n in zip(ids, names) if is_native(n)}
    log(f"A S1s with truth {len(gt)}; native-script raw records {len(rec_name)}")
    pairs = Counter()
    tot = Counter()
    ratio_sum = defaultdict(float)
    n_pairs = 0
    for s1, recs in gt.items():
        sn = s1_name.get(s1, "")
        if is_native(sn):
            continue
        lt = lat_tokens(sn)
        for r in recs:
            rn = rec_name.get(r)
            if rn is None:
                continue
            nt = [t for t in nat_tokens(rn) if is_native(t)]
            if not nt:
                continue
            n_pairs += 1
            for nw, lw, rt in align(nt, lt):
                pairs[(nw, lw)] += 1
                tot[nw] += 1
                ratio_sum[(nw, lw)] += rt
    best = {}
    for (nw, lw), c in pairs.items():
        if nw not in best or c > best[nw][1]:
            best[nw] = (lw, c)
    d = {}
    for nw, (lw, c) in best.items():
        share = c / tot[nw]
        mean_ratio = ratio_sum[(nw, lw)] / c
        if (c >= 2 and share >= 0.5) or (c == 1 and mean_ratio >= 60):
            d[nw] = lw
    json.dump({"dictionary": d, "a_true_native_pairs": n_pairs, "native_tokens_seen": len(tot),
               "mapped": len(d), "rule": "count>=2 & share>=0.5, or count==1 & ratio>=60 (anyascii romanization)",
               "labels": "partition A only"}, open(N / "dictionary.json", "w"), ensure_ascii=False)
    log(f"dictionary: {len(d)} of {len(tot)} native tokens mapped from {n_pairs} A true pairs")


def translit(name, d):
    out = []
    for t in nat_tokens(name):
        if is_native(t):
            out.append(d.get(t) or romanize(t))
        else:
            out.extend(lat_tokens(t))
    return " ".join(x for x in out if x)


STOP = frozenset("private limited pvt ltd llp opc co company the and of inc corp corporation llc plc".split())


def core(text):
    return " ".join(t for t in text.split() if t not in STOP)


def retrieve(q_text, s1_text):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sparse_dot_topn import sp_matmul_topn
    # legal / filler words removed and 3-grams in > 1% of S1 names dropped: keeps the sparse product sparse
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True, min_df=1,
                          max_df=0.01)
    s = vec.fit_transform([core(t) for t in s1_text])
    q = vec.transform([core(t) for t in q_text])
    log(f"  tfidf: S1 {s.shape}, queries {q.shape}, nnz/query {q.nnz / max(q.shape[0], 1):.1f}")
    m = sp_matmul_topn(q, s.T.tocsr(), top_n=K, threshold=0.2, n_threads=THREADS, sort=True).tocsr()
    rows = np.repeat(np.arange(m.shape[0]), np.diff(m.indptr))
    rank = np.concatenate([np.arange(b - a) for a, b in zip(m.indptr[:-1], m.indptr[1:])]) if m.nnz else np.zeros(0, int)
    return rows, m.indices.astype(np.int64), m.data.astype(np.float32), rank


DIG = re.compile(r"\d+")


def name_key(text):
    toks = [t for t in text.split() if t not in STOP]
    return " ".join(sorted(set(toks)))


def addr_norm(a):
    return " ".join(LAT.split(anyascii(a or "").lower())).strip()


def block_retrieve(q_text, q_addr, s1_text, s1_addr, max_group=int(os.environ.get("NAT_MAX_GROUP", "3000")), k=5):
    """Records -> S1s with the identical transliterated core-name key; ranked by address similarity
    (token-set ratio + digit-group Jaccard). Returns rows, cols, score, rank, group size."""
    from rapidfuzz import process
    s_keys = pd.Series([name_key(t) for t in s1_text])
    q_keys = pd.Series([name_key(t) for t in q_text])
    groups = s_keys.groupby(s_keys).indices
    s_addr = [addr_norm(a) for a in s1_addr]
    s_dig = [set(g.lstrip("0") or "0" for g in DIG.findall(a)) for a in s_addr]
    out_r, out_c, out_v, out_k, out_g = [], [], [], [], []
    by_q = q_keys.groupby(q_keys).indices
    for key, qi in by_q.items():
        if not key or key not in groups:
            continue
        si = groups[key]
        if len(si) > max_group:
            continue
        qa = [addr_norm(q_addr[i]) for i in qi]
        sa = [s_addr[j] for j in si]
        m = process.cdist(qa, sa, scorer=fuzz.token_set_ratio, workers=THREADS, dtype=np.float32) / 100
        for a_i, i in enumerate(qi):
            qd = set(g.lstrip("0") or "0" for g in DIG.findall(qa[a_i]))
            dj = np.asarray([len(qd & s_dig[j]) / len(qd | s_dig[j]) if (qd or s_dig[j]) else 0.0 for j in si],
                            np.float32)
            sc = m[a_i] + 0.5 * dj
            top = np.argsort(-sc, kind="stable")[:k]
            out_r.append(np.full(len(top), i))
            out_c.append(si[top])
            out_v.append(sc[top])
            out_k.append(np.arange(len(top)))
            out_g.append(np.full(len(top), len(si)))
    cat = lambda x, dt: np.concatenate(x).astype(dt) if x else np.zeros(0, dt)  # noqa: E731
    return cat(out_r, np.int64), cat(out_c, np.int64), cat(out_v, np.float32), cat(out_k, np.int64), cat(out_g, np.int64)


def run(split):
    d = json.load(open(N / "dictionary.json"))["dictionary"]
    if split == "arena":
        ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["entity_id", "business_name",
                                                                             "business_address", "country"]).to_pandas()
        pool_name = "train_pool"
    else:
        ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet",
                            columns=["entity_id", "business_name", "business_address", "country"]).to_pandas()
        pool_name = "test_pool"
    ids, names, addrs, cc = texts(pool_name)
    nat = np.flatnonzero(np.fromiter((is_native(n) for n in names), bool, count=len(names)) & (cc == 1))
    q_text = [translit(names[i], d) for i in nat]
    india = np.flatnonzero(ref.country.to_numpy() == "India")
    s1_text = [" ".join(lat_tokens(n)) for n in ref.business_name.to_numpy()[india]]
    log(f"{split}: {len(nat)} native-script India records; {len(india)} India S1s")
    s1_addr_all = ref.business_address.fillna("").to_numpy()
    r, c, v, rank, gsize = block_retrieve(q_text, [addrs[i] for i in nat], s1_text, list(s1_addr_all[india]))
    log(f"{split}: name-key blocks -> {len(r)} pairs")
    frame = pd.DataFrame({"s1": india[c], "raw": nat[r], "score": v, "rank": rank, "group": gsize})
    # name/address similarity features for the candidates (label-free)
    frame["nat_ratio"] = [fuzz.ratio(q_text[i], s1_text[j]) / 100 for i, j in zip(r, c)]
    frame["nat_tsr"] = [fuzz.token_set_ratio(q_text[i], s1_text[j]) / 100 for i, j in zip(r, c)]
    s1_addr = ref.business_address.fillna("").to_numpy()
    frame["addr_tsr"] = [fuzz.token_set_ratio(anyascii(addrs[nat[i]]).lower(), anyascii(s1_addr[india[j]]).lower()) / 100
                         for i, j in zip(r, c)]
    frame["rec_id"] = np.asarray(ids, dtype=object)[frame.raw.to_numpy()]
    frame["s1_id"] = ref.entity_id.to_numpy()[frame.s1.to_numpy()]
    frame.to_parquet(N / f"{split}_candidates.parquet", index=False)
    np.save(N / f"{split}_native_raw.npy", nat)
    log(f"{split}: {len(frame)} candidate pairs ({len(frame) / max(len(india), 1):.2f} per India S1)")
    return frame


def existing_keys_arena():
    """(arena S1 idx, raw record row) keys of dense top-50 pairs, B_REST routes and R's B_REST reverse pairs."""
    t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_raw_idx"])
    keys = [t.column("left_idx").to_numpy().astype(np.int64) * 20_000_000 + t.column("right_raw_idx").to_numpy()]
    pool_raw = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    rl, rr = np.load(ROOT / "u8src/b_rest/left.npy"), np.load(ROOT / "u8src/b_rest/right.npy")
    keys.append(rl.astype(np.int64) * 20_000_000 + pool_raw[rr])
    rv = pq.read_table(N / "rev/r_reverse_pairs_b_rest.parquet", columns=["left_idx", "raw_row_idx"])
    keys.append(rv.column("left_idx").to_numpy().astype(np.int64) * 20_000_000 + rv.column("raw_row_idx").to_numpy())
    return np.unique(np.concatenate(keys))


def arena():
    frame = run("arena")
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    base = existing_keys_arena()
    k = frame.s1.to_numpy().astype(np.int64) * 20_000_000 + frame.raw.to_numpy()
    frame["new"] = ~np.isin(k, base)
    brest = part == 1
    b_ids = set(ref.entity_id[brest])
    gt = truth(b_ids)
    true_keys = set()
    for s1, recs in gt.items():
        for rec in recs:
            true_keys.add((s1, rec))
    frame["true"] = [(a, b) in true_keys for a, b in zip(frame.s1_id, frame.rec_id)]
    fb = frame[brest[frame.s1.to_numpy()]]
    # baseline found true links per B_REST S1 (dense + routes + reverse), then + new pairs under each rule
    ids_raw = pq.read_table(N / "texts/train_pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    pos = {v: i for i, v in enumerate(ids_raw)}
    s1_idx = {v: i for i, v in enumerate(ref.entity_id)}
    tk = np.asarray([s1_idx[s] * 20_000_000 + pos[r] for s, recs in gt.items() for r in recs if r in pos], np.int64)
    tc = np.load(ROOT / "data/arena/labels/true_count.npy").astype(np.int64)
    found0 = np.isin(tk, base)
    n_s1 = len(part)

    def oracle(found_keys):
        tp = np.minimum(np.bincount(found_keys // 20_000_000, minlength=n_s1), 11)
        f = np.where(tc == 0, 1.0, np.where(tp == 0, 0.0, 5 * tp / np.maximum(tc + 4 * tp, 1)))
        return float(f[brest].mean())

    o0 = oracle(tk[found0])
    rep = {"b_rest_s1": int(brest.sum()), "b_rest_india_s1": int((brest & (ref.country == "India").to_numpy()).sum()),
           "true_links": int(len(tk)), "baseline_found": int(found0.sum()), "baseline_oracle_f05": o0,
           "native_true_links_missed_by_baseline": None, "rules": {}}
    nat_raw = np.load(N / "arena_native_raw.npy")
    miss = tk[~found0]
    rep["native_true_links_missed_by_baseline"] = int(np.isin(miss % 20_000_000, nat_raw).sum())
    rep["native_true_links_total"] = int(np.isin(tk % 20_000_000, nat_raw).sum())
    for rule, m in (("rank0", fb["rank"] == 0), ("rank<=1", fb["rank"] <= 1), ("rank<=2", fb["rank"] <= 2),
                    ("rank<=4", fb["rank"] <= 4), ("rank0|score>=0.5", (fb["rank"] == 0) | (fb.score >= 0.5)),
                    ("all_k", fb["rank"] >= 0)):
        sel = fb[m & fb.new]
        kk = sel.s1.to_numpy().astype(np.int64) * 20_000_000 + sel.raw.to_numpy()
        new_true = sel.true.to_numpy()
        found = np.union1d(tk[found0], kk[new_true])
        rep["rules"][rule] = {"new_pairs": int(len(sel)), "new_pairs_per_b_rest_india_s1": len(sel) / rep["b_rest_india_s1"],
                              "new_true": int(new_true.sum()), "true_rate": float(new_true.mean()) if len(sel) else None,
                              "oracle_f05": oracle(found), "oracle_gain": oracle(found) - o0}
        log(f"{rule}: {json.dumps(rep['rules'][rule])}")
    frame.to_parquet(N / "arena_candidates.parquet", index=False)
    (N / "arena_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps({k: v for k, v in rep.items() if k != "rules"}))


if __name__ == "__main__":
    {"learn": learn, "arena": arena, "test": lambda: run("test")}[sys.argv[1]]()
