"""Edit-operation (U7), invented-name-at-address (U9) and transliteration (U3) pair feature blocks
(27 Sep, Agent U).

    python edit_features.py            # fits word tables on an A sample, writes prep/{brest,test}/x_ed.npy, x_tr.npy

Block ed (U7/U9): token edit counts; A-fitted false-rates of added/removed name words (smoothed target
encoding fitted on partition-A candidate rows only; B_REST/test never enter the fit); legal-form relation;
house-number relation and delta; address word overlap / ratio without digits; exact address; name ratio,
token-set ratio.
Block tr (U3): native-script flags; ratio after anyascii transliteration; consonant-skeleton ratio.
Test: US/India candidate rows (France rows NaN).
"""
import json
import os
import re
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from anyascii import anyascii
from rapidfuzz import fuzz

from ber.forensic_features import LEGAL_SUFFIXES, fold_ascii

ROOT = Path("/home/ubuntu/run27/U")
PREP = ROOT / "prep"
WORKERS = int(os.environ.get("U_WORKERS", "10"))
A_SAMPLE = 4_000_000
M_SMOOTH = 20.0
DECOY = {1, 3, 5, 7, 9, 11, 13, 21}
ED = ["ed_n_added", "ed_n_removed", "ed_n_common", "ed_te_add_max", "ed_te_add_mean", "ed_te_rem_max",
      "ed_te_add_min", "ed_legal_rel", "ed_num_rel", "ed_num_delta", "ed_num_delta_decoy", "ed_addr_tok_jacc",
      "ed_addr_ratio_nodigit", "ed_addr_exact", "ed_name_ratio", "ed_name_tsr", "ed_n_rec_digits_extra"]
TR = ["tr_native_s1", "tr_native_rec", "tr_anyascii_ratio", "tr_skeleton_ratio"]
VOWELS = re.compile(r"[aeiouy]")
T0 = time.time()
G = {}


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def prep_side(names, addresses):
    """Per-row tuple: (core tokens, legal tokens, folded addr, first number, addr words, digit groups, native, raw)."""
    fn = fold_ascii(names).to_pylist()
    fa = fold_ascii(addresses).to_pylist()
    raw = names.to_pylist()
    out = []
    for n, a, r in zip(fn, fa, raw):
        toks = (n or "").split()
        core = frozenset(t for t in toks if t not in LEGAL_SUFFIXES)
        legal = frozenset(t for t in toks if t in LEGAL_SUFFIXES)
        a = a or ""
        digs = re.findall(r"\d+", a)
        words = frozenset(t for t in a.split() if not t.isdigit())
        native = any(ord(ch) > 0x2FF for ch in (r or ""))
        out.append((core, legal, a, int(digs[0]) if digs else None, words, digs, native, r or "",
                    " ".join(t for t in toks if t not in LEGAL_SUFFIXES)))
    return out


def table_rate(counter_n, counter_f, prior):
    return {w: (counter_f[w] + M_SMOOTH * prior) / (n + M_SMOOTH) for w, n in counter_n.items() if n >= 5}


def _fit_chunk(bounds):
    s, e = bounds
    S, R, L, Y = G["S"], G["R"], G["l"], G["y"]
    na, fa, nr, fr = Counter(), Counter(), Counter(), Counter()
    for i in range(s, e):
        a, b = S[L[i][0]], R[L[i][1]]
        f = 0 if Y[i] else 1
        for w in b[0] - a[0]:
            na[w] += 1
            fa[w] += f
        for w in a[0] - b[0]:
            nr[w] += 1
            fr[w] += f
    return na, fa, nr, fr


def skeleton(s):
    return VOWELS.sub("", s.lower())


def _feat_chunk(bounds):
    s, e = bounds
    S, R, P = G["S"], G["R"], G["pairs"]
    ta, tr_, pa, pr = G["ta"], G["tr"], G["pa"], G["pr"]
    ed = np.full((e - s, len(ED)), np.nan, np.float32)
    tr = np.full((e - s, len(TR)), np.nan, np.float32)
    for j, i in enumerate(range(s, e)):
        a, b = S[P[i][0]], R[P[i][1]]
        added, removed = b[0] - a[0], a[0] - b[0]
        rates_a = [ta.get(w, pa) for w in added]
        rates_r = [tr_.get(w, pr) for w in removed]
        la, lb = a[1], b[1]
        legal = 0 if la == lb else (1 if not la else (2 if not lb else 3))
        if a[3] is None or b[3] is None:
            num_rel, delta, decoy = (4 if a[3] is None and b[3] is None else 5), np.nan, np.nan
        else:
            d = b[3] - a[3]
            da, db = a[5][0], b[5][0]
            num_rel = 0 if d == 0 and da == db else (1 if d == 0 else (2 if (da.startswith(db) or db.startswith(da)) else 3))
            delta, decoy = float(max(-1000, min(1000, d))), float(abs(d) in DECOY)
        wu = a[4] | b[4]
        ed[j] = [len(added), len(removed), len(a[0] & b[0]), max(rates_a) if rates_a else np.nan,
                 float(np.mean(rates_a)) if rates_a else np.nan, max(rates_r) if rates_r else np.nan,
                 min(rates_a) if rates_a else np.nan, legal, num_rel, delta, decoy,
                 len(a[4] & b[4]) / len(wu) if wu else np.nan,
                 fuzz.ratio(re.sub(r"\d+", "", a[2]), re.sub(r"\d+", "", b[2])) / 100.0 if (a[2] and b[2]) else np.nan,
                 float(a[2] == b[2] and a[2] != ""), fuzz.ratio(a[8], b[8]) / 100.0,
                 fuzz.token_set_ratio(a[8], b[8]) / 100.0, len(set(b[5]) - set(a[5]))]
        if a[6] or b[6]:
            xa, xb = anyascii(a[7]).lower(), anyascii(b[7]).lower()
            tr[j] = [float(a[6]), float(b[6]), fuzz.ratio(xa, xb) / 100.0, fuzz.ratio(skeleton(xa), skeleton(xb)) / 100.0]
        else:
            tr[j] = [0.0, 0.0, np.nan, np.nan]
    return s, ed, tr


def chunks(n, k=200):
    step = max(1, n // k + 1)
    return [(s, min(n, s + step)) for s in range(0, n, step)]


def side_tables(ref_path, pool_path, s_idx, r_idx):
    ref = pq.read_table(ref_path, columns=["business_name", "business_address"])
    pool = pq.read_table(pool_path, columns=["business_name", "business_address"])
    us, si = np.unique(s_idx, return_inverse=True)
    ur, ri = np.unique(r_idx, return_inverse=True)
    import pyarrow as pa
    S = prep_side(ref.column("business_name").take(pa.array(us)), ref.column("business_address").take(pa.array(us)))
    R = prep_side(pool.column("business_name").take(pa.array(ur)), pool.column("business_address").take(pa.array(ur)))
    return S, R, list(zip(si.tolist(), ri.tolist()))


def main():
    ctx = get_context("fork")
    # ---- fit word tables on an A sample
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    cl, cr = np.load(PREP / "arena_cand/left.npy"), np.load(PREP / "arena_cand/right.npy")
    crow = np.load(PREP / "arena_cand/cand_rows.npy")
    a_rows = np.flatnonzero(part[cl] == 0)
    rng = np.random.default_rng(20260927)
    a_rows = np.sort(rng.choice(a_rows, min(A_SAMPLE, len(a_rows)), replace=False))
    label = np.load(ROOT / "data/arena/labels/label.npy", mmap_mode="r")
    y = np.asarray(label[crow[a_rows]]).astype(bool)
    b_left, b_right = np.load(PREP / "brest/left.npy"), np.load(PREP / "brest/right.npy")
    s_all = np.r_[cl[a_rows].astype(np.int64), b_left]
    r_all = np.r_[cr[a_rows].astype(np.int64), b_right]
    S, R, pairs = side_tables(ROOT / "data/arena/reference.parquet", ROOT / "data/arena/pool.parquet", s_all, r_all)
    log(f"arena sides: {len(S)} S1s, {len(R)} records; A sample {len(a_rows)} (true {int(y.sum())})")
    G.update(S=S, R=R, l=pairs[:len(a_rows)], y=y)
    na, fa, nr, fr = Counter(), Counter(), Counter(), Counter()
    with ctx.Pool(WORKERS) as p:
        for a1, a2, a3, a4 in p.imap_unordered(_fit_chunk, chunks(len(a_rows))):
            na.update(a1), fa.update(a2), nr.update(a3), fr.update(a4)
    pa_ = sum(fa.values()) / max(1, sum(na.values()))
    pr_ = sum(fr.values()) / max(1, sum(nr.values()))
    ta, tr_ = table_rate(na, fa, pa_), table_rate(nr, fr, pr_)
    log(f"word tables: added {len(ta)} words (prior {pa_:.3f}), removed {len(tr_)} (prior {pr_:.3f})")
    json.dump({"added_prior": pa_, "removed_prior": pr_, "added": ta, "removed": tr_, "a_sample": int(len(a_rows)),
               "smoothing_m": M_SMOOTH, "min_count": 5}, open(PREP / "edit_word_tables.json", "w"))
    G.update(ta=ta, tr=tr_, pa=pa_, pr=pr_)

    def run(split, S_, R_, P_, n_rows, row_idx):
        G.update(S=S_, R=R_, pairs=P_)
        ed = np.full((n_rows, len(ED)), np.nan, np.float32)
        tr = np.full((n_rows, len(TR)), np.nan, np.float32)
        with ctx.Pool(WORKERS) as p:
            for s, e_, t_ in p.imap_unordered(_feat_chunk, chunks(len(P_))):
                ed[row_idx[s:s + len(e_)]] = e_
                tr[row_idx[s:s + len(t_)]] = t_
        np.save(PREP / split / "x_ed.npy", ed)
        np.save(PREP / split / "x_tr.npy", tr)
        log(f"{split}: {len(P_)} rows featurized")

    run("brest", S, R, pairs[len(a_rows):], len(b_left), np.arange(len(b_left)))
    (PREP / "brest/names_ed.json").write_text(json.dumps(ED))
    (PREP / "brest/names_tr.json").write_text(json.dumps(TR))
    # ---- test US/India rows
    tm = json.loads((PREP / "test/meta.json").read_text())
    tc = np.asarray(tm["countries"])[np.load(PREP / "test/country_code.npy")]
    tl, trr = np.load(PREP / "test/left.npy"), np.load(PREP / "test/right.npy")
    rows = np.flatnonzero(np.isin(tc[tl], ["US", "India"]))
    S2, R2, P2 = side_tables(ROOT / "data/test/dense_test/reference.parquet", ROOT / "data/test/dense_test/pool.parquet",
                             tl[rows], trr[rows])
    log(f"test sides: {len(S2)} S1s, {len(R2)} records")
    run("test", S2, R2, P2, len(tl), rows)


if __name__ == "__main__":
    main()
