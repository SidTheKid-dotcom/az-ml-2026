"""Idea 4 (27 Sep, Agent U): v1 / collective / edit-op blocks on the L1 target rows (l1_targets.py order).

v1: the 12 pair_tree_arena features from the saved v1 caches (route rows: the v1 record/S1 context only).
coll: ber.collective_features, tables over the whole arena (all partitions, text only) / over US/India test.
ed: edit_features blocks; the word false-rate tables are fitted on partition-A candidate rows only and applied
OUT-OF-FOLD on A (5 component groups: an A row gets the table fitted without its group); every non-A row
(B_REST, SLICE, test) gets the table fitted on all of A. SLICE/HOLDOUT-2 labels are never read.
Writes prep/l1/{arena,test}_{v1,coll,ed}.npy and prep/l1/a_group.npy (per arena S1; -1 outside A).
"""
import hashlib
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import edit_features as EF  # noqa: E402
from ber.collective_features import build_tables, pair_features  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
L1 = P / "l1"
W = int(os.environ.get("U_WORKERS", "30"))
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def v1_block():
    for split, cache, n_main, src in (("arena", "arena_v1_cache.npz", None, "brest_u8f"),
                                      ("test", "test_v1_cache.npz", None, "test_u8f")):
        out = L1 / f"{split}_v1.npy"
        if out.exists():
            continue
        with np.load(ROOT / "podbackup" / cache) as c:
            v1x, cl, cr = c["v1x"], c["left"], c["right"]
        tl, tr = np.load(L1 / f"targets_{split}_left.npy"), np.load(L1 / f"targets_{split}_right.npy")
        n = len(cl)
        if not (np.array_equal(cl, tl[:n]) and np.array_equal(cr, tr[:n])):
            raise SystemExit(f"{split}: v1 cache rows differ from targets")
        flag = np.load(P / src / "x_route.npy")[:, 0] == 1
        route = np.load(P / src / "x_base.npy", mmap_mode="r")[np.flatnonzero(flag), :12]
        np.save(out, np.vstack([v1x.astype(np.float32), np.asarray(route, np.float32)]))
        log(f"v1 {split}: {n} + {len(route)} rows")


def coll_block():
    out = L1 / "arena_coll.npy"
    if not out.exists():
        ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["business_name", "business_address"])
        pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["business_name", "business_address"])
        t = build_tables(ref.column("business_name"), ref.column("business_address"), pool.column("business_name"),
                         pool.column("business_address"), log=log)
        al, ar = np.load(P / "arena_cand/left.npy").astype(np.int64), np.load(P / "arena_cand/right.npy").astype(np.int64)
        x = pair_features(t, al, ar)
        # parity: the B_REST rows must equal the saved eureka collective block
        part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
        bpos = np.flatnonzero(part[al] == 1)
        saved = np.load(P / "brest/x_coll.npy")
        ok = np.allclose(np.nan_to_num(x[bpos]), np.nan_to_num(saved), atol=1e-5)
        log(f"coll arena: parity with saved B_REST block {ok}")
        if not ok:
            raise SystemExit("collective parity failed")
        flag = np.load(P / "brest_u8f/x_route.npy")[:, 0] == 1
        np.save(out, np.vstack([x, np.load(P / "brest_u8f/x_coll.npy")[flag]]))
    out = L1 / "test_coll.npy"
    if not out.exists():
        flag = np.load(P / "test_u8f/x_route.npy")[:, 0] == 1
        np.save(out, np.vstack([np.load(P / "test/x_coll.npy"), np.load(P / "test_u8f/x_coll.npy")[flag]]))
    log("coll done")


def a_groups():
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    comp = np.load(ROOT / "data/arena/finalize_v1/component_ids.npy").astype(np.int64)
    uc, inv = np.unique(comp, return_inverse=True)
    g = np.asarray([int(hashlib.sha256(f"l1a:{int(c)}".encode()).hexdigest()[:8], 16) % 5 for c in uc], np.int8)[inv]
    g = np.where(part == 0, g, -1).astype(np.int8)
    np.save(L1 / "a_group.npy", g)
    return part, g


def ed_block():
    if (L1 / "arena_ed.npy").exists() and (L1 / "test_ed.npy").exists():
        return
    ctx = get_context("fork")
    part, grp = a_groups()
    al, ar = np.load(L1 / "targets_arena_left.npy"), np.load(L1 / "targets_arena_right.npy")
    crow = np.load(P / "arena_cand/cand_rows.npy")
    n_cand = len(crow)
    S, R, pairs = EF.side_tables(ROOT / "data/arena/reference.parquet", ROOT / "data/arena/pool.parquet", al, ar)
    log(f"arena sides {len(S)} S1s, {len(R)} records")
    label = np.load(ROOT / "data/arena/labels/label.npy", mmap_mode="r")
    a_rows = np.flatnonzero((part[al[:n_cand]] == 0))
    y_a = np.asarray(label[crow[a_rows]]).astype(bool)
    g_rows = grp[al[a_rows]]
    counters = []
    for g in range(5):
        sel = a_rows[g_rows == g]
        EF.G.update(S=S, R=R, l=[pairs[i] for i in sel], y=y_a[g_rows == g])
        tot = [Counter(), Counter(), Counter(), Counter()]
        with ctx.Pool(W) as p:
            for parts in p.imap_unordered(EF._fit_chunk, EF.chunks(len(sel))):
                for acc, c in zip(tot, parts):
                    acc.update(c)
        counters.append(tot)
        log(f"A group {g}: {len(sel)} rows counted")
    full = [sum((c[j] for c in counters), Counter()) for j in range(4)]

    def tables(excl):
        na, fa, nr, fr = (full[j] - counters[excl][j] if excl is not None else full[j] for j in range(4))
        pa_ = sum(fa.values()) / max(1, sum(na.values()))
        pr_ = sum(fr.values()) / max(1, sum(nr.values()))
        return EF.table_rate(na, fa, pa_), EF.table_rate(nr, fr, pr_), pa_, pr_

    def run(pairs_sel, S_, R_, tabs):
        ta, tr_, pa_, pr_ = tabs
        EF.G.update(S=S_, R=R_, pairs=pairs_sel, ta=ta, tr=tr_, pa=pa_, pr=pr_)
        ed = np.full((len(pairs_sel), len(EF.ED)), np.nan, np.float32)
        with ctx.Pool(W) as p:
            for s, e_, _ in p.imap_unordered(EF._feat_chunk, EF.chunks(len(pairs_sel))):
                ed[s:s + len(e_)] = e_
        return ed

    ed = np.full((len(al), len(EF.ED)), np.nan, np.float32)
    row_grp = np.where(np.arange(len(al)) < n_cand, grp[al], -1)
    for g in range(5):
        sel = np.flatnonzero(row_grp == g)
        ed[sel] = run([pairs[i] for i in sel], S, R, tables(g))
        log(f"arena ed group {g}: {len(sel)} rows")
    full_tabs = tables(None)
    sel = np.flatnonzero(row_grp < 0)
    ed[sel] = run([pairs[i] for i in sel], S, R, full_tabs)
    np.save(L1 / "arena_ed.npy", ed)
    json.dump({"added_prior": full_tabs[2], "removed_prior": full_tabs[3], "added": full_tabs[0],
               "removed": full_tabs[1], "a_rows": int(len(a_rows))}, open(L1 / "edit_word_tables_fullA.json", "w"))
    log("arena ed saved")
    del S, R, pairs
    # test: US/India prep rows + route rows (route records beyond the dense pool come from the route set pool)
    tl, tr = np.load(L1 / "targets_test_left.npy"), np.load(L1 / "targets_test_right.npy")
    meta = json.loads((P / "test/meta.json").read_text())
    country = np.asarray(meta["countries"])[np.load(P / "test/country_code.npy")]
    n_dense = meta["n_rec"]
    ui = np.flatnonzero(np.isin(country[tl], ["US", "India"]))
    ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["business_name", "business_address"])
    pool = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["business_name", "business_address"])
    extra_ids = json.loads((L1 / "extra_rec_ids.json").read_text())
    sp = pq.read_table(ROOT / "u8src/test/pool.parquet", columns=["entity_id", "business_name", "business_address"])
    pos = {v: i for i, v in enumerate(sp.column("entity_id").to_pylist())}
    ex = pa.array([pos[v] for v in extra_ids])
    names = pa.chunked_array(pool.column("business_name").chunks + sp.column("business_name").take(ex).chunks)
    addrs = pa.chunked_array(pool.column("business_address").chunks + sp.column("business_address").take(ex).chunks)
    us, si = np.unique(tl[ui], return_inverse=True)
    ur, ri = np.unique(tr[ui], return_inverse=True)
    S2 = EF.prep_side(ref.column("business_name").take(pa.array(us)), ref.column("business_address").take(pa.array(us)))
    R2 = EF.prep_side(names.take(pa.array(ur)), addrs.take(pa.array(ur)))
    log(f"test sides {len(S2)} S1s, {len(R2)} records (dense pool {n_dense})")
    ted = np.full((len(tl), len(EF.ED)), np.nan, np.float32)
    ted[ui] = run(list(zip(si.tolist(), ri.tolist())), S2, R2, full_tabs)
    np.save(L1 / "test_ed.npy", ted)
    log("test ed saved")


if __name__ == "__main__":
    v1_block()
    coll_block()
    ed_block()
    log("ALL_DONE")
