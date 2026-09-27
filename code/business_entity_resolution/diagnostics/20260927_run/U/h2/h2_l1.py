"""HOLDOUT-2 final check: Agent G string features and the A-trained L1 (v3 capacity models) on HOLDOUT-2 rows.

    python h2_l1.py gparity      # G's current code on 200k B_REST rows vs the out1 values used to train L1
    python h2_l1.py gfeat        # G features for HOLDOUT-2 dense + route rows (joint S1 multiplicity)
    python h2_l1.py l1           # L1 stage 1 / listwise (joint) / stage 2 -> H2/join/h2_l1v2.npy ([full, nov1])
Label-free.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
L1 = P / "l1"
H2 = ROOT / os.environ.get("U_H2DIR", "h2")
J = H2 / "join"
N_ARENA = 2_072_821
W = int(os.environ.get("U_WORKERS", "28"))
COLS = ["entity_id", "business_name", "business_address", "country"]
CTX = [4, 5, 6, 8, 9]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def g_features(ref, pool, left, right, path):
    from ber.string_gbdt import Table, pair_features, s1_multiplicity, source_s3
    table = Table(ref.business_name.to_numpy(object), ref.business_address.to_numpy(object),
                  pool.business_name.to_numpy(object), pool.business_address.to_numpy(object), W)
    mult = s1_multiplicity(table, ref.country.to_numpy(object))
    rec_s3 = source_s3(pool.entity_id)
    order = np.lexsort((right, left))
    pair_features(table, left[order], right[order], mult, rec_s3, path, W)
    x = np.load(path)
    out = np.empty_like(x)
    out[order] = x
    np.save(path, out)
    return out


def gparity():
    ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=COLS).to_pandas()
    pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=COLS).to_pandas()
    al, ar = np.load(L1 / "targets_arena_left.npy"), np.load(L1 / "targets_arena_right.npy")
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    rows = np.flatnonzero(part[al[:15_506_727]] == 1)
    rows = np.sort(np.random.default_rng(0).choice(rows, 200_000, replace=False))
    x = g_features(ref, pool, al[rows], ar[rows], J / "gparity_X.npy")
    old = np.asarray(np.load(L1 / "g/g_arena.npy", mmap_mode="r")[rows])
    diff = ~np.isclose(np.nan_to_num(x, nan=-9), np.nan_to_num(old, nan=-9), atol=1e-5)
    names = json.loads((L1 / "g/g_meta.json").read_text())["features"]
    rep = {"rows": int(len(rows)), "rows_any_diff": int(diff.any(1).sum()),
           "per_feature_diff": {n: int(v) for n, v in zip(names, diff.sum(0)) if v}}
    (J / "gparity.json").write_text(json.dumps(rep, indent=1) + "\n")
    log(json.dumps(rep))


def gfeat():
    b = np.load(J / "h2_blocks.npz")
    rows = np.flatnonzero(b["kind"] < 2)
    aref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=COLS).to_pandas()
    ref = pd.concat([aref, pd.read_parquet(H2 / "reference.parquet", columns=COLS)], ignore_index=True)
    pool = pd.read_parquet(J / "joint_pool.parquet", columns=COLS)
    x = g_features(ref, pool, b["left"][rows], b["right"][rows], J / "h2_g_rows.npy")
    full = np.full((len(b["kind"]), x.shape[1]), np.nan, np.float32)
    full[rows] = x
    np.save(J / "h2_g.npy", full)
    log(f"G features for {len(rows)} HOLDOUT-2 dense+route rows")


def listwise(p, left, right):
    from l1_train import listwise as lw
    n_s1, n_rec = int(left.max()) + 1, int(right.max()) + 1
    return lw(p, left, right, n_s1, n_rec)


def l1():
    from catboost import CatBoostClassifier
    import edit_features as EF
    b = np.load(J / "h2_blocks.npz")
    kind, L, R = b["kind"], b["left"], b["right"]
    g = np.load(J / "h2_g.npy")
    # edit-op with the full-A tables (as the L1 training / B_REST L1 inputs)
    import pyarrow as pa
    aref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=COLS).to_pandas()
    ref = pd.concat([aref, pd.read_parquet(H2 / "reference.parquet", columns=COLS)], ignore_index=True)
    full = pd.read_parquet(J / "joint_pool.parquet", columns=COLS)  # L1 rows (dense + route) never reach orphans
    tab = json.loads((L1 / "edit_word_tables_fullA.json").read_text())
    rows = np.flatnonzero(kind < 2)
    EF_tables = tab
    import edit_features
    edit_features.G.update(ta=EF_tables["added"], tr=EF_tables["removed"], pa=EF_tables["added_prior"],
                           pr=EF_tables["removed_prior"])
    us, si = np.unique(L[rows], return_inverse=True)
    ur, ri = np.unique(R[rows], return_inverse=True)
    S = EF.prep_side(pa.array(ref.business_name.iloc[us].fillna("")), pa.array(ref.business_address.iloc[us].fillna("")))
    RR = EF.prep_side(pa.array(full.business_name.iloc[ur].fillna("")), pa.array(full.business_address.iloc[ur].fillna("")))
    EF.G.update(S=S, R=RR, pairs=list(zip(si.tolist(), ri.tolist())))
    _, ed, _ = EF._feat_chunk((0, len(rows)))
    v1 = b["v1"][rows]
    coll = b["coll"][rows]
    out = np.full((len(kind), 2), np.nan, np.float32)
    # arena dense p1 (A: OOF; others: mean) for the joint listwise
    al = np.load(L1 / "targets_arena_left.npy")
    ar = np.load(L1 / "targets_arena_right.npy")
    n_cand = json.loads((L1 / "targets.json").read_text())["arena_cand"]
    apool_n = pq.ParquetFile(ROOT / "data/arena/pool.parquet").metadata.num_rows
    for j, v in enumerate(("full", "nov1")):
        vv = v1 if v == "full" else v1[:, CTX]
        x = np.hstack([g[rows], vv, coll, ed]).astype(np.float32)
        p1 = np.zeros(len(rows), np.float32)
        for f in (0, 1):
            m = CatBoostClassifier()
            m.load_model(str(L1 / f"l1_{v}f100c_s1_fold{f}.cbm"))
            p1 += m.predict_proba(x)[:, 1].astype(np.float32) / 2
        pa1 = np.load(L1 / f"p1_{v}f100c_arena.npy")[:n_cand]
        # records: arena rows use the arena pool index; HOLDOUT-2 rows the joint pool index (identical below
        # apool_n, HOLDOUT-2-only records above it) -> one record index space
        dense = kind[rows] == 0
        LL = np.r_[al[:n_cand], L[rows][dense]]
        RR2 = np.r_[ar[:n_cand], R[rows][dense]]
        pp = np.r_[pa1, p1[dense]]
        lw_dense = listwise(pp, LL, RR2)[n_cand:]
        LL_u = np.r_[LL, L[rows][~dense]]
        RR_u = np.r_[RR2, R[rows][~dense]]
        pp_u = np.r_[pp, p1[~dense]]
        lw_route = listwise(pp_u, LL_u, RR_u)[len(LL):] if (~dense).any() else np.zeros((0, 9), np.float32)
        lwall = np.empty((len(rows), 9), np.float32)
        lwall[dense], lwall[~dense] = lw_dense, lw_route
        m2 = CatBoostClassifier()
        m2.load_model(str(L1 / f"l1_{v}f100c_s2.cbm"))
        p2 = m2.predict_proba(np.hstack([x, p1[:, None], lwall]))[:, 1].astype(np.float32)
        col = np.full(len(kind), np.nan, np.float32)
        col[rows] = p2
        if v == "full":
            col[kind != 0] = np.nan  # no v1 pair scores on route rows (as B_REST)
        out[:, j] = col
        log(f"L1 {v}: {len(rows)} rows scored")
    np.save(J / "h2_l1v2.npy", out)


if __name__ == "__main__":
    {"gparity": gparity, "gfeat": gfeat, "l1": l1}[sys.argv[1]]()
