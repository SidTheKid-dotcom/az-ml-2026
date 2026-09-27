"""Route rows (Agent U's U8f Track-R pairs) as extra K candidate rows, in K-integrate's column layout.

For B_REST (689,474 rows, arena compact pool) and test US/India (1,578,869 rows; 47,944 point outside the dense
test pool) this builds, per route row: the K-robust base columns (v1 context from U's ``v1ctx_*`` mapped onto the
matching x_base columns; C2/Qwen and pair-level v1 scores NaN, as U does), K3 edit-op features + K3 probability
(A-trained models), density-robust claimant/S1-side features computed over dense + route claimants, U's ``ed_*`` /
``coll_*`` blocks, and a ``route`` block (``is_route`` + U's route scores/ranks). B_REST labels come from the arena
record owner (``rec_owner[right] == left``; B_REST S1s only). Cached as .npz per split.

    python -m ber.k_routes build --split brest|test
"""
from __future__ import annotations

import argparse
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber import edit_ops as eo
from ber import record_assign as ra

T0 = time.time()
DATA = ra.DATA
UF = DATA / "ufeat"
CACHE = Path("/home/ubuntu/k/cache")
ROUTE_COLS = ["route_flag", "route_addr_score", "route_addr_rank", "route_empty_score", "route_empty_rank"]
V1CTX = {"v1ctx_s1_candidates_ge_0_5": "s1_candidates_ge_0_5", "v1ctx_rival_blend": "rival_blend",
         "v1ctx_record_claimants_ge_0_5": "record_claimants_ge_0_5", "v1ctx_log1p_name_mult": "log1p_name_mult",
         "v1ctx_record_kind": "record_kind"}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def k3_features(s_names, s_addrs, r_names, r_addrs, workers=30, chunk=50_000):
    jobs = [(s_names[i:i + chunk], s_addrs[i:i + chunk], r_names[i:i + chunk], r_addrs[i:i + chunk])
            for i in range(0, len(s_names), chunk)]
    with get_context("fork").Pool(workers) as pool:
        parts = pool.map(eo.chunk_features, jobs)
    return np.concatenate(parts) if parts else np.zeros((0, eo.NF), np.float32)


def k3_prob(feat, rec_s3):
    import lightgbm as lgb

    models = [lgb.Booster(model_file=str(Path("/home/ubuntu/k/out/k3") / f"k3_half{h}.txt")) for h in (0, 1)]
    x = np.hstack([feat, rec_s3.astype(np.float32)[:, None]])
    return (0.5 * (models[0].predict(x, num_threads=30) + models[1].predict(x, num_threads=30))).astype(np.float32)


def build(split, parquet=None, tag="routes"):
    CACHE.mkdir(parents=True, exist_ok=True)
    R = pq.read_table(parquet or (UF / f"u_route_pairs_{split}.parquet"))
    cols = R.column_names
    rl = R.column("left_idx").to_numpy().astype(np.int64)
    rec_id = np.asarray(R.column("rec_id").to_pylist(), dtype=object)
    if split == "brest":
        ref_path, pool_path = DATA / "arena/reference.parquet", DATA / "arena/pool.parquet"
    else:
        ref_path, pool_path = DATA / "test/reference.parquet", DATA / "test/pool.parquet"
    s_names, s_addrs = eo.load_text(ref_path)
    pool_ids = pq.read_table(pool_path, columns=["entity_id"]).column("entity_id").to_pylist()
    p_names, p_addrs = eo.load_text(pool_path)
    index = {v: i for i, v in enumerate(pool_ids)}
    rr = np.asarray([index.get(v, -1) for v in rec_id], dtype=np.int64)
    outside = rr < 0
    r_names = np.empty(len(rr), dtype=object)
    r_addrs = np.empty(len(rr), dtype=object)
    r_names[~outside] = p_names[rr[~outside]]
    r_addrs[~outside] = p_addrs[rr[~outside]]
    if outside.any():  # test records outside the dense pool: text from U's route search pool
        extra = pq.read_table(DATA / "u8src/test_pool.parquet", columns=["entity_id", "business_name",
                                                                          "business_address"])
        eidx = {v: i for i, v in enumerate(extra.column("entity_id").to_pylist())}
        en = np.asarray(extra.column("business_name").to_pylist(), dtype=object)
        ea = np.asarray(extra.column("business_address").to_pylist(), dtype=object)
        pos = np.asarray([eidx.get(v, -1) for v in rec_id[outside]], dtype=np.int64)
        if np.any(pos < 0):
            raise SystemExit(f"{int((pos < 0).sum())} outside-pool route records have no text")
        r_names[outside] = en[pos]
        r_addrs[outside] = ea[pos]
    log(f"{split}: {len(rr)} route rows, {int(outside.sum())} outside the dense pool")
    feat = k3_features(s_names[rl].tolist(), s_addrs[rl].tolist(), r_names.tolist(), r_addrs.tolist())
    rec_s3 = np.fromiter((v.startswith("S3") for v in rec_id), dtype=bool, count=len(rec_id))
    k3 = k3_prob(feat, rec_s3)
    log("K3 features and probability done")
    # claimant features over dense + route claimants; route records outside the pool get fresh ids
    if split == "brest":
        dl = np.load(DATA / "arena_cand/left.npy").astype(np.int64)
        dr = np.load(DATA / "arena_cand/right.npy").astype(np.int64)
        cand_rows = np.load(DATA / "arena_cand/cand_rows.npy")
        dbase = np.asarray(np.load(DATA / "arena/base_probabilities.npy", mmap_mode="r")[cand_rows], dtype=np.float32)
        dk3 = np.load(Path("/home/ubuntu/k/out/k3/k3_prob_arena_cand.npy"))
        n_s1 = len(np.load(DATA / "arena/s1_partition.npy"))
    else:
        d = DATA / "test_prep"
        dl = np.load(d / "left.npy").astype(np.int64)
        dr = np.load(d / "right.npy").astype(np.int64)
        dbase = np.asarray(np.load(d / "x_base.npy", mmap_mode="r")[:, ra.BASE_NAMES.index("base")], dtype=np.float32)
        dk3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
        n_s1 = len(np.load(d / "country_code.npy"))
    n_pool = len(pool_ids)
    ext = rr.copy()
    if outside.any():
        _, inv = np.unique(rec_id[outside], return_inverse=True)
        ext[outside] = n_pool + inv
    n_rec = int(max(n_pool, ext.max() + 1))
    all_l = np.concatenate([dl, rl])
    all_r = np.concatenate([dr, ext])
    all_b = np.concatenate([dbase, np.zeros(len(rl), np.float32)])
    all_k = np.concatenate([dk3, k3])
    rows = np.arange(len(dl), len(all_l))
    claim = ra.claimant_features(all_r, all_l, all_b, all_k, np.zeros(len(all_l), bool), n_rec, n_s1, rows,
                                 rec_s3.astype(np.float32), np.full(len(rl), -1.0, np.float32),
                                 np.full(len(rl), -1.0, np.float32), ext)
    # x_base / x_q blocks: NaN except the v1 context columns
    xb = np.full((len(rl), len(ra.BASE_NAMES)), np.nan, dtype=np.float32)
    for src, dst in V1CTX.items():
        xb[:, ra.BASE_NAMES.index(dst)] = R.column(src).to_numpy(zero_copy_only=False).astype(np.float32)
    xq = np.full((len(rl), len(ra.Q_NAMES)), np.nan, dtype=np.float32)
    full = np.hstack([xb, xq, feat, claim])  # record_assign.NAMES layout
    fam = {c: R.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in cols
           if c.startswith("ed_") or c.startswith("coll_") or c in ROUTE_COLS}
    out = {"x_names": np.asarray(ra.NAMES), "x": full, "left": rl, "right": rr, "right_ext": ext,
           "rec_id": rec_id.astype(str), "fam_names": np.asarray(list(fam)),
           "fam": np.column_stack(list(fam.values())).astype(np.float32)}
    if split == "brest":
        out["fold"] = R.column("fold").to_numpy().astype(np.int64)
        owner = np.load(DATA / "labels/rec_owner.npy")
        out["label"] = owner[rr] == rl
        part = np.load(DATA / "arena/s1_partition.npy")
        if not np.all(part[rl] == 1):
            raise SystemExit("B_REST route rows include non-B_REST S1s")
        log(f"B_REST route rows true: {int(out['label'].sum())}")
    np.savez(CACHE / f"{tag}_{split}.npz", **out)
    log(f"saved {CACHE / f'{tag}_{split}.npz'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["build", "build_rev"])
    p.add_argument("--split", choices=["brest", "test"], required=True)
    p.add_argument("--parquet", help="route-pair parquet (default: U's U8f file for the split)")
    p.add_argument("--tag", default="routes", help="cache name prefix")
    a = p.parse_args()
    if a.command == "build_rev":
        build_rev(a.split)
    else:
        build(a.split, a.parquet, a.tag)



# ----------------------------------------------------------------------------- Agent R reverse pairs

REV_COLS = ["rev_flag", "rev_rank", "rev_rec_fwd_claimants", "rev_orphan", "rev_top1_gap", "rev_cos_minus_s1fwd50"]


def build_rev(split):
    """Reverse rows (R's rank1|orphan<=10 pairs; test: US/India only) in the same cache layout as routes.

    Record text: the (arena compact / dense test) pool when the record is there, else R's raw text parquet
    (``name | address`` split at the first " | "). ``ed_*`` comes from U's u8r parquet (joined by ids); the
    ``rev`` block is U's rev3 (no absolute cosines). B_REST labels: R's ``label`` column (B_REST S1s only).
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    name = "b_rest" if split == "brest" else "test"
    R = pq.read_table(DATA / f"rpairs/r_reverse_pairs_{name}.parquet").to_pandas()
    R["src_order"] = np.arange(len(R))
    if split == "test":
        R = R[R.country != "France"].reset_index(drop=True)
    ref_path = DATA / ("arena/reference.parquet" if split == "brest" else "test/reference.parquet")
    pool_path = DATA / ("arena/pool.parquet" if split == "brest" else "test/pool.parquet")
    s_names, s_addrs = eo.load_text(ref_path)
    pool_ids = pq.read_table(pool_path, columns=["entity_id"]).column("entity_id").to_pylist()
    p_names, p_addrs = eo.load_text(pool_path)
    index = {v: i for i, v in enumerate(pool_ids)}
    rl = R.left_idx.to_numpy().astype(np.int64)
    rec_id = R.rec_id.to_numpy().astype(object)
    rr = np.asarray([index.get(v, -1) for v in rec_id], dtype=np.int64)
    outside = rr < 0
    r_names = np.empty(len(rr), dtype=object)
    r_addrs = np.empty(len(rr), dtype=object)
    r_names[~outside] = p_names[rr[~outside]]
    r_addrs[~outside] = p_addrs[rr[~outside]]
    if outside.any():
        raw = pq.read_table(DATA / f"rtexts/{'train' if split == 'brest' else 'test'}_pool.parquet",
                            columns=["entity_id", "text"]).to_pandas().drop_duplicates("entity_id").set_index("entity_id")
        txt = raw.text.reindex(rec_id[outside]).fillna("").to_numpy()
        parts = [t.split(" | ", 1) for t in txt]
        r_names[outside] = [p[0] for p in parts]
        r_addrs[outside] = [p[1] if len(p) > 1 else "" for p in parts]
    log(f"{split}: {len(rr)} reverse rows, {int(outside.sum())} to records outside the pool")
    feat = k3_features(s_names[rl].tolist(), s_addrs[rl].tolist(), [x or "" for x in r_names],
                       [x or "" for x in r_addrs])
    rec_s3 = np.fromiter((v.startswith("S3") for v in rec_id), dtype=bool, count=len(rec_id))
    k3 = k3_prob(feat, rec_s3)
    if split == "brest":
        dl = np.load(DATA / "arena_cand/left.npy").astype(np.int64)
        dr = np.load(DATA / "arena_cand/right.npy").astype(np.int64)
        cand_rows = np.load(DATA / "arena_cand/cand_rows.npy")
        dbase = np.asarray(np.load(DATA / "arena/base_probabilities.npy", mmap_mode="r")[cand_rows], dtype=np.float32)
        dk3 = np.load(Path("/home/ubuntu/k/out/k3/k3_prob_arena_cand.npy"))
        n_s1 = len(np.load(DATA / "arena/s1_partition.npy"))
    else:
        d = DATA / "test_prep"
        dl = np.load(d / "left.npy").astype(np.int64)
        dr = np.load(d / "right.npy").astype(np.int64)
        dbase = np.asarray(np.load(d / "x_base.npy", mmap_mode="r")[:, ra.BASE_NAMES.index("base")], dtype=np.float32)
        dk3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
        n_s1 = len(np.load(d / "country_code.npy"))
    n_pool = len(pool_ids)
    ext = rr.copy()
    if outside.any():
        _, inv = np.unique(rec_id[outside], return_inverse=True)
        ext[outside] = n_pool + inv
    n_rec = int(max(n_pool, ext.max() + 1))
    all_l = np.concatenate([dl, rl])
    all_r = np.concatenate([dr, ext])
    all_b = np.concatenate([dbase, np.zeros(len(rl), np.float32)])
    all_k = np.concatenate([dk3, k3])
    rows = np.arange(len(dl), len(all_l))
    claim = ra.claimant_features(all_r, all_l, all_b, all_k, np.zeros(len(all_l), bool), n_rec, n_s1, rows,
                                 rec_s3.astype(np.float32), np.full(len(rl), -1.0, np.float32),
                                 np.full(len(rl), -1.0, np.float32), ext)
    # U's u8r features for these rows (ed_*, rev_*, v1ctx_*) joined by ids
    U = pq.read_table(DATA / f"u8r/u8r_pairs_{split}.parquet").to_pandas()
    U = U[U.row_type == 2] if "row_type" in U else U  # 0 dense, 1 route, 2 reverse
    U.index = U.s1_id + "|" + U.rec_id
    U = U[~U.index.duplicated()]
    key = R.s1_id + "|" + R.rec_id
    J = U.reindex(key)
    log(f"u8r join: {int(J.p.notna().sum()) if 'p' in J else 0} of {len(J)} reverse rows matched")
    xb = np.full((len(rl), len(ra.BASE_NAMES)), np.nan, dtype=np.float32)
    for src, dst in V1CTX.items():
        if src in J:
            xb[:, ra.BASE_NAMES.index(dst)] = J[src].to_numpy(dtype=np.float32)
    xq = np.full((len(rl), len(ra.Q_NAMES)), np.nan, dtype=np.float32)
    full = np.hstack([xb, xq, feat, claim])
    fam = {c: J[c].to_numpy(dtype=np.float32) for c in J.columns if c.startswith("ed_")}
    fam["rev_flag"] = np.ones(len(rl), np.float32)
    fam["rev_rank"] = R.rev_rank.to_numpy(dtype=np.float32)
    fam["rev_rec_fwd_claimants"] = R.rec_fwd_claimants.to_numpy(dtype=np.float32)
    fam["rev_orphan"] = (R.rec_fwd_claimants.to_numpy() == 0).astype(np.float32)
    fam["rev_top1_gap"] = (R.rec_top1_cos - R.cosine).to_numpy(dtype=np.float32)
    fam["rev_cos_minus_s1fwd50"] = (R.cosine - R.s1_fwd50_cos).to_numpy(dtype=np.float32)
    fam["pu"] = J["p"].to_numpy(dtype=np.float32) if "p" in J else np.full(len(rl), np.nan, np.float32)
    out = {"x_names": np.asarray(ra.NAMES), "x": full, "left": rl, "right": rr, "right_ext": ext,
           "rec_id": rec_id.astype(str), "fam_names": np.asarray(list(fam)),
           "fam": np.column_stack(list(fam.values())).astype(np.float32), "src_order": R.src_order.to_numpy()}
    if split == "brest":
        folds = pq.read_table(UF / "u_features_brest.parquet", columns=["left_idx", "fold"]).to_pandas()
        comp = np.load(DATA / "arena/component_ids.npy")
        folds["comp"] = comp[folds.left_idx.to_numpy()]
        cmap = folds.drop_duplicates("comp").set_index("comp").fold
        f = cmap.reindex(comp[rl]).to_numpy()
        miss = np.isnan(f.astype(float))
        f[miss] = comp[rl][miss] % 5  # components without dense rows: any fixed component-level split
        out["fold"] = f.astype(np.int64)
        log(f"reverse rows fold from U's component map; {int(miss.sum())} rows on components without dense rows")
        out["label"] = R.label.to_numpy().astype(bool)
        log(f"B_REST reverse rows true: {int(out['label'].sum())}")
    np.savez(CACHE / f"rev_{split}.npz", **out)
    log(f"saved {CACHE / f'rev_{split}.npz'}")


if __name__ == "__main__":
    main()
