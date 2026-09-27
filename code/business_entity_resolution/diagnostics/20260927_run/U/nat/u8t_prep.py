"""U8t prep (27 Sep, Agent U): add the native-script dictionary pairs (address rank 0, new vs dense / routes / R-reverse)
to the u8r splits.

    python u8t_prep.py brest|test

New rows: C2/Qwen/L1 NaN; v1 record/S1 context (dense-pair blend statistics; records outside the pool: no claimant);
collective with the split's tables (NaN for records outside the pool); edit-op on the TRANSLITERATED record name vs the
S1 name (B_REST word table); route 0; rev3 0; new `nat` block = [flag, address score, address rank, log1p block size,
translit-name ratio, translit-name token-set ratio, address token-set ratio] (other rows: flag 0, rest NaN).
B_REST labels for new rows come from the ground truth of B_REST S1s (nat_dict.py arena). Writes prep/<split>_u8t/.
"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nat_dict import N, existing_keys_arena, texts, translit  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
NAT = ["nat_flag", "nat_addr_score", "nat_addr_rank", "nat_log1p_block", "nat_name_ratio", "nat_name_tsr",
       "nat_addr_tsr"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def test_existing_keys(n_rec_raw=10_000_000):
    """(test S1 idx, raw test record row) keys of test dense top-50, all Track R routes and R's test reverse pairs."""
    pool_raw = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["raw_row_idx"]).column(
        "raw_row_idx").to_numpy()
    l, r = np.load(ROOT / "data/test/index/left.npy"), np.load(ROOT / "data/test/index/right.npy")
    keys = [l.astype(np.int64) * 20_000_000 + pool_raw[r]]
    sp = pq.read_table(ROOT / "u8src/test/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    rl, rr = np.load(ROOT / "u8src/test/left.npy"), np.load(ROOT / "u8src/test/right.npy")
    keys.append(rl.astype(np.int64) * 20_000_000 + sp[rr])
    rv = pq.read_table(N / "rev/r_reverse_pairs_test.parquet", columns=["left_idx", "raw_row_idx"])
    keys.append(rv.column("left_idx").to_numpy().astype(np.int64) * 20_000_000 + rv.column("raw_row_idx").to_numpy())
    return np.unique(np.concatenate(keys))


def main(split):
    from ber.collective_features import build_tables, pair_features
    from ber.forensic_features import EMPTY, address_kinds
    from ber.pair_tree_arena import load_scores, top_two
    import edit_features as EF
    arena = split == "brest"
    src, dst = P / f"{split}_u8r", P / f"{split}_u8t"
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    n_rec0 = meta["n_rec"]
    f = pd.read_parquet(N / ("arena_candidates.parquet" if arena else "test_candidates.parquet"))
    f = f[f["rank"] == 0]
    if arena:
        part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
        f = f[f.new & (part[f.s1.to_numpy()] == 1)]
    else:
        k = f.s1.to_numpy().astype(np.int64) * 20_000_000 + f.raw.to_numpy()
        f = f[~np.isin(k, test_existing_keys())]
    log(f"{split}: {len(f)} new native-script rank-0 rows")
    # record indices: pool rows, then the split's existing extras, then new extras (raw order)
    pool_path = ROOT / ("data/arena/pool.parquet" if arena else "data/test/dense_test/pool.parquet")
    praw = pq.read_table(pool_path, columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    raw_to_pool = np.full(int(max(praw.max(), f.raw.max())) + 1, -1, np.int64)
    raw_to_pool[praw] = np.arange(len(praw))
    extra = json.loads((src / "extra_rec_ids.json").read_text()) if (src / "extra_rec_ids.json").exists() else []
    base_n = n_rec0 - len(extra)
    known = {v: base_n + i for i, v in enumerate(extra)}
    ids, names, addrs, _ = texts("train_pool" if arena else "test_pool")
    R = raw_to_pool[f.raw.to_numpy()]
    new_ids, new_raw = [], []
    for j in np.flatnonzero(R < 0):
        rid = ids[f.raw.iloc[j]]
        if rid not in known:
            known[rid] = n_rec0 + len(new_ids)
            new_ids.append(rid)
            new_raw.append(int(f.raw.iloc[j]))
        R[j] = known[rid]
    n_rec = n_rec0 + len(new_ids)
    L = f.s1.to_numpy().astype(np.int64)
    log(f"{split}: records in pool {int((f.raw.map(lambda x: raw_to_pool[x] >= 0)).sum())}; new extras {len(new_ids)}")
    # v1 context over the split's dense pairs
    if arena:
        t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_idx"])
        pl, pr = t.column("left_idx").to_numpy().astype(np.int64), t.column("right_idx").to_numpy().astype(np.int64)
        blend = load_scores(ROOT / "data/arena/probabilities.npy")
        ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["business_name", "business_address"])
    else:
        pl = np.load(ROOT / "data/test/index/left.npy").astype(np.int64)
        pr = np.load(ROOT / "data/test/index/right.npy").astype(np.int64)
        blend = load_scores(ROOT / "data/test/probabilities.npy")
        ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["business_name", "business_address"])
    n_s1 = int(max(pl.max(), L.max())) + 1
    rec_top1, _ = top_two(pr, blend, n_rec)
    strong = blend >= 0.5
    s1_strong = np.bincount(pl[strong], minlength=n_s1).astype(np.float32)
    rec_strong = np.bincount(pr[strong], minlength=n_rec).astype(np.float32)
    xb0 = np.load(src / "x_base.npy", mmap_mode="r")
    L0 = np.load(src / "left.npy")
    s1m = np.full(n_s1, np.nan, np.float32)
    s1m[L0] = np.asarray(xb0[:, 8])
    rec_addr = [addrs[i] for i in f.raw.to_numpy()]
    kind = address_kinds(pa.array(rec_addr))
    xb = np.full((len(f), xb0.shape[1]), np.nan, np.float32)
    xb[:, 4], xb[:, 5], xb[:, 6], xb[:, 8], xb[:, 9] = s1_strong[L], rec_top1[R], rec_strong[R], s1m[L], kind
    # collective (in-pool records only)
    pool = pq.read_table(pool_path, columns=["business_name", "business_address"])
    xc = np.full((len(f), 25), np.nan, np.float32)
    inpool = R < base_n
    if inpool.any():
        if arena:
            tb = build_tables(ref.column("business_name"), ref.column("business_address"), pool.column("business_name"),
                              pool.column("business_address"), log=log)
            xc[inpool] = pair_features(tb, L[inpool], R[inpool])
        else:
            country = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["country"]).column(
                "country").to_numpy(zero_copy_only=False)
            rc = pq.read_table(pool_path, columns=["country"]).column("country").to_numpy(zero_copy_only=False)
            s1_sel = np.flatnonzero(np.isin(country, ["US", "India"]))
            rec_sel = np.flatnonzero(np.isin(rc, ["US", "India"]))
            s1_map = np.full(len(country), -1, np.int64)
            s1_map[s1_sel] = np.arange(len(s1_sel))
            rec_map = np.full(len(rc), -1, np.int64)
            rec_map[rec_sel] = np.arange(len(rec_sel))
            tk = lambda c, idx: c.take(pa.array(idx))  # noqa: E731
            tb = build_tables(tk(ref.column("business_name"), s1_sel), tk(ref.column("business_address"), s1_sel),
                              tk(pool.column("business_name"), rec_sel), tk(pool.column("business_address"), rec_sel),
                              log=log)
            ok = inpool.copy()
            ok[inpool] = rec_map[R[inpool]] >= 0
            xc[ok] = pair_features(tb, s1_map[L[ok]], rec_map[R[ok]])
    # edit-op on the transliterated record name (B_REST word table)
    d = json.load(open(N / "dictionary.json"))["dictionary"]
    tr_names = [translit(names[i], d) for i in f.raw.to_numpy()]
    tables = json.loads((P / "edit_word_tables.json").read_text())
    us, si = np.unique(L, return_inverse=True)
    S = EF.prep_side(ref.column("business_name").take(pa.array(us)), ref.column("business_address").take(pa.array(us)))
    Rs = EF.prep_side(pa.array(tr_names), pa.array(rec_addr))
    EF.G.update(S=S, R=Rs, pairs=list(zip(si.tolist(), range(len(f)))), ta=tables["added"], tr=tables["removed"],
                pa=tables["added_prior"], pr=tables["removed_prior"])
    _, ed, _ = EF._feat_chunk((0, len(f)))
    nat = np.column_stack([np.ones(len(f)), f.score.to_numpy(), f["rank"].to_numpy(), np.log1p(f.group.to_numpy()),
                           f.nat_ratio.to_numpy(), f.nat_tsr.to_numpy(), f.addr_tsr.to_numpy()]).astype(np.float32)
    n0 = len(L0)
    nat0 = np.full((n0, len(NAT)), np.nan, np.float32)
    nat0[:, 0] = 0
    blocks = {"x_base": xb, "x_q": np.full((len(f), 11), np.nan, np.float32), "x_coll": xc, "x_ed": ed,
              "x_route": np.zeros((len(f), 5), np.float32), "x_l1v2": np.full((len(f), 2), np.nan, np.float32),
              "x_rev3": np.zeros((len(f), 6), np.float32)}
    for k, v in blocks.items():
        np.save(dst / f"{k}.npy", np.vstack([np.load(src / f"{k}.npy"), v.astype(np.float32)]))
    np.save(dst / "x_nat.npy", np.vstack([nat0, nat]))
    np.save(dst / "left.npy", np.r_[L0, L])
    np.save(dst / "right.npy", np.r_[np.load(src / "right.npy"), R])
    np.save(dst / "base.npy", np.r_[np.load(src / "base.npy"), np.full(len(f), np.nan, np.float32)])
    shutil.copy(src / "country_code.npy", dst / "country_code.npy")
    if arena:
        np.save(dst / "label.npy", np.r_[np.load(src / "label.npy"), f.true.to_numpy()])
        mult0 = np.load(src / "mult.npy")
        sm = np.ones(n_s1, mult0.dtype)
        sm[L0] = mult0
        np.save(dst / "mult.npy", np.r_[mult0, sm[L]])
        np.save(dst / "rec_empty.npy", np.r_[np.load(src / "rec_empty.npy"), kind == EMPTY])
        np.save(dst / "name_exact.npy", np.r_[np.load(src / "name_exact.npy"), np.zeros(len(f), bool)])
        for x in ("true_count", "population", "component", "partition"):
            shutil.copy(src / f"{x}.npy", dst / f"{x}.npy")
    for g in src.glob("names_*.json"):
        shutil.copy(g, dst / g.name)
    (dst / "names_nat.json").write_text(json.dumps(NAT))
    (dst / "extra_rec_ids.json").write_text(json.dumps(extra + new_ids))
    (dst / "meta.json").write_text(json.dumps({**meta, "n_rec": int(n_rec), "rows": int(n0 + len(f)), "nat_rows": int(len(f)),
                                               "nat_true": int(f.true.sum()) if arena else None,
                                               "nat_new_records": len(new_ids),
                                               "ed_on": "transliterated record name"}) + "\n")
    log(f"{split}_u8t saved: {n0 + len(f)} rows ({len(f)} native-script)")


if __name__ == "__main__":
    main(sys.argv[1])
