"""U8r (27 Sep, Agent U): add Agent R's reverse-retrieval pairs (rule rank1|orphan<=10) to the U8f splits.

    python u8r_prep.py brest|test

Reverse rows (US/India): C2/Qwen/L1 blocks NaN; v1 block = record/S1 context only (as U8f route rows); collective
from the same tables (NaN for orphan records, which are outside the compact/dense pool, as for U8f route records
outside the dense pool); edit-op block with the same word tables (orphan text from R's raw text parquet); route block
zeros; new block `rev` = [is_rev, rev_rank, cosine, rec_top1_cos, s1_fwd50_cos, rec_fwd_claimants, is_orphan,
rec_top1_cos - cosine]. Orphan records get new record indices after the existing ones (test: shared with U8f route
records outside the dense pool by entity id). Labels: B_REST reverse rows use R's `label` column (B_REST S1s only).
Writes prep/<split>_u8r/.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parent))
from u8_prep import ed_tr  # noqa: E402
from ber.collective_features import build_tables, pair_features  # noqa: E402
from ber.forensic_features import EMPTY, address_kinds  # noqa: E402
from ber.pair_tree_arena import top_two  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
RS = ROOT / "rsrc"
REV = ["rev_flag", "rev_rank", "rev_cosine", "rev_rec_top1_cos", "rev_s1_fwd50_cos", "rev_rec_fwd_claimants",
       "rev_orphan", "rev_top1_gap"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def split_text(texts):
    names, addrs = [], []
    for t in texts:
        t = t[len("passage: "):] if t.startswith("passage: ") else t
        n, _, a = t.partition(" | ")
        names.append(n)
        addrs.append(a)
    return names, addrs


def main(split):
    src, dst = P / f"{split}_u8f", P / f"{split}_u8r"
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "meta.json").read_text())
    n_rec0 = meta["n_rec"]
    r = pq.read_table(RS / f"r_reverse_pairs_{'b_rest' if split == 'brest' else 'test'}.parquet")
    r = r.filter(pa.compute.is_in(r.column("country"), value_set=pa.array(["US", "India"])))
    Lr = r.column("left_idx").to_numpy().astype(np.int64)
    ri = r.column("right_idx").to_numpy().astype(np.int64)
    rec_ids = np.asarray(r.column("rec_id").to_pylist(), dtype=object)
    raw_idx = r.column("raw_row_idx").to_numpy()
    orphan = ri < 0
    # orphan record indices: shared with existing extra records (test) by entity id, else new
    extra_path = src / "extra_rec_ids.json"
    extra = json.loads(extra_path.read_text()) if extra_path.exists() else []
    base_n = n_rec0 - len(extra)
    known = {v: base_n + i for i, v in enumerate(extra)}
    new_ids, new_raw = [], []
    Rr = ri.copy()
    for j in np.flatnonzero(orphan):
        v = rec_ids[j]
        if v not in known:
            known[v] = n_rec0 + len(new_ids)
            new_ids.append(v)
            new_raw.append(int(raw_idx[j]))
        Rr[j] = known[v]
    n_rec = n_rec0 + len(new_ids)
    new_set = set(new_ids)
    to_extra = int(sum(1 for j in np.flatnonzero(orphan) if rec_ids[j] not in new_set))
    log(f"{split}: {len(Lr)} reverse rows ({int(orphan.sum())} to orphans; {len(new_ids)} new records; "
        f"{to_extra} rows to existing extra records)")
    # texts: S1 side and record side (existing records from the pool parquets; new orphans from R's raw texts)
    if split == "brest":
        ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["business_name", "business_address"])
        pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["business_name", "business_address"])
        rtxt = pq.read_table(RS / "texts/train_pool.parquet", columns=["text"]).column("text")
        pool_names, pool_addr = pool.column("business_name"), pool.column("business_address")
        ex_names, ex_addr = [], []
    else:
        ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["business_name", "business_address"])
        pool = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["business_name", "business_address"])
        rtxt = pq.read_table(RS / "texts/test_pool.parquet", columns=["text"]).column("text")
        sp = pq.read_table(ROOT / "u8src/test/pool.parquet", columns=["entity_id", "business_name", "business_address"])
        pos = {v: i for i, v in enumerate(sp.column("entity_id").to_pylist())}
        ex = pa.array([pos[v] for v in extra])
        pool_names, pool_addr = pool.column("business_name"), pool.column("business_address")
        ex_names, ex_addr = sp.column("business_name").take(ex).to_pylist(), sp.column("business_address").take(ex).to_pylist()
    nn, na_ = split_text(rtxt.take(pa.array(new_raw, type=pa.int64())).to_pylist()) if new_raw else ([], [])
    all_names = pa.chunked_array(pool_names.chunks + [pa.array(ex_names + nn, type=pool_names.type)])
    all_addr = pa.chunked_array(pool_addr.chunks + [pa.array(ex_addr + na_, type=pool_addr.type)])
    if len(all_names) != n_rec:
        raise SystemExit(f"record text count {len(all_names)} != {n_rec}")
    ed_r, _ = ed_tr(ref.column("business_name"), ref.column("business_address"), all_names, all_addr, Lr, Rr)
    log("ed done")
    kind_r = address_kinds(all_addr.take(pa.array(Rr)))
    # collective: non-orphans only (same tables as U8f)
    xc_r = np.full((len(Lr), 25), np.nan, np.float32)
    ok = Rr < base_n
    if split == "brest":
        t = build_tables(ref.column("business_name"), ref.column("business_address"), pool_names, pool_addr, log=log)
        xc_r[ok] = pair_features(t, Lr[ok], Rr[ok])
    else:
        tc = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["country"]).column("country")
        rc = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["country"]).column("country")
        s1c, rcc = np.asarray(tc.to_pylist(), dtype=object), np.asarray(rc.to_pylist(), dtype=object)
        s1_sel, rec_sel = np.flatnonzero(np.isin(s1c, ["US", "India"])), np.flatnonzero(np.isin(rcc, ["US", "India"]))
        s1_map = np.full(len(s1c), -1, np.int64)
        s1_map[s1_sel] = np.arange(len(s1_sel))
        rec_map = np.full(n_rec, -1, np.int64)
        rec_map[rec_sel] = np.arange(len(rec_sel))
        tk = lambda c, idx: c.take(pa.array(idx))  # noqa: E731
        t = build_tables(tk(ref.column("business_name"), s1_sel), tk(ref.column("business_address"), s1_sel),
                         tk(pool_names, rec_sel), tk(pool_addr, rec_sel), log=log)
        ok = ok & (rec_map[np.minimum(Rr, n_rec - 1)] >= 0)
        xc_r[ok] = pair_features(t, s1_map[Lr[ok]], rec_map[Rr[ok]])
    log("coll done")
    # v1 record/S1 context from the dense pairs (orphans: no dense claimant)
    if split == "brest":
        tp = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_idx"])
        pl, pr = tp.column("left_idx").to_numpy().astype(np.int64), tp.column("right_idx").to_numpy().astype(np.int64)
        blend = np.nan_to_num(np.load(ROOT / "data/arena/probabilities.npy").astype(np.float32), nan=0.0)
    else:
        pl = np.load(ROOT / "data/test/index/left.npy").astype(np.int64)
        pr = np.load(ROOT / "data/test/index/right.npy").astype(np.int64)
        blend = np.nan_to_num(np.load(ROOT / "data/test/probabilities.npy").astype(np.float32), nan=0.0)
    n_s1 = int(max(pl.max(), Lr.max())) + 1
    rec_top1, _ = top_two(pr, blend, n_rec)
    strong = blend >= 0.5
    s1_strong = np.bincount(pl[strong], minlength=n_s1).astype(np.float32)
    rec_strong = np.bincount(pr[strong], minlength=n_rec).astype(np.float32)
    xb0 = np.load(src / "x_base.npy")
    L0 = np.load(src / "left.npy")
    s1m = np.full(n_s1, np.nan, np.float32)
    s1m[L0] = xb0[:, 8]
    xb_r = np.full((len(Lr), xb0.shape[1]), np.nan, np.float32)
    xb_r[:, 4], xb_r[:, 5], xb_r[:, 6] = s1_strong[Lr], rec_top1[Rr], rec_strong[Rr]
    xb_r[:, 8], xb_r[:, 9] = s1m[Lr], kind_r
    cols = {c: r.column(c).to_numpy().astype(np.float32) for c in ("rev_rank", "cosine", "rec_top1_cos",
                                                                    "s1_fwd50_cos", "rec_fwd_claimants")}
    rev_r = np.column_stack([np.ones(len(Lr), np.float32), cols["rev_rank"], cols["cosine"], cols["rec_top1_cos"],
                             cols["s1_fwd50_cos"], cols["rec_fwd_claimants"], orphan.astype(np.float32),
                             cols["rec_top1_cos"] - cols["cosine"]])
    n0 = len(L0)
    blocks = {"x_base": (xb0, xb_r), "x_coll": (np.load(src / "x_coll.npy"), xc_r),
              "x_ed": (np.load(src / "x_ed.npy"), ed_r),
              "x_q": (np.load(src / "x_q.npy"), np.full((len(Lr), 11), np.nan, np.float32)),
              "x_route": (np.load(src / "x_route.npy"), np.zeros((len(Lr), 5), np.float32)),
              "x_l1": (np.load(src / "x_l1.npy"), np.full((len(Lr), 2), np.nan, np.float32)),
              "x_rev": (np.zeros((n0, len(REV)), np.float32), rev_r)}
    for k, (a, b) in blocks.items():
        np.save(dst / f"{k}.npy", np.vstack([a, b.astype(np.float32)]))
    np.save(dst / "left.npy", np.r_[L0, Lr])
    np.save(dst / "right.npy", np.r_[np.load(src / "right.npy"), Rr])
    if split == "brest":
        np.save(dst / "label.npy", np.r_[np.load(src / "label.npy"), r.column("label").to_numpy(zero_copy_only=False)])
        mult0 = np.load(src / "mult.npy")
        s1mult = np.ones(n_s1, mult0.dtype)
        s1mult[L0] = mult0
        np.save(dst / "mult.npy", np.r_[mult0, s1mult[Lr]])
        np.save(dst / "rec_empty.npy", np.r_[np.load(src / "rec_empty.npy"), kind_r == EMPTY])
        np.save(dst / "name_exact.npy", np.r_[np.load(src / "name_exact.npy"), np.zeros(len(Lr), bool)])
        for f in ("true_count", "population", "component", "partition"):
            shutil.copy(src / f"{f}.npy", dst / f"{f}.npy")
    np.save(dst / "base.npy", np.r_[np.load(src / "base.npy"), np.full(len(Lr), np.nan, np.float32)])
    shutil.copy(src / "country_code.npy", dst / "country_code.npy")
    for f in src.glob("names_*.json"):
        shutil.copy(f, dst / f.name)
    (dst / "names_rev.json").write_text(json.dumps(REV))
    if split == "test":
        (dst / "extra_rec_ids.json").write_text(json.dumps(extra + new_ids))
    (dst / "meta.json").write_text(json.dumps({**meta, "n_rec": int(n_rec), "rows": int(n0 + len(Lr)),
                                               "reverse_rows": int(len(Lr)), "reverse_orphan_rows": int(orphan.sum()),
                                               "new_orphan_records": len(new_ids),
                                               "reverse_true": int(r.column("label").to_numpy(zero_copy_only=False).sum())
                                               if split == "brest" else None}) + "\n")
    log(f"{split}_u8r saved: {n0 + len(Lr)} rows")


if __name__ == "__main__":
    main(sys.argv[1])
