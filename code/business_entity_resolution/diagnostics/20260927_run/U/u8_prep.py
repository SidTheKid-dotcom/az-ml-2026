"""U8 prep (27 Sep, Agent U): add Track R route pairs (address >= 0.6; empty >= 0.8 at rank <= 2; new vs dense+union)
as extra candidate rows, with every feature block rebuilt on the union.

    python u8_prep.py brest            # arena B_REST route pairs (C2 scored: eureka/c2_b_rest_new)
    python u8_prep.py test C2_DIR      # US/India test route pairs, C2 scores in C2_DIR (test_ui order)

Route rows: v1 block NaN (no v1 tree/blend exists for them), Qwen unscored (-1), C2 from the route scoring,
collective / edit / translit / U1 blocks from the same code, plus a route block
(is_route, address score/rank, empty score/rank). C2 and Qwen rival/rank blocks are recomputed on the union
(parity with the saved blocks is asserted on the original rows). Writes prep/<split>_u8/.
Labels: B_REST route rows from rec_owner restricted to B_REST owners. SLICE/HOLDOUT-2 never read.
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
from rapidfuzz import fuzz

sys.path.insert(0, str(Path(__file__).parent))
import edit_features as EF  # noqa: E402
from u2_ea_name import norm  # noqa: E402
from ber.collective_features import build_tables, pair_features  # noqa: E402
from ber.forensic_features import EMPTY, address_kinds  # noqa: E402
from ber.pair_tree_c2 import c2_features  # noqa: E402
from ber.qwen_ensemble import q_features  # noqa: E402

ROOT = Path("/home/ubuntu/run27/U")
NOC2 = os.environ.get("U8_NOC2", "") == "1"  # route rows without any C2 score (weights unreachable)
SFX = "_u8n" if NOC2 else "_u8"
PREP, SRC = ROOT / "prep", ROOT / "u8src"
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def route_info(parquets, left, right, n_rec):
    """(addr_score, addr_rank, empty_score, empty_rank) per route pair from the branch parquets (route index space)."""
    key = left.astype(np.int64) * n_rec + right
    out = np.full((len(left), 4), np.nan, np.float32)
    for j, p in enumerate(parquets):
        t = pq.read_table(p, columns=["s1", "rec", "score", "rank"])
        k = t.column("s1").to_numpy() * n_rec + t.column("rec").to_numpy()
        o = np.argsort(k)
        k = k[o]
        pos = np.minimum(np.searchsorted(k, key), len(k) - 1)
        hit = k[pos] == key
        out[hit, 2 * j] = t.column("score").to_numpy()[o][pos[hit]]
        out[hit, 2 * j + 1] = t.column("rank").to_numpy()[o][pos[hit]]
    return out


def ed_tr(ref_names, ref_addr, pool_names, pool_addr, s_idx, r_idx):
    tables = json.loads((PREP / "edit_word_tables.json").read_text())
    us, si = np.unique(s_idx, return_inverse=True)
    ur, ri = np.unique(r_idx, return_inverse=True)
    S = EF.prep_side(ref_names.take(pa.array(us)), ref_addr.take(pa.array(us)))
    R = EF.prep_side(pool_names.take(pa.array(ur)), pool_addr.take(pa.array(ur)))
    EF.G.update(S=S, R=R, pairs=list(zip(si.tolist(), ri.tolist())), ta=tables["added"], tr=tables["removed"],
                pa=tables["added_prior"], pr=tables["removed_prior"])
    _, ed, tr = EF._feat_chunk((0, len(s_idx)))
    return ed, tr


def u1_rows(ref_names, pool_names, s_idx, r_idx, z, rec_ea):
    x = np.full((len(s_idx), 7), np.nan, np.float32)
    rows = np.flatnonzero(rec_ea)
    if len(rows) == 0:
        return x
    s, r = s_idx[rows], r_idx[rows]
    us, ui = np.unique(s, return_inverse=True)
    ur, ri = np.unique(r, return_inverse=True)
    sn, rn = norm(ref_names.take(pa.array(us))), norm(pool_names.take(pa.array(ur)))
    this = np.fromiter((fuzz.ratio(sn[a], rn[b]) / 100.0 for a, b in zip(ui, ri)), np.float32, count=len(rows))
    zr = z["rows"]  # index into search arrays per route row (or -1)
    ok = zr[rows] >= 0
    q = np.where(ok, zr[rows], 0)
    x[rows, 0] = np.where(ok, z["bsim"][q], np.nan)
    x[rows, 1] = np.where(ok, z["ssim"][q], np.nan)
    x[rows, 2] = x[rows, 0] - x[rows, 1]
    x[rows, 3] = np.where(ok, z["nties"][q], np.nan)
    x[rows, 4] = this
    x[rows, 5] = np.where(ok, (z["best"][q] == s).astype(np.float32), np.nan)
    x[rows, 6] = x[rows, 0] - this
    return x


def union_blocks(split, L0, R0, Lr, Rr, c2r, n_s1, n_rec):
    d = PREP / split
    xb = np.load(d / "x_base.npy")
    xq = np.load(d / "x_q.npy")
    base0 = np.load(d / "base.npy")
    c2_0 = np.where(xb[:, 12] < 0, np.nan, xb[:, 12]).astype(np.float32)
    q_0 = np.where(xq[:, 0] < 0, np.nan, xq[:, 0]).astype(np.float32)
    L, R = np.r_[L0, Lr], np.r_[R0, Rr]
    c2 = np.r_[c2_0, c2r].astype(np.float32)
    q = np.r_[q_0, np.full(len(Lr), np.nan, np.float32)]
    base = np.r_[base0, np.full(len(Lr), np.nan, np.float32)]
    x2 = c2_features(L, R, c2, base, n_s1, n_rec)
    x2q = q_features(L, R, c2, q, base, n_s1, n_rec)
    n0 = len(L0)
    # parity on original rows after dropping route rows' influence is not expected (new rivals); check a no-route rebuild
    x2_chk = c2_features(L0, R0, c2_0, base0, n_s1, n_rec)
    same = np.allclose(np.nan_to_num(x2_chk, nan=-9), np.nan_to_num(xb[:, 12:20], nan=-9), atol=1e-6)
    changed = int((np.abs(np.nan_to_num(x2[:n0], nan=-9) - np.nan_to_num(xb[:, 12:20], nan=-9)) > 1e-6).any(1).sum())
    log(f"{split}: C2 block rebuild parity on original rows {same}; rows whose C2 block changed with routes {changed}")
    if not same:
        raise SystemExit("C2 block rebuild does not reproduce the saved block")
    v1 = np.vstack([xb[:, :12], np.full((len(Lr), 12), np.nan, np.float32)])
    return L, R, np.hstack([v1, x2]).astype(np.float32), x2q.astype(np.float32), changed


def save_union(split_out, blocks, extra):
    out = PREP / split_out
    out.mkdir(parents=True, exist_ok=True)
    for k, v in {**blocks, **extra}.items():
        np.save(out / f"{k}.npy", v)


def brest():
    d = PREP / "brest"
    meta = json.loads((d / "meta.json").read_text())
    n_s1, n_rec = meta["n_s1"], meta["n_rec"]
    L0, R0 = np.load(d / "left.npy"), np.load(d / "right.npy")
    Lr, Rr = np.load(SRC / "b_rest/left.npy"), np.load(SRC / "b_rest/right.npy")
    k0 = np.sort(L0 * n_rec + R0)
    kr = Lr * n_rec + Rr
    dup = np.isin(kr, k0)
    log(f"route rows {len(Lr)}; already candidate rows {int(dup.sum())} (dropped)")
    sc = np.load(SRC / "c2_b_rest_new/scores.npy")
    msk = np.load(SRC / "c2_b_rest_new/scored_mask.npy")
    keep = ~dup
    Lr, Rr = Lr[keep], Rr[keep]
    c2r = np.where(msk[keep], sc[keep], np.nan).astype(np.float32)
    if NOC2:
        c2r[:] = np.nan
    info = route_info([SRC / "b_rest/b_rest_address.parquet", SRC / "b_rest/b_rest_empty.parquet"],
                      Lr, Rr, n_rec)
    L, R, xbase, xq, changed = union_blocks("brest", L0, R0, Lr, Rr, c2r, n_s1, n_rec)
    ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["business_name", "business_address"])
    pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["business_name", "business_address"])
    rec_ea_all = address_kinds(pool.column("business_address")) == EMPTY
    tables = build_tables(ref.column("business_name"), ref.column("business_address"), pool.column("business_name"),
                          pool.column("business_address"), log=log)
    xc_r = pair_features(tables, Lr, Rr)
    ed_r, tr_r = ed_tr(ref.column("business_name"), ref.column("business_address"), pool.column("business_name"),
                       pool.column("business_address"), Lr, Rr)
    z = dict(np.load(ROOT / "u2/u2_name_search.npz"))
    z["rows"] = np.where(z["best"][Rr] >= -1, Rr, -1)
    u1_r = u1_rows(ref.column("business_name"), pool.column("business_name"), Lr, Rr, z, rec_ea_all[Rr])
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    own = np.load(ROOT / "data/arena/labels/rec_owner.npy")
    own = np.where((own >= 0) & (part[np.maximum(own, 0)] == 1), own, -1)
    lab_r = own[Rr] == Lr
    mult0 = np.load(d / "mult.npy")
    s1_mult = np.ones(n_s1, mult0.dtype)
    s1_mult[L0] = mult0
    blocks = {"x_base": xbase, "x_q": xq,
              "x_coll": np.vstack([np.load(d / "x_coll.npy"), xc_r]),
              "x_ed": np.vstack([np.load(d / "x_ed.npy"), ed_r]),
              "x_tr": np.vstack([np.load(d / "x_tr.npy"), tr_r]),
              "x_u1": np.vstack([np.load(d / "x_u1.npy"), u1_r]),
              "x_route": np.vstack([np.zeros((len(L0), 5), np.float32),
                                    np.column_stack([np.ones(len(Lr), np.float32), info])])}
    extra = {"left": L, "right": R, "label": np.r_[np.load(d / "label.npy"), lab_r],
             "rec_empty": np.r_[np.load(d / "rec_empty.npy"), rec_ea_all[Rr]],
             "name_exact": np.r_[np.load(d / "name_exact.npy"), np.zeros(len(Lr), bool)],
             "mult": s1_mult[L], "base": np.r_[np.load(d / "base.npy"), np.full(len(Lr), np.nan, np.float32)]}
    save_union("brest" + SFX, blocks, extra)
    out = PREP / ("brest" + SFX)
    for f in ("true_count", "population", "component", "partition", "country_code"):
        shutil.copy(d / f"{f}.npy", out / f"{f}.npy")
    for f in d.glob("names_*.json"):
        shutil.copy(f, out / f.name)
    (out / "names_route.json").write_text(json.dumps(["route_flag", "route_addr_score", "route_addr_rank",
                                                      "route_empty_score", "route_empty_rank"]))
    (out / "meta.json").write_text(json.dumps({**meta, "rows": int(len(L)), "route_rows": int(len(Lr)),
                                               "route_true": int(lab_r.sum()), "route_dup_dropped": int(dup.sum()),
                                               "original_rows_with_changed_c2_block": changed}) + "\n")
    log(f"brest{SFX} saved: {len(L)} rows ({len(Lr)} route, {int(lab_r.sum())} true)")


def test(c2_dir):
    d = PREP / "test"
    meta = json.loads((d / "meta.json").read_text())
    n_s1, n_rec = meta["n_s1"], meta["n_rec"]
    dense_ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["entity_id"]).column("entity_id")
    set_ref = pq.read_table(SRC / "test/reference.parquet", columns=["entity_id", "country"])
    if not set_ref.column("entity_id").equals(dense_ref):
        raise SystemExit("route set reference order differs from dense test reference")
    country = np.asarray(set_ref.column("country").to_pylist(), dtype=object)
    la, ra = np.load(SRC / "test/left.npy"), np.load(SRC / "test/right.npy")
    rows = np.flatnonzero(np.isin(country[la], ["US", "India"]))
    Lr, Rs = la[rows], ra[rows]
    set_pool = pq.read_table(SRC / "test/pool.parquet")
    dense_raw = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["raw_row_idx"]).column(
        "raw_row_idx").to_numpy()
    raw_to_dense = {int(v): i for i, v in enumerate(dense_raw)}
    set_raw = set_pool.column("raw_row_idx").to_numpy()
    set_to_main = np.asarray([raw_to_dense.get(int(v), -1) for v in set_raw], np.int64)
    outside = np.flatnonzero(set_to_main < 0)
    set_to_main[outside] = n_rec + np.arange(len(outside))
    Rr = set_to_main[Rs]
    n_rec2 = n_rec + len(outside)
    log(f"test route US/India rows {len(Lr)}; records outside dense pool {len(outside)} "
        f"(route rows to them {int((Rr >= n_rec).sum())})")
    L0, R0 = np.load(d / "left.npy"), np.load(d / "right.npy")
    k0 = np.sort(L0 * n_rec2 + R0)
    dup = np.isin(Lr * n_rec2 + Rr, k0)
    if c2_dir == "none":
        sc, msk = np.zeros(len(rows), np.float32), np.zeros(len(rows), bool)
    else:
        sc = np.load(Path(c2_dir) / "scores.npy")
        msk = np.load(Path(c2_dir) / "scored_mask.npy")
    if len(sc) != len(rows):
        raise SystemExit("C2 route scores are not aligned with the US/India route rows")
    # the branch parquets index records by RAW test pool row (not the route set pool row)
    info_all = route_info([SRC / "test/test_address.parquet", SRC / "test/test_empty.parquet"], Lr,
                          set_raw[Rs].astype(np.int64), int(set_raw.max()) + 1)
    log(f"route info found for {float((~np.isnan(info_all[:, [0, 2]]).any(1)).mean()):.4f} of route rows")
    keep = ~dup
    Lr, Rr, Rs, info = Lr[keep], Rr[keep], Rs[keep], info_all[keep]
    c2r = np.where(msk[keep], sc[keep], np.nan).astype(np.float32)
    L, R, xbase, xq, changed = union_blocks("test", L0, R0, Lr, Rr, c2r, n_s1, n_rec2)
    # text for route rows comes from the route set pool (it holds every route record)
    ref = pq.read_table(ROOT / "data/test/dense_test/reference.parquet", columns=["business_name", "business_address",
                                                                                  "country"])
    sp_names, sp_addr = set_pool.column("business_name"), set_pool.column("business_address")
    ed_r, tr_r = ed_tr(ref.column("business_name"), ref.column("business_address"), sp_names, sp_addr, Lr, Rs)
    rec_ea_r = (address_kinds(sp_addr) == EMPTY)[Rs]
    zt = dict(np.load(ROOT / "u2/u1_test_name_search.npz"))
    zt["rows"] = np.where(Rr < n_rec, Rr, -1)
    u1_r = u1_rows(ref.column("business_name"), sp_names, Lr, Rs, zt, rec_ea_r)
    # collective: tables over US/India dense test S1s + records (as test_coll.py); outside records -> NaN
    pool = pq.read_table(ROOT / "data/test/dense_test/pool.parquet", columns=["business_name", "business_address",
                                                                                "country"])
    s1c = np.asarray(ref.column("country").to_pylist(), dtype=object)
    rc = np.asarray(pool.column("country").to_pylist(), dtype=object)
    s1_sel, rec_sel = np.flatnonzero(np.isin(s1c, ["US", "India"])), np.flatnonzero(np.isin(rc, ["US", "India"]))
    s1_map = np.full(len(s1c), -1, np.int64)
    s1_map[s1_sel] = np.arange(len(s1_sel))
    rec_map = np.full(n_rec2, -1, np.int64)
    rec_map[rec_sel] = np.arange(len(rec_sel))
    tk = lambda t, c, idx: t.column(c).take(pa.array(idx))  # noqa: E731
    tables = build_tables(tk(ref, "business_name", s1_sel), tk(ref, "business_address", s1_sel),
                          tk(pool, "business_name", rec_sel), tk(pool, "business_address", rec_sel), log=log)
    xc_r = np.full((len(Lr), 25), np.nan, np.float32)
    ok = rec_map[Rr] >= 0
    xc_r[ok] = pair_features(tables, s1_map[Lr[ok]], rec_map[Rr[ok]])
    blocks = {"x_base": xbase, "x_q": xq,
              "x_coll": np.vstack([np.load(d / "x_coll.npy"), xc_r]),
              "x_ed": np.vstack([np.load(d / "x_ed.npy"), ed_r]),
              "x_tr": np.vstack([np.load(d / "x_tr.npy"), tr_r]),
              "x_u1": np.vstack([np.load(d / "x_u1.npy"), u1_r]),
              "x_route": np.vstack([np.zeros((len(L0), 5), np.float32),
                                    np.column_stack([np.ones(len(Lr), np.float32), info])])}
    save_union("test" + SFX, blocks, {"left": L, "right": R, "country_code": np.load(d / "country_code.npy"),
                                   "base": np.r_[np.load(d / "base.npy"), np.full(len(Lr), np.nan, np.float32)]})
    out = PREP / ("test" + SFX)
    extra_ids = set_pool.column("entity_id").take(pa.array(outside)).to_pylist()
    (out / "extra_rec_ids.json").write_text(json.dumps(extra_ids))
    (out / "meta.json").write_text(json.dumps({**meta, "n_rec": int(n_rec2), "rows": int(len(L)),
                                               "route_rows": int(len(Lr)), "route_dup_dropped": int(dup.sum()),
                                               "extra_records": int(len(outside)),
                                               "original_rows_with_changed_c2_block": changed,
                                               "c2_dir": str(c2_dir)}) + "\n")
    log(f"test{SFX} saved: {len(L)} rows ({len(Lr)} route)")


if __name__ == "__main__":
    brest() if sys.argv[1] == "brest" else test(sys.argv[2])
