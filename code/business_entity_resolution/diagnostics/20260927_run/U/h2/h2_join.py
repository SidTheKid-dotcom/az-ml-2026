"""HOLDOUT-2 final check (27 Sep, Agent U): HOLDOUT-2 S1s joined to the arena, every U feature block rebuilt with
the arena's code, the finalist applied, one-owner decode together with the B_REST OOF, and a sealed scorer.

Stages (label-free unless stated):
  joint     v1 dense rows. Joint reference = arena S1s (0..2,072,820) + HOLDOUT-2 S1s (2,072,821 + i); joint pool =
            arena compact pool + HOLDOUT-2-only records (raw order); joint pairs = arena dense pairs + HOLDOUT-2 dense
            pairs with their frozen-v1 blend/tree (GPU step B). `pair_tree_arena.build_pairs` on the joint set gives
            the candidate rule and the 12 v1 features (rivals, name multiplicity and exact keys over the joint S1s).
            Parity: the B_REST rows are compared with the saved arena v1 block.
  routes    Track R branches (`retrieval_eureka.run_branches`, KEEP rule) for the HOLDOUT-2 S1s over the joint pool;
            new = not a HOLDOUT-2 dense pair (union-k3 does not exist for HOLDOUT-2).
  reverse   R's rule rank1|orphan<=10 from GPU step A (reverse top-20 over the joint S1s); orphan = in no S1's
            forward top-50 (stored P2 for arena S1s, GPU step A for HOLDOUT-2 S1s); new = not dense, not route.
  blocks    collective (tables over joint S1s + joint pool), edit-op (the B_REST word table), route, rev3 and v1
            context blocks for all HOLDOUT-2 rows; C2 / Qwen / L1 blocks come from `assemble --c2 --q --l1`.
  assemble  prep/h2_u8r/ in the brest_u8r layout (x_* blocks, left/right in the joint index space).
  apply     the finalist's fold models (mean) -> p; decode (threshold, one owner, cap 11) jointly with the B_REST
            OOF rows of the same model; writes the HOLDOUT-2 links (no labels).
  score     --open-labels only: HOLDOUT-2 ground truth (train_ground_truth.tsv rows of HOLDOUT-2 S1s only) ->
            macro F0.5, P, R, false links, by country, S1-bootstrap CI; vs the model's B_REST OOF score.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("EUREKA_THREADS", "30")
ROOT = Path("/home/ubuntu/run27/U")
P = ROOT / "prep"
H2 = ROOT / os.environ.get("U_H2DIR", "h2")
J = H2 / "join"
N_ARENA = 2_072_821
COLS = ["entity_id", "business_name", "business_address", "country"]
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def load_frames():
    aref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=COLS).to_pandas()
    href = pd.read_parquet(H2 / "reference.parquet", columns=COLS)
    ref = pd.concat([aref, href], ignore_index=True)
    apool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=COLS + ["raw_row_idx"]).to_pandas()
    hpool = pd.read_parquet(H2 / "v1data/pool.parquet")
    extra = hpool[~hpool.raw_row_idx.isin(apool.raw_row_idx)].sort_values("raw_row_idx")
    pool = pd.concat([apool, extra], ignore_index=True)
    raw_to_joint = np.full(10_320_219, -1, np.int64)
    raw_to_joint[pool.raw_row_idx.to_numpy()] = np.arange(len(pool))
    return ref, pool, raw_to_joint, len(apool)


# ----------------------------------------------------------------------------- joint (v1)
def joint_cmd(a):
    from ber.pair_tree_arena import build_pairs, load_scores
    J.mkdir(parents=True, exist_ok=True)
    ref, pool, raw_to_joint, n_apool = load_frames()
    t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_idx"])
    al, ar = t.column("left_idx").to_numpy().astype(np.int64), t.column("right_idx").to_numpy().astype(np.int64)
    ab, ae = load_scores(ROOT / "data/arena/probabilities.npy"), load_scores(ROOT / "data/arena/base_probabilities.npy")
    hp = pq.read_table(H2 / "v1data/pairs.parquet", columns=["left_idx", "right_raw_idx"])
    sp = pq.read_table(H2 / "v1out/scored_pairs.parquet", columns=["left_idx", "right_idx"]) \
        if (H2 / "v1out/scored_pairs.parquet").exists() else None
    hl = hp.column("left_idx").to_numpy().astype(np.int64)
    hr_raw = hp.column("right_raw_idx").to_numpy().astype(np.int64)
    hb, he = load_scores(H2 / "v1out/probabilities.npy"), load_scores(H2 / "v1out/base_probabilities.npy")
    if len(hb) != len(hl) or len(he) != len(hl):
        raise SystemExit("v1 scores not aligned with HOLDOUT-2 pairs")
    if sp is not None:
        vp = pq.read_table(H2 / "v1data/pairs.parquet", columns=["left_idx", "right_idx"])
        if not (np.array_equal(sp.column("left_idx").to_numpy(), vp.column("left_idx").to_numpy())
                and np.array_equal(sp.column("right_idx").to_numpy(), vp.column("right_idx").to_numpy())):
            raise SystemExit("scored pair order differs from the HOLDOUT-2 pair file")
    hr = raw_to_joint[hr_raw]
    if np.any(hr < 0):
        raise SystemExit("HOLDOUT-2 record outside the joint pool")
    left = np.r_[al, hl + N_ARENA]
    right = np.r_[ar, hr]
    blend, base = np.r_[ab, hb].astype(np.float32), np.r_[ae, he].astype(np.float32)
    n_ap = len(al)
    log(f"joint: {len(ref)} S1s, {len(pool)} records ({len(pool) - n_apool} HOLDOUT-2-only), {len(left)} pairs")
    pairs = build_pairs(ref[["business_name", "business_address", "country"]],
                        pool[["business_name", "business_address", "country"]], left, right, blend, base)
    cand = pairs.cand_rows
    h_rows = cand[cand >= n_ap]
    b_rows = cand[cand < n_ap]
    from ber.forensic_features import EMPTY
    np.savez(J / "h2_dense.npz", pair_rows=h_rows - n_ap, left=left[h_rows], right=right[h_rows],
             v1x=pairs.features(h_rows), name_exact=pairs.name_exact[h_rows], rec_empty=pairs.rec_kind[h_rows] == EMPTY,
             mult=np.asarray(pairs.s1_mult)[left[h_rows]], blend=blend[h_rows], base=base[h_rows])
    # parity on B_REST candidate rows vs the saved arena v1 block (only rows whose context HOLDOUT-2 can touch differ)
    saved_rows = np.load(P / "arena_cand/cand_rows.npy")
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    rep = {"joint_s1": int(len(ref)), "joint_pool": int(len(pool)), "h2_only_records": int(len(pool) - n_apool),
           "h2_dense_pairs": int(len(hl)), "h2_candidate_rows": int(len(h_rows)),
           "arena_candidate_rows_saved": int(len(saved_rows)), "arena_candidate_rows_joint": int(len(b_rows)),
           "arena_candidate_rows_common": int(np.isin(saved_rows, b_rows).sum())}
    spos = np.flatnonzero(np.isin(saved_rows, b_rows) & (part[al[saved_rows]] == 1))
    new = pairs.features(saved_rows[spos])
    old = np.asarray(np.load(P / "l1/arena_v1.npy", mmap_mode="r")[spos])
    diff = ~np.isclose(np.nan_to_num(new, nan=-9), np.nan_to_num(old, nan=-9), atol=1e-6)
    rep["brest_v1_parity"] = {"rows": int(len(spos)), "rows_any_diff": int(diff.any(1).sum()),
                              "per_column_diff": diff.sum(0).tolist()}
    h2_s1 = np.unique(left[h_rows])
    rep["h2_s1_with_candidates"] = int(len(h2_s1))
    (J / "joint_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    # joint dense-pair context reused by the other stages
    np.savez(J / "joint_context.npz", h2_pair_left=hl + N_ARENA, h2_pair_right=hr, h2_blend=hb, h2_base=he)
    pool.to_parquet(J / "joint_pool.parquet", index=False)
    log(json.dumps(rep))


# ----------------------------------------------------------------------------- routes
def routes_cmd(a):
    from ber.retrieval_eureka import KEEP, run_branches
    ref, _, _, _ = load_frames()
    pool = pd.read_parquet(J / "joint_pool.parquet")
    q = np.arange(N_ARENA, len(ref))
    branches = run_branches(ref, pool, q)
    ctx = np.load(J / "joint_context.npz")
    n = len(pool)
    dense = np.unique(ctx["h2_pair_left"] * n + ctx["h2_pair_right"])
    kept, info = [], {}
    for b, f in branches.items():
        lo, mr = KEEP[b]
        f = f[(f.score >= lo) & (f["rank"] <= mr)]
        k = f.s1.to_numpy(np.int64) * n + f.rec.to_numpy(np.int64)
        f = f[~np.isin(k, dense)]
        kept.append(f.s1.to_numpy(np.int64) * n + f.rec.to_numpy(np.int64))
        info[b] = f
    keys = np.unique(np.concatenate(kept))
    L, R = keys // n, keys % n
    ri = np.full((len(keys), 4), np.nan, np.float32)
    for j, b in enumerate(("address", "empty")):
        f = info[b]
        k = f.s1.to_numpy(np.int64) * n + f.rec.to_numpy(np.int64)
        pos = np.searchsorted(keys, k)
        ri[pos, 2 * j] = f.score.to_numpy()
        ri[pos, 2 * j + 1] = f["rank"].to_numpy()
    np.savez(J / "h2_routes.npz", left=L, right=R, route_info=ri)
    log(f"HOLDOUT-2 routes: {len(keys)} new pairs ({len(keys) / 50000:.2f} per S1)")


# ----------------------------------------------------------------------------- reverse
def reverse_cmd(a):
    g = H2 / "gpu_out"
    pool = pd.read_parquet(J / "joint_pool.parquet", columns=["entity_id", "raw_row_idx"])
    raw_to_joint = np.full(10_320_219, -1, np.int64)
    raw_to_joint[pool.raw_row_idx.to_numpy()] = np.arange(len(pool))
    rev_idx = np.load(g / "h2_rev_idx.npy")
    rev_score = np.load(g / "h2_rev_score.npy").astype(np.float32)
    fwd_idx, fwd_score = np.load(g / "h2_fwd_idx.npy"), np.load(g / "h2_fwd_score.npy")
    t = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["right_raw_idx"])
    a_raw = t.column("right_raw_idx").to_numpy().astype(np.int64)
    claim = np.bincount(a_raw, minlength=10_320_219) + np.bincount(fwd_idx.reshape(-1), minlength=10_320_219)
    orphan = claim == 0
    j = np.arange(rev_idx.shape[1])[None, :]
    s1 = rev_idx.astype(np.int64)
    keep = (s1 >= N_ARENA) & ((j == 0) | (orphan[:, None] & (j < 10)))
    rec_raw, rank = np.nonzero(keep)
    S = s1[rec_raw, rank]
    cos = rev_score[rec_raw, rank]
    top1 = rev_score[rec_raw, 0]
    h = S - N_ARENA
    n = len(pool)
    # new vs HOLDOUT-2 dense top-50 (raw keys) and HOLDOUT-2 routes (joint keys)
    dense_raw = np.unique(np.repeat(np.arange(fwd_idx.shape[0]), 50) * 10_320_219 + fwd_idx.reshape(-1))
    new = ~np.isin(h * 10_320_219 + rec_raw, dense_raw)
    rt = np.load(J / "h2_routes.npz")
    rjk = raw_to_joint[rec_raw]
    in_route = (rjk >= 0) & np.isin(S * n + np.maximum(rjk, 0), rt["left"] * n + rt["right"])
    new &= ~in_route
    rec_raw, rank, S, cos, top1, h, rjk = (x[new] for x in (rec_raw, rank, S, cos, top1, h, rjk))
    rev = np.column_stack([np.ones(len(S)), rank + 1, claim[rec_raw], orphan[rec_raw], top1 - cos,
                           cos - fwd_score[h, 49]]).astype(np.float32)
    np.savez(J / "h2_reverse.npz", left=S, rec_raw=rec_raw, right_joint=rjk, rev3=rev, cosine=cos)
    log(f"HOLDOUT-2 reverse: {len(S)} new pairs ({len(S) / 50000:.2f} per S1; to records outside the joint pool "
        f"{int((rjk < 0).sum())})")


# ----------------------------------------------------------------------------- blocks
def blocks_cmd(a):
    import edit_features as EF
    from u8_prep import ed_tr
    from ber.collective_features import build_tables, pair_features
    from ber.forensic_features import EMPTY, address_kinds
    from ber.pair_tree_arena import top_two
    ref, _, _, _ = load_frames()
    pool = pd.read_parquet(J / "joint_pool.parquet")
    d = np.load(J / "h2_dense.npz")
    rt = np.load(J / "h2_routes.npz")
    rv = np.load(J / "h2_reverse.npz")
    # orphan records of reverse rows: append after the joint pool (text from the raw TSVs)
    orph_raw = np.unique(rv["rec_raw"][rv["right_joint"] < 0])
    from h2_pairs import read_pool_rows
    orph = read_pool_rows(orph_raw) if len(orph_raw) else pool.iloc[:0]
    n_pool = len(pool)
    full = pd.concat([pool, orph], ignore_index=True)
    rmap = {int(r): n_pool + i for i, r in enumerate(orph_raw)}
    rv_right = np.where(rv["right_joint"] >= 0, rv["right_joint"],
                        np.asarray([rmap.get(int(r), -1) for r in rv["rec_raw"]], np.int64))
    L = np.r_[d["left"], rt["left"], rv["left"]].astype(np.int64)
    R = np.r_[d["right"], rt["right"], rv_right].astype(np.int64)
    kind = np.r_[np.zeros(len(d["left"]), np.int8), np.ones(len(rt["left"]), np.int8),
                 np.full(len(rv["left"]), 2, np.int8)]
    log(f"HOLDOUT-2 rows: dense {int((kind == 0).sum())}, route {int((kind == 1).sum())}, reverse {int((kind == 2).sum())}")
    # collective: tables over the joint S1s and joint pool (orphans NaN, as for B_REST reverse rows)
    t = build_tables(pa.array(ref.business_name.fillna("")), pa.array(ref.business_address.fillna("")),
                     pa.array(pool.business_name.fillna("")), pa.array(pool.business_address.fillna("")), log=log)
    xc = np.full((len(L), 25), np.nan, np.float32)
    ok = R < n_pool
    xc[ok] = pair_features(t, L[ok], R[ok])
    # edit-op: the same word table as the B_REST block
    ed, _ = ed_tr(pa.chunked_array([pa.array(ref.business_name.fillna(""))]),
                  pa.chunked_array([pa.array(ref.business_address.fillna(""))]),
                  pa.chunked_array([pa.array(full.business_name.fillna(""))]),
                  pa.chunked_array([pa.array(full.business_address.fillna(""))]), L, R)
    # v1 block: dense rows own v1 features; route/reverse rows the v1 record/S1 context over joint dense pairs
    ctx = np.load(J / "joint_context.npz")
    tp = pq.read_table(ROOT / "data/arena/pairs.parquet", columns=["left_idx", "right_idx"])
    from ber.pair_tree_arena import load_scores
    pl = np.r_[tp.column("left_idx").to_numpy().astype(np.int64), ctx["h2_pair_left"]]
    pr = np.r_[tp.column("right_idx").to_numpy().astype(np.int64), ctx["h2_pair_right"]]
    pb = np.r_[load_scores(ROOT / "data/arena/probabilities.npy"), ctx["h2_blend"]].astype(np.float32)
    n_rec = len(full)
    rec_top1, _ = top_two(pr, pb, n_rec)
    strong = pb >= 0.5
    s1_strong = np.bincount(pl[strong], minlength=len(ref)).astype(np.float32)
    rec_strong = np.bincount(pr[strong], minlength=n_rec).astype(np.float32)
    s1m = np.full(len(ref), np.nan, np.float32)
    s1m[d["left"]] = d["v1x"][:, 8]
    v1 = np.full((len(L), 12), np.nan, np.float32)
    v1[kind == 0] = d["v1x"]
    nd = kind > 0
    kind_rec = address_kinds(pa.array(full.business_address.fillna("")))
    v1[nd, 4], v1[nd, 5], v1[nd, 6] = s1_strong[L[nd]], rec_top1[R[nd]], rec_strong[R[nd]]
    v1[nd, 8], v1[nd, 9] = s1m[L[nd]], kind_rec[R[nd]]
    route = np.zeros((len(L), 5), np.float32)
    route[kind == 1, 0] = 1
    route[kind == 1, 1:] = rt["route_info"]
    rev3 = np.zeros((len(L), 6), np.float32)
    rev3[kind == 2] = rv["rev3"]
    rec_empty = np.r_[d["rec_empty"], kind_rec[R[kind > 0]] == EMPTY]
    name_exact = np.r_[d["name_exact"], np.zeros(int((kind > 0).sum()), bool)]
    np.savez(J / "h2_blocks.npz", left=L, right=R, kind=kind, v1=v1, coll=xc, ed=ed, route=route, rev3=rev3,
             rec_empty=rec_empty, name_exact=name_exact, base=np.r_[d["base"], np.full(int(nd.sum()), np.nan, np.float32)])
    full[["entity_id", "raw_row_idx"]].to_parquet(J / "joint_pool_ids.parquet", index=False)
    log("blocks saved")


# ----------------------------------------------------------------------------- assemble
def c2_block(L, R, kind, c2_h2, q_h2, n_s1, n_rec):
    """C2 / Qwen rival blocks over the joint candidate rows: B_REST rows carry their saved scores, HOLDOUT-2 dense
    rows the new scores, A / SLICE rows none (the B_REST regime); route / reverse rows get NaN (as in training)."""
    from ber.pair_tree_c2 import c2_features
    from ber.qwen_ensemble import q_features
    bl, br = np.load(P / "brest/left.npy"), np.load(P / "brest/right.npy")
    bx, bq = np.load(P / "brest/x_base.npy", mmap_mode="r"), np.load(P / "brest/x_q.npy", mmap_mode="r")
    bc2 = np.where(np.asarray(bx[:, 12]) < 0, np.nan, np.asarray(bx[:, 12])).astype(np.float32)
    bqq = np.where(np.asarray(bq[:, 0]) < 0, np.nan, np.asarray(bq[:, 0])).astype(np.float32)
    bbase = np.load(P / "brest/base.npy")
    d = kind == 0
    hb = np.load(J / "h2_blocks.npz")["base"][d]
    LL, RR = np.r_[bl, L[d]], np.r_[br, R[d]]
    c2 = np.r_[bc2, c2_h2].astype(np.float32)
    q = np.r_[bqq, q_h2].astype(np.float32)
    base = np.r_[bbase, hb].astype(np.float32)
    x2 = c2_features(LL, RR, c2, base, n_s1, n_rec)[len(bl):]
    xq = q_features(LL, RR, c2, q, base, n_s1, n_rec)[len(bl):]
    out2 = np.full((len(L), 8), np.nan, np.float32)
    outq = np.full((len(L), 11), np.nan, np.float32)
    out2[d], outq[d] = x2, xq
    return out2, outq


def assemble_cmd(a):
    b = np.load(J / "h2_blocks.npz")
    L, R, kind = b["left"], b["right"], b["kind"]
    ref = pd.read_parquet(H2 / "reference.parquet", columns=["country"])
    n_s1 = N_ARENA + len(ref)
    n_rec = int(R.max()) + 1
    dd = int((kind == 0).sum())
    c2 = np.load(a.c2) if a.c2 else np.full(dd, np.nan, np.float32)
    q = np.load(a.q) if a.q else np.full(dd, np.nan, np.float32)
    if len(c2) != dd or len(q) != dd:
        raise SystemExit(f"C2/Qwen score arrays must have one value per HOLDOUT-2 dense row ({dd})")
    x2, xq = c2_block(L, R, kind, c2, q, n_s1, max(n_rec, int(np.load(P / "brest/right.npy").max()) + 1))
    out = P / os.environ.get("U_H2PREP", "h2_u8r")
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "x_base.npy", np.hstack([b["v1"], x2]).astype(np.float32))
    np.save(out / "x_v1.npy", b["v1"].astype(np.float32))
    np.save(out / "x_q.npy", xq)
    np.save(out / "x_coll.npy", b["coll"])
    np.save(out / "x_ed.npy", b["ed"])
    np.save(out / "x_route.npy", b["route"])
    np.save(out / "x_rev3.npy", b["rev3"])
    l1 = np.load(a.l1) if a.l1 else np.full((len(L), 2), np.nan, np.float32)
    np.save(out / "x_l1v2.npy", l1.astype(np.float32))
    for k in ("left", "right", "kind", "rec_empty", "name_exact", "base"):
        np.save(out / f"{k}.npy", b[k])
    countries = json.loads((P / "brest_u8r/meta.json").read_text())["countries"]
    aref_c = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["country"]).column("country").to_pylist()
    joint_c = np.asarray(aref_c + ref.country.tolist(), dtype=object)
    np.save(out / "country_code.npy", np.asarray([countries.index(c) for c in joint_c], np.int8))
    (out / "meta.json").write_text(json.dumps({"countries": countries, "n_s1": n_s1, "rows": int(len(L)),
                                               "dense": dd, "route": int((kind == 1).sum()),
                                               "reverse": int((kind == 2).sum()), "c2": a.c2, "q": a.q, "l1": a.l1,
                                               "labels_read": False}) + "\n")
    log(f"assembled prep/h2_u8r: {len(L)} rows; C2 {'given' if a.c2 else 'MISSING'}, Qwen {'given' if a.q else 'MISSING'}, "
        f"L1 {'given' if a.l1 else 'MISSING'}")


# ----------------------------------------------------------------------------- apply
def apply_cmd(a):
    from catboost import CatBoostClassifier
    from ber.pair_tree_arena import decide
    from ber.pair_tree_c2 import proba
    work = ROOT / "u4" / a.work
    rep = json.loads((work / "report.json").read_text())
    thr = float(a.threshold if a.threshold is not None else rep["thresholds"]["full_oof_threshold"])
    d = P / os.environ.get("U_H2PREP", "h2_u8r")
    blocks = []
    for blk in rep["variant"].split("+"):
        name = {"base": "x_base", "q": "x_q", "coll": "x_coll"}.get(blk, f"x_{blk}")
        blocks.append(np.load(d / f"{name}.npy"))
    x = np.hstack(blocks).astype(np.float32)
    per_country = (work / "model_fold0_US.cbm").exists()
    preds = []
    meta = json.loads((d / "meta.json").read_text())
    s1c = np.asarray(meta["countries"])[np.load(d / "country_code.npy")]
    L = np.load(d / "left.npy")
    for k in range(5):
        pk = np.zeros(len(x), np.float32)
        for c in (("US", "India") if per_country else (None,)):
            m = CatBoostClassifier()
            m.load_model(str(work / (f"model_fold{k}_{c}.cbm" if c else f"model_fold{k}.cbm")))
            sel = (s1c[L] == c) if c else np.ones(len(x), bool)
            pk[sel] = proba(m, x[sel])
        preds.append(pk)
    p = np.mean(preds, 0)
    # joint one-owner decode with the model's B_REST OOF rows (the B_REST S1s compete for the same records)
    # common record id = raw train pool row (B_REST split and HOLDOUT-2 joint pool use different extra indices)
    split = a.brest_split
    bl, br = np.load(P / split / "left.npy"), np.load(P / split / "right.npy")
    boof = np.load(work / "oof.npy")
    apool_raw = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["raw_row_idx"]).column("raw_row_idx").to_numpy()
    b_extra = json.loads((P / split / "extra_rec_ids.json").read_text()) if (P / split / "extra_rec_ids.json").exists() else []
    if b_extra:
        tp = pq.read_table(ROOT / "rsrc/texts/train_pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
        pos = {v: i for i, v in enumerate(tp)}
        extra_raw = np.asarray([pos[v] for v in b_extra], np.int64)
    else:
        extra_raw = np.zeros(0, np.int64)
    b_raw = np.where(br < len(apool_raw), apool_raw[np.minimum(br, len(apool_raw) - 1)],
                     extra_raw[np.maximum(br - len(apool_raw), 0)] if len(extra_raw) else -1)
    h_raw = pd.read_parquet(J / "joint_pool_ids.parquet", columns=["raw_row_idx"]).raw_row_idx.to_numpy()[
        np.load(d / "right.npy")]
    LL, RR, pp = np.r_[bl, L], np.r_[b_raw, h_raw].astype(np.int64), np.r_[boof, p]
    acc = decide(pp, LL, RR, thr, None)[len(bl):]
    np.savez(work / "h2_predictions.npz", p=p, p_folds=np.stack(preds).astype(np.float16), accept=acc, left=L,
             right=np.load(d / "right.npy"), threshold=thr)
    ref = pd.read_parquet(H2 / "reference.parquet", columns=["country"])
    k_acc = np.bincount(L[acc] - N_ARENA, minlength=len(ref))
    out = {"model": str(work), "threshold": thr, "rows": int(len(x)), "accepted_links": int(acc.sum()),
           "mean_list": float(k_acc.mean()), "empty_rows": float((k_acc == 0).mean()),
           "by_country": {c: {"mean_list": float(k_acc[(ref.country == c).to_numpy()].mean())} for c in ("US", "India")},
           "labels_read": False}
    (work / "h2_apply.json").write_text(json.dumps(out, indent=1) + "\n")
    log(json.dumps(out))


# ----------------------------------------------------------------------------- score (labels)
def score_cmd(a):
    if not a.open_labels:
        raise SystemExit("refusing: HOLDOUT-2 labels are sealed; pass --open-labels only after the lead's go")
    work = ROOT / "u4" / a.work
    pr = np.load(work / "h2_predictions.npz")
    ref = pd.read_parquet(H2 / "reference.parquet", columns=["entity_id", "country"])
    h2_ids = set(ref.entity_id)
    ids = pd.read_parquet(J / "joint_pool_ids.parquet").entity_id.to_numpy()
    truth = {}
    gt_path = Path(os.environ.get("U_H2_GT", str(ROOT / "h2/labels/train_ground_truth.tsv")))
    for chunk in pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False,
                             chunksize=500_000):
        cols = list(chunk.columns)
        s1col = cols[0]
        chunk = chunk[chunk[s1col].isin(h2_ids)]
        for s1, recs in zip(chunk[s1col], chunk[cols[1]]):
            truth.setdefault(s1, set()).update(v for v in str(recs).split(",") if v)
    L, R, acc = pr["left"], pr["right"], pr["accept"]
    pred = {}
    for l, r in zip(L[acc] - N_ARENA, R[acc]):
        pred.setdefault(int(l), []).append(ids[r])
    f, tp_all, k_all, t_all = [], 0, 0, 0
    for i, s1 in enumerate(ref.entity_id):
        t = truth.get(s1, set())
        p = set(pred.get(i, []))
        tp = len(t & p)
        if not t and not p:
            f.append(1.0)
        else:
            f.append(5 * tp / (len(t) + 4 * len(p)) if (len(t) + len(p)) else 0.0)
        tp_all, k_all, t_all = tp_all + tp, k_all + len(p), t_all + len(t)
    f = np.asarray(f)
    rng = np.random.default_rng(20260929)
    boot = [f[rng.integers(0, len(f), len(f))].mean() for _ in range(1000)]
    out = {"macro_f05": float(f.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
           "precision": tp_all / max(k_all, 1), "recall": tp_all / max(t_all, 1), "false_links": int(k_all - tp_all),
           "by_country": {c: float(f[(ref.country == c).to_numpy()].mean()) for c in ("US", "India")},
           "brest_oof_same_model": json.loads((work / "report.json").read_text())["result"]["macro_f05"],
           "labels_read": f"HOLDOUT-2 ground truth rows only, opened by --open-labels ({gt_path})"}
    (work / "h2_score.json").write_text(json.dumps(out, indent=1) + "\n")
    log(json.dumps(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["joint", "routes", "reverse", "blocks", "assemble", "apply", "score"])
    ap.add_argument("--c2")
    ap.add_argument("--q")
    ap.add_argument("--l1")
    ap.add_argument("--work")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--open-labels", action="store_true")
    ap.add_argument("--brest-split", default="brest_u8r")
    a = ap.parse_args()
    {"joint": joint_cmd, "routes": routes_cmd, "reverse": reverse_cmd, "blocks": blocks_cmd,
     "assemble": assemble_cmd, "apply": apply_cmd, "score": score_cmd}[a.stage](a)


if __name__ == "__main__":
    main()
