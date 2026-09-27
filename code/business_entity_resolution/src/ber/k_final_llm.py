"""K-final + Agent J LLM judge (``llm`` family). Copy of ``k_final`` (K's release source is left untouched) with one
added family: J's Qwen3-Reranker-4B LoRA pair score, joined by (s1_id, rec_id) from ``data/llmj/llm_{brest,test}
.parquet`` (label-free band rule, identical on B_REST and test; rows outside the band get NaN and llm:has 0), plus
per-S1 rank / gap to the S1's top llm_p and the gap to the best other S1 claiming the same record. Same folds,
rounds, threshold grid, decoding and bootstrap as k_final. ``K_THREADS`` sets LightGBM threads (default 30).

Original k_final docstring:
K final US/India integration: K-robust + feature families over three row sources (dense candidates, U8f route
rows, Agent R reverse rows), 5-fold on U's frozen fold column, paired component bootstrap per family, test rows.

Families (``--families``): ed (U edit/decoy), l1 (U level-1 A-trained probabilities, dense only), l2 (Agent L
run2 listwise lw_p / lw_p_none / lw_rival_p on all three sources), pu (U's u8r_v2 OOF probability, stacked; valid
only with U's fold column, used here), rev (R rev3 block), route (U8f route block). Sources: ``--sources
routes,rev``. Test: mean of the fold models; dense rows at the full-OOF threshold, extra-source rows at
``--extra-thr`` (rate-matching guard) if given.

    python -m ber.k_final evaluate --families ed,l1,l2,rev,route --sources routes,rev --out DIR [--baseline per_s1.npy]
    python -m ber.k_final test --out DIR [--extra-thr 0.955]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import record_assign as ra
from ber.arena_decision import component_bootstrap, evaluate, one_owner
from ber.k_integrate import BASE_COLS, brest_matrix, fit, load_family, test_matrix, u_folds, adversarial

T0 = time.time()
DATA = ra.DATA
CACHE = Path("/home/ubuntu/k/cache")
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.5, 0.951, 0.025), 3))
ROUTE_BLOCK = ["route_flag", "route_addr_score", "route_addr_rank", "route_empty_score", "route_empty_rank"]
REV_BLOCK = ["rev_rank", "rev_rec_fwd_claimants", "rev_orphan", "rev_top1_gap", "rev_cos_minus_s1fwd50"]
L_COLS = ["lw_p", "lw_p_none", "lw_rival_p"]
LLM_COLS = ["llm:p", "llm:logit", "llm:has", "llm:rank_s1", "llm:gap_s1_top", "llm:rec_gap"]
THREADS = int(os.environ.get("K_THREADS", "30"))


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def llm_fill(split, x, names, left, s1_str, rec_str, out=None, src=None):
    """Fill the llm family in place. ``s1_str``/``rec_str`` are per-row id strings (all sources)."""
    t = pq.read_table(DATA / "llmj" / f"llm_{split}.parquet", columns=["s1_id", "rec_id", "llm_p", "llm_logit"]).to_pandas()
    idx = pd.Index(t.s1_id + "|" + t.rec_id)
    if idx.has_duplicates:
        raise SystemExit("duplicate (s1, rec) in llm file")
    pos = idx.get_indexer(pd.Index(s1_str + "|" + rec_str))
    has = pos >= 0
    lp = np.where(has, t.llm_p.to_numpy(dtype=np.float32)[np.maximum(pos, 0)], np.nan).astype(np.float32)
    lg = np.where(has, t.llm_logit.to_numpy(dtype=np.float32)[np.maximum(pos, 0)], np.nan).astype(np.float32)
    x[:, names.index("llm:p")] = lp
    x[:, names.index("llm:logit")] = lg
    x[:, names.index("llm:has")] = has.astype(np.float32)
    h = np.flatnonzero(has)
    rank = np.full(len(x), np.nan, dtype=np.float32)
    gap = np.full(len(x), np.nan, dtype=np.float32)
    rgap = np.full(len(x), np.nan, dtype=np.float32)
    if h.size:
        lh = left[h]
        rank[h] = np.minimum(ra.rank_within(lh, lp[h]), 50)
        top, _ = ra.top_two(lh, lp[h], int(left.max()) + 1)
        gap[h] = lp[h] - top[lh]
        # record side: best llm_p of a different S1 on the same record (pairs deduped by (s1, rec))
        _, rk = np.unique(rec_str[h], return_inverse=True)
        pk = pd.Series(s1_str[h] + "|" + rec_str[h])
        first = ~pk.duplicated().to_numpy()
        t1, t2 = ra.top_two(rk[first], lp[h][first], int(rk.max()) + 1)
        me = lp[h]
        riv = np.where(me >= t1[rk], t2[rk], t1[rk])
        rgap[h] = np.where(riv < 0, np.nan, me - riv)
    x[:, names.index("llm:rank_s1")] = rank
    x[:, names.index("llm:gap_s1_top")] = gap
    x[:, names.index("llm:rec_gap")] = rgap
    log(f"llm {split}: {int(has.sum())} of {len(x)} rows scored ({len(t)} in file, {len(t) - len(np.unique(pos[has]))} unjoined)")
    if out is not None:
        np.savez(Path(out) / f"llm_feats_{split}.npz", x=x[:, [names.index(c) for c in LLM_COLS]], left=left, src=src)


def columns(fams, sources):
    names = list(BASE_COLS)
    if "ed" in fams:
        names += [f"ed:{c}" for c in pq.ParquetFile(DATA / "ufeat/u_features_brest.parquet").schema_arrow.names
                  if c.startswith("ed_")]
    if "l1" in fams:
        names += ["l1:full", "l1:nov1"]
    if "l2" in fams:
        names += [f"l2:{c}" for c in L_COLS]
    if "pu" in fams:
        names += ["pu:p"]
    if "g2" in fams:
        names += ["g2:sg_p1", "g2:sg_prob"]
    if "c2v2" in fams:
        names += ["c2v2:s", "c2v2:rank_s1", "c2v2:gap_s1_top"]
    if "llm" in fams:
        names += LLM_COLS
    names += [f"src:{s}" for s in sources]
    if "route" in fams and "routes" in sources:
        names += [f"route:{c}" for c in ROUTE_BLOCK]
    if "rev" in fams and "rev" in sources:
        names += [f"rev:{c}" for c in REV_BLOCK]
    return names


def u8r_p(split, keys):
    t = pq.read_table(DATA / f"u8r/u8r_pairs_{split}.parquet", columns=["s1_id", "rec_id", "p"]).to_pandas()
    s = pd.Series(t.p.to_numpy(dtype=np.float32), index=t.s1_id + "|" + t.rec_id)
    s = s[~s.index.duplicated()]
    return s.reindex(keys).to_numpy(dtype=np.float32)


_G2 = {}


def g2_vals(split, keys):
    """G v2 stage-1/stage-2 probabilities (A-trained, out-of-sample on B_REST) joined by (s1_id, rec_id)."""
    if split not in _G2:
        import glob

        parts = sorted(glob.glob(str(DATA / "gfeat" / ("b_rest" if split == "brest" else "test") / "part-*.parquet")))
        t = pd.concat([pq.read_table(f, columns=["s1_id", "rec_id", "sg_p1", "sg_prob"]).to_pandas() for f in parts])
        t.index = t.s1_id + "|" + t.rec_id
        _G2[split] = t[~t.index.duplicated()][["sg_p1", "sg_prob"]].astype(np.float32)
        log(f"G v2 {split}: {len(_G2[split])} pairs from {len(parts)} parts")
    return _G2[split].reindex(keys).to_numpy(dtype=np.float32)



def c2v2_dense(split, pair_keys):
    tag = "brest" if split == "brest" else "test"
    sc = np.load(DATA / "c2v2" / f"{tag}_scores.npy", mmap_mode="r")
    mk = np.load(DATA / "c2v2" / f"{tag}_mask.npy", mmap_mode="r")
    v = np.asarray(sc[pair_keys], dtype=np.float32)
    v[~np.asarray(mk[pair_keys], dtype=bool)] = np.nan
    return v


def c2v2_extra(split, source, keys):
    tag = "brest" if split == "brest" else "test"
    t = pq.read_table(DATA / "c2v2" / f"{tag}_{'routes' if source == 'routes' else 'reverse'}.parquet").to_pandas()
    rc = "rec_id" if "rec_id" in t.columns else "record_id"
    col = [c for c in t.columns if c not in ("s1_id", rc, "left_idx", "right_idx", "country", "row_type")]
    col = "c2v2" if "c2v2" in col else ("score" if "score" in col else col[0])
    s = pd.Series(t[col].to_numpy(dtype=np.float32), index=t["s1_id"] + "|" + t[rc])
    s = s[~s.index.duplicated()]
    return s.reindex(keys).to_numpy(dtype=np.float32)


def c2v2_derived(x, names, left):
    """S1-side rank and gap to the S1's top C2 v2 score over all rows of the model (complete per S1)."""
    j = names.index("c2v2:s")
    s = np.nan_to_num(x[:, j], nan=-1.0)
    rank = ra.rank_within(left, s)
    top, _ = ra.top_two(left, s, int(left.max()) + 1)
    x[:, names.index("c2v2:rank_s1")] = np.minimum(rank, 50)
    x[:, names.index("c2v2:gap_s1_top")] = np.where(np.isnan(x[:, j]), np.nan, s - top[left])

def l2_arrays(split, source, n, order=None):
    pre = {"dense": "", "routes": "route_", "rev": "rev_"}[source]
    tag = "brest" if split == "brest" else "test"
    out = np.full((n, len(L_COLS)), np.nan, dtype=np.float32)
    for j, c in enumerate(L_COLS):
        path = DATA / "l2" / f"{tag}_{pre}lw_{c[3:]}.npy"
        if not path.exists():
            log(f"missing {path}: NaN")
            continue
        v = np.load(path).astype(np.float32)
        out[:, j] = v[order] if order is not None else v
    return out


def dense_block(split, names, fams, sources, base_x, pair_keys, ids):
    """Dense rows in ``names`` layout; ``pair_keys`` = pair_row (brest) or cand_row (test)."""
    m = np.full((len(base_x), len(names)), np.nan, dtype=np.float32)
    m[:, :len(BASE_COLS)] = base_x
    if "ed" in fams:
        v, cols, n = load_family("ed", split, pair_keys)
        for j, c in enumerate(cols):
            m[:, names.index(c)] = v[:, j]
    if "l1" in fams:
        tag = "brest" if split == "brest" else "test"
        order = None if split == "brest" else ids["rows"]
        for c in ("full", "nov1"):
            v = np.load(DATA / "l1" / f"l1_{tag}_{c}.npy").astype(np.float32)
            m[:, names.index(f"l1:{c}")] = v if order is None else v[order]
    if "l2" in fams:
        v = l2_arrays(split, "dense", len(base_x) if split == "brest" else len(ids["all_rows"]),
                      None if split == "brest" else None)
        v = v if split == "brest" else v[ids["rows"]]
        for j, c in enumerate(L_COLS):
            m[:, names.index(f"l2:{c}")] = v[:, j]
    if "pu" in fams:
        m[:, names.index("pu:p")] = u8r_p(split, ids["keys"])
    if "c2v2" in fams:
        m[:, names.index("c2v2:s")] = c2v2_dense(split, pair_keys)
    if "g2" in fams:
        v = g2_vals(split, ids["keys"])
        m[:, names.index("g2:sg_p1")] = v[:, 0]
        m[:, names.index("g2:sg_prob")] = v[:, 1]
        log(f"g2 dense coverage {float(np.mean(~np.isnan(v[:, 1]))):.4f}")
    for s in sources:
        m[:, names.index(f"src:{s}")] = 0.0
    return m


def extra_block(split, source, names, fams, sources):
    z = np.load(CACHE / f"{'routes' if source == 'routes' else 'rev'}_{split}.npz", allow_pickle=False)
    xn, x = list(z["x_names"]), z["x"]
    fn, fam = list(z["fam_names"]), z["fam"]
    m = np.full((len(x), len(names)), np.nan, dtype=np.float32)
    for j, n in enumerate(names):
        if n in xn:
            m[:, j] = x[:, xn.index(n)]
        elif n.startswith("ed:") and "ed" in fams and n[3:] in fn:
            m[:, j] = fam[:, fn.index(n[3:])]
        elif n.startswith("route:") and source == "routes" and n[6:] in fn:
            m[:, j] = fam[:, fn.index(n[6:])]
        elif n.startswith("rev:") and source == "rev" and n[4:] in fn:
            m[:, j] = fam[:, fn.index(n[4:])]
    for s in sources:
        m[:, names.index(f"src:{s}")] = 1.0 if s == source else 0.0
    if "l2" in fams:
        n_src = {"routes": {"brest": 689474, "test": 1578869}, "rev": None}[source]
        order = z["src_order"] if source == "rev" else None
        v = l2_arrays(split, source, len(x), order)
        for j, c in enumerate(L_COLS):
            m[:, names.index(f"l2:{c}")] = v[:, j]
    ids = np.asarray(z["rec_id"]).astype(object)
    return m, z, ids


def evaluate_cmd(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fams = [f for f in a.families.split(",") if f]
    sources = [s for s in a.sources.split(",") if s]
    names = columns(fams, sources)
    B, xb, partition, component, n_s1 = brest_matrix()
    ref_ids = np.asarray(pq.read_table(DATA / "arena/reference.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    pool_ids = np.asarray(pq.read_table(DATA / "arena/pool.parquet", columns=["entity_id"]).column("entity_id")
                          .to_pylist(), dtype=object)
    keys = ref_ids[B["left"]] + "|" + pool_ids[B["right"]]
    x = dense_block("brest", names, fams, sources, xb, B["pair_rows"], {"keys": keys})
    del xb
    left, right, label = B["left"], B["right"], B["label"]
    fold = u_folds(B["pair_rows"])
    nd = len(left)
    info = {}
    rec_all = [pool_ids[B["right"]].astype(object)]
    src_all = [np.zeros(nd, dtype=np.int8)]
    for si, s in enumerate(sources):
        m, z, rid = extra_block("brest", s, names, fams, sources)
        rec_all.append(rid)
        src_all.append(np.full(len(m), si + 1, dtype=np.int8))
        if "pu" in fams:
            m[:, names.index("pu:p")] = u8r_p("brest", ref_ids[z["left"]] + "|" + rid)
        if "c2v2" in fams:
            m[:, names.index("c2v2:s")] = c2v2_extra("brest", s, ref_ids[z["left"]] + "|" + rid)
        if "g2" in fams:
            v = g2_vals("brest", ref_ids[z["left"]] + "|" + rid)
            m[:, names.index("g2:sg_p1")], m[:, names.index("g2:sg_prob")] = v[:, 0], v[:, 1]
        x = np.vstack([x, m])
        left = np.concatenate([left, z["left"]])
        right = np.concatenate([right, z["right_ext"]])
        label = np.concatenate([label, z["label"]])
        fold = np.concatenate([fold, z["fold"]])
        info[s] = {"rows": int(len(m)), "true": int(z["label"].sum())}
        log(f"source {s}: +{len(m)} rows ({int(z['label'].sum())} true)")
    if "c2v2" in fams:
        c2v2_derived(x, names, left)
        log(f"c2v2 coverage {float(np.mean(~np.isnan(x[:, names.index('c2v2:s')]))):.4f}")
    if "llm" in fams:
        llm_fill("brest", x, names, left, ref_ids[left].astype(object), np.concatenate(rec_all), out,
                 np.concatenate(src_all))
    tc = np.load(DATA / "labels/true_count.npy")
    country = np.asarray(pq.read_table(DATA / "arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    pop = partition == 1
    s1_fold = np.full(n_s1, -1, dtype=np.int64)
    s1_fold[left] = fold
    p = np.zeros(len(left))
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        model = fit(x[tr], label[tr], names, threads=THREADS, rounds=a.rounds)
        model.save_model(str(out / f"fold{f}.txt"))
        p[te] = model.predict(x[te], num_threads=THREADS)
        log(f"fold {f}: fit {len(tr)}")
    np.save(out / "oof.npy", p.astype(np.float32))
    per_rule = {t: evaluate(ra.decode(p, left, right, ("thr", t), n_s1), left, label, tc, return_per_s1=True)["per_s1"]
                for t in THRESHOLDS}
    pick = {f: max(THRESHOLDS, key=lambda t: per_rule[t][pop & (s1_fold != f) & (s1_fold >= 0)].mean())
            for f in range(5)}
    final = np.zeros(len(left), dtype=bool)
    for f in range(5):
        final |= ra.decode(p, left, right, ("thr", pick[f]), n_s1) & (fold == f)
    final = ra.cap(one_owner(final, left, right, p), p, left)
    res = evaluate(final, left, label, tc, population=pop, country=country, return_per_s1=True)
    per = res.pop("per_s1")
    np.save(out / "per_s1.npy", per)
    dl, dr, dy = B["left"], B["right"], B["label"]
    thr = np.where(component[dl] % 2 == 1, 0.75, 0.725)
    base_acc = ra.cap(one_owner(B["oof_c2tree"] >= thr, dl, dr, B["oof_c2tree"]), B["oof_c2tree"], dl)
    per_tree = evaluate(base_acc, dl, dy, tc, return_per_s1=True)["per_s1"]
    boots = {"vs_pair_tree": component_bootstrap(per_tree, per, component, n=a.bootstrap, seed=20260927, population=pop)}
    for b in (a.baseline or []):
        boots[f"vs_{Path(b).parent.name}"] = component_bootstrap(np.load(b), per, component, n=a.bootstrap,
                                                                  seed=20260927, population=pop)
    full_t = max(THRESHOLDS, key=lambda t: per_rule[t][pop].mean())
    src_acc = {s: int(final[nd + sum(info[q]["rows"] for q in sources[:i]):
                            nd + sum(info[q]["rows"] for q in sources[:i + 1])].sum()) for i, s in enumerate(sources)}
    keysr = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy")
    report = {"families": fams, "sources": sources, "source_info": info, "features": names, "rounds": a.rounds,
              "result": dict({k: res[k] for k in keysr}, by_country={c: v["macro_f05"] for c, v in res["by_country"].items()}),
              "bootstrap": boots, "picks": {str(f): pick[f] for f in pick}, "full_threshold": full_t,
              "accepted_by_source_oof": src_acc, "labels_read": "B_REST only"}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(f"RESULT {fams} {sources}: {res['macro_f05']:.6f}; " + "; ".join(
        f"{k} {v['delta']:+.6f} [{v['ci_low']:+.6f}, {v['ci_high']:+.6f}]" for k, v in boots.items()))


def test_cmd(a):
    import lightgbm as lgb

    out = Path(a.out)
    rep = json.loads((out / "report.json").read_text())
    fams, sources, names = rep["families"], rep["sources"], rep["features"]
    T = test_matrix()
    rows = T["rows"]
    ref_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    pool_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                          .to_pylist(), dtype=object)
    left, right = T["left"][rows], T["right"][rows]
    keys = ref_ids[left] + "|" + pool_ids[right]
    x = dense_block("test", names, fams, sources, T["x"], T["cand_rows"][rows],
                    {"keys": keys, "rows": rows, "all_rows": T["left"]})
    models = [lgb.Booster(model_file=str(out / f"fold{f}.txt")) for f in range(5)]
    rid = pool_ids[right].astype(object)
    nd = len(x)
    mats, src_rows = [x], {}
    for s in sources:
        m, z, r = extra_block("test", s, names, fams, sources)
        if "pu" in fams:
            m[:, names.index("pu:p")] = u8r_p("test", ref_ids[z["left"]] + "|" + r)
        if "c2v2" in fams:
            m[:, names.index("c2v2:s")] = c2v2_extra("test", s, ref_ids[z["left"]] + "|" + r)
        if "g2" in fams:
            v = g2_vals("test", ref_ids[z["left"]] + "|" + r)
            m[:, names.index("g2:sg_p1")], m[:, names.index("g2:sg_prob")] = v[:, 0], v[:, 1]
        start = sum(len(q) for q in mats)
        mats.append(m)
        left = np.concatenate([left, z["left"]])
        right = np.concatenate([right, z["right_ext"]])
        rid = np.concatenate([rid, r])
        src_rows[s] = (start, start + len(m))
    X = np.vstack(mats)
    del mats, x
    if "c2v2" in fams:
        c2v2_derived(X, names, left)
    if "llm" in fams:
        src = np.zeros(len(X), dtype=np.int8)
        for si, s in enumerate(sources):
            src[src_rows[s][0]:src_rows[s][1]] = si + 1
        llm_fill("test", X, names, left, ref_ids[left].astype(object), rid, out, src)
    p = np.mean([m.predict(X, num_threads=THREADS) for m in models], axis=0)
    del X
    # dedupe (s1, record) across sources: keep the highest probability
    k = pd.Series(ref_ids[left] + "|" + rid)
    dup = k.duplicated(keep=False).to_numpy()
    if dup.any():
        order = np.lexsort((-p, k.to_numpy()))
        first = ~k.iloc[order].duplicated().to_numpy()
        keep = np.zeros(len(p), dtype=bool)
        keep[order[first]] = True
    else:
        keep = np.ones(len(p), dtype=bool)
    n_s1 = len(T["code"])
    np.save(out / "test_p.npy", p.astype(np.float32))
    np.save(out / "test_left.npy", left)
    (out / "test_src_rows.json").write_text(json.dumps({k: list(map(int, v)) for k, v in src_rows.items()}) + "\n")
    t = float(rep["full_threshold"])
    p_dec = np.where(keep, p, 0.0)
    if a.extra_thr is not None:
        p_dec[nd:] = np.where(p[nd:] >= a.extra_thr, p_dec[nd:], 0.0)
    if a.route_thr is not None and "routes" in src_rows:  # route rows only (test acceptance 1.5x B_REST at t)
        r0, r1 = src_rows["routes"]
        p_dec[r0:r1] = np.where(p[r0:r1] >= a.route_thr, p_dec[r0:r1], 0.0)
    # records outside the dense pool get ids beyond the pool size per source; unify by record id string
    uniq, rkey = np.unique(rid, return_inverse=True)
    acc = ra.decode(p_dec, left, rkey, ("thr", t), n_s1)
    order = np.flatnonzero(acc)
    order = order[np.lexsort((-p[order], left[order]))]
    starts = np.searchsorted(left[order], np.arange(n_s1), side="left")
    stops = np.searchsorted(left[order], np.arange(n_s1), side="right")
    tag = (f"_extra{a.extra_thr}" if a.extra_thr is not None else "") + (f"_route{a.route_thr}" if a.route_thr is not None else "")
    path = out / f"us_india_rows{tag}.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i in np.flatnonzero(T["code"] != 0):
            f.write(f"{ref_ids[i]}\t{','.join(rid[order[starts[i]:stops[i]]])}\n")
    oof = np.load(out / "oof.npy")
    code = T["code"]
    per_src = {}
    for s, (a0, a1) in src_rows.items():
        cc = code[left[a0:a1]]
        per_src[s] = {"rows": int(a1 - a0), "accepted": int(acc[a0:a1].sum()),
                      "per_1k_India": float(1000 * acc[a0:a1][cc == 1].mean()) if (cc == 1).any() else None,
                      "per_1k_US": float(1000 * acc[a0:a1][cc == 2].mean()) if (cc == 2).any() else None}
    lengths = (stops - starts)[code != 0]
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "links": int(lengths.sum()),
              "longest": int(lengths.max()), "threshold": t, "extra_thr": a.extra_thr, "route_thr": a.route_thr, "per_source": per_src,
              "share_dense_p_ge_t_test": float((p[:nd] >= t).mean()), "share_p_ge_t_oof_all": float((oof >= t).mean()),
              "labels_read": False}
    (out / f"test_report{tag}.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    log(json.dumps(result, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["evaluate", "test"])
    p.add_argument("--families", default="ed")
    p.add_argument("--sources", default="routes")
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=500)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--baseline", action="append")
    p.add_argument("--extra-thr", type=float)
    p.add_argument("--route-thr", type=float)
    a = p.parse_args()
    {"evaluate": evaluate_cmd, "test": test_cmd}[a.command](a)


if __name__ == "__main__":
    main()
