"""K1 integration layer: K-robust pair GBDT + accepted external feature families -> US/India rows.

Base features = K-robust (``record_assign``: v1+C2 x_base, Qwen x_q, K3 edit ops + K3 prob, density-robust
claimant/S1-side features). Families are joined by pair keys (B_REST ``pair_row``, test ``cand_row``):

* ``ed``   U's edit/decoy block ``ed_*`` (``run27/U/features``), ``coll`` U's collective block ``coll_*``,
* ``pu4``  U's stacked OOF probability ``p_u4_ed`` (valid only with U's fold column, which is used here),
* ``g``    Agent G string-feature table (parquet with ``pair_row``/``cand_row`` + numeric columns),
* ``l``    Agent L listwise score(s) (same parquet convention, or .npy aligned to U prep order).

Protocol (B_REST only, labels of other partitions never read): 5-fold cross-fit on U's frozen fold column
(hash of claim-graph component), each fold's threshold chosen on the other folds' OOF; paired component
bootstrap vs the C2 pair tree, U4-ed (U's per-S1) and K-robust on the same folds. Adversarial AUC (B_REST vs
test, within country) per family. ``test`` = mean of the 5 fold models, full-OOF threshold, one owner + cap 11.

    python -m ber.k_integrate evaluate --families ed,coll --out DIR
    python -m ber.k_integrate test --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber import record_assign as ra
from ber.arena_decision import component_bootstrap, evaluate, one_owner

T0 = time.time()
DATA = ra.DATA
UF = DATA / "ufeat"
DENSE = ["rec_out_best_base", "rec_out_best_k3", "rec_n_claim_grp", "rec_n_claim_all", "rec_best_c2_grp",
         "rec_best_q_grp", "rec_k3_mass", "rank_base_in_rec", "rank_k3_in_rec", "rec_base_ge_0_5", "rec_k3_ge_0_5",
         "is_none"]
BASE_COLS = [n for n in ra.NAMES if n not in DENSE]
PARAMS = dict(ra.PARAMS, objective="binary")
ROUNDS = 500
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.5, 0.951, 0.025), 3))


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def family_columns(fam, names):
    if fam == "ed":
        return [c for c in names if c.startswith("ed_")]
    if fam == "coll":
        return [c for c in names if c.startswith("coll_")]
    if fam == "pu4":
        return ["p_u4_ed"]
    raise KeyError(fam)


def join(keys_needed, table_keys, values):
    """Rows of ``values`` aligned to ``keys_needed`` (NaN where the key is absent)."""
    order = np.argsort(table_keys)
    sk = table_keys[order]
    pos = np.minimum(np.searchsorted(sk, keys_needed), len(sk) - 1)
    ok = sk[pos] == keys_needed
    out = np.full((len(keys_needed), values.shape[1]), np.nan, dtype=np.float32)
    out[ok] = values[order[pos[ok]]]
    return out, int(ok.sum())


def load_family(fam, split, keys_needed):
    """(matrix, column names, rows matched) for family ``fam`` on ``split`` in {brest, test}."""
    if fam in ("ed", "coll", "pu4"):
        path = UF / f"u_features_{split}.parquet"
        names = pq.ParquetFile(path).schema_arrow.names
        cols = family_columns(fam, names)
        key = "pair_row" if split == "brest" else "cand_row"
        t = pq.read_table(path, columns=[key] + cols)
        vals = np.column_stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in cols])
        m, n = join(keys_needed, t.column(key).to_numpy().astype(np.int64), vals)
        return m, [f"{fam}:{c}" for c in cols], n
    if fam == "l":  # Agent L listwise features, .npy aligned to U prep order (lw_group_size excluded: density)
        cols = ["lw_p", "lw_p_none", "lw_rival_p"]
        if split == "brest":
            table_keys = np.load(DATA / "brest_prep/pair_rows.npy").astype(np.int64)
        else:
            table_keys = np.load(DATA / "test_prep/cand_rows.npy").astype(np.int64)
        vals = np.column_stack([np.load(DATA / "lfeat" / f"{split}_{c}.npy").astype(np.float32) for c in cols])
        if len(vals) != len(table_keys):
            raise SystemExit(f"L {split} arrays have {len(vals)} rows, expected {len(table_keys)}")
        m, n = join(keys_needed, table_keys, vals)
        return m, [f"l:{c}" for c in cols], n
    if fam in ("g", "l_parquet"):
        path = DATA / f"{fam}feat" / f"{split}.parquet"
        t = pq.read_table(path)
        key = "pair_row" if split == "brest" else "cand_row"
        cols = [c for c in t.column_names if c not in (key, "left_idx", "right_idx", "s1_id", "rec_id", "fold")]
        vals = np.column_stack([t.column(c).to_numpy(zero_copy_only=False).astype(np.float32) for c in cols])
        m, n = join(keys_needed, t.column(key).to_numpy().astype(np.int64), vals)
        return m, [f"{fam}:{c}" for c in cols], n
    raise KeyError(fam)


def u_folds(pair_rows):
    t = pq.read_table(UF / "u_features_brest.parquet", columns=["pair_row", "fold"])
    f, n = join(pair_rows, t.column("pair_row").to_numpy().astype(np.int64),
                t.column("fold").to_numpy().astype(np.float32)[:, None])
    if n != len(pair_rows):
        raise SystemExit(f"U fold column covers {n} of {len(pair_rows)} B_REST rows")
    return f[:, 0].astype(np.int64)


def fit(x, y, names, threads=30, rounds=ROUNDS):
    import lightgbm as lgb

    cat = [names.index(c) for c in ("legal_rel", "hn_rel") if c in names]
    ds = lgb.Dataset(x, y.astype(np.float32), feature_name=[n.replace(":", "__") for n in names],
                     categorical_feature=cat, free_raw_data=True)
    return lgb.train(dict(PARAMS, num_threads=threads), ds, num_boost_round=rounds)


def auc(score, y):
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    npos, nneg = (y == 1).sum(), (y == 0).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def adversarial(xa, ca, xt, ct, names, seed=0):
    """AUC of B_REST (0) vs test (1) within country; ``ca``/``ct`` are 'India'/'US' per row."""
    import lightgbm as lgb

    rng = np.random.default_rng(seed)
    out = {}
    for c in ("India", "US"):
        pa, pt = np.flatnonzero(ca == c), np.flatnonzero(ct == c)
        ia = rng.choice(pa, min(300_000, len(pa)), replace=False)
        it = rng.choice(pt, min(300_000, len(pt)), replace=False)
        X = np.vstack([xa[ia], xt[it]])
        y = np.r_[np.zeros(len(ia)), np.ones(len(it))]
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]
        n = int(0.7 * len(y))
        m = lgb.train({"objective": "binary", "learning_rate": 0.1, "num_leaves": 63, "verbose": -1, "num_threads": 30},
                      lgb.Dataset(X[:n], y[:n], feature_name=[s.replace(":", "__") for s in names]), 120)
        gain = sorted(zip(names, m.feature_importance("gain")), key=lambda kv: -kv[1])[:5]
        out[c] = {"auc": auc(m.predict(X[n:]), y[n:]), "top": [g[0] for g in gain]}
    return out


def route_block(split, names, fams):
    """Route rows as a matrix in the column layout ``names`` (BASE_COLS + family columns + route block)."""
    z = np.load(Path("/home/ubuntu/k/cache") / f"routes_{split}.npz", allow_pickle=False)
    xn = list(z["x_names"])
    x = z["x"]
    fam_names = list(z["fam_names"])
    fam = z["fam"]
    m = np.full((len(x), len(names)), np.nan, dtype=np.float32)
    for j, n in enumerate(names):
        if n in xn:
            m[:, j] = x[:, xn.index(n)]
        elif ":" in n:
            f, c = n.split(":", 1)
            if f == "route" and c == "is_route":
                m[:, j] = 1.0
            elif c in fam_names and (f in fams or f == "route"):
                m[:, j] = fam[:, fam_names.index(c)]
    return m, z


ROUTE_NAMES = ["route:is_route"] + [f"route:{c}" for c in ("route_flag", "route_addr_score", "route_addr_rank",
                                                           "route_empty_score", "route_empty_rank")]


# ----------------------------------------------------------------------------- matrices

def brest_matrix():
    B, x, half, grp_key, partition, component, n_s1 = ra.build_brest()
    x = x[:, [ra.NAMES.index(n) for n in BASE_COLS]]
    return B, x, partition, component, n_s1


def test_matrix():
    """K-robust base features on test US/India rows (U test_prep order restricted to US/India)."""
    d = DATA / "test_prep"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    cand_rows = np.load(d / "cand_rows.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")
    xb = np.asarray(np.load(d / "x_base.npy", mmap_mode="r"), dtype=np.float32)
    xq = np.asarray(np.load(d / "x_q.npy", mmap_mode="r"), dtype=np.float32)
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    k3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
    rec_ids = pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    rows = np.flatnonzero(code[left] != 0)
    claim = ra.claimant_features(right, left, xb[:, ra.BASE_NAMES.index("base")], k3, np.zeros(len(left), bool),
                                 len(rec_ids), len(code), rows, rec_s3[right[rows]],
                                 np.nan_to_num(xb[rows, ra.BASE_NAMES.index("c2")], nan=-1.0),
                                 np.nan_to_num(xq[rows, 0], nan=-1.0), right[rows])
    x = np.hstack([xb[rows], xq[rows], np.asarray(feat[rows], dtype=np.float32), claim])
    x = x[:, [ra.NAMES.index(n) for n in BASE_COLS]]
    return {"x": x, "rows": rows, "left": left, "right": right, "cand_rows": cand_rows, "code": code}


# ----------------------------------------------------------------------------- commands

def evaluate_cmd(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fams = [f for f in a.families.split(",") if f]
    B, x, partition, component, n_s1 = brest_matrix()
    left, right, label = B["left"], B["right"], B["label"]
    names = list(BASE_COLS)
    fam_info = {}
    for fam in fams:
        m, cols, n = load_family(fam, "brest", B["pair_rows"])
        x = np.hstack([x, m])
        names += cols
        fam_info[fam] = {"columns": len(cols), "rows_matched": n}
        log(f"family {fam}: {len(cols)} cols, {n} of {len(left)} rows matched")
    fold = u_folds(B["pair_rows"])
    rec_empty = B["rec_empty"]
    if a.routes:
        x = np.hstack([x, np.zeros((len(x), 1), np.float32), np.full((len(x), len(ROUTE_NAMES) - 1), np.nan,
                                                                     np.float32)])
        names += ROUTE_NAMES
        rm, z = route_block("brest", names, fams)
        dense_keys = left * 20_000_000 + right
        rkeys = z["left"] * 20_000_000 + z["right"]
        if np.isin(rkeys, dense_keys).any():
            raise SystemExit("route rows overlap the dense candidate rows")
        x = np.vstack([x, rm])
        left = np.concatenate([left, z["left"]])
        right = np.concatenate([right, z["right"]])
        label = np.concatenate([label, z["label"]])
        fold = np.concatenate([fold, z["fold"]])
        rec_empty = np.concatenate([rec_empty, rm[:, names.index("rec_addr_empty")] > 0.5])
        fam_info["routes"] = {"rows": int(len(rm)), "true": int(z["label"].sum())}
        log(f"routes: +{len(rm)} rows ({int(z['label'].sum())} true)")
    true_count = np.load(DATA / "labels/true_count.npy")
    country = np.asarray(pq.read_table(DATA / "arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    pop = partition == 1
    # per-S1 fold: every row of an S1 shares its component, hence its fold
    s1_fold = np.full(n_s1, -1, dtype=np.int64)
    s1_fold[left] = fold
    p = np.zeros(len(left), dtype=np.float64)
    if a.reuse and (out / "oof.npy").exists() and all((out / f"fold{f}.txt").exists() for f in range(5)):
        p = np.load(out / "oof.npy").astype(np.float64)
        if len(p) != len(left):
            raise SystemExit("saved OOF does not match the rows")
        log("reused saved fold models and OOF")
    else:
        for f in range(5):
            tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
            model = fit(x[tr], label[tr], names, rounds=a.rounds)
            model.save_model(str(out / f"fold{f}.txt"))
            p[te] = model.predict(x[te], num_threads=30)
            log(f"fold {f}: fit {len(tr)}, predicted {len(te)}")
        np.save(out / "oof.npy", p.astype(np.float32))
    rules = [("thr", t) for t in THRESHOLDS]
    per_rule = {r: evaluate(ra.decode(p, left, right, r, n_s1), left, label, true_count, return_per_s1=True)["per_s1"]
                for r in rules}
    pick = {f: max(rules, key=lambda r: per_rule[r][pop & (s1_fold != f) & (s1_fold >= 0)].mean()) for f in range(5)}
    final = np.zeros(len(left), dtype=bool)
    for f in range(5):
        final |= ra.decode(p, left, right, pick[f], n_s1) & (fold == f)
    final = ra.cap(one_owner(final, left, right, p), p, left)
    res = evaluate(final, left, label, true_count, population=pop, country=country, return_per_s1=True)
    per = res.pop("per_s1")
    np.save(out / "per_s1.npy", per)
    # paired comparisons
    dl, dr, dy = B["left"], B["right"], B["label"]
    thr = np.where(component[dl] % 2 == 1, 0.75, 0.725)
    base_acc = ra.cap(one_owner(B["oof_c2tree"] >= thr, dl, dr, B["oof_c2tree"]), B["oof_c2tree"], dl)
    per_tree = evaluate(base_acc, dl, dy, true_count, return_per_s1=True)["per_s1"]
    boots = {"vs_pair_tree": component_bootstrap(per_tree, per, component, n=a.bootstrap, seed=20260927,
                                                 population=pop)}
    u4 = UF / "per_s1_u4_ed.npy"
    if u4.exists():
        boots["vs_u4_ed"] = component_bootstrap(np.load(u4), per, component, n=a.bootstrap, seed=20260927,
                                                population=pop)
    if a.baseline and Path(a.baseline).exists():
        boots["vs_baseline"] = component_bootstrap(np.load(a.baseline), per, component, n=a.bootstrap,
                                                   seed=20260927, population=pop)
    ea, lab = rec_empty, label & rec_empty
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy",
            "empty_predictions")
    report = {"families": fams, "routes": bool(a.routes), "family_info": fam_info, "features": names, "rounds": a.rounds, "params": PARAMS,
              "folds": "U frozen fold column (hash u4-20260927 of claim-graph component) % 5",
              "result": dict({k: res[k] for k in keys}, by_country={c: {k: v[k] for k in ("macro_f05", "precision",
                                                                                              "recall", "false_links")}
                                                                    for c, v in res["by_country"].items()}),
              "ea": {"ea_recall_in_candidates": float((final & lab).sum() / max(1, lab.sum())),
                     "ea_precision": float((final & lab).sum() / max(1, (final & ea).sum()))},
              "bootstrap": boots, "picks": {str(f): list(pick[f]) for f in range(5)},
              "full_pick_for_test": list(max(rules, key=lambda r: per_rule[r][pop].mean())),
              "full_oof_grid": {str(r[1]): float(per_rule[r][pop].mean()) for r in rules},
              "labels_read": "B_REST only"}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(f"RESULT {fams}{' +routes' if a.routes else ''}: {res['macro_f05']:.6f}; " + "; ".join(
        f"{k} {v['delta']:+.6f} [{v['ci_low']:+.6f}, {v['ci_high']:+.6f}]" for k, v in boots.items()))
    if a.adversarial and a.routes:
        zt = np.load(Path("/home/ubuntu/k/cache") / "routes_test.npz", allow_pickle=False)
        rt, _ = route_block("test", names, fams)
        code = np.load(DATA / "test_prep/country_code.npy")
        ct_r = np.where(code[zt["left"]] == 1, "India", "US")
        nb = len(B["left"])
        j = [names.index(c) for c in ROUTE_NAMES[1:]] + [names.index("k3"), names.index("rival_k3_all"),
                                                        names.index("k3_minus_rival_all")]
        cols = [names[i] for i in j]
        report["adversarial_route_rows"] = adversarial(x[nb:][:, j], country[left[nb:]], rt[:, j], ct_r, cols)
        log(f"adversarial route rows: {report['adversarial_route_rows']}")
    if a.adversarial and fams:
        T = test_matrix()
        ct = np.where(T["code"][T["left"][T["rows"]]] == 1, "India", "US")
        ca = country[left]
        adv = {}
        for fam in fams:
            mt, cols, n = load_family(fam, "test", T["cand_rows"][T["rows"]])
            j = [names.index(c) for c in cols]
            adv[fam] = dict(adversarial(x[:, j], ca, mt, ct, cols), test_rows_matched=n)
            log(f"adversarial {fam}: {adv[fam]}")
        report["adversarial"] = adv
    report["elapsed_seconds"] = round(time.time() - T0, 1)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(f"RESULT {fams}: {res['macro_f05']:.6f}; " + "; ".join(f"{k} {v['delta']:+.6f} [{v['ci_low']:+.6f}, {v['ci_high']:+.6f}]"
                                                             for k, v in boots.items()))


def test_cmd(a):
    import lightgbm as lgb

    out = Path(a.out)
    rep = json.loads((out / "report.json").read_text())
    kind, val = rep["full_pick_for_test"]
    T = test_matrix()
    x = T["x"]
    names = list(BASE_COLS)
    for fam in rep["families"]:
        m, cols, n = load_family(fam, "test", T["cand_rows"][T["rows"]])
        x = np.hstack([x, m])
        names += cols
        log(f"test family {fam}: {n} of {len(T['rows'])} rows matched")
    routes = bool(rep.get("routes"))
    if routes:
        x = np.hstack([x, np.zeros((len(x), 1), np.float32), np.full((len(x), len(ROUTE_NAMES) - 1), np.nan,
                                                                     np.float32)])
        names += ROUTE_NAMES
    if names != rep["features"]:
        raise SystemExit("test feature order differs from the evaluated model")
    models = [lgb.Booster(model_file=str(out / f"fold{f}.txt")) for f in range(5)]
    s1_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                        .to_pylist(), dtype=object)
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    rows = T["rows"]
    left, right = T["left"][rows], T["right"][rows]
    rid = rec_ids[right]
    p = np.mean([m.predict(x, num_threads=30) for m in models], axis=0)
    n_dense = len(left)
    if routes:
        rm, z = route_block("test", names, rep["families"])
        dense_keys = left * 20_000_000 + right
        if np.isin(z["left"] * 20_000_000 + z["right_ext"], dense_keys).any():
            raise SystemExit("test route rows overlap the dense candidate rows")
        p = np.concatenate([p, np.mean([m.predict(rm, num_threads=30) for m in models], axis=0)])
        left = np.concatenate([left, z["left"]])
        right = np.concatenate([right, z["right_ext"]])
        rid = np.concatenate([rid, z["rec_id"].astype(object)])
    n_s1 = len(T["code"])
    np.save(out / "test_p.npy", p.astype(np.float32))
    p_dec = p.copy()
    if routes and a.route_thr is not None:  # stricter threshold on route rows only (label-free rate matching)
        p_dec[n_dense:] = np.where(p[n_dense:] >= a.route_thr, p[n_dense:], 0.0)
    acc = ra.decode(p_dec, left, right, (kind, val), n_s1)
    order = np.flatnonzero(acc)
    order = order[np.lexsort((-p[order], left[order]))]
    starts = np.searchsorted(left[order], np.arange(n_s1), side="left")
    stops = np.searchsorted(left[order], np.arange(n_s1), side="right")
    path = out / (f"us_india_rows_routethr{a.route_thr}.tsv" if a.route_thr is not None else "us_india_rows.tsv")
    with open(path.with_suffix(".tmp"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i in np.flatnonzero(T["code"] != 0):
            f.write(f"{s1_ids[i]}\t{','.join(rid[order[starts[i]:stops[i]]])}\n")
    path.with_suffix(".tmp").replace(path)
    route_acc = np.flatnonzero(acc[n_dense:]) + n_dense
    with open(out / (f"accepted_route_pairs_routethr{a.route_thr}.tsv" if a.route_thr is not None
                     else "accepted_route_pairs.tsv"), "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tentity_id\n")
        for r in route_acc:
            f.write(f"{s1_ids[left[r]]}\t{rid[r]}\n")
    import hashlib

    lengths = (stops - starts)[T["code"] != 0]
    oof = np.load(out / "oof.npy")
    result = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": int(len(lengths)),
              "links": int(lengths.sum()), "longest": int(lengths.max()), "accepted_route_links": int(len(route_acc)),
              "accepted_route_links_outside_dense_pool": int((right[route_acc] >= len(rec_ids)).sum()),
              "rule": [kind, val], "share_p_ge_rule_test": float((p[:n_dense] >= val).mean()),
              "share_p_ge_rule_oof": float((oof >= val).mean()), "labels_read": False}
    (out / (f"test_report_routethr{a.route_thr}.json" if a.route_thr is not None else "test_report.json")).write_text(
        json.dumps(result, indent=2, default=float) + "\n")
    log(json.dumps(result, default=float))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["evaluate", "test"])
    p.add_argument("--families", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--rounds", type=int, default=ROUNDS)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--baseline", help="per-S1 .npy of a model to compare against (paired)")
    p.add_argument("--adversarial", action="store_true")
    p.add_argument("--route-thr", type=float, help="test: stricter threshold for route rows only")
    p.add_argument("--reuse", action="store_true", help="reuse saved fold models/OOF in --out")
    p.add_argument("--routes", action="store_true", help="add U's U8f route rows (needs ber.k_routes caches)")
    a = p.parse_args()
    {"evaluate": evaluate_cmd, "test": test_cmd}[a.command](a)


if __name__ == "__main__":
    main()
