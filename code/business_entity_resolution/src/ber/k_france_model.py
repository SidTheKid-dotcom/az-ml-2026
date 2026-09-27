"""C2-free claimant-aware K model for France: v1 pair features + K3 edit ops + K3 prob + density-robust claimant
features (no C2, no Qwen), trained on US/India B_REST (5-fold on U's fold column), gated by leave-one-country-out
(US -> India, India -> US; threshold from the training country's OOF), then applied to French test candidates with
label-free diagnostics vs the scored S1 file (added/removed decoy-fingerprint rates, probes, list length).

    python -m ber.k_france_model --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber import record_assign as ra
from ber.arena_decision import component_bootstrap, evaluate
from ber.k3_test import DATA, keys, read_links, write_rows
from ber.k_integrate import u_folds

T0 = time.time()
V1 = ra.BASE_NAMES[:12]
CLAIM = ["k3", "rec_s3", "rec_best_base", "rec_best_k3", "rival_base_all", "rival_k3_all", "base_minus_rival_all",
         "k3_minus_rival_all", "s1_n_rows", "s1_k3_mass", "s1_k3_ge_0_5", "k3_rank_in_s1", "k3_gap_to_s1_top"]
COLS = V1 + ra.K3F + CLAIM
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.5, 0.951, 0.025), 3))
PARAMS = dict(ra.PARAMS, objective="binary")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def fit(x, y, rounds=500):
    import lightgbm as lgb

    names = COLS
    ds = lgb.Dataset(x, y.astype(np.float32), feature_name=names,
                     categorical_feature=[names.index("legal_rel"), names.index("hn_rel")], free_raw_data=True)
    return lgb.train(dict(PARAMS, num_threads=30), ds, num_boost_round=rounds)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    B, x, half, grp_key, partition, component, n_s1 = ra.build_brest()
    x = x[:, [ra.NAMES.index(n) for n in COLS]]
    left, right, label = B["left"], B["right"], B["label"]
    fold = u_folds(B["pair_rows"])
    tc = np.load(DATA / "labels/true_count.npy")
    country = np.asarray(pq.read_table(DATA / "arena/reference.parquet", columns=["country"]).column("country")
                         .to_pylist(), dtype=object)
    pop = partition == 1
    rc = country[left]
    p = np.zeros(len(left))
    models = []
    for f in range(5):
        tr, te = np.flatnonzero(fold != f), np.flatnonzero(fold == f)
        m = fit(x[tr], label[tr])
        m.save_model(str(out / f"fold{f}.txt"))
        models.append(m)
        p[te] = m.predict(x[te], num_threads=30)
        log(f"fold {f} done")
    per_rule = {t: evaluate(ra.decode(p, left, right, ("thr", t), n_s1), left, label, tc, return_per_s1=True)["per_s1"]
                for t in THRESHOLDS}
    best = max(THRESHOLDS, key=lambda t: per_rule[t][pop].mean())
    res = evaluate(ra.decode(p, left, right, ("thr", best), n_s1), left, label, tc, population=pop, country=country)
    report = {"features": COLS, "b_rest_5fold": {k: res[k] for k in ("macro_f05", "precision", "recall", "false_links")},
              "b_rest_by_country": {c: v["macro_f05"] for c, v in res["by_country"].items()},
              "threshold_full_oof": best, "note": "full-OOF threshold (optimistic by the pick); LOCO below is the gate"}
    log(f"B_REST C2-free K: {res['macro_f05']:.6f} at {best}")
    # LOCO
    loco = {}
    for tr_c, te_c in (("US", "India"), ("India", "US")):
        tr = np.flatnonzero(rc == tr_c)
        te = np.flatnonzero(rc == te_c)
        m = fit(x[tr], label[tr])
        pt = m.predict(x[te], num_threads=30)
        pop_tr = pop & (country == tr_c)
        t_tr = max(THRESHOLDS, key=lambda t: per_rule[t][pop_tr].mean())  # from the training country's OOF
        pop_te = pop & (country == te_c)
        r = evaluate(ra.decode(pt, left[te], right[te], ("thr", t_tr), n_s1), left[te], label[te], tc,
                     population=pop_te)
        same = res["by_country"][te_c]["macro_f05"]
        loco[f"{tr_c}_to_{te_c}"] = {"threshold": t_tr, "macro_f05": r["macro_f05"], "precision": r["precision"],
                                     "recall": r["recall"], "in_country_5fold": same, "drop": r["macro_f05"] - same}
        log(f"LOCO {tr_c}->{te_c}: {r['macro_f05']:.6f} (in-country {same:.6f})")
    report["loco"] = loco
    # France
    d = DATA / "test_prep"
    tl = np.load(d / "left.npy").astype(np.int64)
    tr_ = np.load(d / "right.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")
    xb = np.load(d / "x_base.npy", mmap_mode="r")
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    k3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    s1_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                        .to_pylist(), dtype=object)
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    rows = np.flatnonzero(code[tl] == 0)
    base = np.asarray(xb[:, ra.BASE_NAMES.index("base")], dtype=np.float32)
    claim = ra.claimant_features(tr_, tl, base, k3, np.zeros(len(tl), bool), len(rec_ids), len(code), rows,
                                 rec_s3[tr_[rows]], np.full(len(rows), -1.0, np.float32),
                                 np.full(len(rows), -1.0, np.float32), tr_[rows])
    cn = [n for n in ra.CLAIM_NAMES]
    xf = np.hstack([np.asarray(xb[rows][:, :12], dtype=np.float32), np.asarray(feat[rows], dtype=np.float32),
                    claim[:, [cn.index(n) for n in CLAIM]]])
    pf = np.zeros(len(tl))
    pf[rows] = np.mean([m.predict(xf, num_threads=30) for m in models], axis=0)
    np.save(out / "france_prob.npy", pf[rows].astype(np.float32))
    fr = code[tl] == 0
    acc = ra.decode(pf, tl, tr_, ("thr", best), len(code)) & fr
    s1_index = {v: i for i, v in enumerate(s1_ids)}
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    n_rec = len(rec_ids)
    ck = keys(tl, tr_, n_rec)
    order = np.argsort(ck)
    sk = ck[order]

    def rows_of(kk):
        pos = np.minimum(np.searchsorted(sk, kk), len(sk) - 1)
        return order[pos[sk[pos] == kk]]

    sl, sr = read_links(DATA / "s1file/matching_results.tsv", s1_index, rec_index)
    in_s1 = np.zeros(len(tl), bool)
    in_s1[rows_of(np.unique(keys(sl, sr, n_rec)))] = True
    from ber.edit_ops import FEATURES as F

    def rates(mask):
        r = np.flatnonzero(mask)
        if not len(r):
            return {"links": 0}
        f = np.asarray(feat[r], dtype=np.float32)
        return {"links": int(len(r)), "hn_delta_shift": float(np.mean(f[:, F.index("hn_in_delta")] == 1)),
                "hn_even_shift": float(np.mean(f[:, F.index("hn_even_shift")] == 1)),
                "decoy_word_added": float(np.mean(f[:, F.index("decoy_added")] > 0)),
                "legal_swapped": float(np.mean(f[:, F.index("legal_rel")] == 5)),
                "record_empty_address": float(np.mean(f[:, F.index("rec_addr_empty")] == 1)),
                "mean_p": float(pf[r].mean())}

    base_fr = in_s1 & fr
    probes = {}
    for name, fn in (("A_T2r", "probeA_T2r.tsv"), ("Cprime", "probeCprime.tsv")):
        import pandas as pd

        pr = pd.read_csv(DATA / "probes" / fn, sep="\t", dtype=str)
        kk = np.asarray([s1_index[s] * n_rec + rec_index[r] for s, r in zip(pr.iloc[:, 0], pr.iloc[:, 1])])
        rr = rows_of(np.unique(kk))
        probes[name] = {"mean_p": float(pf[rr].mean()), "share_accepted": float(acc[rr].mean())}
    lens = np.bincount(tl[acc], minlength=len(code))[code == 0]
    report["france"] = {"links": int(acc.sum()), "s1_file_links": int(base_fr.sum()), "added": rates(acc & ~in_s1),
                        "removed": rates(base_fr & ~acc), "probes": probes, "mean_list": float(lens.mean()),
                        "empty_rows": int((lens == 0).sum()),
                        "file": write_rows(out / "france_rows.tsv", s1_ids, rec_ids, tl, tr_, acc,
                                           pf.astype(np.float32), code == 0)}
    report["labels_read"] = "B_REST only"
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps({k: report[k] for k in ("b_rest_5fold", "loco")}, default=float))
    log(json.dumps(report["france"], default=float)[:1500])


if __name__ == "__main__":
    main()
