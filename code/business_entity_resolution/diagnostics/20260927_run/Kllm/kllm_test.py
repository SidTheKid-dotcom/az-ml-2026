"""Apply the frozen B_REST overlay (brest_report.json; weights fitted on all B_REST band rows, full-B_REST threshold)
to the K-final F mixture US/India test rows. No test labels exist; label-free checks only.

Usage: kllm_test.py VARIANT   (e.g. no_us_dense/llm)
"""
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/home/ubuntu/kllm")
from kllm_brest import auc, decide, logit  # noqa: E402

T0 = time.time()
H = "/home/ubuntu/run27"
W = "/home/ubuntu/kllm"
OUT = f"{W}/out"


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def main():
    variant = sys.argv[1]
    rep_b = json.load(open(f"{OUT}/{sys.argv[2] if len(sys.argv) > 2 else 'brest_report.json'}"))
    v = rep_b["variants"][variant]
    w, cols, t = np.asarray(v["weights_full"]), v["cols"], float(v["full_threshold"])
    ref = pq.read_table(f"{H}/U/data/test/dense_test/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    ctry = pd.Series(ref.country.to_numpy(), index=ref.entity_id.to_numpy())
    # K-final F mixture US/India rows (validated file)
    K = pd.read_csv(f"{W}/in/us_india_mix_F_C.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert list(K.columns) == ["source1_entity_id", "matched_entity_ids"]
    s1_order = K.source1_entity_id.to_numpy()
    assert len(set(s1_order)) == len(s1_order)
    kl = K.assign(rec=K.matched_entity_ids.str.split(",")).explode("rec")
    kl = kl[kl.rec.fillna("").str.len() > 0][["source1_entity_id", "rec"]].rename(columns={"source1_entity_id": "s1_id", "rec": "rec_id"})
    kl["korder"] = np.arange(len(kl))
    log(f"K rows {len(K)} S1s, {len(kl)} links")
    kpairs = set(zip(kl.s1_id, kl.rec_id))
    L = pd.read_parquet(f"{W}/in/llm_test.parquet")
    L["country"] = ctry.reindex(L.s1_id.to_numpy()).to_numpy()
    assert L.country.notna().all()
    assert L.s1_id.isin(set(s1_order)).all()
    vm = np.ones(len(L), bool) if variant.startswith("all_band") else ~((L.row_type == 0) & (L.country == "US")).to_numpy()
    if variant.startswith("no_us_dense_no_route"):
        vm &= (L.row_type != 1).to_numpy()
    B = L[vm].reset_index(drop=True)
    rt = B.row_type.to_numpy()
    ind = (B.country == "India").to_numpy().astype(float)
    a = np.fromiter(((x, y) in kpairs for x, y in zip(B.s1_id, B.rec_id)), bool, len(B)).astype(float)
    Lu = logit(B.u_p.to_numpy())
    zl = B.llm_logit.to_numpy().astype(np.float64)
    r1, r2 = (rt == 1).astype(float), (rt == 2).astype(float)
    X = np.column_stack([a, Lu, a * Lu, r1, r2, ind, r1 * a, r2 * a, ind * a, r1 * Lu, r2 * Lu,
                         zl, zl * a, r1 * zl, r2 * zl, ind * zl, zl * Lu])[:, cols]
    q = 1 / (1 + np.exp(-(np.hstack([X, np.ones((len(X), 1))]) @ w)))
    B["q"], B["a"] = q, a
    log(f"band rows {len(B)} (variant {variant}); K-accepted {int(a.sum())}; q>=t {int((q >= t).sum())} at t {t}")
    # universe: K links outside the band keep priority 1; band rows carry q
    band_pairs = set(zip(B.s1_id, B.rec_id))
    kl_nb = kl[[p not in band_pairs for p in zip(kl.s1_id, kl.rec_id)]]
    U = pd.concat([pd.DataFrame({"s1_id": kl_nb.s1_id, "rec_id": kl_nb.rec_id, "p": 1.0, "thr": 0.5, "src": -1,
                                 "korder": kl_nb.korder}),
                   pd.DataFrame({"s1_id": B.s1_id, "rec_id": B.rec_id, "p": B.q, "thr": t, "src": B.row_type,
                                 "korder": -1})], ignore_index=True)
    s1c, s1u = pd.factorize(U.s1_id)
    recc, recu = pd.factorize(U.rec_id)
    is_s2 = np.asarray([r.startswith("S2") for r in recu], bool)[recc]
    sel = decide(U.p.to_numpy(), s1c.astype(np.int64), recc.astype(np.int64), U.thr.to_numpy(), is_s2)
    A = U[sel].copy()
    A = A.sort_values(["s1_id", "p", "korder"], ascending=[True, False, True]).drop_duplicates(["s1_id", "rec_id"])
    new_pairs = set(zip(A.s1_id, A.rec_id))
    added = new_pairs - kpairs
    removed = kpairs - new_pairs
    log(f"new links {len(new_pairs)}; added {len(added)}; removed {len(removed)}")
    # write in K's S1 order
    grp = A.groupby("s1_id", sort=False).rec_id.apply(list).to_dict()
    path = f"/home/ubuntu/kllm/out/kfJ_us_india_rows_{variant.split('/')[0]}.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1_order:
            f.write(f"{s}\t{','.join(grp.get(s, []))}\n")
    # validation
    chk = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    assert list(chk.source1_entity_id) == list(s1_order)
    ll = chk.assign(rec=chk.matched_entity_ids.str.split(",")).explode("rec")
    ll = ll[ll.rec.fillna("").str.len() > 0]
    assert not ll.duplicated(["source1_entity_id", "rec"]).any()
    assert not ll.rec.duplicated().any(), "one owner violated"
    n_per = ll.groupby("source1_entity_id").size()
    n_s2 = ll[ll.rec.str.startswith("S2")].groupby("source1_entity_id").size()
    n_s3 = ll[ll.rec.str.startswith("S3")].groupby("source1_entity_id").size()
    assert n_per.max() <= 11 and (n_s2.max() if len(n_s2) else 0) <= 5 and (n_s3.max() if len(n_s3) else 0) <= 6
    assert set(ll.rec.str[:2]) <= {"S2", "S3"}
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    # extra candidate pairs: accepted links outside the dense test candidates (U test_prep pairs)
    tl = np.load(f"{H}/U/prep/test/left.npy").astype(np.int64)
    tr = np.load(f"{H}/U/prep/test/right.npy").astype(np.int64)
    pool_index = pd.Index(pq.read_table(f"{H}/U/data/test/dense_test/pool.parquet", columns=["entity_id"]).column(0).to_pylist())
    ref_index = pd.Index(ref.entity_id.to_numpy())
    li = ref_index.get_indexer(ll.source1_entity_id.to_numpy()).astype(np.int64)
    ri = pool_index.get_indexer(ll.rec.to_numpy()).astype(np.int64)
    dense_keys = np.unique(tl * 10_000_000 + tr)
    lk = np.where(ri >= 0, li * 10_000_000 + ri, -1)
    in_dense = np.isin(lk, dense_keys) & (ri >= 0)
    ex = ll[~in_dense][["source1_entity_id", "rec"]].rename(columns={"rec": "entity_id"})
    ex_path = path + ".extra_candidate_pairs.tsv"
    ex.to_csv(ex_path, sep="\t", index=False, lineterminator="\n")
    # consistency with K's previous extras file (US/India part) for links kept from K
    old = pd.read_csv(f"{W}/in/old_extras.tsv", sep="\t", dtype=str)
    old = old[old.source1_entity_id.isin(set(s1_order))]
    old_set = set(zip(old.source1_entity_id, old.entity_id))
    ex_set = set(zip(ex.source1_entity_id, ex.entity_id))
    kept_k = new_pairs & kpairs
    cons = {"old_extras_usin": len(old_set), "old_extras_in_K_file": len(old_set & kpairs),
            "kept_K_links_flagged_extra": len(kept_k & ex_set), "kept_K_links_in_old_extras": len(kept_k & old_set),
            "mismatch_kept": len((kept_k & ex_set) ^ (kept_k & old_set))}
    log(f"extras {len(ex)}; consistency {cons}")
    # label-free diagnostics
    Bk = B.assign(new=[(x, y) in new_pairs for x, y in zip(B.s1_id, B.rec_id)])
    diag = {}
    for (c, ty), g in Bk.groupby(["country", "row_type"]):
        diag[f"{c}/{ty}"] = {"rows": len(g), "K_acc_per_1k": 1000 * float(g.a.mean()), "new_acc_per_1k": 1000 * float(g.new.mean())}
    addl = pd.DataFrame(list(added), columns=["s1_id", "rec_id"]).merge(B[["s1_id", "rec_id", "row_type", "country", "q"]], on=["s1_id", "rec_id"], how="left")
    reml = pd.DataFrame(list(removed), columns=["s1_id", "rec_id"]).merge(B[["s1_id", "rec_id", "row_type", "country", "q"]], on=["s1_id", "rec_id"], how="left")
    by = lambda d: {f"{c}/{ty}": int(n) for (c, ty), n in d.groupby(["country", "row_type"]).size().items()}
    # adversarial AUC (B_REST vs test) of the llm features within country x row type
    Lb = pd.read_parquet(f"{W}/in/llm_brest.parquet")
    refb = pq.read_table(f"{H}/U/data/arena/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    Lb["country"] = pd.Series(refb.country.to_numpy(), index=refb.entity_id.to_numpy()).reindex(Lb.s1_id.to_numpy()).to_numpy()
    adv = {}
    for (c, ty), g in L.groupby(["country", "row_type"]):
        gb = Lb[(Lb.country == c) & (Lb.row_type == ty)]
        sc = np.concatenate([gb.llm_logit.to_numpy(), g.llm_logit.to_numpy()])
        yy = np.concatenate([np.zeros(len(gb)), np.ones(len(g))])
        adv[f"{c}/{ty}"] = {"brest": len(gb), "test": len(g), "auc_llm_logit": auc(sc, yy)}
    res = {"variant": variant, "threshold": t, "path": path, "sha256": sha, "rows": len(chk), "links": int(len(ll)),
           "K_links": len(kpairs), "added": len(added), "removed": len(removed), "added_by": by(addl), "removed_by": by(reml),
           "added_q_mean": float(addl.q.mean()) if len(addl) else None, "removed_q_mean": float(reml.q.mean()) if len(reml) else None,
           "extras": {"path": ex_path, "n": len(ex), "sha256": hashlib.sha256(open(ex_path, "rb").read()).hexdigest(), "consistency": cons},
           "band_rates": diag, "adversarial_llm_logit": adv, "longest": int(n_per.max()), "labels_read": False,
           "base_file": "run27/K/final/us_india_mix_F_C.tsv (K-final F mixture US/India rows)"}
    json.dump(res, open(f"{OUT}/test_report_{variant.replace('/', '_')}.json", "w"), indent=1, default=float)
    log(json.dumps({k: res[k] for k in ("sha256", "links", "K_links", "added", "removed", "added_by", "removed_by")}))
    log("TEST DONE")


if __name__ == "__main__":
    main()
