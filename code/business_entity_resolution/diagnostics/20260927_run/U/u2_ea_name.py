"""U2 audit (27 Sep): empty-address (EA) records vs a typo-tolerant name search over ALL same-country arena S1s.

For every EA record of the arena pool: top-10 S1s (all partitions, text only) by char-3gram TF-IDF cosine,
reranked by RapidFuzz ratio on folded, legal-stripped names -> best S1, best and second similarity.
Then, on B_REST (labels of B_REST pairs only; SLICE/HOLDOUT-2 labels never used):
  * headroom: missed EA true links of the current pair tree (saved OOF, per-half 0.725, one owner, cap 11)
    whose record's best S1 is its owner with margin >= eps -> oracle macro-F0.5 gain if they were added;
  * rule: every currently-unaccepted EA record whose best S1 is a B_REST S1 with margin >= eps and
    sim >= floor -> precision, count and realized macro-F0.5 delta of adding them all.
Writes OUT/u2_name_search.npz (per-record best/second) and OUT/u2_report.json.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from ber.arena_decision import entity_f05
from ber.forensic_features import EMPTY, _strip_legal, address_kinds, fold_ascii
from ber.pair_tree_c2 import Arena, accept_per_row

ROOT = Path("/home/ubuntu/run27/U")
PREP = ROOT / "prep" / "brest"
OUT = ROOT / "u2"
THREADS = int(os.environ.get("U_THREADS", "10"))  # first run: 20 threads via argv (same results)
T0 = time.time()


def log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def norm(names):
    folded = fold_ascii(names).to_pylist()
    return np.asarray([_strip_legal(s or "") for s in folded], dtype=object)


def search(s1_names, rec_names, topn=10):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), dtype=np.float32, sublinear_tf=True, min_df=1)
    s = vec.fit_transform(s1_names)
    q = vec.transform(rec_names)
    log(f"  tfidf S1 {s.shape} queries {q.shape}")
    m = sp_matmul_topn(q, s.T.tocsr(), top_n=topn, threshold=0.2, n_threads=THREADS, sort=True).tocsr()
    log(f"  topn nnz {m.nnz}")
    best = np.full(len(rec_names), -1, np.int64)
    bsim = np.zeros(len(rec_names), np.float32)
    ssim = np.zeros(len(rec_names), np.float32)
    nties = np.zeros(len(rec_names), np.int16)
    for i in range(len(rec_names)):
        a, b = m.indptr[i], m.indptr[i + 1]
        if a == b:
            continue
        cand = m.indices[a:b]
        sims = np.fromiter((fuzz.ratio(rec_names[i], s1_names[j]) / 100.0 for j in cand), np.float32, count=b - a)
        o = np.argsort(-sims, kind="stable")
        best[i], bsim[i] = cand[o[0]], sims[o[0]]
        ssim[i] = sims[o[1]] if len(o) > 1 else 0.0
        nties[i] = int((sims >= sims[o[0]] - 1e-6).sum())
    return best, bsim, ssim, nties


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(ROOT / "data/arena/reference.parquet", columns=["business_name", "country"])
    pool = pq.read_table(ROOT / "data/arena/pool.parquet", columns=["business_name", "business_address", "country"])
    s1_country = np.asarray(ref.column("country").to_pylist(), dtype=object)
    rec_country = np.asarray(pool.column("country").to_pylist(), dtype=object)
    rec_ea = address_kinds(pool.column("business_address")) == EMPTY
    cache = OUT / "u2_name_search.npz"
    if cache.exists():
        z = np.load(cache)
        best, bsim, ssim, nties = z["best"], z["bsim"], z["ssim"], z["nties"]
    else:
        s1_norm = norm(ref.column("business_name"))
        ea_idx = np.flatnonzero(rec_ea)
        rec_norm = norm(pool.column("business_name").take(ea_idx))
        best = np.full(len(rec_ea), -1, np.int64)
        bsim = np.zeros(len(rec_ea), np.float32)
        ssim = np.zeros(len(rec_ea), np.float32)
        nties = np.zeros(len(rec_ea), np.int16)
        for c in sorted(set(s1_country)):
            si = np.flatnonzero(s1_country == c)
            qi = np.flatnonzero(rec_country[ea_idx] == c)
            log(f"country {c}: {len(si)} S1s, {len(qi)} EA records")
            b, bs, ss, nt = search(s1_norm[si], rec_norm[qi])
            r = ea_idx[qi]
            best[r] = np.where(b >= 0, si[np.maximum(b, 0)], -1)
            bsim[r], ssim[r], nties[r] = bs, ss, nt
        np.savez(cache, best=best, bsim=bsim, ssim=ssim, nties=nties, rec_ea=rec_ea)
        log("name search saved")

    ld = lambda n: np.load(PREP / f"{n}.npy")  # noqa: E731
    meta = json.loads((PREP / "meta.json").read_text())
    arena = Arena(ld("left"), ld("right"), ld("label"), ld("true_count"), ld("rec_empty"), ld("name_exact"),
                  ld("mult"), ld("population"), np.asarray(meta["countries"])[ld("country_code")])
    part = np.load(ROOT / "data/arena/finalize_v1/s1_partition.npy")
    comp = ld("component")
    half_row = (comp[arena.left] % 2).astype(np.int8)
    acc = accept_per_row(arena, ld("oof_c2tree"), np.full(len(arena.left), 0.725, np.float32), "model")
    # (both halves picked 0.725 in the saved crossfit_state; parity 0.985511 checked in prep)
    del half_row
    n_s1, n_rec = len(part), len(rec_ea)
    rec_owner_all = np.load(ROOT / "data/arena/labels/rec_owner.npy")
    # only B_REST owners are used (SLICE/HOLDOUT-2 labels never enter a statistic)
    owner = np.where((rec_owner_all >= 0) & (part[np.maximum(rec_owner_all, 0)] == 1), rec_owner_all, -1)
    owner_is_a = (rec_owner_all >= 0) & (part[np.maximum(rec_owner_all, 0)] == 0)
    del rec_owner_all
    pop = part == 1
    t = arena.true_count.astype(np.int64)
    k0 = np.bincount(arena.left[acc], minlength=n_s1)
    tp0 = np.bincount(arena.left[acc & arena.label], minlength=n_s1)
    f0 = entity_f05(tp0, k0, t)
    base = float(f0[pop].mean())
    log(f"base {base:.6f}")
    acc_owner = np.full(n_rec, -1, np.int64)
    acc_owner[arena.right[acc]] = arena.left[acc]
    acc_true = np.zeros(n_rec, bool)
    acc_true[arena.right[acc & arena.label]] = True
    in_cand = np.zeros(n_rec, bool)
    s1_country = np.asarray(s1_country)

    def delta(add_s1, add_rec, add_true):
        """Macro change on B_REST when (s1, rec) links are added and any other accepted owner of rec is dropped."""
        dk = np.bincount(add_s1, minlength=n_s1) - np.bincount(acc_owner[add_rec][acc_owner[add_rec] >= 0],
                                                                 minlength=n_s1)
        dtp = np.bincount(add_s1[add_true], minlength=n_s1)
        drop = acc_owner[add_rec] >= 0
        dtp -= np.bincount(acc_owner[add_rec][drop & acc_true[add_rec]], minlength=n_s1)
        f1 = entity_f05(tp0 + dtp, k0 + dk, t)
        out = {"delta": float(f1[pop].mean() - base)}
        for c in ("India", "US"):
            m = pop & (s1_country == c)
            out[f"delta_{c}"] = float(f1[m].mean() - f0[m].mean())
        return out

    margin = bsim - ssim
    report = {"base_macro_f05": base, "scope": "arena B_REST; name search over all same-country arena S1s "
              "(A+B_REST+SLICE names, text only); B_REST-owner labels only", "ea_records": int(rec_ea.sum())}
    # missed EA true links (B_REST owners)
    ea_true = rec_ea & (owner >= 0)
    missed = ea_true & ~((acc_owner == owner) & (owner >= 0))
    cand_pairs = set()
    lab_rows = np.flatnonzero(arena.label)
    in_cand[arena.right[lab_rows]] = True  # true pair retrieved (records have one owner)
    report["missed_ea_true_links"] = {"n": int(missed.sum()), "in_candidate_rows": int((missed & in_cand).sum()),
                                      "not_retrieved": int((missed & ~in_cand).sum()),
                                      "record_accepted_by_other_s1": int((missed & (acc_owner >= 0)).sum()),
                                      "best_is_owner": int((missed & (best == owner)).sum())}
    grid = {}
    for floor in (0.0, 0.8, 0.9):
        for eps in (0.0, 0.05, 0.1, 0.2):
            key = f"floor{floor}_eps{eps}"
            ok = (bsim >= floor) & ((margin >= eps) if eps > 0 else (margin > 1e-6))
            hit = np.flatnonzero(missed & (best == owner) & ok)
            head = delta(owner[hit], hit, np.ones(len(hit), bool))
            # simple rule over every currently-unaccepted EA record with a B_REST best S1
            rule = np.flatnonzero(rec_ea & (acc_owner < 0) & (best >= 0) & ok & (part[np.maximum(best, 0)] == 1))
            truth = owner[rule] == best[rule]
            real = delta(best[rule], rule, truth)
            grid[key] = {"headroom_links": int(len(hit)), "headroom_in_candidates": int(in_cand[hit].sum()),
                         "headroom": head,
                         "rule_adds": int(len(rule)), "rule_precision": float(truth.mean()) if len(rule) else None,
                         "rule_true": int(truth.sum()), "rule_record_owned_by_A_s1": int(owner_is_a[rule].sum()),
                         "rule_record_no_owner_in_brest_or_A": int(((owner[rule] < 0) & ~owner_is_a[rule]).sum()),
                         "rule_pair_in_candidates": None,
                         "rule_realized": real,
                         "rule_by_country": {c: {"adds": int((s1_country[best[rule]] == c).sum()),
                                                 "precision": float(truth[s1_country[best[rule]] == c].mean())
                                                 if (s1_country[best[rule]] == c).any() else None}
                                             for c in ("India", "US")}}
            log(f"{key}: {json.dumps(grid[key])}")
    report["grid"] = grid
    # all missed EA links oracle (upper bound regardless of name search)
    allm = np.flatnonzero(missed)
    report["oracle_all_missed_ea"] = delta(owner[allm], allm, np.ones(len(allm), bool))
    (OUT / "u2_report.json").write_text(json.dumps(report, indent=1) + "\n")
    log("done")


if __name__ == "__main__":
    main()
