"""U10 + U6 audits (27 Sep, Agent U) on arena B_REST, current pair trees (saved OOF, 0.725, one owner, cap 11).

U10: allocate every B_REST S1's loss (1 - F0.5) to its false links (loss removed if they were dropped) and
     its missed true links (the rest, split equally), then bucket links:
       missed: not retrieved (EA / non-EA) | retrieved EA shared-core | EA unique-core | native script |
               invented name at the S1's digit key | other (sub-split) ; false links by kind.
U6:  oracle gain of adding every missed true link among B_REST candidate rows with C2 < 0.01 (by band),
     against the C2 tree and the Qwen tree.
B_REST owners only (rec_owner restricted to partition 1); SLICE/HOLDOUT-2 labels never used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import entity_f05
from ber.collective_features import FEATURES as COLL
from ber.forensic_features import EMPTY, address_kinds
from ber.pair_tree_c2 import FEATURES as BASE_F, Arena, accept_per_row

PREP = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/az-ml-new/run27/U/prep/brest")
ARENA = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/az-ml-new/arena")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "/workspace/az-ml-new/run27/U/audits")
OUT.mkdir(parents=True, exist_ok=True)
ld = lambda n: np.load(PREP / f"{n}.npy")  # noqa: E731
meta = json.loads((PREP / "meta.json").read_text())
country = np.asarray(meta["countries"])[ld("country_code")]
arena = Arena(ld("left"), ld("right"), ld("label"), ld("true_count"), ld("rec_empty"), ld("name_exact"), ld("mult"),
              ld("population"), country)
part = np.load(ARENA / "finalize_v1/s1_partition.npy")
pop = part == 1
n_s1 = len(part)
L, R, Y = arena.left, arena.right, arena.label
t = arena.true_count.astype(np.int64)
xb = np.load(PREP / "x_base.npy")
c2 = np.asarray(xb[:, BASE_F.index("c2")])
xc = np.load(PREP / "x_coll.npy")
fc = {n: np.asarray(xc[:, i]) for i, n in enumerate(COLL) if n in ("s1_core_other", "core_jaccard", "key_eq",
                                                                     "main_eq", "core_eq", "core_empty")}
rec_owner = np.load(ARENA / "labels/rec_owner.npy")
owner = np.where((rec_owner >= 0) & (part[np.maximum(rec_owner, 0)] == 1), rec_owner, -1)
del rec_owner
n_rec = len(owner)
pool = pq.read_table(ARENA / "v1/pool.parquet", columns=["business_name", "business_address"])
rec_ea = address_kinds(pool.column("business_address")) == EMPTY
rec_names = pool.column("business_name")
ref_names = pq.read_table(ARENA / "v1/reference.parquet", columns=["business_name"]).column("business_name")


def native(names, idx):
    vals = names.take(np.asarray(idx, dtype=np.int64)).to_pylist()
    return np.fromiter((any(ord(ch) > 0x2FF for ch in (v or "")) for v in vals), bool, count=len(vals))


report = {}
for tree in ("oof_c2tree", "oof_qtree"):
    acc = accept_per_row(arena, ld(tree), np.full(len(L), 0.725, np.float32), "model")
    k = np.bincount(L[acc], minlength=n_s1)
    tp = np.bincount(L[acc & Y], minlength=n_s1)
    f = entity_f05(tp, k, t)
    fprime = entity_f05(tp, tp, t)  # same S1 with its false links removed
    loss_false = np.where(pop, fprime - f, 0.0)
    loss_miss = np.where(pop, 1.0 - fprime, 0.0)
    N = pop.sum()
    rep = {"macro_f05": float(f[pop].mean()), "total_loss": float((1 - f[pop]).mean()),
           "loss_false_links": float(loss_false.sum() / N), "loss_missed_links": float(loss_miss.sum() / N)}
    acc_owner = np.full(n_rec, -1, np.int64)
    acc_owner[R[acc]] = L[acc]
    # missed true links: (owner[r], r) for records with a B_REST owner, not accepted for that owner
    recs = np.flatnonzero((owner >= 0) & (acc_owner != owner))
    s_of = owner[recs]
    missed_n = t - tp
    w = np.where(missed_n[s_of] > 0, loss_miss[s_of] / np.maximum(missed_n[s_of], 1), 0.0)
    # sanity: candidate rows of true pairs
    row_of = np.full(n_rec, -1, np.int64)
    trow = np.flatnonzero(Y)
    row_of[R[trow]] = trow
    rows = row_of[recs]
    retrieved = rows >= 0
    ea = rec_ea[recs]
    cat = np.full(len(recs), "", dtype=object)
    cat[~retrieved & ea] = "not_retrieved_EA"
    cat[~retrieved & ~ea] = "not_retrieved_nonEA"
    rr = np.where(retrieved, rows, 0)
    shared = fc["s1_core_other"][rr] > 0
    cat[retrieved & ea & shared] = "EA_shared_core"
    cat[retrieved & ea & ~shared] = "EA_unique_core"
    ne = retrieved & ~ea
    nat = np.zeros(len(recs), bool)
    idx = np.flatnonzero(ne)
    nat[idx] = native(rec_names, recs[idx]) | native(ref_names, s_of[idx])
    cat[ne & nat] = "native_script"
    at_key = (fc["key_eq"][rr] > 0) | (fc["main_eq"][rr] > 0)
    invented = ne & ~nat & at_key & (fc["core_jaccard"][rr] == 0)
    cat[invented] = "invented_name_at_key"
    rest = ne & ~nat & ~invented
    stolen = acc_owner[recs] >= 0
    jac = fc["core_jaccard"][rr]
    c2r = c2[rr]
    sub = np.full(len(recs), "", dtype=object)
    sub[rest & stolen] = "other:record_owned_by_other_s1"
    free = rest & ~stolen
    sub[free & at_key & (jac > 0)] = "other:same_key_partial_name"
    sub[free & ~at_key & (jac >= 0.5)] = "other:name_similar_address_differs"
    sub[free & ~at_key & (jac < 0.5)] = "other:name_and_address_differ"
    cat[rest] = sub[rest]
    buckets = {}
    for c in sorted(set(cat)):
        m = cat == c
        buckets[c] = {"links": int(m.sum()), "loss": round(float(w[m].sum() / N), 6),
                      "US": round(float(w[m & (country[s_of] == "US")].sum() / N), 6),
                      "India": round(float(w[m & (country[s_of] == "India")].sum() / N), 6),
                      "c2_lt_0.01": int((m & retrieved & (c2r < 0.01)).sum()),
                      "c2_ge_0.5": int((m & retrieved & (c2r >= 0.5)).sum())}
    rep["missed_buckets"] = buckets
    rep["missed_loss_check"] = float(w.sum() / N)
    # false links
    fl = np.flatnonzero(acc & ~Y)
    fs = L[fl]
    fp_n = k - tp
    wf = loss_false[fs] / np.maximum(fp_n[fs], 1)
    fea = arena.rec_empty[fl]
    fsing = t[fs] == 0
    rep["false_buckets"] = {
        name: {"links": int(m.sum()), "loss": round(float(wf[m].sum() / N), 6)}
        for name, m in (("singleton_S1_EA", fsing & fea), ("singleton_S1_nonEA", fsing & ~fea),
                        ("matched_S1_EA", ~fsing & fea), ("matched_S1_nonEA", ~fsing & ~fea))}
    # U6: missed true candidate rows with C2 < 0.01, oracle add (drop any other accepted owner)
    miss_rows = np.flatnonzero(Y & ~acc)
    u6 = {}
    for band, m in (("unscored(tree<0.01)", c2[miss_rows] < 0), ("[0,0.001)", (c2[miss_rows] >= 0) & (c2[miss_rows] < 0.001)),
                    ("[0.001,0.005)", (c2[miss_rows] >= 0.001) & (c2[miss_rows] < 0.005)),
                    ("[0.005,0.01)", (c2[miss_rows] >= 0.005) & (c2[miss_rows] < 0.01)),
                    ("all_lt_0.01_scored", (c2[miss_rows] >= 0) & (c2[miss_rows] < 0.01)),
                    ("all_lt_0.01_incl_unscored", c2[miss_rows] < 0.01)):
        rws = miss_rows[m]
        add_s, add_r = L[rws], R[rws]
        prev = acc_owner[add_r]
        dk = np.bincount(add_s, minlength=n_s1) - np.bincount(prev[prev >= 0], minlength=n_s1)
        prev_true = np.zeros(len(rws), bool)  # a record has one owner, so another S1's accepted claim is false
        dtp = np.bincount(add_s, minlength=n_s1)
        f1 = entity_f05(tp + dtp, k + dk, t)
        u6[band] = {"links": int(len(rws)), "EA": int(arena.rec_empty[rws].sum()),
                    "oracle_delta": round(float(f1[pop].mean() - f[pop].mean()), 6),
                    "candidate_rows_in_band": int(((c2 < 0.01) & (c2 >= 0) if band.startswith("all_lt_0.01_s") else
                                                   (c2 < 0.01)).sum()) if band.startswith("all") else None}
    rep["u6"] = u6
    report[tree] = rep
    print(tree, json.dumps(rep, indent=1), flush=True)
(OUT / "u10_u6.json").write_text(json.dumps(report, indent=1) + "\n")
