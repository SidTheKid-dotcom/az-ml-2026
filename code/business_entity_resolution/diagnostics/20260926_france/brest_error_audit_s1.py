"""Same audit for the submitted US/India method: pair_tree_c2 'model' OOF on B_REST at 0.725 + one owner."""
import json
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
from ber.pair_tree_arena import load_scores, decide
A = Path("/workspace/az-ml-new/arena"); W = Path("/opt/az-ml-new/agents/pair_tree_c2/step040000")
rows = np.load(W / "b_rest_pair_rows.npy"); oof = np.load(W / "oof.npy")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy()[rows].astype(np.int64); ra = t.column("right_idx").to_numpy()[rows].astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); assert np.all(part[la] == 1)
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool); tc = np.load(A / "labels/true_count.npy")
c2 = load_scores(Path("/opt/az-ml-new/runs/c2_step040000/b_rest_scores/scores.npy"))[rows]
n = len(tc); acc = decide(oof, la, ra, 0.725, n)
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
P = part == 1
tp = np.bincount(la[acc & lab], minlength=n); pred = np.bincount(la[acc], minlength=n); cand_true = np.bincount(la[lab], minlength=n)
f = np.where(tc == 0, (pred == 0).astype(float), np.where(pred == 0, 0.0, 5 * tp / np.maximum(tc + 4 * pred, 1)))
loss = 1 - f; fp = pred - tp; mr = tc - cand_true; mm_n = cand_true - tp
out = {"macro_f05": float(f[P].mean())}
cats = {"singleton_with_false_link": P & (tc == 0) & (pred > 0), "has_false_link": P & (tc > 0) & (fp > 0),
        "misses_only_matcher": P & (tc > 0) & (fp == 0) & (mm_n > 0) & (mr == 0),
        "misses_only_retrieval": P & (tc > 0) & (fp == 0) & (mr > 0) & (mm_n == 0), "misses_both": P & (tc > 0) & (fp == 0) & (mr > 0) & (mm_n > 0)}
out["loss_share"] = {k: [int(m.sum()), round(float(loss[m].sum() / loss[P].sum()), 4)] for k, m in cats.items()}
out["loss_per_s1_total"] = float(loss[P].mean())
out["links"] = {"true": int(tc[P].sum()), "missed_retrieval(not in candidate rows)": int(mr[P].sum()), "missed_matcher": int(mm_n[P].sum()), "false": int(fp[P].sum())}
emptyp = np.array([not (a or "").strip() for a in pool.business_address.to_numpy()])
owner = np.full(len(pool), -1, np.int64); owner[ra[acc]] = la[acc]
mm = np.flatnonzero(lab & ~acc); fl = np.flatnonzero(acc & ~lab)
out["matcher_misses"] = {"n": int(len(mm)), "record_address_empty": round(float(emptyp[ra[mm]].mean()), 4), "record_owned_by_other_s1": round(float((owner[ra[mm]] >= 0).mean()), 4)}
out["false_links"] = {"n": int(len(fl)), "record_address_empty": round(float(emptyp[ra[fl]].mean()), 4)}
out["pool_empty_address_share"] = round(float(emptyp.mean()), 4)
print(json.dumps(out, indent=1))
rng = np.random.default_rng(5)
def show(i, tag):
    s, j = la[i], ra[i]; o = owner[j]
    print(f"{tag} p={oof[i]:.3f} c2={c2[i]:.3f} | S1: {ref.business_name[s]} | {ref.business_address[s]} ({ref.country[s]})\n      REC: {pool.business_name[j]} | {pool.business_address[j]}"
          + (f"\n      OWNED BY: {ref.business_name[o]} | {ref.business_address[o]}" if o >= 0 and o != s else ""))
print("\n=== MISSED (non-empty address) ===")
ne = mm[~emptyp[ra[mm]]]
for i in rng.choice(ne, min(25, len(ne)), replace=False): show(i, "MISS")
print("\n=== MISSED (empty address) ===")
em = mm[emptyp[ra[mm]]]
for i in rng.choice(em, min(12, len(em)), replace=False): show(i, "MISS")
print("\n=== FALSE ===")
for i in rng.choice(fl, 15, replace=False): show(i, "FALSE")
