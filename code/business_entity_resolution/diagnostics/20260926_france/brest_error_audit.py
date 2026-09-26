"""B_REST error audit for the plain C2 blend (w 0.9, t 0.1) + one owner: where is the macro-F0.5 loss,
and what do missed true links and false links look like. Arena labels only; SLICE/HOLDOUT-2 unread."""
import json, re
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
from ber.pair_tree_arena import load_scores, decide
A = Path("/workspace/az-ml-new/arena"); R = Path("/opt/az-ml-new/runs/c2_step040000")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy().astype(np.int64); ra = t.column("right_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); rows = np.flatnonzero(part[la] == 1); la, ra = la[rows], ra[rows]
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool); tc = np.load(A / "labels/true_count.npy")
c2 = load_scores(R / "b_rest_scores/scores.npy")[rows]; tree = load_scores(A / "v1/base_probabilities.npy")[rows]
b = np.where(tree >= .01, .9 * c2 + .1 * tree, 0).astype(np.float32); n = len(tc)
acc = decide(b, la, ra, 0.1, n)
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
pop = part == 1
tp = np.bincount(la[acc & lab], minlength=n); pred = np.bincount(la[acc], minlength=n); cand_true = np.bincount(la[lab], minlength=n)
f = np.where(tc == 0, (pred == 0).astype(float), np.where(pred == 0, 0.0, 5 * tp / np.maximum(tc + 4 * pred, 1)))
loss = 1 - f
fp = pred - tp; miss_retr = tc - cand_true; miss_match = cand_true - tp
P = pop
out = {"macro_f05": float(f[P].mean()), "total_loss_per_s1": float(loss[P].mean())}
cats = {"singleton_with_false_link": P & (tc == 0) & (pred > 0),
        "has_false_link": P & (tc > 0) & (fp > 0),
        "misses_only_matcher": P & (tc > 0) & (fp == 0) & (miss_match > 0) & (miss_retr == 0),
        "misses_only_retrieval": P & (tc > 0) & (fp == 0) & (miss_retr > 0) & (miss_match == 0),
        "misses_both": P & (tc > 0) & (fp == 0) & (miss_retr > 0) & (miss_match > 0)}
out["loss_share"] = {k: [int(m.sum()), round(float(loss[m].sum() / loss[P].sum()), 4)] for k, m in cats.items()}
out["links"] = {"true": int(tc[P].sum()), "missed_retrieval": int(miss_retr[P].sum()), "missed_matcher": int(miss_match[P].sum()), "false": int(fp[P].sum())}
# owner of each record under acc
owner = np.full(len(pool), -1, np.int64); owner[ra[acc]] = la[acc]
mm = np.flatnonzero(lab & ~acc & P[la])
stolen = owner[ra[mm]] >= 0
empty_addr = np.array([not (pool.business_address[j] or "").strip() for j in ra[mm]])
out["matcher_misses"] = {"n": int(len(mm)), "record_owned_by_other_s1": round(float(stolen.mean()), 4), "record_address_empty": round(float(empty_addr.mean()), 4),
                         "c2_lt_0.01": round(float((c2[mm] < .01).mean()), 4), "c2_0.01_0.1": round(float(((c2[mm] >= .01) & (c2[mm] < .1)).mean()), 4), "tree_lt_0.01(ungated)": round(float((tree[mm] < .01).mean()), 4)}
print(json.dumps(out, indent=1))
rng = np.random.default_rng(11)
def show(i, tag):
    s, j = la[i], ra[i]
    o = owner[j]
    print(f"{tag} c2={c2[i]:.3f} tree={tree[i]:.3f} | S1: {ref.business_name[s]} | {ref.business_address[s]} ({ref.country[s]})\n      REC: {pool.business_name[j]} | {pool.business_address[j]}"
          + (f"\n      OWNED BY: {ref.business_name[o]} | {ref.business_address[o]}" if o >= 0 and o != s else ""))
print("\n=== MISSED TRUE LINKS (retrieved, rejected) ===")
for i in rng.choice(mm, 30, replace=False): show(i, "MISS")
fl = np.flatnonzero(acc & ~lab & P[la])
print("\n=== FALSE LINKS ===")
for i in rng.choice(fl, 15, replace=False): show(i, "FALSE")
