"""B_REST true records that never reach the pair-tree candidate rows: what are they?"""
import json, re, unicodedata
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
A = Path("/workspace/az-ml-new/arena"); W = Path("/opt/az-ml-new/agents/pair_tree_c2/step040000")
rows = np.load(W / "b_rest_pair_rows.npy")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
L = t.column("left_idx").to_numpy().astype(np.int64); R = t.column("right_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy")
owner = np.load(A / "labels/rec_owner.npy").astype(np.int64)
recs = np.flatnonzero((owner >= 0) & (part[np.maximum(owner, 0)] == 1))
n_rec = len(owner)
in_cand = np.zeros(n_rec, bool); in_cand[R[rows]] = True    # B_REST candidate rows: (left, right) pairs
cand_key = set((L[rows] * n_rec + R[rows]).tolist()) if False else None
keys_c = np.sort(L[rows] * n_rec + R[rows]); keys_d = None
k = owner[recs] * n_rec + recs
pos = np.minimum(np.searchsorted(keys_c, k), len(keys_c) - 1); hit_c = keys_c[pos] == k
bm = np.flatnonzero(part[L] == 1); keys_all = np.sort(L[bm] * n_rec + R[bm])
pos = np.minimum(np.searchsorted(keys_all, k), len(keys_all) - 1); hit_d = keys_all[pos] == k
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
def toks(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z0-9]+", s))
miss = recs[~hit_c]; miss_dense = recs[~hit_d]
def cat(j):
    s = owner[j]; rn, sn = toks(pool.business_name[j]), toks(ref.business_name[s])
    ra = (pool.business_address[j] or "").strip()
    return ("empty_addr" if not ra else "addr") + "|" + ("name_disjoint" if not (rn & sn) else "name_overlap")
import collections
out = {"true_records_b_rest": int(len(recs)), "not_in_candidate_rows": int(len(miss)), "not_in_dense_top50": int(len(miss_dense)),
       "dropped_by_candidate_gate": int(len(miss) - len(miss_dense)),
       "dense_miss_categories": dict(collections.Counter(cat(j) for j in miss_dense)),
       "gate_drop_categories": dict(collections.Counter(cat(j) for j in np.setdiff1d(miss, miss_dense)))}
print(json.dumps(out, indent=1))
rng = np.random.default_rng(2)
print("\n=== NOT IN DENSE TOP-50 ===")
for j in rng.choice(miss_dense, 30, replace=False):
    s = owner[j]; print(f"S1: {ref.business_name[s]} | {ref.business_address[s]} ({ref.country[s]})\n   REC: {pool.business_name[j]} | {pool.business_address[j]}")
