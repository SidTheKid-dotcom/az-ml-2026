import numpy as np, pyarrow.parquet as pq
from pathlib import Path
from ber.pair_tree_arena import load_scores
T = Path("/opt/az-ml-new/test_inputs"); R = Path("/opt/az-ml-new/runs/c2_step040000")
ref = pq.read_table(T / "dense_test/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(T / "dense_test/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
left = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64)
right = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
c2 = load_scores(R / "test_scores/scores.npy"); tree = load_scores(T / "base_probabilities.npy")
b = np.where(tree >= 0.01, 0.9 * c2 + 0.1 * tree, 0)
fr = (ref.country.to_numpy() == "France")
border = fr[left] & (b >= 0.05) & (b < 0.7)
rng = np.random.default_rng(7)
s1s = rng.choice(np.unique(left[border]), 40, replace=False)
for s in s1s:
    m = np.flatnonzero((left == s) & (b >= 0.02)); m = m[np.argsort(-b[m])]
    print(f"\nS1 {ref.business_name[s]} | {ref.business_address[s]}")
    for i in m[:12]:
        print(f"  {b[i]:.3f}  {pool.business_name[right[i]]} | {pool.business_address[right[i]]}")
