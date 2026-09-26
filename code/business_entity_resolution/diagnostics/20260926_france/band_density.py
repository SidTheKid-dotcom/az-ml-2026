import numpy as np, pyarrow.parquet as pq, json
from pathlib import Path
from ber.pair_tree_arena import load_scores
A = Path("/workspace/az-ml-new/arena"); T = Path("/opt/az-ml-new/test_inputs"); R = Path("/opt/az-ml-new/runs/c2_step040000")
BANDS = [0.01, 0.05, 0.1, 0.3, 0.7, 1.01]
def dens(left, c2, tree, country, pop_mask):
    blend = np.where(tree >= 0.01, 0.9 * c2 + 0.1 * tree, 0)
    out = {}
    for c in np.unique(country[pop_mask]):
        s1 = pop_mask & (country == c); ns = int(s1.sum()); m = s1[left]
        b = blend[m]; out[str(c)] = {"s1": ns, **{f"[{lo},{hi})": round(float(((b >= lo) & (b < hi)).sum() / ns), 4) for lo, hi in zip(BANDS[:-1], BANDS[1:])}}
    return out
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx"]); la = t.column("left_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); rows = np.flatnonzero(part[la] == 1); la = la[rows]
ca = pq.read_table(A / "v1/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
arena = dens(la, load_scores(R / "b_rest_scores/scores.npy")[rows], load_scores(A / "v1/base_probabilities.npy")[rows], ca, part == 1)
lt = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64)
ct = pq.read_table(T / "dense_test/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
test = dens(lt, load_scores(R / "test_scores/scores.npy"), load_scores(T / "base_probabilities.npy"), ct, np.ones(len(ct), bool))
print(json.dumps({"arena_b_rest": arena, "test": test}, indent=1))
