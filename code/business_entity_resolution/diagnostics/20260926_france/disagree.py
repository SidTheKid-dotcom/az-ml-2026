"""Tree says yes (>=0.9), C2 says no (<0.05): size per S1, how often the S1 is the record's best claimant,
and (arena B_REST only) truth rate. SLICE/HOLDOUT-2 unread; test has no labels."""
import numpy as np, pyarrow.parquet as pq, json
from pathlib import Path
from ber.pair_tree_arena import load_scores
A = Path("/workspace/az-ml-new/arena"); T = Path("/opt/az-ml-new/test_inputs"); R = Path("/opt/az-ml-new/runs/c2_step040000")
def other_best(right, b):
    o = np.lexsort((-b, right)); r = right[o]
    first = np.r_[True, r[1:] != r[:-1]]
    top1 = np.zeros(right.max() + 1, np.float32); top2 = np.zeros_like(top1)
    top1[r[first]] = b[o][first]
    second = np.r_[False, first[:-1]] & ~first
    top2[r[second]] = b[o][second]
    is_top = np.zeros(len(b), bool); is_top[o[first]] = True
    return np.where(is_top, top2[right], top1[right])
def report(left, right, c2, tree, country, n_s1_by_c, label=None):
    b = np.where(tree >= 0.01, 0.9 * c2 + 0.1 * tree, 0).astype(np.float32)
    ob = other_best(right, b)
    cl = (tree >= 0.9) & (c2 < 0.05)
    out = {}
    for c in np.unique(country):
        m = cl & (country[left] == c)
        free = m & (ob < 0.7)
        d = {"cluster_per_s1": round(float(m.sum() / n_s1_by_c[c]), 4), "share_no_better_claimant": round(float(free.sum() / max(m.sum(), 1)), 4),
             "accepted_at_0.1_share": round(float((m & (b >= 0.1)).sum() / max(m.sum(), 1)), 4)}
        if label is not None:
            d["true_rate"] = round(float(label[m].mean()), 4); d["true_rate_no_better_claimant"] = round(float(label[free].mean()), 4)
            d["true_rate_claimed_elsewhere"] = round(float(label[m & ~free].mean()), 4)
            for lo, hi in ((0, .005), (.005, .02), (.02, .05)):
                k = m & free & (c2 >= lo) & (c2 < hi); d[f"true_free_c2[{lo},{hi})"] = [int(k.sum()), round(float(label[k].mean()), 4)]
        out[str(c)] = d
    return out
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy().astype(np.int64); ra = t.column("right_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); rows = np.flatnonzero(part[la] == 1)
ca = pq.read_table(A / "v1/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool)
arena = report(la[rows], ra[rows], load_scores(R / "b_rest_scores/scores.npy")[rows], load_scores(A / "v1/base_probabilities.npy")[rows],
               ca, {c: int(((part == 1) & (ca == c)).sum()) for c in ("US", "India")}, lab)
del la, ra
lt = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); rt = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
ct = pq.read_table(T / "dense_test/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
test = report(lt, rt, load_scores(R / "test_scores/scores.npy"), load_scores(T / "base_probabilities.npy"), ct, {c: int((ct == c).sum()) for c in np.unique(ct)})
print(json.dumps({"arena_b_rest": arena, "test": test}, indent=1))
