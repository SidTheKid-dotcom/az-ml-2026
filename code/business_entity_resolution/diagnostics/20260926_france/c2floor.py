"""Accepted at blend>=0.1 but C2 below a floor: arena truth rate (B_REST) and test counts by country."""
import numpy as np, pyarrow.parquet as pq, json
from pathlib import Path
from ber.pair_tree_arena import load_scores, decide
from ber.arena_decision import evaluate
A = Path("/workspace/az-ml-new/arena"); T = Path("/opt/az-ml-new/test_inputs"); R = Path("/opt/az-ml-new/runs/c2_step040000")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy().astype(np.int64); ra = t.column("right_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); rows = np.flatnonzero(part[la] == 1); la, ra = la[rows], ra[rows]
ca = pq.read_table(A / "v1/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool); tc = np.load(A / "labels/true_count.npy")
c2 = load_scores(R / "b_rest_scores/scores.npy")[rows]; tree = load_scores(A / "v1/base_probabilities.npy")[rows]
b = np.where(tree >= .01, .9 * c2 + .1 * tree, 0).astype(np.float32)
n = len(tc); pop = part == 1
acc = decide(b, la, ra, 0.1, n)
out = {"arena": {}}
for fl in (0.002, 0.005, 0.01, 0.02):
    m = acc & (c2 < fl)
    d = {c: [int((m & (ca[la] == c)).sum()), round(float(lab[m & (ca[la] == c)].mean()), 4)] for c in ("US", "India")}
    b2 = np.where(c2 < fl, 0, b).astype(np.float32)
    acc2 = decide(b2, la, ra, 0.1, n)
    r1 = evaluate(acc, la, lab, tc, population=pop, country=ca); r2 = evaluate(acc2, la, lab, tc, population=pop, country=ca)
    d["macro_f05_base_vs_floor"] = [round(r1["macro_f05"], 6), round(r2["macro_f05"], 6)]
    out["arena"][fl] = d
del la, ra, lab, c2, tree, b
lt = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); rt = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
ct = pq.read_table(T / "dense_test/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
c2 = load_scores(R / "test_scores/scores.npy"); tree = load_scores(T / "base_probabilities.npy")
b = np.where(tree >= .01, .9 * c2 + .1 * tree, 0).astype(np.float32)
acc = decide(b, lt, rt, 0.1, len(ct))
out["test_accepted"] = {c: int((acc & (ct[lt] == c)).sum()) for c in ("France", "US", "India")}
out["test_accepted_c2_below"] = {fl: {c: int((acc & (c2 < fl) & (ct[lt] == c)).sum()) for c in ("France", "US", "India")} for fl in (0.002, 0.005, 0.01, 0.02)}
print(json.dumps(out, indent=1))
