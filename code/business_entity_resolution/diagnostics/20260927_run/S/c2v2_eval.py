"""Agent S: plain-blend (0.9*C2 + 0.1*tree >= 0.10, tree gate 0.01, one owner, cap 11) F0.5 on arena v1 B_REST S1s
(population mask) for a C2 v2 score dir vs the stored C2 step-40k B_REST scores. Only B_REST labels are read.
Usage: c2v2_eval.py SCORES_DIR POPULATION.npy OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from ber.arena_decision import evaluate
from ber.pair_tree_arena import decide
A = Path("/dev/shm/c2v2/in")
pop = np.load(sys.argv[2]); part = np.load(A / "s1_partition.npy")
assert (part[pop] == 1).all(), "population must be B_REST only"
t = pq.read_table(A / "arena/pairs.parquet", columns=["left_idx", "right_idx"])
left = t.column(0).to_numpy().astype(np.int64); right = t.column(1).to_numpy().astype(np.int64)
rows = np.flatnonzero(pop[left])
tree = np.load(A / "arena/base_probabilities.npy", mmap_mode="r")[rows]
label = np.load(A / "labels/label.npy", mmap_mode="r")[rows]
tc = np.load(A / "labels/true_count.npy"); country = pq.read_table(A / "arena/reference.parquet", columns=["country"]).column(0).to_numpy(zero_copy_only=False)
lsub, rsub = left[rows], right[rows]
out = {}
per = {}
for name, d in (("candidate", Path(sys.argv[1])), ("c2_40k_stored", Path("/opt/az-ml-new/runs/c2_step040000/b_rest_scores"))):
    s = np.nan_to_num(np.asarray(np.load(d / "scores.npy", mmap_mode="r")[rows], np.float32))
    m = np.asarray(np.load(d / "scored_mask.npy", mmap_mode="r")[rows])
    assert m[tree >= .01].all(), f"{name} misses gated pairs"
    prob = np.where(tree >= .01, .9 * s + .1 * tree, 0).astype(np.float32)
    acc_sub = decide(prob, lsub, rsub, .10, len(tc))
    acc = np.zeros(len(left), bool); acc[rows] = acc_sub
    lab_full = np.zeros(len(left), bool); lab_full[rows] = label
    res = evaluate(acc, left, lab_full, tc, population=pop, country=country, return_per_s1=True)
    per[name] = np.asarray(res.pop("per_s1"))[pop]; out[name] = res
d = per["candidate"] - per["c2_40k_stored"]
out["candidate_minus_c2_40k"] = {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))), "s1": int(pop.sum())}
Path(sys.argv[3]).write_text(json.dumps(out, indent=1, default=float) + "\n")
print(json.dumps({k: (v["macro_f05"] if "macro_f05" in v else v) for k, v in out.items()}))
