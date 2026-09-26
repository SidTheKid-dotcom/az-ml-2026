"""Arena B_REST with a fraction of S1s withheld (their records become unowned distractors),
to mimic the test pool's ~5.8 records per S1 versus train's 4.67. Plain C2 blend (w 0.9) + one owner.
Labels are arena B_REST only; SLICE and HOLDOUT-2 are not read."""
import json, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from ber.arena_decision import evaluate
from ber.pair_tree_arena import decide, load_scores
from ber.c2_stack import blend
A = Path("/workspace/az-ml-new/arena"); S = Path("/opt/az-ml-new/runs/c2_step040000/b_rest_scores")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
left_all = t.column("left_idx").to_numpy().astype(np.int64); right_all = t.column("right_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); brest = part == 1
rows = np.flatnonzero(brest[left_all]); left, right = left_all[rows], right_all[rows]; del left_all, right_all
label = np.load(A / "labels/label.npy", mmap_mode="r")[rows]; true_count = np.load(A / "labels/true_count.npy")
tree = load_scores(A / "v1/base_probabilities.npy")[rows]; c2 = load_scores(S / "scores.npy")[rows]
country = pq.read_table(A / "v1/reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
n = len(true_count); prob0 = blend(c2, tree, 0.01, 0.9)
# records owned by B_REST S1s, and distractor ratio
out = {}
rng = np.random.default_rng(20260926)
u = rng.random(n)
for d in (0.0, 0.2, 0.3):
    dropped = brest & (u < d); pop = brest & ~dropped
    prob = np.where(dropped[left], 0, prob0).astype(np.float32)
    res = {}
    for th in (0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        acc = decide(prob, left, right, th, n)
        r = evaluate(acc, left, label, true_count, population=pop, country=country)
        res[th] = {k: r[k] for k in r if k in ("macro_f05", "precision", "recall", "false_links", "singleton_accuracy")}
        print(time.strftime("%T"), d, th, json.dumps(res[th], default=float), flush=True)
    out[d] = res
Path("/workspace/az-ml-new/distractor_sim/report.json").write_text(json.dumps(out, indent=1, default=float))
