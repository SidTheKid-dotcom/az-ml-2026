"""Count Track R test route pairs under the B_REST kept rule (address >= 0.6; empty >= 0.8 at rank <= 2), US/India."""
import json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
d = Path("/workspace/az-ml-new/eureka/retrieval/test")
left, right = np.load(d / "left.npy"), np.load(d / "right.npy")
score, rank, branch = np.load(d / "score.npy"), np.load(d / "rank.npy"), np.load(d / "branch.npy")
ref = pq.read_table(d / "reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
print(len(left), len(score), np.unique(branch, return_counts=True), left.dtype, score.dtype)
c = ref[left]
for b in np.unique(branch):
    m = branch == b
    print("branch", b, "score quantiles", np.quantile(score[m], [0, .5, .9, 1]).round(3), "rank max", rank[m].max())
