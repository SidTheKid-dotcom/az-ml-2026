"""U8 prep: US/India subset of the Track R test route pairs (kept rule already applied) for C2 scoring.
Writes /workspace/az-ml-new/run27/U/u8/test_ui/{left,right,tree,rows}.npy (indices into
/workspace/az-ml-new/eureka/retrieval/test/{reference,pool}.parquet). Label-free."""
import hashlib, json
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
d = Path("/workspace/az-ml-new/eureka/retrieval/test")
out = Path("/workspace/az-ml-new/run27/U/u8/test_ui")
out.mkdir(parents=True, exist_ok=True)
left, right, tree = np.load(d / "left.npy"), np.load(d / "right.npy"), np.load(d / "tree.npy")
ref = pq.read_table(d / "reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
rows = np.flatnonzero(np.isin(ref[left], ["US", "India"]))
for n, a in (("left", left[rows]), ("right", right[rows]), ("tree", tree[rows]), ("rows", rows)):
    np.save(out / (n + ".npy"), a)
man = {"pairs": int(len(rows)), "by_country": {c: int((ref[left[rows]] == c).sum()) for c in ("US", "India")},
       "tree_ge_0.01": int((tree[rows] >= 0.01).sum()),
       "files": {n: hashlib.sha256((out / (n + ".npy")).read_bytes()).hexdigest() for n in ("left", "right", "tree", "rows")}}
(out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
print(json.dumps(man, indent=1))
