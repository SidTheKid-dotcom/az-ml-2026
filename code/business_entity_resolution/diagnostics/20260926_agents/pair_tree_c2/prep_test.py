"""Build the C2-independent test v1 cache, and a DRY-RUN C2 score dir (v1 blend as stand-in) for pipeline timing only."""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.pair_tree_arena import load_scores
from ber.pair_tree_c2 import COLUMNS, log, v1_cache

scores = Path("/opt/az-ml-new/test_inputs")
cache_path = Path("/opt/az-ml-new/agents/pair_tree_c2/step005000/test_v1_cache.npz")
ref = pq.read_table(scores / "dense_test" / "reference.parquet", columns=COLUMNS).to_pandas()
pool = pq.read_table(scores / "dense_test" / "pool.parquet", columns=COLUMNS).to_pandas()
left = np.asarray(np.load(scores / "index" / "left.npy", mmap_mode="r")).astype(np.int64)
right = np.asarray(np.load(scores / "index" / "right.npy", mmap_mode="r")).astype(np.int64)
blend, base = load_scores(scores / "probabilities.npy"), load_scores(scores / "base_probabilities.npy")
assert len(left) == len(right) == len(blend) == len(base)
cache = v1_cache(ref, pool, left, right, blend, base, cache_path)
fake = Path("/opt/az-ml-new/agents/pair_tree_c2/dryrun_fake_test_scores")
fake.mkdir(parents=True, exist_ok=True)
gated = base >= 0.01
np.save(fake / "scores.npy", np.where(gated, blend, 0).astype(np.float32))
np.save(fake / "scored_mask.npy", gated)
(fake / "complete.json").write_text(json.dumps({"state": "complete", "DRYRUN": "v1 blend stand-in, not C2",
                                                "checkpoint": None}) + "\n")
log(f"test cache rows {len(cache['cand_rows'])}; fake scores written")
