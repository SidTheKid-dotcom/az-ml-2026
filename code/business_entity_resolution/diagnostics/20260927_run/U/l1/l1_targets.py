"""Idea 4 (27 Sep, Agent U): row universe for the A-trained level-1 model (label-free).

arena targets = every arena candidate row (prep/arena_cand order, all partitions) + the B_REST route rows
(prep/brest_u8f route section order). test targets = every test prep row (K's 14,513,431-row order) + the US/India
test route rows (prep/test_u8f route section order; record index in the dense pool extended by extra_rec_ids).
Writes prep/l1/targets_{arena,test}_{left,right}.npy and a manifest.
"""
import json
import shutil
from pathlib import Path

import numpy as np

P = Path("/home/ubuntu/run27/U/prep")
out = P / "l1"
out.mkdir(parents=True, exist_ok=True)
al, ar = np.load(P / "arena_cand/left.npy").astype(np.int64), np.load(P / "arena_cand/right.npy").astype(np.int64)
bf = np.load(P / "brest_u8f/x_route.npy")[:, 0] == 1
bl, br = np.load(P / "brest_u8f/left.npy"), np.load(P / "brest_u8f/right.npy")
np.save(out / "targets_arena_left.npy", np.r_[al, bl[bf]])
np.save(out / "targets_arena_right.npy", np.r_[ar, br[bf]])
tl, tr = np.load(P / "test/left.npy"), np.load(P / "test/right.npy")
tf = np.load(P / "test_u8f/x_route.npy")[:, 0] == 1
ul, ur = np.load(P / "test_u8f/left.npy"), np.load(P / "test_u8f/right.npy")
np.save(out / "targets_test_left.npy", np.r_[tl, ul[tf]])
np.save(out / "targets_test_right.npy", np.r_[tr, ur[tf]])
shutil.copy(P / "test_u8f/extra_rec_ids.json", out / "extra_rec_ids.json")
man = {"arena_rows": int(len(al) + bf.sum()), "arena_cand": int(len(al)), "arena_route": int(bf.sum()),
       "test_rows": int(len(tl) + tf.sum()), "test_prep": int(len(tl)), "test_route": int(tf.sum()),
       "test_dense_records": json.loads((P / "test/meta.json").read_text())["n_rec"]}
(out / "targets.json").write_text(json.dumps(man, indent=1) + "\n")
print(man)
