"""Agent S: map stored C2 step-40k B_REST scores (arena v1 pair order) onto arena_fr_v3 pairs via source_pair_row.
Usage: s_baseline_map.py ARENA_FR_DIR C2_B_REST_SCORES_DIR OUT_DIR. Only world-0 (replay/monitor) rows can map;
French tuning-world rows get mask False (C2 40k weights are lost, so it cannot be scored there)."""
import json, sys
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
A, B, O = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
O.mkdir(parents=True, exist_ok=True)
t = pq.read_table(A / "arena/pairs.parquet", columns=["left_idx", "world", "source_pair_row"])
left = t.column(0).to_numpy(); world = t.column(1).to_numpy(); src = t.column(2).to_numpy()
part = np.load(A / "s1_partition.npy"); tree = np.load(A / "arena/v1_tree.npy")
bs = np.load(B / "scores.npy", mmap_mode="r"); bm = np.load(B / "scored_mask.npy", mmap_mode="r")
assert len(bs) == 103_641_050, len(bs)
mon = (part[left] == 1)
assert (world[mon] == 0).all() and (src[mon] >= 0).all()
scores = np.zeros(len(left), np.float32); mask = np.zeros(len(left), bool)
rows = np.flatnonzero(mon)
scores[rows] = np.asarray(bs[src[rows]]); mask[rows] = np.asarray(bm[src[rows]])
need = mon & (tree >= .01)
R = {"monitor_pairs": int(mon.sum()), "monitor_gated": int(need.sum()), "gated_scored": int((need & mask).sum()),
     "source": str(B), "note": "C2 step 40k scores mapped by source_pair_row; tuning-world rows unscored"}
assert (need & ~mask).sum() == 0, R
np.save(O / "scores.npy", scores); np.save(O / "scored_mask.npy", mask)
(O / "complete.json").write_text(json.dumps(R, indent=1) + "\n"); print(json.dumps(R))
