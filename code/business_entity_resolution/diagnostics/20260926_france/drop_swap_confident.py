"""Probe: S1 public file minus French template-swap links with plain blend >= 0.7 (disjoint from the T2r band probe)."""
import hashlib, json
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from ber.pair_tree_c2 import COLUMNS, c2_blend, c2_on_rows, v1_cache
W = Path("/opt/az-ml-new/agents/pair_tree_c2"); T = Path("/opt/az-ml-new/test_inputs"); TS = Path("/opt/az-ml-new/runs/c2_step040000/test_scores")
s1_ids = np.asarray(pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id"]).column("entity_id").to_pylist(), dtype=object)
rec_ids = np.asarray(pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist(), dtype=object)
n_rec = len(rec_ids)
cache = v1_cache(None, None, None, None, None, None, W / "step005000/test_v1_cache.npz")
left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
c2 = c2_on_rows(TS, cache["cand_rows"], cache["base"], must_score=np.ones(len(left), bool), n_pairs=int(cache["n"][2]))
pp = c2_blend(np.where(np.isnan(c2), -1.0, c2), cache["base"], 0.9)
keys = left * n_rec + right; o = np.argsort(keys); ks = keys[o]
s1_pos = {v: i for i, v in enumerate(s1_ids)}; rec_pos = {v: i for i, v in enumerate(rec_ids)}
swaps = json.loads(Path("/workspace/az-ml-new/eureka/template_swap_fr.json").read_text())
q = np.array([s1_pos[s] * n_rec + rec_pos[r] for s, r in swaps], dtype=np.int64)
pos = np.searchsorted(ks, q); assert np.all(ks[pos] == q)
conf = {tuple(sw) for sw, v in zip(swaps, pp[o[pos]]) if v >= 0.7}
src = Path("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv").read_text(encoding="utf-8").split("\n")
out, dropped = [src[0]], 0
for line in src[1:]:
    if not line: continue
    s, ids = line.split("\t"); lst = ids.split(",") if ids else []
    keep = [r for r in lst if (s, r) not in conf]; dropped += len(lst) - len(keep)
    out.append(s + "\t" + ",".join(keep))
p = Path("/workspace/az-ml-new/eureka/probes/drop_T3conf/matching_results.tsv"); p.parent.mkdir(parents=True, exist_ok=True)
with open(p, "w", encoding="utf-8", newline="\n") as h: h.write("\n".join(out) + "\n")
print(json.dumps({"confident_swaps": len(conf), "dropped": dropped, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}))
