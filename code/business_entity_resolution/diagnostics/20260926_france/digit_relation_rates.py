"""D4: digit relation (S1 address digits vs record address digits) of ACCEPTED links, by country, in the S1
test file; plus arena B_REST truth rate per relation under the submitted US/India decision. Labels: arena only."""
import json, re, collections
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
from ber.pair_tree_arena import decide
def dig(s): return tuple(str(int(d)) for d in re.findall(r"\d+", s or ""))
def rel(a, b):
    da, db = dig(a), dig(b)
    if not db: return "rec_no_digits"
    if not da: return "s1_no_digits"
    if da == db: return "same"
    sa, sb = set(da), set(db)
    if sb < sa: return "rec_subset"
    if sa < sb: return "rec_superset"
    if da[:1] == db[:1]: return "same_first_other_diff"
    return "first_number_differs"
out = {}
A = Path("/workspace/az-ml-new/arena"); W = Path("/opt/az-ml-new/agents/pair_tree_c2/step040000")
rows = np.load(W / "b_rest_pair_rows.npy"); oof = np.load(W / "oof.npy")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy()[rows].astype(np.int64); ra = t.column("right_idx").to_numpy()[rows].astype(np.int64); del t
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool)
acc = decide(oof, la, ra, 0.725, int(la.max()) + 1)
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_address", "country"]).to_pandas()
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_address"]).to_pandas()
idx = np.flatnonzero(acc)
r = np.array([rel(ref.business_address[i], pool.business_address[j]) for i, j in zip(la[idx], ra[idx])])
ctry = ref.country.to_numpy()[la[idx]]
for c in ("US", "India"):
    m = ctry == c; d = {}
    for k in sorted(set(r[m])):
        mk = m & (r == k); d[k] = [round(float(mk.sum() / m.sum()), 4), round(float(lab[idx][mk].mean()), 4)]
    out[f"arena_{c}_share_truth"] = d
T = Path("/opt/az-ml-new/test_inputs")
ref = pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id", "business_address", "country"]).to_pandas().set_index("entity_id")
pool = pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id", "business_address"]).to_pandas().set_index("entity_id")
cnt = collections.defaultdict(collections.Counter)
for line in open("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv").read().split("\n")[1:]:
    if not line: continue
    sid, ids = line.split("\t")
    if not ids: continue
    a = ref.business_address[sid]; c = ref.country[sid]
    for x in ids.split(","): cnt[c][rel(a, pool.business_address[x])] += 1
out["test_share"] = {c: {k: round(v / sum(cc.values()), 4) for k, v in sorted(cc.items())} for c, cc in cnt.items()}
print(json.dumps(out, indent=1))
