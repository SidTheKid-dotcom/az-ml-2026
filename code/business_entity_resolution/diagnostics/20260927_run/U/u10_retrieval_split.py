"""U10 follow-up (27 Sep, Agent U): split B_REST true links that are not candidate rows into
(a) in the dense top-50 pairs but dropped by the candidate rule (tree < 0.01, no exact name/key), and
(b) not in the dense top-50 at all, by country, record kind (EA / not) and native script. Also counts true
links whose record is not in the compact arena pool. B_REST owners only.
"""
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.forensic_features import EMPTY, address_kinds

ROOT = Path("/home/ubuntu/run27/U")
A = ROOT / "data/arena"
part = np.load(A / "finalize_v1/s1_partition.npy")
tc = np.load(A / "labels/true_count.npy").astype(np.int64)
own = np.load(A / "labels/rec_owner.npy")
own = np.where((own >= 0) & (part[np.maximum(own, 0)] == 1), own, -1)
t = pq.read_table(A / "v1/pairs.parquet" if (A / "v1/pairs.parquet").exists() else A / "pairs.parquet",
                  columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy().astype(np.int64)
ra = t.column("right_idx").to_numpy().astype(np.int64)
del t
in_dense = np.zeros(len(own), bool)
m = own[ra] == la
in_dense[ra[m]] = True
b_right = np.load(ROOT / "prep/brest/right.npy")
b_label = np.load(ROOT / "prep/brest/label.npy")
in_cand = np.zeros(len(own), bool)
in_cand[b_right[b_label]] = True
pool = pq.read_table(A / "pool.parquet", columns=["business_name", "business_address", "country"])
ea = address_kinds(pool.column("business_address")) == EMPTY
recs = np.flatnonzero((own >= 0) & ~in_cand)
names = pool.column("business_name").take(recs).to_pylist()
native = np.fromiter((any(ord(ch) > 0x2FF for ch in (v or "")) for v in names), bool, count=len(recs))
ref_c = np.asarray(pq.read_table(A / "reference.parquet", columns=["country"]).column("country").to_pylist())
out = {}
for where, mask in (("dense_but_candidate_rule_dropped", in_dense[recs]), ("not_in_dense_top50", ~in_dense[recs])):
    for c in ("US", "India"):
        cm = mask & (ref_c[own[recs]] == c)
        out[f"{where}|{c}"] = {"links": int(cm.sum()), "EA": int((cm & ea[recs]).sum()),
                               "native_script_record": int((cm & native).sum())}
pop = part == 1
out["true_links_b_rest"] = int(tc[pop].sum())
out["true_links_with_record_in_pool"] = int((own >= 0).sum())
out["true_links_record_not_in_pool"] = int(tc[pop].sum() - (own >= 0).sum())
(ROOT / "u2" / "u10_retrieval_split.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
