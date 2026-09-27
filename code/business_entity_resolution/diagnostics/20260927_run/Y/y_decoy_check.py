"""Agent Y (label-free): which decoy words drive the added French links (Laya rows vs S1 file), with examples.
Same decoy test as s_rebuild.falsifier (record name adds a decoy word absent from the S1 name).
Usage: y_decoy_check.py ROWS.tsv OUT.json"""
import json, re, sys, unicodedata
from collections import Counter
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
Y = Path("/home/ubuntu/run27/Y")
sys.path.insert(0, "/home/ubuntu/run27/Y/repo/diagnostics/20260927_reset")
import sibling_check as sib
DECOY = {"groupe", "holding", "developpement", "participations", "distribution", "international", "france"}

def links(path, fr=None):
    out = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, ids = line.rstrip("\n").split("\t")
            if ids and (fr is None or s in fr):
                out.update((s, r) for r in ids.split(","))
    return out
ref = pq.read_table(Y / "in/test/dense_test/reference.parquet", columns=["entity_id", "business_name", "business_address", "country"]).to_pandas()
fr = set(ref.entity_id[ref.country == "France"])
new = links(sys.argv[1]); old = links(Y / "in/opt/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv", fr)
add = sorted(new - old)
pool = pq.read_table(Y / "in/test/dense_test/pool.parquet", columns=["entity_id", "business_name", "business_address"]).to_pandas()
rn = dict(zip(ref.entity_id, ref.business_name.fillna(""))); ra = dict(zip(ref.entity_id, ref.business_address.fillna("")))
pn = dict(zip(pool.entity_id, pool.business_name.fillna(""))); pa = dict(zip(pool.entity_id, pool.business_address.fillna("")))
sc = pd.read_parquet(Y / "results/laya_french_pair_scores.parquet", columns=["s1_id", "rec_id", "laya", "tree"])
sc = sc.set_index(sc.s1_id + "|" + sc.rec_id)
other = {"c2fr_S": links(Y / "in/other/c2fr_french_rows.tsv"), "k_vote3_t08acr": links(Y / "in/other/k_vote3_t08acr.tsv")}
g = pq.read_table(Y / "in/G/g_v1_pair_scores.parquet", columns=["s1_id", "rec_id", "accept"]).to_pandas()
other["g_v1"] = set(zip(g.s1_id[g.accept], g.rec_id[g.accept]))
support = Counter()
words, ex, n_dec = Counter(), [], 0
for s, r in add:
    w = (set(sib.norm(pn.get(r, ""))) - set(sib.norm(rn[s]))) & DECOY
    if w:
        n_dec += 1; words.update(w)
        support.update(k for k, v in other.items() if (s, r) in v)
        if len(ex) < 25:
            k = s + "|" + r
            ex.append({"s1": rn[s] + " | " + ra[s], "rec": pn.get(r, "") + " | " + pa.get(r, ""), "words": sorted(w),
                       "laya": float(sc.laya.get(k, np.nan)), "tree": float(sc.tree.get(k, np.nan))})
R = {"added": len(add), "added_with_decoy_word": n_dec, "decoy_additions_also_accepted_by": dict(support), "rate": n_dec / max(len(add), 1), "by_word": dict(words.most_common()),
     "examples": ex, "labels_read": False}
Path(sys.argv[2]).write_text(json.dumps(R, indent=1, ensure_ascii=False) + "\n")
print(json.dumps({k: v for k, v in R.items() if k != "examples"}, ensure_ascii=False))
for e in ex[:25]:
    print(json.dumps(e, ensure_ascii=False))
