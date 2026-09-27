"""Agent Y (label-free): link-level agreement of Laya French rows with the other French families and the S1 file.
Families: S1 file French rows (#5 rule on stored C2 40k), C2-FR rows (Agent S, run27/S/results/french_rows.tsv),
G v1 accept (run27/G/v1/pair_scores.parquet), K 3-family vote rows (run27/K/france_vote_3fam, t0.8_acr).
Reports links, pairwise overlaps and probe A / C' acceptance (pricing-model group inferences, not labels).
Usage: y_agree.py OUT.json NAME=ROWS.tsv [NAME=ROWS.tsv ...]"""
import json, sys
from pathlib import Path
import pandas as pd, pyarrow.parquet as pq
Y = Path("/home/ubuntu/run27/Y")

def rows_links(path, fr=None):
    out = set()
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, ids = line.rstrip("\n").split("\t")
            if ids and (fr is None or s in fr):
                out.update(s + "|" + r for r in ids.split(","))
    return out

ref = pq.read_table(Y / "in/test/dense_test/reference.parquet", columns=["entity_id", "country"]).to_pandas()
fr = set(ref.entity_id[ref.country == "France"])
fam = {"s1_file": rows_links(Y / "in/opt/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv", fr),
       "c2fr_S": rows_links(Y / "in/other/c2fr_french_rows.tsv"),
       "k_vote3_t08acr": rows_links(Y / "in/other/k_vote3_t08acr.tsv")}
g = pq.read_table(Y / "in/G/g_v1_pair_scores.parquet", columns=["s1_id", "rec_id", "accept"]).to_pandas()
g = g[g.accept]
fam["g_v1"] = set((g.s1_id + "|" + g.rec_id).tolist())
for spec in sys.argv[2:]:
    name, path = spec.split("=", 1)
    fam[name] = rows_links(path)
probe = {}
for name, fn in (("A_T2r", "probeA_T2r.tsv"), ("Cprime", "probeCprime.tsv")):
    t = pd.read_csv(Y / "in/probe_links" / fn, sep="\t", dtype=str)
    probe[name] = set((t.iloc[:, 0] + "|" + t.iloc[:, 1]).tolist())
R = {"links": {k: len(v) for k, v in fam.items()},
     "probe_accept": {k: {p: len(v & ps) / len(ps) for p, ps in probe.items()} for k, v in fam.items()},
     "pairwise": {}}
names = list(fam)
for i, a in enumerate(names):
    for b in names[i + 1:]:
        A, B = fam[a], fam[b]
        R["pairwise"][f"{a}__{b}"] = {"both": len(A & B), "only_" + a: len(A - B), "only_" + b: len(B - A),
                                      "jaccard": len(A & B) / max(len(A | B), 1)}
s1 = fam["s1_file"]
R["additions_vs_s1_file_support"] = {}
for k in names:
    if k == "s1_file":
        continue
    add = fam[k] - s1
    R["additions_vs_s1_file_support"][k] = {"added": len(add), **{f"also_in_{o}": len(add & fam[o]) for o in names if o not in (k, "s1_file")}}
R["removals_vs_s1_file_support"] = {}
for k in names:
    if k == "s1_file":
        continue
    rem = s1 - fam[k]
    R["removals_vs_s1_file_support"][k] = {"removed": len(rem), **{f"also_removed_by_{o}": len(rem - fam[o])
                                                                   for o in names if o not in (k, "s1_file")}}
R["labels_read"] = False
Path(sys.argv[1]).write_text(json.dumps(R, indent=1) + "\n")
print(json.dumps(R, indent=1))
