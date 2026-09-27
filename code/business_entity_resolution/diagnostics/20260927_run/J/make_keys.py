"""Agent J: keyed (s1_id, rec_id) lists of uncertain pairs to score with the 4B judge.

US/India: Agent U u8r_v2 rows with p in [LO, HI] (B_REST: OOF p; test: fold-mean p).
France: G fr_v3_hybrid pair universe; tier 1 = any family uncertain or families disagree, tier 2 = rest with
any family >= 0.05 (scored last, optional).
"""
import sys, json
import numpy as np, pandas as pd, pyarrow.parquet as pq

IN, OUT = sys.argv[1], sys.argv[2]
LO, HI = 0.02, 0.98
rep = {}
for s in ("brest", "test"):
    t = pq.read_table(f"{IN}/u/u8r_pairs_{s}.parquet", columns=["s1_id", "rec_id", "row_type", "rec_outside_pool", "p"]).to_pandas()
    k = t[(t.p >= LO) & (t.p <= HI)].reset_index(drop=True)
    k = k.rename(columns={"p": "u_p"})
    k.to_parquet(f"{OUT}/keys_{s}_unc.parquet", index=False)
    rep[s] = {"rows": len(t), "uncertain": len(k), "by_row_type": {int(a): int(b) for a, b in k.row_type.value_counts().items()}}
g = pq.read_table(f"{IN}/g/pair_scores.parquet", columns=["s1_id", "rec_id", "route", "p1", "prob", "accept"]).to_pandas()

def rows_tsv(path):
    pairs = []
    with open(path) as f:
        next(f)
        for line in f:
            a, _, b = line.rstrip("\n").partition("\t")
            if b:
                for r in b.split(","):
                    r = r.strip()
                    if r:
                        pairs.append((a, r))
    return pd.MultiIndex.from_tuples(pairs, names=["s1_id", "rec_id"])

key = pd.MultiIndex.from_arrays([g.s1_id, g.rec_id])
c2fr = key.isin(rows_tsv(f"{IN}/sfr/french_rows.tsv"))
qfr = key.isin(rows_tsv(f"{IN}/sfr/qwenfr_t070_french_rows.tsv"))
gacc = g.accept.to_numpy()
votes = gacc.astype(int) + c2fr.astype(int) + qfr.astype(int)
unc = lambda v: (v >= .02) & (v <= .98)
tier1 = unc(g.prob.to_numpy()) | unc(g.p1.to_numpy()) | ((votes > 0) & (votes < 3))
anyhi = (g.prob.to_numpy() >= .05) | (g.p1.to_numpy() >= .05) | (votes > 0)
tier2 = anyhi & ~tier1
g["c2fr_accept"], g["qfr_accept"], g["tier"] = c2fr, qfr, np.where(tier1, 1, np.where(tier2, 2, 0)).astype(np.int8)
fr = g[g.tier > 0].sort_values("tier", kind="stable").reset_index(drop=True)
fr = fr[["s1_id", "rec_id", "route", "p1", "prob", "accept", "c2fr_accept", "qfr_accept", "tier"]]
fr.to_parquet(f"{OUT}/keys_fr.parquet", index=False)
rep["fr"] = {"universe": len(g), "tier1": int(tier1.sum()), "tier2": int(tier2.sum()),
             "c2fr_links": int(c2fr.sum()), "qfr_links": int(qfr.sum()), "g_accept": int(gacc.sum()),
             "votes_hist": {int(a): int(b) for a, b in zip(*np.unique(votes, return_counts=True))}}
json.dump(rep, open(f"{OUT}/keys_report.json", "w"), indent=1)
print(json.dumps(rep, indent=1))
