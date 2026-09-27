"""Agent J: French deliverable from deliver_NAME/llm_fr.parquet -> llm_fr_tierT.parquet (s1_id, record_id, llm_p, ...)
plus label-free agreement stats vs the G / C2-FR / Qwen-FR votes. Usage: fr_publish.py NAME TIER OUT.parquet"""
import hashlib, json, sys
import numpy as np, pandas as pd
name, tier, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = pd.read_parquet(f"deliver_{name}/llm_fr.parquet")
d = d[d.tier == tier].reset_index(drop=True)
o = pd.DataFrame({"s1_id": d.s1_id, "record_id": d.rec_id, "llm_p": d.llm_p, "llm_logit": d.llm_logit,
                  "route": d.route, "tier": d.tier})
o.to_parquet(out, index=False)
votes = d.accept.astype(int) + d.c2fr_accept.astype(int) + d.qfr_accept.astype(int)
acc = d.llm_p >= .5
stats = {"rows": len(o), "sha256": hashlib.sha256(open(out, "rb").read()).hexdigest(),
         "llm_p_ge_0.5": int(acc.sum()),
         "llm_accept_rate_by_family_votes": {int(v): {"rows": int((votes == v).sum()), "llm_accept": float(acc[votes == v].mean())}
                                             for v in sorted(votes.unique())},
         "llm_accept_rate_by_G_prob_bin": {str(k): float(v) for k, v in acc.groupby(pd.cut(d.prob, [-.01, .02, .1, .5, .9, .98, 1.01])).mean().items()},
         "corr_logit_llm_vs_G_prob": float(np.corrcoef(d.llm_logit, np.log(np.clip(d.prob, 1e-6, 1 - 1e-6) / (1 - np.clip(d.prob, 1e-6, 1 - 1e-6))))[0, 1])}
json.dump(stats, open(out.replace(".parquet", "_stats.json"), "w"), indent=1)
print(json.dumps(stats, indent=1))
