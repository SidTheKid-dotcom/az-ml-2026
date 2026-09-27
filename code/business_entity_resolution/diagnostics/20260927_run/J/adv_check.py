"""Agent J: adversarial check of llm_logit, B_REST vs test US/India uncertain rows (label-free).

AUC of "row is test" by llm_logit, within country x row type, raw and stratified by U-probability decile
(mean over deciles of the within-decile AUC, 0.5 = no shift). Also acceptance per 1k rows at llm_p >= 0.5.
Usage: adv_check.py TEXTS_DIR LLM_BREST.parquet LLM_TEST.parquet OUT.json
"""
import json
import sys

import numpy as np
import pandas as pd


def auc(score, y):
    s = pd.Series(score).rank().to_numpy()
    y = np.asarray(y, bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((s[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    texts, fb, ft, out = sys.argv[1:5]
    cc = {}
    for side in ("train", "test"):
        t = pd.read_parquet(f"{texts}/{side}_s1.parquet", columns=["entity_id", "cc"])
        cc[side] = pd.Series(t.cc.to_numpy(), index=t.entity_id.to_numpy())
    b = pd.read_parquet(fb)
    t = pd.read_parquet(ft)
    b["cc"] = cc["train"].reindex(b.s1_id.to_numpy()).to_numpy()
    t["cc"] = cc["test"].reindex(t.s1_id.to_numpy()).to_numpy()
    rep = {"cc_values_brest": {int(k): int(v) for k, v in b.cc.value_counts().items()},
           "cc_values_test": {int(k): int(v) for k, v in t.cc.value_counts().items()}}
    both = pd.concat([b.assign(is_test=0), t.assign(is_test=1)], ignore_index=True)
    both["dec"] = pd.qcut(both.u_p, 10, labels=False, duplicates="drop")
    res = {}
    for (c, r), g in both.groupby(["cc", "row_type"]):
        within = [auc(h.llm_logit.to_numpy(), h.is_test.to_numpy()) for _, h in g.groupby("dec") if h.is_test.nunique() == 2]
        res[f"cc{int(c)}_rt{int(r)}"] = {
            "rows_brest": int((g.is_test == 0).sum()), "rows_test": int((g.is_test == 1).sum()),
            "auc_raw": auc(g.llm_logit.to_numpy(), g.is_test.to_numpy()),
            "auc_u_p_raw": auc(g.u_p.to_numpy(), g.is_test.to_numpy()),
            "auc_within_u_decile_mean": float(np.nanmean(within)) if within else None,
            "accept_per_1k_brest": float((g[g.is_test == 0].llm_p >= .5).mean() * 1000),
            "accept_per_1k_test": float((g[g.is_test == 1].llm_p >= .5).mean() * 1000)}
    rep["by_country_rowtype"] = res
    json.dump(rep, open(out, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
