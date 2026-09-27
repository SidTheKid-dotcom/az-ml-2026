"""Agent Y: does Laya add signal on Agent U's uncertain B_REST band (0.02 <= u8r_v2 p <= 0.98)? Reads B_REST TRAIN labels
(arena rec_owner; B_REST is the US/India evaluation partition, not SLICE/HOLDOUT-2). Reports AUC of u8r_v2 p, of Laya,
and of a 2-feature logistic combination fitted and scored with 5-fold S1-grouped cross-validation.
Usage: y_u_auc.py OUT.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
Y = Path("/home/ubuntu/run27/Y"); AR = Path("/home/ubuntu/run27/S/in/arena")
t = pd.read_parquet(Y / "results/laya_us_india_uncertain_brest.parquet")
t = t[t.has_text].reset_index(drop=True)
ref = pq.read_table(AR / "v1/reference.parquet", columns=["entity_id"]).column(0).to_pylist()
pool = pq.read_table(AR / "v1/pool.parquet", columns=["entity_id"]).column(0).to_pylist()
li = pd.Series(np.arange(len(ref)), index=ref).reindex(t.s1_id).to_numpy()
ri = pd.Series(np.arange(len(pool)), index=pool).reindex(t.rec_id).to_numpy()
owner = np.load(AR / "labels/rec_owner.npy")
y = (owner[ri.astype(np.int64)] == li.astype(np.int64)).astype(int)
def lg(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
X = np.column_stack([lg(t.u8r_v2_p.to_numpy()), lg(t.laya.to_numpy())])
oof = np.zeros(len(t))
for tr, te in GroupKFold(5).split(X, y, groups=t.s1_id):
    oof[te] = LogisticRegression(C=1.0).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
R = {"rows": int(len(t)), "positives": int(y.sum()), "auc_u8r_v2_p": float(roc_auc_score(y, t.u8r_v2_p)),
     "auc_laya": float(roc_auc_score(y, t.laya)), "auc_logit_combo_5fold_grouped": float(roc_auc_score(y, oof)),
     "by_row_type": {int(k): {"rows": int(len(g)), "auc_u8r": float(roc_auc_score(y[g.index], g.u8r_v2_p)) if 0 < y[g.index].sum() < len(g) else None,
                              "auc_laya": float(roc_auc_score(y[g.index], g.laya)) if 0 < y[g.index].sum() < len(g) else None}
                     for k, g in t.groupby("row_type")},
     "note": "B_REST TRAIN labels; Laya trained on arena partition A only; u8r_v2 p is U's cross-fitted OOF"}
Path(sys.argv[1]).write_text(json.dumps(R, indent=1) + "\n"); print(json.dumps(R))
