"""Label-free shift diagnostic: B_REST vs test rates per country x row type on the primary band (B_REST OOF decisions
re-derived from q_no_us_dense_llm.npy with the fold picks), and adversarial AUC of u_p / q."""
import json
import numpy as np, pandas as pd, pyarrow.parquet as pq
import kllm_brest as kb
H, W = kb.H, kb.W
b = json.load(open(f"{W}/out/brest_report.json"))["variants"]["no_us_dense/llm"]
refb = pq.read_table(f"{H}/U/data/arena/reference.parquet", columns=["entity_id", "country"]).to_pandas()
Lb = pd.read_parquet(f"{W}/in/llm_brest.parquet")
Lb["country"] = pd.Series(refb.country.to_numpy(), index=refb.entity_id.to_numpy()).reindex(Lb.s1_id.to_numpy()).to_numpy()
Lb = Lb[~((Lb.row_type == 0) & (Lb.country == "US"))]
Lt = pd.read_parquet(f"{W}/in/llm_test.parquet")
reft = pq.read_table(f"{H}/U/data/test/dense_test/reference.parquet", columns=["entity_id", "country"]).to_pandas()
Lt["country"] = pd.Series(reft.country.to_numpy(), index=reft.entity_id.to_numpy()).reindex(Lt.s1_id.to_numpy()).to_numpy()
Lt = Lt[~((Lt.row_type == 0) & (Lt.country == "US"))]
w, cols = np.asarray(b["weights_full"]), b["cols"]
def feats(D, a):
    rt = D.row_type.to_numpy(); ind = (D.country == "India").to_numpy().astype(float)
    Lu = kb.logit(D.u_p.to_numpy()); zl = D.llm_logit.to_numpy().astype(float)
    r1, r2 = (rt == 1).astype(float), (rt == 2).astype(float)
    X = np.column_stack([a, Lu, a*Lu, r1, r2, ind, r1*a, r2*a, ind*a, r1*Lu, r2*Lu, zl, zl*a, r1*zl, r2*zl, ind*zl, zl*Lu])[:, cols]
    return kb.predict_logreg(w, X)
# K accept bits: B_REST not saved -> use q>=t share with a from full-model scoring as a rough rate; test from K file
K = pd.read_csv(f"{W}/in/us_india_mix_F_C.tsv", sep="\t", dtype=str, keep_default_na=False)
kl = K.assign(rec=K.matched_entity_ids.str.split(",")).explode("rec")
kp = set(zip(kl.source1_entity_id, kl.rec))
at = np.fromiter(((x, y) in kp for x, y in zip(Lt.s1_id, Lt.rec_id)), bool, len(Lt)).astype(float)
Lt["a"] = at; Lt["q"] = feats(Lt, at)
t = b["full_threshold"]
out = {}
for (c, ty), g in Lt.groupby(["country", "row_type"]):
    gb = Lb[(Lb.country == c) & (Lb.row_type == ty)]
    yy = np.concatenate([np.zeros(len(gb)), np.ones(len(g))])
    out[f"{c}/{ty}"] = {"brest_rows": len(gb), "test_rows": len(g),
        "u_p_ge_0.5_per_1k": [round(1000*float((gb.u_p >= .5).mean()), 1), round(1000*float((g.u_p >= .5).mean()), 1)],
        "llm_p_ge_0.5_per_1k": [round(1000*float((gb.llm_p >= .5).mean()), 1), round(1000*float((g.llm_p >= .5).mean()), 1)],
        "auc_u_p": round(kb.auc(np.concatenate([gb.u_p, g.u_p]), yy), 3),
        "test_K_acc_per_1k": round(1000*float(g.a.mean()), 1), "test_q_ge_t_per_1k": round(1000*float((g.q >= t).mean()), 1),
        "test_q_ge_t_and_not_K_per_1k": round(1000*float(((g.q >= t) & (g.a == 0)).mean()), 1)}
    print(f"{c}/{ty}", out[f"{c}/{ty}"], flush=True)
json.dump(out, open(f"{W}/out/shift_diag.json", "w"), indent=1)
