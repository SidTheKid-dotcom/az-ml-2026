"""INFO ONLY (not applicable to test: K test probabilities lost with cpu2). The brief's restack on logit(K p), with
kf_C OOF as K: band rows (J's) re-scored by a cross-fitted logistic on [logit Kp, its square, row-type/India one-hots
x logit Kp] (+ llm_logit and its interactions); other rows keep Kp; per-fold threshold (other folds) for all rows;
one owner, cap 11, S2<=5/S3<=6. Baseline = Kp with the same protocol. B_REST labels only."""
import json, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
import kllm_brest as kb

H, W = kb.H, kb.W
T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)
ref = pq.read_table(f"{H}/U/data/arena/reference.parquet", columns=["entity_id", "country"]).to_pandas()
n_s1 = len(ref); s1_index = pd.Index(ref.entity_id.to_numpy())
pool_ids = np.asarray(pq.read_table(f"{H}/U/data/arena/pool.parquet", columns=["entity_id"]).column(0).to_pylist(), dtype=object)
pool_index = pd.Index(pool_ids)
part = np.load(f"{H}/U/data/arena/finalize_v1/s1_partition.npy"); comp = np.load(f"{H}/U/data/arena/finalize_v1/component_ids.npy")
tc = np.load(f"{H}/U/data/arena/labels/true_count.npy"); owner = np.load(f"{H}/U/data/arena/labels/rec_owner.npy")
s1_fold = np.load(f"{H}/U/folds/brest_fold5_u4-20260927.npy").astype(np.int64); pop = part == 1; country = ref.country.to_numpy()
d = f"{H}/U/prep/brest"
dl, dr, dy = np.load(f"{d}/left.npy").astype(np.int64), np.load(f"{d}/right.npy").astype(np.int64), np.load(f"{d}/label.npy").astype(bool)
R = pq.read_table(f"{H}/R/data/routes/u_route_pairs_brest.parquet", columns=["left_idx", "rec_id"]).to_pandas()
rl = R.left_idx.to_numpy().astype(np.int64); rpi = pool_index.get_indexer(R.rec_id.to_numpy())
ry = np.where(rpi >= 0, owner[np.clip(rpi, 0, None)] == rl, False)
V = pq.read_table(f"{H}/R/out/r_reverse_pairs_b_rest.parquet", columns=["s1_id", "rec_id", "label"]).to_pandas()
vl = s1_index.get_indexer(V.s1_id.to_numpy()).astype(np.int64); vy = V.label.to_numpy().astype(bool)
s1 = np.concatenate([dl, rl, vl]); rec_s = np.concatenate([pool_ids[dr], R.rec_id.to_numpy().astype(object), V.rec_id.to_numpy().astype(object)])
y = np.concatenate([dy, ry, vy]); src = np.concatenate([np.zeros(len(dl), np.int8), np.ones(len(rl), np.int8), np.full(len(vl), 2, np.int8)])
pK = np.load(f"{W}/in/kfC_oof.npy").astype(np.float64)
rec, rec_u = pd.factorize(rec_s); is_s2 = np.asarray([r.startswith("S2") for r in rec_u], bool)[rec]
L = pd.read_parquet(f"{W}/in/llm_brest.parquet")
key_rows = pd.DataFrame({"s1_id": ref.entity_id.to_numpy()[s1], "rec_id": rec_s, "row_type": src, "i": np.arange(len(s1))})
m = L.merge(key_rows, on=["s1_id", "rec_id", "row_type"], how="inner").drop_duplicates("i")
rows = m.i.to_numpy().astype(np.int64); zl = m.llm_logit.to_numpy().astype(np.float64)
rt = src[rows]; ind = (country[s1[rows]] == "India").astype(float); Lk = kb.logit(pK[rows])
r1, r2 = (rt == 1).astype(float), (rt == 2).astype(float)
X = np.column_stack([Lk, Lk*Lk, r1, r2, ind, r1*Lk, r2*Lk, ind*Lk, zl, zl*Lk, r1*zl, r2*zl, ind*zl])
yb = y[rows].astype(float); rf = s1_fold[s1[rows]]; row_fold = s1_fold[s1]
def crossfit(p):
    per = {t: kb.per_s1(kb.decide(p, s1, rec, np.full(len(p), t), is_s2), s1, y, tc, n_s1) for t in kb.GRID}
    pick = {f: float(max(kb.GRID, key=lambda t: per[t][pop & (s1_fold != f) & (s1_fold >= 0)].mean())) for f in range(5)}
    thr = np.array([pick[f] for f in range(5)])[np.clip(row_fold, 0, 4)]
    sel = kb.decide(p, s1, rec, thr, is_s2); fo = kb.per_s1(sel, s1, y, tc, n_s1)
    tp = int((sel & y).sum())
    return fo, {"macro_f05": float(fo[pop].mean()), "precision": tp/int(sel.sum()), "recall": tp/int(tc[pop].sum()),
                "false_links": int((sel & ~y).sum()), "US": float(fo[pop & (country == "US")].mean()),
                "India": float(fo[pop & (country == "India")].mean()), "picks": pick}
rep = {"info_only": "restack on logit(kf_C p); not applicable to test (K test p lost)"}
fK, rep["K"] = crossfit(pK); log(f"K {rep['K']}")
for vname, vm in (("all_band", np.ones(len(rows), bool)), ("no_us_dense", ~((rt == 0) & (ind == 0)))):
    res = {}
    for arm, cols in (("K_only_restack", list(range(8))), ("K_llm_restack", list(range(13)))):
        q = np.empty(vm.sum()); Xv, yv, fv = X[vm][:, cols], yb[vm], rf[vm]
        for f in range(5):
            w = kb.fit_logreg(Xv[fv != f], yv[fv != f]); q[fv == f] = kb.predict_logreg(w, Xv[fv == f])
        p = pK.copy(); p[rows[vm]] = q
        res[arm], rep[f"{vname}/{arm}"] = crossfit(p)
        rep[f"{vname}/{arm}"]["vs_K"] = kb.bootstrap(res[arm] - fK, comp, pop)
        log(f"{vname}/{arm} {rep[f'{vname}/{arm}']}")
    rep[f"{vname}/llm_vs_K_only_restack"] = kb.bootstrap(res["K_llm_restack"] - res["K_only_restack"], comp, pop)
    log(f"{vname} llm vs K-only restack {rep[f'{vname}/llm_vs_K_only_restack']}")
json.dump(rep, open(f"{W}/out/prestack_info.json", "w"), indent=1, default=float)
log("PRESTACK DONE")
