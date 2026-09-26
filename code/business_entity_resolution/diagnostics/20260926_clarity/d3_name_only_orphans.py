"""D3: does the empty-address name-only rule lose precision when more entities are orphans?
Arena B_REST only (SLICE/HOLDOUT-2 labels unread). Withhold 0/20/30% of B_REST S1s (their records become
orphans), recompute per-country S1 name multiplicity without them, apply name_only_rule on top of the
submitted US/India decision (pair_tree_c2 OOF at 0.725 + one owner), and test an orphan-aware guard."""
import json, re, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
from pathlib import Path
from ber.pair_tree_arena import decide
from ber.arena_decision import evaluate, name_only_rule
from ber.name_only import norm_name, name_multiplicity, is_empty_address
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
A = Path("/workspace/az-ml-new/arena"); W = Path("/opt/az-ml-new/agents/pair_tree_c2")
OUT = Path("/workspace/az-ml-new/eureka/clarity"); OUT.mkdir(parents=True, exist_ok=True)
import sys
BASE = sys.argv[1] if len(sys.argv) > 1 else "pair_tree"
rows = np.load(W / "step040000/b_rest_pair_rows.npy"); oof = np.load(W / "step040000/oof.npy").astype(np.float32)
cache = np.load(W / "step005000/arena_v1_cache.npz")
cand = cache["cand_rows"]; assert np.all(np.diff(cand) > 0)
pos = np.searchsorted(cand, rows); assert np.all(cand[pos] == rows)
left = cache["left"][pos].astype(np.int64); right = cache["right"][pos].astype(np.int64)
rec_empty = cache["rec_empty"][pos]; name_exact = cache["name_exact"][pos]; cached_mult = cache["s1_mult"]
label = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool)
THR = 0.725
if BASE == "plain_c2":
    from ber.pair_tree_arena import load_scores
    c2 = load_scores(Path("/opt/az-ml-new/runs/c2_step040000/b_rest_scores/scores.npy"))[rows]
    tree = load_scores(A / "v1/base_probabilities.npy")[rows]
    oof = np.where(tree >= 0.01, 0.9 * np.maximum(c2, 0) + 0.1 * tree, 0).astype(np.float32); THR = 0.1
tc = np.load(A / "labels/true_count.npy"); part = np.load(A / "finalize_v1/s1_partition.npy")
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
country = ref.country.to_numpy(); n = len(ref)
log("S1 names")
s1_name = ref.business_name.map(norm_name).to_numpy()
m0 = name_multiplicity(country, s1_name)
log(f"multiplicity parity with cache: {bool(np.array_equal(m0, cached_mult))}")
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
def digs(a): return frozenset(str(int(d)) for d in re.findall(r"\d+", a or ""))
s1_digits_cache = {}
def s1d(s):
    if s not in s1_digits_cache: s1_digits_cache[s] = digs(ref.business_address[s])
    return s1_digits_cache[s]
log("pool names (for guard)")
rec_name = pool.business_name.map(norm_name).to_numpy()
rec_addr_ok = ~pool.business_address.map(is_empty_address).to_numpy()
by_name = pd.Series(np.arange(len(pool)))[rec_addr_ok].groupby(rec_name[rec_addr_ok]).apply(list) if False else None
rng = np.random.default_rng(20260926); u = rng.random(n)
results = {"base_decision": BASE, "multiplicity_parity": bool(np.array_equal(m0, cached_mult))}
# name -> address-bearing pool records (restricted to names of candidate add S1s, built lazily per level)
addr_idx = np.flatnonzero(rec_addr_ok); addr_names = rec_name[addr_idx]
name_to_recs = None
for frac in (0.0, 0.2, 0.3):
    withheld = (part == 1) & (u < frac); pop = (part == 1) & ~withheld
    keep = ~withheld[left]
    L, R, P, Y = left[keep], right[keep], oof[keep], label[keep]
    E, X = rec_empty[keep], name_exact[keep]
    active = ~withheld
    mult = np.zeros(n, np.int64); mult[active] = name_multiplicity(country[active], s1_name[active])
    acc = decide(P, L, R, THR, n)
    base = evaluate(acc, L, Y, tc, population=pop, country=country)
    ruled, info = name_only_rule(acc, L, R, E, X, mult[L])
    rr = evaluate(ruled, L, Y, tc, population=pop, country=country)
    add = info["added"]; drop = info["dropped"]; orem = info["owner_removed"]
    # orphan-aware guard on add candidates (hash-based joins)
    cand_add = E & X & (mult[L] == 1)
    ci = np.flatnonzero(cand_add)
    need = pd.Series(s1_name[L[ci]]).unique()
    sel = pd.Series(addr_names).isin(set(need)).to_numpy()
    recs = pd.DataFrame({"j": addr_idx[sel], "name": addr_names[sel]})
    cands = pd.DataFrame({"i": ci, "s": L[ci], "name": s1_name[L[ci]]})
    mg = cands.merge(recs, on="name", how="inner")
    acc_keys = pd.Index(L[acc].astype(np.int64) * len(pool) + R[acc].astype(np.int64))
    mg = mg[~pd.Index(mg.s.to_numpy().astype(np.int64) * len(pool) + mg.j.to_numpy().astype(np.int64)).isin(acc_keys)]
    fire = np.zeros(len(L), bool)
    rd_cache = {}
    for i, s, j in zip(mg.i.to_numpy(), mg.s.to_numpy(), mg.j.to_numpy()):
        if fire[i]: continue
        rd = rd_cache.get(j)
        if rd is None: rd = rd_cache[j] = digs(pool.business_address[j])
        if rd and not (rd & s1d(s)): fire[i] = True
    log(f"level {frac}: add candidates {len(ci)}, joined rows {len(mg)}, guard fired {int(fire.sum())}")
    Xg = X & ~(fire & cand_add)
    ruled_g, info_g = name_only_rule(acc, L, R, E, Xg, mult[L])
    rg = evaluate(ruled_g, L, Y, tc, population=pop, country=country)
    addg = info_g["added"]
    res = {"withheld_frac": frac, "population_s1": int(pop.sum()),
           "base_macro_f05": base["macro_f05"], "name_only_macro_f05": rr["macro_f05"], "guarded_macro_f05": rg["macro_f05"],
           "name_only_delta": rr["macro_f05"] - base["macro_f05"], "guarded_delta": rg["macro_f05"] - base["macro_f05"],
           "adds": int(add.sum()), "add_precision": float(Y[add].mean()) if add.any() else None,
           "drops": int(drop.sum()), "drops_true": int((drop & Y).sum()), "owner_removed": int(orem.sum()), "owner_removed_true": int((orem & Y).sum()),
           "guard_fired_on_add_candidates": int((fire & cand_add).sum()),
           "guarded_adds": int(addg.sum()), "guarded_add_precision": float(Y[addg].mean()) if addg.any() else None,
           "guard_blocked_adds": int((add & ~addg).sum()), "guard_blocked_true": int((add & ~addg & Y).sum()),
           "false_links": {"base": base.get("false_links"), "name_only": rr.get("false_links"), "guarded": rg.get("false_links")}}
    results[str(frac)] = res
    log(json.dumps(res))
(OUT / ("d3.json" if BASE == "pair_tree" else f"d3_{BASE}.json")).write_text(json.dumps(results, indent=1, default=float) + "\n")
log("done")
