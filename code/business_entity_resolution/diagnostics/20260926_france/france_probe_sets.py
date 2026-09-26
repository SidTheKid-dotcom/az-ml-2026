"""Size, overlap and label-free delta curves of candidate France-only probes on the S1 public file (0.978287), and
writes the drop-probe files. Delta(p): p = truth rate of the dropped links; per link with predicted list length n,
dropping a false link gains 4/(5n-1) (n>=2), a true one loses 1/(5n-4); n=1 counted as gain 0.5 / loss 1.
No labels are read."""
import hashlib, json, re, sys, unicodedata
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from rapidfuzz import fuzz
from ber.pair_tree_c2 import COLUMNS, c2_blend, c2_on_rows, decode, v1_cache, write_tsv, validate_tsv, sha256, log
LEGAL = set("inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl sas sasu sa sci eurl ei snc gmbh opc lp pllc pc".split())
SUFFIX = set("group groupe holdings holding fils sons france india usa international cie associes developpement distribution".split())
FILLER = set("et and the".split())
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()
def name_info(s):
    t = norm(s); return (frozenset(x for x in t if x not in LEGAL | SUFFIX | FILLER), frozenset(x for x in t if x in SUFFIX))
def digits(s): return tuple(str(int(d)) for d in re.findall(r"\d+", s or ""))
def rule(ln, la, rn, ra):
    (lc, ls), (rc, rs) = ln, rn
    if not lc or lc != rc or (rs - ls): return False
    ld = digits(la)
    return bool(ld) and ld == digits(ra) and fuzz.token_set_ratio(" ".join(norm(la)), " ".join(norm(ra))) >= 70
S1_SHA = "559f37b9b66f4e18fc28c0b25ae60d08543c386e0bdb32c619990b4c9cbd1a78"
W = Path("/opt/az-ml-new/agents/pair_tree_c2"); T = Path("/opt/az-ml-new/test_inputs")
S1RUN = Path("/opt/az-ml-new/runs/s1_pair_tree_40k"); TS = Path("/opt/az-ml-new/runs/c2_step040000/test_scores")
OUT = Path("/workspace/az-ml-new/eureka/probes"); OUT.mkdir(parents=True, exist_ok=True)
rep = json.loads((S1RUN / "report.json").read_text()); pw, pt = rep["plain_c2_name_only"]["weight"], rep["plain_c2_name_only"]["threshold"]
ref_table = pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id"] + COLUMNS)
pool_table = pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id"] + COLUMNS)
s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object); rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
country = np.asarray(ref_table.column("country").to_pylist(), dtype=object); n_s1, n_rec = len(s1_ids), len(rec_ids)
ref, pool = ref_table.select(COLUMNS).to_pandas(), pool_table.select(COLUMNS).to_pandas()
cache = v1_cache(None, None, None, None, None, None, W / "step005000/test_v1_cache.npz")
left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64); n_pairs = int(cache["n"][2]); base = cache["base"]
c2 = c2_on_rows(TS, cache["cand_rows"], base, must_score=np.ones(len(left), bool), n_pairs=n_pairs)
c2v = np.where(np.isnan(c2), -1.0, c2)
plain_prob = c2_blend(c2v, base, pw); mult = cache["s1_mult"][left]
prob = np.load(S1RUN / "probabilities_candidates.npy")
accept = decode(prob, rep["threshold"], rep["variant"], left, right, cache["rec_empty"], cache["name_exact"], mult)
plain = decode(plain_prob, pt, "model_then_name_only", left, right, cache["rec_empty"], cache["name_exact"], mult)
france = country[left] == "France"
la = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); ra = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
pair_keys = np.sort(la * n_rec + ra); del la, ra
rec_index = {v: i for i, v in enumerate(rec_ids)}
final = np.where(france, plain, accept); final_prob = np.where(france, plain_prob, prob)
par = OUT / "parity.tsv"; write_tsv(par, s1_ids, rec_ids, left, right, final, final_prob)
assert sha256(par) == S1_SHA, "parity failed"; par.unlink(); log("parity OK")
fa = plain & france
lengths = np.bincount(left[final], minlength=n_s1)
# classes on French accepted rows
sel = np.flatnonzero(fa & (base >= 0.3))
hit = np.array([rule(name_info(ref.business_name[i]), ref.business_address[i] or "", name_info(pool.business_name[j]), pool.business_address[j] or "")
                for i, j in zip(left[sel], right[sel])], dtype=bool)
is_rule = np.zeros(len(left), bool); is_rule[sel[hit]] = True
swaps = json.loads(Path("/workspace/az-ml-new/eureka/template_swap_fr.json").read_text())
s1_pos = {v: i for i, v in enumerate(s1_ids)}
keys_fa = np.flatnonzero(fa); k_fa = left[keys_fa] * n_rec + right[keys_fa]; o = np.argsort(k_fa); k_sorted = k_fa[o]
ks = np.array([s1_pos[s] * n_rec + rec_index[r] for s, r in swaps], dtype=np.int64)
pos = np.searchsorted(k_sorted, ks); ok = (pos < len(k_sorted)) & (k_sorted[np.minimum(pos, len(k_sorted) - 1)] == ks)
is_swap = np.zeros(len(left), bool); is_swap[keys_fa[o[pos[ok]]]] = True
name_only = fa & (plain_prob < pt)
sets = {"T1_tree_yes_c2_no": fa & (base >= 0.9) & (c2v < 0.05),
        "T1r_tree_yes_c2_no_minus_rule": fa & (base >= 0.9) & (c2v < 0.05) & ~is_rule,
        "T2_blend_below_0.7": fa & (plain_prob < 0.7),
        "T2r_band_0.1_0.7_minus_rule": fa & (plain_prob >= pt) & (plain_prob < 0.7) & ~is_rule,
        "T3_template_swap": is_swap, "name_only_adds": name_only, "rule_pairs_accepted": fa & is_rule}
def curve(m):
    n = lengths[left[m]]
    g = np.where(n >= 2, 4.0 / (5 * n - 1), 0.5); l = np.where(n >= 2, 1.0 / (5 * n - 4), 1.0)
    G, L = g.sum(), l.sum(); N = 1732544
    return {"links": int(m.sum()), "s1s": int(len(np.unique(left[m]))), "mean_list_len": round(float(n.mean()), 2),
            "break_even_truth_rate": round(float(G / (G + L)), 3),
            **{f"delta_public_if_{int(p*100)}pct_true": round(float(((1 - p) * G - p * L) / N), 5) for p in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0)}}
out = {"french_accepted_links": int(fa.sum()), "sets": {k: curve(m) for k, m in sets.items()},
       "overlaps": {"T1&T3": int((sets["T1_tree_yes_c2_no"] & is_swap).sum()), "T2&T3": int((sets["T2_blend_below_0.7"] & is_swap).sum()),
                    "T1&rule": int((sets["T1_tree_yes_c2_no"] & is_rule).sum()), "T1&T2": int((sets["T1_tree_yes_c2_no"] & sets["T2_blend_below_0.7"]).sum()),
                    "T3&rule": int((is_swap & is_rule).sum())},
       "T3_blend_bands": {b: int((is_swap & (plain_prob >= lo) & (plain_prob < hi)).sum()) for b, lo, hi in (("<0.1", -9, .1), ("0.1-0.3", .1, .3), ("0.3-0.7", .3, .7), ("0.7-0.9", .7, .9), (">=0.9", .9, 9))},
       "french_blend_bands_accepted": {b: int((fa & (plain_prob >= lo) & (plain_prob < hi)).sum()) for b, lo, hi in (("<0.1", -9, .1), ("0.1-0.3", .1, .3), ("0.3-0.7", .3, .7), ("0.7-0.9", .7, .9), (">=0.9", .9, 9))}}
# contested French records (>=2 French S1 claimants with plain_prob >= 0.1)
cm = france & (plain_prob >= pt); cc = np.bincount(right[cm], minlength=n_rec)
out["french_records_with_2plus_claimants"] = int((cc >= 2).sum())
print(json.dumps(out, indent=1), flush=True)
(OUT / "probe_sets.json").write_text(json.dumps(out, indent=1))
for name, key in (("drop_T1", "T1_tree_yes_c2_no"), ("drop_T1r", "T1r_tree_yes_c2_no_minus_rule"), ("drop_T2r", "T2r_band_0.1_0.7_minus_rule")):
    f = final & ~sets[key]; p = OUT / name / "matching_results.tsv"
    L = write_tsv(p, s1_ids, rec_ids, left, right, f, final_prob); chk = validate_tsv(p, s1_ids, rec_index, pair_keys, n_rec)
    print(name, json.dumps({"sha256": sha256(p), "passed": chk["passed"], "links": chk["links"], "problems": chk["problems"], "french_links": int(L[country == "France"].sum())}), flush=True)
