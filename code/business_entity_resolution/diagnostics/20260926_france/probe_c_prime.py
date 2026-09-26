"""Probe C': from the S1 public file (0.978287), drop ONE random confident (C2 blend >= 0.7) non-legal-form French
link per French S1 whose list has >= 2 links (seed 20260926). No lists are emptied. French rows only. No labels read."""
import hashlib, json, re, unicodedata
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from rapidfuzz import fuzz
from ber.pair_tree_c2 import COLUMNS, c2_blend, c2_on_rows, v1_cache
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
W = Path("/opt/az-ml-new/agents/pair_tree_c2"); T = Path("/opt/az-ml-new/test_inputs"); TS = Path("/opt/az-ml-new/runs/c2_step040000/test_scores")
rt = pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id"] + COLUMNS); pt = pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id"] + COLUMNS)
s1_ids = np.asarray(rt.column("entity_id").to_pylist(), dtype=object); rec_ids = np.asarray(pt.column("entity_id").to_pylist(), dtype=object)
country = np.asarray(rt.column("country").to_pylist(), dtype=object); ref, pool = rt.select(COLUMNS).to_pandas(), pt.select(COLUMNS).to_pandas()
n_rec = len(rec_ids); s1_pos = {v: i for i, v in enumerate(s1_ids)}; rec_pos = {v: i for i, v in enumerate(rec_ids)}
cache = v1_cache(None, None, None, None, None, None, W / "step005000/test_v1_cache.npz")
left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
c2 = c2_on_rows(TS, cache["cand_rows"], cache["base"], must_score=np.ones(len(left), bool), n_pairs=int(cache["n"][2]))
pp = c2_blend(np.where(np.isnan(c2), -1.0, c2), cache["base"], 0.9)
keys = left * n_rec + right; o = np.argsort(keys); ks = keys[o]
src = Path("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv").read_text(encoding="utf-8").split("\n")
rng = np.random.default_rng(20260926)
out, dropped, lens, skipped_no_candidate = [src[0]], [], [], 0
NI = {}
def ni(kind, i, s):
    k = (kind, i)
    if k not in NI: NI[k] = name_info(s)
    return NI[k]
for line in src[1:]:
    if not line: continue
    s, ids = line.split("\t"); lst = ids.split(",") if ids else []
    i = s1_pos[s]
    if country[i] != "France" or len(lst) < 2:
        out.append(line); continue
    q = np.array([i * n_rec + rec_pos[r] for r in lst], dtype=np.int64)
    pos = np.searchsorted(ks, q); rows = o[np.minimum(pos, len(ks) - 1)]
    ok = ks[np.minimum(pos, len(ks) - 1)] == q
    cand = [r for r, row, f in zip(lst, rows, ok) if f and pp[row] >= 0.7]
    rng.shuffle(cand)
    pick = None
    for r in cand:
        j = rec_pos[r]
        if not rule(ni(0, i, ref.business_name[i]), ref.business_address[i] or "", ni(1, j, pool.business_name[j]), pool.business_address[j] or ""):
            pick = r; break
    if pick is None:
        skipped_no_candidate += 1; out.append(line); continue
    dropped.append((s, pick)); lens.append(len(lst))
    out.append(s + "\t" + ",".join(r for r in lst if r != pick))
p = Path("/workspace/az-ml-new/eureka/probes/drop_Cprime/matching_results.tsv"); p.parent.mkdir(parents=True, exist_ok=True)
with open(p, "w", encoding="utf-8", newline="\n") as h: h.write("\n".join(out) + "\n")
n = np.array(lens); g = 4.0 / (5 * n - 1); l = 1.0 / (5 * n - 4); N = 1732544
G, L = g.sum() / N, l.sum() / N
rep = {"dropped_links": len(dropped), "french_lists_touched": len(dropped), "lists_without_eligible_link": skipped_no_candidate,
       "mean_list_len": round(float(n.mean()), 2), "delta_if_all_false": round(G, 5), "delta_if_all_true": round(-L, 5),
       "slope": round(G + L, 5), "break_even_p": round(G / (G + L), 3), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
(p.parent / "report.json").write_text(json.dumps(rep, indent=1)); (p.parent / "dropped.json").write_text(json.dumps(dropped))
print(json.dumps(rep))
