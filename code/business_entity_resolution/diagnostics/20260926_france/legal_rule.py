"""Generator rule for legal-form / corporate-suffix changes. Pairs whose core name (minus legal forms and
corporate suffix words) is equal and whose address has the same digits and similar text. Arena B_REST
truth rate per class vs French test class sizes; acceptance under blend>=0.1 + one owner."""
import re, unicodedata, json
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
from rapidfuzz import fuzz
from ber.pair_tree_arena import load_scores, decide
LEGAL = set("inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl sas sasu sa sci eurl ei snc gmbh opc lp pllc pc".split())
SUFFIX = set("group groupe holdings holding fils sons france india usa international cie associes developpement distribution".split())
FILLER = set("et and the".split())
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()
def name_info(s):
    t = norm(s); return (frozenset(x for x in t if x not in LEGAL and x not in SUFFIX and x not in FILLER),
                         frozenset(x for x in t if x in LEGAL), frozenset(x for x in t if x in SUFFIX))
def digits(s): return tuple(str(int(d)) for d in re.findall(r"\d+", s or ""))
def classify(ln, la, rn, ra):
    lc, ll, ls = ln; rc, rl, rs = rn
    if not lc or lc != rc: return None
    ld, rd = digits(la), digits(ra)
    if not ld or ld != rd or fuzz.token_set_ratio(" ".join(norm(la)), " ".join(norm(ra))) < 70: return None
    legal = "same" if ll == rl else "rec_dropped" if not rl else "rec_added" if not ll else "rec_dropped_some" if rl < ll else "changed"
    suf = "suffix_added" if rs - ls else "suffix_same_or_dropped"
    return f"{legal}|{suf}"
def run(ref, pool, left, right, sel, extra):
    li, ri = left[sel], right[sel]
    ul, ur = np.unique(li), np.unique(ri)
    LN = {i: name_info(ref.business_name[i]) for i in ul}; RN = {j: name_info(pool.business_name[j]) for j in ur}
    cls = np.array([classify(LN[i], ref.business_address[i] or "", RN[j], pool.business_address[j] or "") or "" for i, j in zip(li, ri)], dtype=object)
    out = {}
    for c in sorted(set(cls) - {""}):
        m = cls == c
        out[c] = {"n": int(m.sum()), **{k: round(float(v[m].mean()), 4) for k, v in extra.items()}}
    return out
A = Path("/workspace/az-ml-new/arena"); T = Path("/opt/az-ml-new/test_inputs"); R = Path("/opt/az-ml-new/runs/c2_step040000")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy().astype(np.int64); ra = t.column("right_idx").to_numpy().astype(np.int64); del t
part = np.load(A / "finalize_v1/s1_partition.npy"); rows = np.flatnonzero(part[la] == 1); la, ra = la[rows], ra[rows]
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool)
c2 = load_scores(R / "b_rest_scores/scores.npy")[rows]; tree = load_scores(A / "v1/base_probabilities.npy")[rows]
b = np.where(tree >= .01, .9 * c2 + .1 * tree, 0).astype(np.float32); acc = decide(b, la, ra, 0.1, len(part))
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_name", "business_address"]).to_pandas()
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
sel = tree >= 0.3
arena = run(ref, pool, la, ra, sel, {"true": lab[sel], "accepted": acc[sel], "c2_ge_0.5": (c2 >= .5)[sel]})
print(json.dumps({"arena_b_rest": arena}, indent=1), flush=True)
del la, ra, lab, c2, tree, b, acc, ref, pool
lt = np.asarray(np.load(T / "index/left.npy", mmap_mode="r")).astype(np.int64); rt = np.asarray(np.load(T / "index/right.npy", mmap_mode="r")).astype(np.int64)
ref = pq.read_table(T / "dense_test/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(T / "dense_test/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
c2 = load_scores(R / "test_scores/scores.npy"); tree = load_scores(T / "base_probabilities.npy")
b = np.where(tree >= .01, .9 * c2 + .1 * tree, 0).astype(np.float32); acc = decide(b, lt, rt, 0.1, len(ref))
fr = (ref.country.to_numpy() == "France")[lt]
sel = fr & (tree >= 0.3)
test = run(ref, pool, lt, rt, sel, {"accepted": acc[sel], "c2_ge_0.5": (c2 >= .5)[sel]})
print(json.dumps({"test_france": test}, indent=1))
