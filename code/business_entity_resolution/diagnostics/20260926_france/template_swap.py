"""Template-word swap class: accepted link whose address digits equal the S1's, and whose core name differs from the
S1's by substituting whole words that are NOT typos of each other (Levenshtein ratio < 0.6) where every inserted word
is a common template word (a whole word in >= MINS other S1 names of that country). Arena B_REST truth rate under the
submitted US/India decision; counts in the S1 test file by country. Labels: arena only."""
import json, re, unicodedata, collections, sys
import numpy as np, pyarrow.parquet as pq
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from ber.pair_tree_arena import decide
MINS = 20
LEGAL = set("inc incorporated corp corporation co company llc ltd limited pvt private llp plc sarl sas sasu sa sci eurl ei snc gmbh opc lp pllc pc association".split())
SUFFIX = set("group groupe holdings holding fils sons france india usa international cie associes developpement distribution participations services service center partners enterprises".split())
FILLER = set("et and the of de du des la le les d l".split())
def norm(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower().replace(".", "")
    return re.sub(r"[^a-z0-9]+", " ", s).split()
def core(s): return [t for t in norm(s) if t not in LEGAL and t not in SUFFIX and t not in FILLER]
def dig(s): return tuple(str(int(d)) for d in re.findall(r"\d+", s or ""))
def vocab(names):
    c = collections.Counter()
    for n in names: c.update(set(core(n)))
    return c
def classify(sn, sa, rn, ra, voc):
    ds, dr = dig(sa), dig(ra)
    if not ds or ds != dr: return "other"
    a, b = core(sn), core(rn)
    sa_, sb_ = set(a), set(b)
    miss, extra = sa_ - sb_, sb_ - sa_
    if not miss or not extra: return "other"
    for e in extra:
        if max(Levenshtein.normalized_similarity(e, m) for m in miss) >= 0.6: return "typo_like"
    if all(voc[e] >= MINS for e in extra) and len(sa_ & sb_) >= 1: return "template_swap"
    return "rare_word_swap"
out = {}
A = Path("/workspace/az-ml-new/arena"); W = Path("/opt/az-ml-new/agents/pair_tree_c2/step040000")
rows = np.load(W / "b_rest_pair_rows.npy"); oof = np.load(W / "oof.npy")
t = pq.read_table(A / "v1/pairs.parquet", columns=["left_idx", "right_idx"])
la = t.column("left_idx").to_numpy()[rows].astype(np.int64); ra = t.column("right_idx").to_numpy()[rows].astype(np.int64); del t
lab = np.asarray(np.load(A / "labels/label.npy", mmap_mode="r")[rows]).astype(bool)
acc = decide(oof, la, ra, 0.725, int(la.max()) + 1)
ref = pq.read_table(A / "v1/reference.parquet", columns=["business_name", "business_address", "country"]).to_pandas()
pool = pq.read_table(A / "v1/pool.parquet", columns=["business_name", "business_address"]).to_pandas()
vocs = {c: vocab(ref.business_name[ref.country == c]) for c in ("US", "India")}
idx = np.flatnonzero(acc); ctry = ref.country.to_numpy()[la[idx]]
cls = np.array([classify(ref.business_name[i], ref.business_address[i], pool.business_name[j], pool.business_address[j], vocs[c])
                for i, j, c in zip(la[idx], ra[idx], ctry)])
for c in ("US", "India"):
    m = ctry == c
    out[f"arena_{c}"] = {k: [int((m & (cls == k)).sum()), round(float(lab[idx][m & (cls == k)].mean()), 4)] for k in ("template_swap", "rare_word_swap", "typo_like")}
ex = idx[cls == "template_swap"][:0]
T = Path("/opt/az-ml-new/test_inputs")
tref = pq.read_table(T / "dense_test/reference.parquet", columns=["entity_id", "business_name", "business_address", "country"]).to_pandas().set_index("entity_id")
tpool = pq.read_table(T / "dense_test/pool.parquet", columns=["entity_id", "business_name", "business_address"]).to_pandas().set_index("entity_id")
tv = {c: vocab(tref.business_name[tref.country == c]) for c in ("US", "India", "France")}
cnt = collections.defaultdict(collections.Counter); swaps = []
for line in open("/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv").read().split("\n")[1:]:
    if not line: continue
    sid, ids = line.split("\t")
    if not ids: continue
    c = tref.country[sid]
    for x in ids.split(","):
        k = classify(tref.business_name[sid], tref.business_address[sid], tpool.business_name[x], tpool.business_address[x], tv[c])
        cnt[c][k] += 1
        if c == "France" and k == "template_swap": swaps.append((sid, x))
out["test_counts"] = {c: dict(v) for c, v in cnt.items()}
Path("/workspace/az-ml-new/eureka/template_swap_fr.json").write_text(json.dumps(swaps))
print(json.dumps(out, indent=1))
import random; random.seed(3)
for sid, x in random.sample(swaps, min(25, len(swaps))):
    print(f"S1: {tref.business_name[sid]} | {tref.business_address[sid]}\n   REC: {tpool.business_name[x]} | {tpool.business_address[x]}")
