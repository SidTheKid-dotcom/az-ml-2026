"""D1: are test records that never reach any dense top-50 list (unreached) plausible true copies?
Plausible owner = S1 of the same country with the same house number (first digit run) AND a shared city
component AND (fuzzy street-token match OR name core-token overlap). Label-free (test)."""
import json, re, unicodedata, random, time
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.csv as pc, pyarrow as pa
from pathlib import Path
from rapidfuzz.distance import Levenshtein
from ber.name_only import norm_name
T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.0f}s] {m}", flush=True)
RAW = Path("/workspace/az-ml-new/union/raw"); TI = Path("/opt/az-ml-new/test_inputs/dense_test")
TR = Path("/workspace/az-ml-new/eureka/retrieval/test"); OUT = Path("/workspace/az-ml-new/eureka/clarity")
S1FILE = "/opt/az-ml-new/runs/s1_pair_tree_40k/model_us_india_c2_france/matching_results.tsv"
COLS = ["entity_id", "business_name", "business_address", "country"]
PO = pc.ParseOptions(delimiter="\t", quote_char=False)
def read(p): return pc.read_csv(p, parse_options=PO, convert_options=pc.ConvertOptions(column_types={c: pa.string() for c in COLS}, include_columns=COLS)).to_pandas()
raw = pd.concat([read(RAW / "test_source2.tsv"), read(RAW / "test_source3.tsv")], ignore_index=True).fillna("")
dense_ids = set(pq.read_table(TI / "pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist())
ref = pq.read_table(TI / "reference.parquet", columns=COLS).to_pandas().fillna("")
log(f"raw {len(raw)} dense {len(dense_ids)} ref {len(ref)}")
raw["reached"] = raw.entity_id.isin(dense_ids)
acc_owner = {}
for line in open(S1FILE, encoding="utf-8").read().split("\n")[1:]:
    if line:
        s, ids = line.split("\t")
        for r in (ids.split(",") if ids else []): acc_owner[r] = s
raw["accepted_by"] = raw.entity_id.map(acc_owner)
REGIONS = {"hauts de france", "nouvelle aquitaine", "pays de la loire", "ile de france", "grand est", "normandie", "bretagne",
           "centre val de loire", "bourgogne franche comte", "auvergne rhone alpes", "occitanie", "provence alpes cote d azur", "corse"}
US_STATES = {"alabama","alaska","arizona","arkansas","california","colorado","connecticut","delaware","florida","georgia","hawaii","idaho","illinois","indiana","iowa","kansas","kentucky","louisiana","maine","maryland","massachusetts","michigan","minnesota","mississippi","missouri","montana","nebraska","nevada","new hampshire","new jersey","new mexico","new york","north carolina","north dakota","ohio","oklahoma","oregon","pennsylvania","rhode island","south carolina","south dakota","tennessee","texas","utah","vermont","virginia","washington","west virginia","wisconsin","wyoming","district of columbia"}
STREET = set("rue r avenue av ave boulevard bd blvd allee allees all quai q place pl impasse imp cours crs chemin ch route rte ruelle square sq de du des la le les l d street st road rd drive dr lane ln court ct way circle cir no n bis ter".split())
def fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", re.sub(r"[-'’_/]", " ", s)).strip()
def comps(a): return [re.sub(r"[^a-z0-9 ]", " ", fold(c)).split() for c in (a or "").split(",")]
def hnum(a):
    m = re.search(r"\d+", a or ""); return str(int(m.group(0))) if m else None
def is_region(tokens, country):
    j = " ".join(tokens)
    return (j in REGIONS) if country == "France" else (len(j) <= 2 or j in US_STATES)
def s1_parts(a, country):
    cs = comps(a); cities, street = set(), []
    for t in cs:
        if not t: continue
        if any(x.isdigit() for x in t):
            if not street: street = [x for x in t if not x.isdigit() and x not in STREET and len(x) >= 3]
        elif not is_region(t, country): cities.add(" ".join(t))
    return cities, street
def rec_parts(a):
    cs = comps(a); return {" ".join(t) for t in cs if t and not any(x.isdigit() for x in t)}, {x for t in cs for x in t}
def name_toks(n): return {t for t in norm_name(n).split() if len(t) >= 3}
out = {}
index = {}
for country in ("France", "US", "India"):
    sub = ref[ref.country == country]
    idx = {}
    for i, a, nm in zip(sub.index.to_numpy(), sub.business_address.to_numpy(), sub.business_name.to_numpy()):
        h = hnum(a)
        if h is None: continue
        cities, street = s1_parts(a, country)
        for c in cities: idx.setdefault((h, c), []).append(i)
    index[country] = idx
    log(f"{country}: index keys {len(idx)}")
S1INFO = {}
def s1info(i):
    if i not in S1INFO:
        S1INFO[i] = (s1_parts(ref.business_address[i], ref.country[i])[1], name_toks(ref.business_name[i]))
    return S1INFO[i]
def owners(row, strict_street=False):
    h = hnum(row.business_address)
    if h is None: return None, []
    cities, toks = rec_parts(row.business_address)
    rn = name_toks(row.business_name)
    cands = set()
    for c in cities: cands.update(index[row.country].get((h, c), ()))
    good, street_ok = [], []
    for s in cands:
        st, nt = s1info(s)
        sm = any(Levenshtein.normalized_similarity(a, b) >= 0.8 for a in st for b in toks if abs(len(a) - len(b)) <= 3)
        if sm or (rn & nt): good.append((s, sm, len(rn & nt)))
    good.sort(key=lambda x: (-x[1], -x[2]))
    return h, good
def stats(df, name, check_owner=False):
    n = len(df); has_h = one = ge1 = street1 = own_in = 0
    for row in df.itertuples(index=False):
        h, good = owners(row)
        if h is None: continue
        has_h += 1
        if good: ge1 += 1
        if len(good) == 1: one += 1
        if any(g[1] for g in good): street1 += 1
        if check_owner and row.accepted_by is not None:
            sidx = S1POS.get(row.accepted_by)
            if any(g[0] == sidx for g in good): own_in += 1
    d = {"n": n, "share_with_house_number": round(has_h / max(n, 1), 4), "share_ge1_plausible": round(ge1 / max(n, 1), 4),
         "share_exactly1": round(one / max(n, 1), 4), "share_street_fuzzy_owner": round(street1 / max(n, 1), 4)}
    if check_owner: d["share_accepted_owner_among_plausible"] = round(own_in / max(n, 1), 4)
    out[name] = d; log(f"{name}: {d}")
    return d
S1POS = {e: i for i, e in enumerate(ref.entity_id.to_numpy())}
rng = np.random.default_rng(1)
fr = raw[raw.country == "France"]; us = raw[raw.country == "US"]; ind = raw[raw.country == "India"]
U_FR = fr[~fr.reached]; U_US = us[~us.reached]; U_IN = ind[~ind.reached]
A_FR = fr[fr.accepted_by.notna()]; N_FR = fr[fr.reached & fr.accepted_by.isna()]
A_US = us[us.accepted_by.notna()]; N_US = us[us.reached & us.accepted_by.isna()]
def samp(df, k): return df.iloc[rng.choice(len(df), min(k, len(df)), replace=False)]
out["counts"] = {"unreached_france": int(len(U_FR)), "unreached_us": int(len(U_US)), "unreached_india": int(len(U_IN)),
                 "france_records": int(len(fr)), "france_accepted": int(len(A_FR)), "france_reached_not_accepted": int(len(N_FR))}
pU = stats(U_FR, "unreached_france")
pA = stats(samp(A_FR, 60000), "france_accepted_sample", check_owner=True)
pN = stats(samp(N_FR, 60000), "france_reached_not_accepted_sample")
stats(U_US, "unreached_us")
stats(samp(A_US, 40000), "us_accepted_sample", check_owner=True)
stats(samp(N_US, 40000), "us_reached_not_accepted_sample")
stats(samp(U_IN, 40000), "unreached_india_sample")
# mixture estimate of the true-copy share among unreached French records (label-free, rough)
k = "share_ge1_plausible"
mix = (pU[k] - pN[k]) / max(pA[k] - pN[k], 1e-9)
out["france_unreached_true_share_mixture_estimate"] = round(float(mix), 4)
out["france_unreached_true_links_estimate"] = int(round(mix * len(U_FR)))
# Track R coverage of unreached French records
tp = pq.read_table(TR / "pool.parquet", columns=["entity_id"]).column("entity_id").to_numpy(zero_copy_only=False)
tl, tr_, rk = np.load(TR / "left.npy"), np.load(TR / "right.npy"), np.load(TR / "rank.npy")
pair_rec = tp[tr_]; ufr = set(U_FR.entity_id)
m_all = np.fromiter((r in ufr for r in pair_rec), bool, len(pair_rec))
cov_all = set(pair_rec[m_all]); cov_top3 = set(pair_rec[m_all & (rk <= 2)])
# does Track R link unreached records with exactly one plausible owner to that owner?
exact_owner = {}
for row in U_FR.itertuples(index=False):
    h, good = owners(row)
    if good and len(good) == 1: exact_owner[row.entity_id] = good[0][0]
hit = hit3 = 0
pairs_by_rec = {}
for l_, r_, k_ in zip(tl[m_all], pair_rec[m_all], rk[m_all]): pairs_by_rec.setdefault(r_, []).append((int(l_), int(k_)))
for r, s in exact_owner.items():
    ps = pairs_by_rec.get(r, [])
    if any(l_ == s for l_, _ in ps): hit += 1
    if any(l_ == s and k_ <= 2 for l_, k_ in ps): hit3 += 1
out["track_r_coverage_france_unreached"] = {"records_in_any_track_r_pair": len(cov_all), "records_in_top3_pairs": len(cov_top3),
    "exactly1_plausible_records": len(exact_owner), "exactly1_owner_pair_present_any": hit, "exactly1_owner_pair_present_top3": hit3}
owner_list_len = {}
for line in open(S1FILE, encoding="utf-8").read().split("\n")[1:]:
    if line:
        s, ids = line.split("\t"); owner_list_len[s] = len(ids.split(",")) if ids else 0
ol = [owner_list_len[ref.entity_id[s]] for s in exact_owner.values()]
fr_all = [owner_list_len[e] for e in ref.entity_id[ref.country == "France"]]
out["exactly1_owner_current_list_len_mean"] = round(float(np.mean(ol)), 3) if ol else None
out["france_all_list_len_mean"] = round(float(np.mean(fr_all)), 3)
(OUT / "d1.json").write_text(json.dumps(out, indent=1) + "\n")
log(json.dumps({k: out[k] for k in out if k.startswith(("france_unreached", "track_r", "exactly1", "france_all"))}))
random.seed(3)
print("\n=== 20 random unreached French records with best plausible owner ===")
for row in U_FR.iloc[rng.choice(len(U_FR), 20, replace=False)].itertuples(index=False):
    h, good = owners(row)
    o = good[0][0] if good else None
    print(f"REC: {row.business_name} | {row.business_address}\n   OWNER: " + (f"{ref.business_name[o]} | {ref.business_address[o]} (street_fuzzy={good[0][1]}, n_plausible={len(good)}, list_len={owner_list_len[ref.entity_id[o]]})" if o is not None else "none"))
