import sys, json, collections, random
import pandas as pd, pyarrow.csv as pc, pyarrow as pa, pyarrow.compute as pcc
sys.path.insert(0, "code/business_entity_resolution/src")
from ber.forensic_features import parse_addresses
T = "data/raw/dataset/test"; PO = pc.ParseOptions(delimiter="\t", quote_char=False)
def read(p, keep=None):
    t = pc.read_csv(p, parse_options=PO, convert_options=pc.ConvertOptions(column_types={c: pa.string() for c in ["entity_id","business_address","country"]}, include_columns=["entity_id","business_address","country"]))
    if keep is not None: t = t.filter(pcc.is_in(t["entity_id"], value_set=pa.array(list(keep))))
    return t.to_pandas().set_index("entity_id")
s1 = read(f"{T}/test_source1.tsv"); country = s1.country
random.seed(0); out = {}
files = {"S1": "/home/sidvish/Downloads/s1-pair-tree/matching_results.tsv", "S2": "/home/sidvish/Downloads/s2-france-ft/matching_results.tsv"}
samples = {}
for name, path in files.items():
    links = collections.defaultdict(list)
    for l in open(path).read().split("\n")[1:]:
        if not l: continue
        sid, ids = l.split("\t")
        if ids:
            parts = ids.split(",")
            for r in parts: links[country[sid]].append((sid, r, len(parts)))
    samples[name] = {c: random.sample(v, min(200000, len(v))) for c, v in links.items()}
    samples[name]["_totals"] = {c: len(v) for c, v in links.items()}
need = {r for name in samples for c, v in samples[name].items() if c != "_totals" for _, r, _ in v}
pool = pd.concat([read(f"{T}/test_source{i}.tsv", need) for i in (2, 3)])
for name in files:
    out[name] = {"totals": samples[name]["_totals"]}
    for c, v in samples[name].items():
        if c == "_totals": continue
        sids = [x[0] for x in v]; rids = [x[1] for x in v]
        st_s, _ = parse_addresses(s1.business_address.reindex(sids).fillna("").to_numpy(), [c]*len(v))
        st_r, _ = parse_addresses(pool.business_address.reindex(rids).fillna("").to_numpy(), [c]*len(v))
        cnt = collections.Counter()
        for a, b in zip(st_s, st_r):
            cnt["empty"] += (not a or not b); cnt["disjoint"] += bool(a and b and not (a & b)); cnt["n"] += 1
        out[name][c] = {"sample": cnt["n"], "street_disjoint_rate": round(cnt["disjoint"]/cnt["n"], 4), "street_missing_rate": round(cnt["empty"]/cnt["n"], 4)}
print(json.dumps(out, indent=1))
