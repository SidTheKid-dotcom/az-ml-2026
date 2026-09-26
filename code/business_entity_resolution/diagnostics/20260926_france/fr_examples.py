import sys, random, collections
import pyarrow.csv as pc, pyarrow as pa, pyarrow.compute as pcc
sys.path.insert(0, "code/business_entity_resolution/src")
from ber.forensic_features import parse_addresses
T = "data/raw/dataset/test"; PO = pc.ParseOptions(delimiter="\t", quote_char=False)
COLS = ["entity_id","business_name","business_address","country"]
def read(p, keep=None):
    t = pc.read_csv(p, parse_options=PO, convert_options=pc.ConvertOptions(column_types={c: pa.string() for c in COLS}, include_columns=COLS))
    if keep is not None: t = t.filter(pcc.is_in(t["entity_id"], value_set=pa.array(list(keep))))
    return t.to_pandas().set_index("entity_id")
s1 = read(f"{T}/test_source1.tsv")
fr = set(s1.index[s1.country == "France"])
rows = []
for l in open("/home/sidvish/Downloads/s1-pair-tree/matching_results.tsv").read().split("\n")[1:]:
    if l:
        sid, ids = l.split("\t")
        if sid in fr and ids: rows.append((sid, ids.split(",")))
random.seed(1); samp = random.sample(rows, 4000)
need = {r for _, ids in samp for r in ids}
import pandas as pd; pool = pd.concat([read(f"{T}/test_source2.tsv", need), read(f"{T}/test_source3.tsv", need)])
st1, _ = parse_addresses(s1.business_address.reindex([s for s,_ in samp]).fillna("").to_numpy(), ["France"]*len(samp))
st1 = dict(zip([s for s,_ in samp], st1))
allr = [r for _, ids in samp for r in ids]
str_, _ = parse_addresses(pool.business_address.reindex(allr).fillna("").to_numpy(), ["France"]*len(allr)); str_ = dict(zip(allr, str_))
mixed = pure = 0; ex = []
for sid, ids in samp:
    a = st1[sid]
    dis = [r for r in ids if a and str_[r] and not (a & str_[r])]
    ok = [r for r in ids if a and str_[r] and (a & str_[r])]
    if dis:
        if ok: mixed += 1
        else: pure += 1
        ex.append((sid, ids, dis))
print(f"S1s with a disjoint link: {len(ex)}/{len(samp)}; list also has a street-matching link: {mixed}; all links disjoint: {pure}")
print("list-length dist of S1s with disjoint:", collections.Counter(min(len(i),6) for _,i,_ in ex))
for sid, ids, dis in random.sample(ex, 30):
    print("\nS1 :", s1.business_name[sid], "|", s1.business_address[sid], " street=", sorted(st1[sid]))
    for r in ids:
        tag = "DIS" if r in dis else "   "
        print(f"  {tag} {pool.business_name[r]} | {pool.business_address[r]}  street={sorted(str_[r])}")
