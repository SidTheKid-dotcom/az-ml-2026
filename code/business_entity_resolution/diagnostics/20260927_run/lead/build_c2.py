"""FINAL_C2 = FINAL_C + U's accepted India native-script dictionary links (u8t), added only where the record is unowned
in FINAL_C, the S1 list has < 11 links and the per-source caps (S2 <= 5, S3 <= 6) hold. French rows untouched."""
import collections, hashlib, sys
import pyarrow.parquet as pq
SRC = "/home/sidvish/Downloads/u3-final-candidates/FINAL_C_usmixFC_frB_swapEA.tsv"
OUT = "/home/sidvish/Downloads/u4-final/FINAL_C2_usmixFC_nat_frC.tsv"
t = pq.read_table("u8t/nat_candidates_test_rank0.parquet", columns=["s1_id", "rec_id"]).to_pydict()
nat = set(zip(t["s1_id"], t["rec_id"]))
u8t = {}
with open("u8t/us_india_rows.tsv") as f:
    next(f)
    for l in f:
        a, b = l.rstrip("\n").split("\t"); u8t[a] = [x for x in b.split(",") if x]
acc = [(s, r) for s, L in u8t.items() for r in L if (s, r) in nat]
order, L = [], {}
with open(SRC) as f:
    header = next(f)
    for l in f:
        a, b = l.rstrip("\n").split("\t"); order.append(a); L[a] = [x for x in b.split(",") if x]
owner = {r: s for s, v in L.items() for r in v}
st = collections.Counter()
for s, r in sorted(acc, key=lambda x: x[0]):
    if r in owner: st["skip: record owned in FINAL_C"] += 1; continue
    lst = L[s]
    if len(lst) >= 11: st["skip: list full"] += 1; continue
    src = r[:2]; n_src = sum(1 for x in lst if x[:2] == src)
    if (src == "S2" and n_src >= 5) or (src == "S3" and n_src >= 6): st["skip: source cap"] += 1; continue
    lst.append(r); owner[r] = s; st["added"] += 1
    st["added to empty S1"] += len(lst) == 1
print("accepted native links in u8t:", len(acc), dict(st))
with open(OUT, "w", newline="\n") as f:
    f.write(header)
    for s in order: f.write(f"{s}\t{','.join(L[s])}\n")
print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())
