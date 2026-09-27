"""France: add unowned records whose exact core name + first house number match exactly one S1 (street match or not;
train truth 0.944-0.999), or match 2+ S1s with exactly one street-matching S1 (train truth 0.947-0.996).
Legal form not swapped; record has an address. Cap 11, S2 <= 5, S3 <= 6. Writes FINAL_C3."""
import collections, hashlib, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, legal, fold, D
SRC = "/home/sidvish/Downloads/u4-final/FINAL_C2_usmixFC_nat_frC.tsv"
OUT = "/home/sidvish/Downloads/u4-final/FINAL_C3_usmixFC_nat_frCexact.tsv"
def fnum(a):
    m = re.findall(r"\d+", fold(a)); return m[0] if m else None
s1 = {}; first_tok = collections.Counter()
for e, n, a, c in rows(f"{D}/test/test_source1.tsv"):
    if c == "France":
        parts = [p for p in (a or "").split(",") if p.strip()]
        s1[e] = (n, parts)
        if parts: first_tok.update(t for t in fold(parts[0]).split() if t.isalpha())
stop = {t for t, _ in first_tok.most_common(60)}
def stt(parts): return {t for t in fold(parts[0]).split() if t.isalpha() and len(t) > 2 and t not in stop} if parts else set()
idx = collections.defaultdict(list)
for e, (n, parts) in s1.items():
    k = core(n); h = fnum(",".join(parts))
    if k and h: idx[(k, h)].append((e, stt(parts), legal(n)))
order, L = [], {}
with open(SRC) as f:
    header = next(f)
    for l in f:
        a, b = l.rstrip("\n").split("\t"); order.append(a); L[a] = [x for x in b.split(",") if x]
owner = {r: s for s, v in L.items() for r in v}
adds = []
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/test/test_{src}.tsv"):
        if c != "France" or not (addr or "").strip() or eid in owner: continue
        k = core(name); h = fnum(addr)
        cands = idx.get((k, h)) if k and h else None
        if not cands: continue
        lg = legal(name); toks = set(fold(addr).split())
        cands = [(e, s_t) for e, s_t, lg1 in cands if not (lg1 and lg and lg1 != lg)]
        if not cands: continue
        match = [e for e, s_t in cands if s_t and len(s_t & toks) / len(s_t) >= 0.5]
        if len(cands) == 1: adds.append((cands[0][0], eid, "1cand_" + ("street" if match else "nostreet")))
        elif len(match) == 1: adds.append((match[0], eid, "2cand_street"))
st = collections.Counter()
for s, r, kind in adds:
    lst = L[s]
    if len(lst) >= 11: st["skip_full"] += 1; continue
    n_src = sum(1 for x in lst if x[:2] == r[:2])
    if (r[:2] == "S2" and n_src >= 5) or (r[:2] == "S3" and n_src >= 6): st["skip_srccap"] += 1; continue
    lst.append(r); owner[r] = s; st["added_" + kind] += 1; st["to_empty_S1"] += len(lst) == 1
print(dict(st))
with open(OUT, "w", newline="\n") as f:
    f.write(header)
    for s in order: f.write(f"{s}\t{','.join(L[s])}\n")
print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())
