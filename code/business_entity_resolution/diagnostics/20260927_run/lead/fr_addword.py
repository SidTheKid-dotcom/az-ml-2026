"""France test: record core = S1 core + one added token. Per token: pool pairs at same number vs +delta number
(label-free shift share), and FINAL_C links at the same number."""
import collections, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, num, D, FINAL_C
DELTA = {1, 3, 5, 7, 9, 11, 13, 21}
C = {s: set(x for x in r.split(",") if x) for s, r in rows(FINAL_C)}
idx = collections.defaultdict(list); info = {}
for e, n, a, c in rows(f"{D}/test/test_source1.tsv"):
    if c == "France":
        k = core(n)
        if k: idx[k].append(e); info[e] = num(a)
st = collections.Counter(); links = []
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/test/test_{src}.tsv"):
        if c != "France": continue
        k = core(name)
        if len(k) < 2: continue
        k2 = num(addr)
        for i, t in enumerate(k):
            for s in idx.get(k[:i] + k[i + 1:], ()):
                k1 = info[s]
                if k1 is None or k2 is None: rel = "nonum"
                elif k1 == k2: rel = "same"
                elif (k2 - k1) in DELTA: rel = "delta"
                else: rel = "other"
                st[(t, rel)] += 1
                if eid in C.get(s, ()):
                    st[(t, rel + "_linked")] += 1
                    links.append((s, eid, t, rel))
toks = {t for t, _ in st}
res = []
for t in toks:
    ns, nd = st[(t, "same")], st[(t, "delta")]
    if ns + nd >= 30:
        res.append((st[(t, "same_linked")] + st[(t, "nonum_linked")] + st[(t, "other_linked")] + st[(t, "delta_linked")], t, ns, nd, nd / (ns + nd)))
res.sort(reverse=True)
print("token linked_total same_pairs delta_pairs shift_share same_linked")
for L, t, ns, nd, sh in res[:60]:
    print(t, L, ns, nd, f"{sh:.2f}", st[(t, "same_linked")])
with open("fr_addword_links.tsv", "w") as f:
    for s, e, t, rel in links: f.write(f"{s}\t{e}\t{t}\t{rel}\n")
