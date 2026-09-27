"""Per-country link mix of a submission: links per S1 by link category (label-free)."""
import collections, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, fold, num, D
SUB = sys.argv[1]
s1 = {}
for e, n, a, c in rows(f"{D}/test/test_source1.tsv"): s1[e] = (n, a, c)
L = {s: [x for x in r.split(",") if x] for s, r in rows(SUB)}
need = {x for v in L.values() for x in v}
rec = {}
for src in ("source2", "source3"):
    for e, n, a, c in rows(f"{D}/test/test_{src}.tsv"):
        if e in need: rec[e] = (n, a, src)
def cat(n1, a1, n2, a2):
    if not (a2 or "").strip() or a2.strip() in ("<NULL>",): addr = "EA"
    else: addr = "addr"
    c1, c2 = set(core(n1)), set(core(n2))
    raw2 = (n2 or "").lower()
    if re.search(r"\.(com|fr|in|net|org|co)\b|www", raw2): nm = "domain"
    elif c1 and c1 == c2: nm = "exact"
    elif c1 and c2 and len(c1 & c2) / len(c1 | c2) >= 0.5: nm = "similar"
    elif c1 and c2 and not (c1 & c2): nm = "disjoint"
    else: nm = "weak"
    return addr + ":" + nm
cnt = collections.Counter(); nS = collections.Counter(); dist = collections.Counter(); srcc = collections.Counter()
for s, (n1, a1, c) in s1.items():
    lst = L.get(s, [])
    nS[c] += 1; dist[(c, min(len(lst), 7))] += 1
    for r in lst:
        n2, a2, src = rec[r]; cnt[(c, cat(n1, a1, n2, a2))] += 1; srcc[(c, src)] += 1
cats = sorted({k[1] for k in cnt})
print("category".ljust(16), *[c.rjust(8) for c in ("US", "India", "France")])
for k in cats: print(k.ljust(16), *[f"{cnt[(c, k)] / nS[c]:8.4f}" for c in ("US", "India", "France")])
print("list-size share")
for m in range(8): print(str(m).ljust(16), *[f"{dist[(c, m)] / nS[c]:8.4f}" for c in ("US", "India", "France")])
for s in ("source2", "source3"): print(s.ljust(16), *[f"{srcc[(c, s)] / nS[c]:8.4f}" for c in ("US", "India", "France")])
