"""France: admin component of addresses (région / other / none), and exact-copy pairs left unlinked by a submission.
Exact copy = same core name (legal forms stripped, legal form not swapped), same first house number, S1 street tokens
contained in the record. Reports linked / unlinked by the record's admin type."""
import collections, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, legal, fold, D
SUB = sys.argv[1]; CTRY = sys.argv[2] if len(sys.argv) > 2 else "France"
s1 = {}; last = collections.Counter(); first_tok = collections.Counter()
for e, n, a, c in rows(f"{D}/test/test_source1.tsv"):
    if c != CTRY: continue
    parts = [p.strip() for p in (a or "").split(",") if p.strip()]
    s1[e] = (n, a, parts)
    if parts: last[fold(parts[-1]).strip()] += 1
    if parts: first_tok.update(t for t in fold(parts[0]).split() if t.isalpha())
regions = {r for r, k in last.most_common(40) if k >= 200}
street_stop = {t for t, _ in first_tok.most_common(40)}
print("S1 last components (top):", last.most_common(8))
def fnum(a):
    m = re.findall(r"\d+", fold(a)); return m[0] if m else None
def street_toks(parts):
    return {t for t in fold(parts[0]).split() if t.isalpha() and len(t) > 2 and t not in street_stop} if parts else set()
idx = collections.defaultdict(list)
for e, (n, a, parts) in s1.items():
    k = core(n); h = fnum(a)
    if k and h: idx[(k, h)].append((e, street_toks(parts), legal(n), fold(parts[-1]).strip() if parts else ""))
L = {s: set(x for x in r.split(",") if x) for s, r in rows(SUB) if s in s1}
owner = {r: s for s, v in L.items() for r in v}
st = collections.Counter(); rec_last = collections.Counter()
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/test/test_{src}.tsv"):
        if c != CTRY: continue
        comps = [fold(p).strip() for p in (addr or "").split(",") if p.strip()]
        admin = "region" if any(p in regions for p in comps) else ("none" if len(comps) <= 2 else "other_admin")
        rec_last[admin] += 1
        k = core(name); h = fnum(addr or "")
        if not k or not h: continue
        cands = idx.get((k, h))
        if not cands: continue
        toks = set(fold(addr).split()); lg = legal(name)
        M = [(e, reg) for e, stt, lg1, reg in cands if stt and len(stt & toks) / len(stt) >= 0.5 and not (lg1 and lg and lg1 != lg)]
        if len(M) != 1: st[("ambiguous" if M else "no_street_match", admin)] += 1; continue
        e, reg = M[0]
        if eid in L.get(e, ()): st[("linked", admin)] += 1
        elif eid in owner: st[("owned_by_other", admin)] += 1
        elif len(L.get(e, ())) >= 11: st[("list_full", admin)] += 1
        else: st[("UNLINKED_unowned", admin)] += 1
print("record admin types:", dict(rec_last))
for k in sorted(st): print(k, st[k])
