"""Records whose exact core name + first house number match 2+ S1s (legal form not swapped) and whose street matches
exactly one of them. Is the record linked to that S1 (right), to another candidate (wrong sibling), or unlinked?
train mode gives the truth rate of 'owner == street-matching S1'."""
import collections, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, legal, fold, D
split, countries = sys.argv[1], set(sys.argv[2].split(","))
SUB = sys.argv[3] if len(sys.argv) > 3 else None
OUT = sys.argv[4] if len(sys.argv) > 4 else None
def fnum(a):
    m = re.findall(r"\d+", fold(a)); return m[0] if m else None
s1 = {}; first_tok = collections.Counter()
for e, n, a, c in rows(f"{D}/{split}/{split}_source1.tsv"):
    if c in countries:
        parts = [p for p in (a or "").split(",") if p.strip()]
        s1[e] = (n, parts)
        if parts: first_tok.update(t for t in fold(parts[0]).split() if t.isalpha())
stop = {t for t, _ in first_tok.most_common(60)}
def stt(parts): return {t for t in fold(parts[0]).split() if t.isalpha() and len(t) > 2 and t not in stop} if parts else set()
idx = collections.defaultdict(list)
for e, (n, parts) in s1.items():
    k = core(n); h = fnum(",".join(parts))
    if k and h: idx[(k, h)].append((e, stt(parts), legal(n)))
owner_of = {}; L = {}; own = {}
if split == "train":
    for s, recs in rows(f"{D}/train/train_ground_truth.tsv"):
        for r in recs.split(","):
            if r: owner_of[r] = s
else:
    L = {s: [x for x in r.split(",") if x] for s, r in rows(SUB) if s in s1}
    own = {r: s for s, v in L.items() for r in v}
st = collections.Counter(); fixes = []
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/{split}/{split}_{src}.tsv"):
        if c not in countries or not (addr or "").strip(): continue
        k = core(name); h = fnum(addr)
        cands = idx.get((k, h)) if k and h else None
        if not cands or len(cands) < 2: continue
        lg = legal(name); toks = set(fold(addr).split())
        cands = [(e, s_t) for e, s_t, lg1 in cands if not (lg1 and lg and lg1 != lg)]
        if len(cands) < 2: continue
        match = [e for e, s_t in cands if s_t and len(s_t & toks) / len(s_t) >= 0.5]
        if len(match) != 1:
            st["street matches 0 or 2+ candidates"] += 1; continue
        tgt = match[0]; cset = {e for e, _ in cands}
        st["n"] += 1
        if split == "train":
            o = owner_of.get(eid)
            st["truth: owner = street-matching S1"] += o == tgt
            st["truth: owner = other candidate"] += (o in cset and o != tgt)
            st["truth: owner outside candidates"] += (o not in cset)
        else:
            o = own.get(eid)
            if o == tgt: st["linked to street-matching S1"] += 1
            elif o in cset:
                st["linked to OTHER sibling"] += 1; fixes.append((eid, o, tgt))
            elif o: st["linked outside candidates"] += 1
            else: st["unlinked"] += 1
for k, v in st.items(): print(f"{k:40s} {v}")
if OUT:
    with open(OUT, "w") as f:
        for eid, o, tgt in fixes: f.write(f"{eid}\t{o}\t{tgt}\n")
