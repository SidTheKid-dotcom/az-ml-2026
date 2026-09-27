"""Name variants at the same address: record and S1 share the first house number and street tokens, core names differ
but are similar (rapidfuzz token_sort_ratio). Only records with exactly one such S1 (sim >= 70). Legal form not swapped.
train: truth per band (US/India). test: link state per band + candidate file for the given country."""
import collections, re, sys
sys.path.insert(0, ".")
from rapidfuzz import fuzz
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
        s1[e] = (n, parts, c)
        if parts: first_tok.update(t for t in fold(parts[0]).split() if t.isalpha())
stop = {t for t, _ in first_tok.most_common(60)}
def stt(parts): return {t for t in fold(parts[0]).split() if t.isalpha() and len(t) > 2 and t not in stop} if parts else set()
idx = collections.defaultdict(list); info = {}
for e, (n, parts, c) in s1.items():
    h = fnum(",".join(parts)); st_ = stt(parts)
    if not h or not st_: continue
    info[e] = (" ".join(core(n)), st_, legal(n))
    for t in st_: idx[(c, h, t)].append(e)
owner_of = {}; L = {}; own = {}
if split == "train":
    for s, recs in rows(f"{D}/train/train_ground_truth.tsv"):
        for r in recs.split(","):
            if r: owner_of[r] = s
else:
    L = {s: set(x for x in r.split(",") if x) for s, r in rows(SUB) if s in s1}
    own = {r: s for s, v in L.items() for r in v}
def band(x):
    for b in (95, 90, 85, 80, 70):
        if x >= b: return f">={b}"
    return None
st = collections.Counter(); out = open(OUT, "w") if OUT else None
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/{split}/{split}_{src}.tsv"):
        if c not in countries or not (addr or "").strip(): continue
        h = fnum(addr)
        if not h: continue
        toks = set(fold(addr).split())
        cands = set()
        for t in toks:
            cands.update(idx.get((c, h, t), ()))
        if not cands: continue
        k2 = " ".join(core(name)); lg = legal(name)
        good = []
        for e in cands:
            k1, st_, lg1 = info[e]
            if len(st_ & toks) / len(st_) < 0.5 or (lg1 and lg and lg1 != lg): continue
            sim = 100.0 if k1 == k2 else fuzz.token_sort_ratio(k1, k2)
            if sim >= 70: good.append((sim, e, k1 == k2))
        if len(good) != 1 or good[0][2]: continue
        sim, e, _ = good[0]; b = band(sim); key = (c, b)
        st[key + ("n",)] += 1
        if split == "train":
            st[key + ("true",)] += owner_of.get(eid) == e
        else:
            o = own.get(eid)
            if o == e: st[key + ("linked",)] += 1
            elif o: st[key + ("owned_elsewhere",)] += 1
            else:
                st[key + ("unowned",)] += 1
                if out: out.write(f"{e}\t{eid}\t{sim:.1f}\n")
for key in sorted({k[:-1] for k in st}):
    n = st[key + ("n",)]
    print(*key, n, {k[-1]: f"{v} ({v / n:.3f})" for k, v in st.items() if k[:-1] == key and k[-1] != "n"}, flush=True)
