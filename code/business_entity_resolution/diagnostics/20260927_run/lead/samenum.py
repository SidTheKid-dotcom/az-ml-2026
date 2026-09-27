"""Records whose exact core name + first house number match >= 1 S1 (legal form not swapped).
Bucket by number of such S1s (1 / 2+) and street match with the best one.
train (US/India): truth = record's owner is that S1. test: FINAL_C link state."""
import collections, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, legal, fold, D
split, countries = sys.argv[1], set(sys.argv[2].split(","))
SUB = sys.argv[3] if len(sys.argv) > 3 else None
def fnum(a):
    m = re.findall(r"\d+", fold(a)); return m[0] if m else None
s1 = {}; first_tok = collections.Counter()
for e, n, a, c in rows(f"{D}/{split}/{split}_source1.tsv"):
    if c in countries:
        parts = [p for p in (a or "").split(",") if p.strip()]
        s1[e] = (n, a, parts, c)
        if parts: first_tok.update(t for t in fold(parts[0]).split() if t.isalpha())
stop = {t for t, _ in first_tok.most_common(60)}
def stt(parts): return {t for t in fold(parts[0]).split() if t.isalpha() and len(t) > 2 and t not in stop} if parts else set()
idx = collections.defaultdict(list)
for e, (n, a, parts, c) in s1.items():
    k = core(n); h = fnum(a)
    if k and h: idx[(c, k, h)].append((e, stt(parts), legal(n)))
owner_of = {}; L = {}; own = {}
if split == "train":
    for s, recs in rows(f"{D}/train/train_ground_truth.tsv"):
        for r in recs.split(","):
            if r: owner_of[r] = s
else:
    L = {s: set(x for x in r.split(",") if x) for s, r in rows(SUB) if s in s1}
    own = {r: s for s, v in L.items() for r in v}
st = collections.Counter()
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/{split}/{split}_{src}.tsv"):
        if c not in countries or not (addr or "").strip(): continue
        k = core(name); h = fnum(addr)
        if not k or not h: continue
        cands = idx.get((c, k, h))
        if not cands: continue
        lg = legal(name); toks = set(fold(addr).split())
        cands = [(e, s_t) for e, s_t, lg1 in cands if not (lg1 and lg and lg1 != lg)]
        if not cands: continue
        match = [e for e, s_t in cands if s_t and len(s_t & toks) / len(s_t) >= 0.5]
        nc = "1cand" if len(cands) == 1 else "2+cand"
        sm = "street_match" if match else "no_street_match"
        tgt = match[0] if len(match) == 1 else (cands[0][0] if len(cands) == 1 else None)
        key = (c, nc, sm)
        st[key + ("n",)] += 1
        if split == "train":
            st[key + ("true_to_some_cand",)] += owner_of.get(eid) in {e for e, _ in cands}
            if tgt: st[key + ("true_to_target",)] += owner_of.get(eid) == tgt
        else:
            if eid in own and own[eid] in {e for e, _ in cands}: st[key + ("linked_to_a_cand",)] += 1
            elif eid in own: st[key + ("owned_elsewhere",)] += 1
            else: st[key + ("unowned",)] += 1
for key in sorted({k[:-1] for k in st}):
    n = st[key + ("n",)]
    extra = {k[-1]: v for k, v in st.items() if k[:-1] == key and k[-1] != "n"}
    print(*key, n, {k: f"{v} ({v / n:.3f})" for k, v in extra.items()})
