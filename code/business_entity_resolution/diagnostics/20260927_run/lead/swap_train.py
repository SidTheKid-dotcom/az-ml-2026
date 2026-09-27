"""Train precision (US/India) of FINAL_C's French word-swap removal rule, transposed to US/India:
same first house number, equal core token count, exactly one differing token, not abbreviation, difflib ratio < 0.6,
both tokens in >= 20 S1 names. Split by whether either token is a data-derived copy word."""
import collections, difflib, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, fold, D
COPY = set("the corporation center centre services service incorporated company partners trust society association council "
           "federation district authority board commission com and of health medicine specialists physicians family".split())
def first_num(a):
    m = re.findall(r"\d+", fold(a)); return m[0] if m else None
def is_abbrev(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    it = iter(l); return len(s) <= 4 and s[0] == l[0] and all(ch in it for ch in s)
owner = {}
for s1, recs in rows(f"{D}/train/train_ground_truth.tsv"):
    for r in recs.split(","):
        if r: owner[r] = s1
vocab = {c: collections.Counter() for c in ("US", "India")}
S1 = []
for e, n, a, c in rows(f"{D}/train/train_source1.tsv"):
    if c in vocab:
        k = core(n)
        if len(k) >= 1:
            vocab[c].update(set(k)); S1.append((e, k, first_num(a), c))
idx = collections.defaultdict(list)
for e, k, n, c in S1:
    if n is None: continue
    for i in range(len(k)):
        idx[(c, n, k[:i] + ("*",) + k[i + 1:])].append((e, k[i]))
del S1
st = collections.Counter()
for src in ("source2", "source3"):
    for eid, name, addr, c in rows(f"{D}/train/train_{src}.tsv"):
        if c not in vocab or not (addr or "").strip(): continue
        n = first_num(addr)
        if n is None: continue
        k = core(name)
        for i in range(len(k)):
            for s, t1 in idx.get((c, n, k[:i] + ("*",) + k[i + 1:]), ()):
                t2 = k[i]
                if t1 == t2 or is_abbrev(t1, t2) or difflib.SequenceMatcher(None, t1, t2).ratio() >= 0.6: continue
                if vocab[c][t1] < 20 or vocab[c][t2] < 20: continue
                g = "copyword" if (t1 in COPY or t2 in COPY) else "plain"
                y = owner.get(eid) == s
                st[(c, g, "n")] += 1; st[(c, g, "y")] += y
for c in ("US", "India"):
    for g in ("plain", "copyword"):
        n = st[(c, g, "n")]; print(c, g, n, st[(c, g, "y")], f"{st[(c, g, 'y')] / max(n, 1):.3f}")
