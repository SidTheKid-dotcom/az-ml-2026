"""FINAL_A2 = FINAL_A with FINAL_C's broad French same-number one-word-swap removal applied to French rows."""
import collections, difflib, hashlib, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, fold, D
LEGAL = set("sarl sas sasu sa sci eurl ei snc".split())
SUFFIX = set("cie services associes fils et france".split())
def ctoks(n): return [t for t in fold(n).split() if t not in LEGAL]
def fnum(a):
    m = re.findall(r"\d+", fold(a)); return m[0] if m else None
def is_abbrev(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    it = iter(l); return len(s) <= 4 and s[0] == l[0] and all(ch in it for ch in s)
fr = {}; vocab = collections.Counter()
for e, n, a, c in rows(f"{D}/test/test_source1.tsv"):
    if c == "France": fr[e] = (n, a); vocab.update(set(ctoks(n)))
SRC = "/home/sidvish/Downloads/u3-final-candidates/FINAL_A_usmixFC_frK5t075.tsv"
OUT = "/home/sidvish/Downloads/u3-final-candidates/FINAL_A2_usmixFC_frK5t075_swap.tsv"
L = {s: [x for x in r.split(",") if x] for s, r in rows(SRC)}
order = [s for s, _ in rows(SRC)]
need = {x for s in fr for x in L[s]}; rec = {}
for src in ("source2", "source3"):
    for e, n, a, c in rows(f"{D}/test/test_{src}.tsv"):
        if e in need: rec[e] = (n, a)
removed = 0
for s in fr:
    n1, a1 = fr[s]; k1 = ctoks(n1); h1 = fnum(a1); keep = []
    for r in L[s]:
        n2, a2 = rec[r]; drop = False
        if (a2 or "").strip() and h1 is not None and fnum(a2) == h1:
            k2 = ctoks(n2)
            if len(k1) == len(k2):
                d = [(x, y) for x, y in zip(k1, k2) if x != y]
                if len(d) == 1:
                    x, y = d[0]
                    if (x not in SUFFIX and y not in SUFFIX and not is_abbrev(x, y)
                            and difflib.SequenceMatcher(None, x, y).ratio() < 0.6 and vocab[x] >= 20 and vocab[y] >= 20):
                        drop = True
        if drop: removed += 1
        else: keep.append(r)
    L[s] = keep
with open(OUT, "w", newline="\n") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for s in order: f.write(f"{s}\t{','.join(L[s])}\n")
print("removed", removed, "french links now", sum(len(L[s]) for s in fr), "per S1", round(sum(len(L[s]) for s in fr) / len(fr), 4))
print("sha256", hashlib.sha256(open(OUT, "rb").read()).hexdigest())
