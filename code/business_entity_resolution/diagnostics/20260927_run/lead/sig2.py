"""Same-core-name pairs by number-edit class x street agreement x name uniqueness.
train: truth rate (US/India). test: pool pairs and FINAL_C links per country."""
import collections, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, fold, D, FINAL_C
from num_ops import op

DELTA = {1, 3, 5, 7, 9, 11, 13, 21}
def addr_parts(a):
    f = fold(a); m = re.findall(r"\d+", f)
    words = set(t for t in f.split() if t.isalpha() and len(t) > 2)
    return (m[0] if m else None), words

def cls(n1, n2):
    if n1 is None or n2 is None: return "no_number"
    o = op(n1, n2)
    if o == "one_digit_sub" or o == "same_len_other" or o == "other":
        d = int(n2) - int(n1)
        if d in DELTA: return o + "|delta"
    return o

def run(split, countries, owner_of=None, links=None):
    idx = collections.defaultdict(list); info = {}
    for eid, name, addr, c in rows(f"{D}/{split}/{split}_source1.tsv"):
        if c in countries:
            k = core(name)
            if k: idx[(c, k)].append(eid); info[eid] = addr_parts(addr)
    st = collections.Counter()
    for src in ("source2", "source3"):
        for eid, name, addr, c in rows(f"{D}/{split}/{split}_{src}.tsv"):
            if c not in countries: continue
            k = core(name)
            S = idx.get((c, k)) if k else None
            if not S: continue
            n2, w2 = addr_parts(addr)
            for s in S:
                n1, w1 = info[s]
                if n1 == n2 and n1 is not None: continue
                cl = cls(n1, n2)
                street = "street_eq" if (w1 and w2 and len(w1 & w2) / len(w1 | w2) >= 0.5) else "street_diff"
                uniq = "uniq_name" if len(S) == 1 else "shared_name"
                key = (c, cl, street, uniq)
                st[key + ("n",)] += 1
                if owner_of is not None: st[key + ("y",)] += owner_of.get(eid) == s
                if links is not None: st[key + ("y",)] += eid in links.get(s, ())
    for key in sorted({k[:-1] for k in st}):
        n = st[key + ("n",)]
        if n >= 100: print(*key, n, st[key + ("y",)], f"{st[key + ('y',)] / n:.3f}")

if __name__ == "__main__":
    if sys.argv[1] == "train":
        owner_of = {}
        for s1, recs in rows(f"{D}/train/train_ground_truth.tsv"):
            for r in recs.split(","):
                if r: owner_of[r] = s1
        run("train", {"US", "India"}, owner_of=owner_of)
    else:
        links = {s: set(x for x in r.split(",") if x) for s, r in rows(FINAL_C)}
        run("test", {"US", "India", "France"}, links=links)
