"""Number-noise copy rule: record with the same core name as exactly one street-matching S1, a copy-style house-number
edit, and no same-name S1 at the record's exact number.

train: rule precision per country and edit class (labels). test: emit French rule pairs to fr_numfix_pairs.tsv.
"""
import collections
import re
import sys

sys.path.insert(0, "/tmp/claude-1000/-home-sidvish-Documents-Work-Personal-az-ml-new/41da6ea4-f15c-42cd-93af-ae21d9c57893/scratchpad")
from num_ops import op
from sig_rates import D, core, fold, rows

DELTA = {1, 3, 5, 7, 9, 11, 13, 21}
TOP_STOP = 200


def words(a):
    return [t for t in fold(a).split() if t.isalpha() and len(t) > 2]


def first_num(a):
    m = re.findall(r"\d+", fold(a))
    return m[0] if m else None


def cls(n1, n2):
    o = op(n1, n2)
    if o in ("one_digit_sub", "same_len_other", "other", "transpose") and (int(n2) - int(n1)) in DELTA:
        return o + "|delta"
    return o


def run(split, countries, owner_of=None, out=None):
    s1_rows = [(e, n, a, c) for e, n, a, c in rows(f"{D}/{split}/{split}_source1.tsv") if c in countries]
    df = {c: collections.Counter() for c in countries}
    for e, n, a, c in s1_rows:
        df[c].update(set(words(a)))
    stop = {c: {t for t, _ in df[c].most_common(TOP_STOP)} for c in countries}
    idx = collections.defaultdict(list)
    for e, n, a, c in s1_rows:
        k = core(n)
        if k:
            idx[(c, k)].append((e, first_num(a), frozenset(t for t in words(a) if t not in stop[c])))
    del s1_rows
    st = collections.Counter()
    for src in ("source2", "source3"):
        for eid, name, addr, c in rows(f"{D}/{split}/{split}_{src}.tsv"):
            if c not in countries:
                continue
            k = core(name)
            S = idx.get((c, k)) if k else None
            if not S:
                continue
            n2 = first_num(addr)
            if n2 is None or any(n1 == n2 for _, n1, _ in S):
                continue
            key2 = frozenset(t for t in words(addr) if t not in stop[c])
            if not key2:
                continue
            M = [(s, n1) for s, n1, k1 in S if k1 and len(k1 & key2) / len(k1 | key2) >= 0.5]
            if len(M) != 1 or M[0][1] is None:
                continue
            s, n1 = M[0]
            cl = cls(n1, n2)
            shared = "uniq" if len(S) == 1 else "shared"
            st[(c, cl, shared, "n")] += 1
            if owner_of is not None:
                st[(c, cl, shared, "y")] += owner_of.get(eid) == s
            if out is not None:
                out.write(f"{s}\t{eid}\t{cl}\t{shared}\t{n1}\t{n2}\n")
    for key in sorted({k[:-1] for k in st}):
        n = st[key + ("n",)]
        line = [*key, n]
        if owner_of is not None:
            line += [st[key + ("y",)], f"{st[key + ('y',)] / n:.3f}"]
        print(*line, flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        owner_of = {}
        for s1, recs in rows(f"{D}/train/train_ground_truth.tsv"):
            for r in recs.split(","):
                if r:
                    owner_of[r] = s1
        run("train", {"US", "India"}, owner_of=owner_of)
    else:
        with open(sys.argv[2], "w") as out:
            run("test", {"France"}, out=out)
