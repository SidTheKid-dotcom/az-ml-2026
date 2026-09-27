"""Truth rates of name/number signature buckets on train (US/India), and the same buckets in FINAL_C's French links.

Buckets use pairs whose record core name (folded, legal forms stripped) equals the S1 core name, or equals it plus
one added word. Number relation = record first house number minus S1 first house number.
"""
import collections
import csv
import re
import sys
import unicodedata

csv.field_size_limit(10**9)
D = "/home/sidvish/Documents/Work/Personal/az-ml-new/data/raw/dataset"
FINAL_C = "/home/sidvish/Downloads/u3-final-candidates/FINAL_C_usmixFC_frB_swapEA.tsv"
LEGAL = set("sarl sas sasu sa sci eurl ei snc llc inc corp co ltd pvt private limited llp lp pllc pc".split())
DELTA = {1, 3, 5, 7, 9, 11, 13, 21}


def fold(s):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]+", " ", s)


def core(n):
    return tuple(t for t in fold(n).split() if t not in LEGAL)


def legal(n):
    return frozenset(t for t in fold(n).split() if t in LEGAL)


def num(a):
    m = re.findall(r"\d+", fold(a))
    return int(m[0]) if m else None


def rel(k1, k2):
    if k1 is None or k2 is None:
        return "no_number"
    d = k2 - k1
    if d == 0:
        return "same"
    a, b = str(k1), str(k2)
    if a.startswith(b) or b.startswith(a):
        return "truncated"
    if d in DELTA:
        return "plus_delta_set"
    if 0 < d <= 12 and d % 2 == 0:
        return "plus_even_2_12"
    if -21 <= d < 0:
        return "minus_1_21"
    return "other_diff"


def rows(path):
    with open(path, newline="") as f:
        r = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(r)
        yield from r


def measure(split, countries, owner_of=None, links=None, count_all=False):
    """owner_of: record -> true S1 (train). links: S1 -> set(records) (test submission)."""
    s1_core = collections.defaultdict(list)
    s1_info = {}
    for eid, name, addr, c in rows(f"{D}/{split}/{split}_source1.tsv"):
        if c not in countries:
            continue
        k = core(name)
        if k:
            s1_core[(c, k)].append(eid)
            s1_info[eid] = (num(addr), legal(name))
    stats = collections.Counter()
    added_tok = collections.Counter()
    for src in ("source2", "source3"):
        for eid, name, addr, c in rows(f"{D}/{split}/{split}_{src}.tsv"):
            if c not in countries:
                continue
            k = core(name)
            if not k:
                continue
            k2 = num(addr)
            lg = legal(name)
            cands = [(s, "exact") for s in s1_core.get((c, k), ())]
            if len(k) >= 2:
                for i, t in enumerate(k):
                    for s in s1_core.get((c, k[:i] + k[i + 1:]), ()):
                        cands.append((s, "added:" + t))
            for s, kind in cands:
                k1, lg1 = s1_info[s]
                r = rel(k1, k2)
                lgs = "legal_swap" if (lg1 and lg and lg1 != lg) else "legal_ok"
                bucket = (c, "exact" if kind == "exact" else "added_word", r, lgs)
                if owner_of is not None:
                    y = owner_of.get(eid) == s
                    stats[bucket + ("n",)] += 1
                    stats[bucket + ("true",)] += y
                    if kind != "exact" and r in ("same", "plus_delta_set"):
                        added_tok[(kind[6:], r, "n")] += 1
                        added_tok[(kind[6:], r, "true")] += y
                else:
                    if count_all:
                        stats[bucket + ("n",)] += 1
                    if eid in links.get(s, ()):
                        stats[bucket + ("linked",)] += 1
    return stats, added_tok


def main():
    mode = sys.argv[1]
    if mode == "train":
        owner_of = {}
        for s1, recs in rows(f"{D}/train/train_ground_truth.tsv"):
            for r in recs.split(","):
                if r:
                    owner_of[r] = s1
        stats, added = measure("train", {"US", "India"}, owner_of=owner_of)
        keys = sorted({k[:-1] for k in stats})
        print("country bucket number_rel legal n true rate")
        for k in keys:
            n = stats[k + ("n",)]
            if n >= 200:
                print(*k, n, stats[k + ("true",)], f"{stats[k + ('true',)] / n:.3f}")
        print("\nadded word (>=300 pairs): token rel n rate")
        for (t, r, what), n in sorted(added.items(), key=lambda x: -x[1]):
            if what == "n" and n >= 300:
                print(t, r, n, f"{added[(t, r, 'true')] / n:.3f}")
    else:
        links = {}
        for s1, recs in rows(FINAL_C):
            links[s1] = set(x for x in recs.split(",") if x)
        stats, _ = measure("test", {"France", "US", "India"}, links=links, count_all=True)
        print("country bucket number_rel legal pool_pairs linked accept_rate")
        for k in sorted({k[:-1] for k in stats}):
            n = stats[k + ("n",)]
            print(*k, n, stats[k + ("linked",)], f"{stats[k + ('linked',)] / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
