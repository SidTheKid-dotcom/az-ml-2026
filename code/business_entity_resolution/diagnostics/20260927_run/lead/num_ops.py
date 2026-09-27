"""Train: number edit operations on TRUE exact-core-name pairs whose first house number differs, and example rows."""
import collections, csv, random, re, sys
sys.path.insert(0, ".")
from sig_rates import rows, core, num, fold, D

def op(a, b):  # a = S1 number string, b = record number string
    if b == a: return "same"
    if a.startswith(b): return "rec_prefix_of_s1"
    if b.startswith(a): return "s1_prefix_of_rec"
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1: return "one_digit_sub"
        if len(d) == 2 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]] and d[1] == d[0] + 1: return "transpose"
        return "same_len_other"
    if len(b) == len(a) - 1 and any(a[:i] + a[i+1:] == b for i in range(len(a))): return "digit_deleted"
    if len(b) == len(a) + 1 and any(b[:i] + b[i+1:] == a for i in range(len(b))): return "digit_inserted"
    return "other"

if __name__ == "__main__":
    random.seed(0)
    gt = {}
    for s1, recs in rows(f"{D}/train/train_ground_truth.tsv"):
        if random.random() < 0.15:
            gt[s1] = [r for r in recs.split(",") if r]
    s1 = {}
    for eid, name, addr, c in rows(f"{D}/train/train_source1.tsv"):
        if eid in gt: s1[eid] = (name, addr, c)
    need = {r for v in gt.values() for r in v}
    rec = {}
    for src in ("source2", "source3"):
        for eid, name, addr, c in rows(f"{D}/train/train_{src}.tsv"):
            if eid in need: rec[eid] = (name, addr, c)
    cnt = collections.Counter(); ex = collections.defaultdict(list); nS1 = collections.Counter()
    for s, L in gt.items():
        if s not in s1: continue
        n1, a1, c = s1[s]; nS1[c] += 1
        m1 = re.findall(r"\d+", fold(a1))
        for r in L:
            if r not in rec: continue
            n2, a2, _ = rec[r]
            if core(n2) != core(n1): continue
            m2 = re.findall(r"\d+", fold(a2))
            if not m1 or not m2: cnt[(c, "no_number")] += 1; continue
            o = op(m1[0], m2[0]); cnt[(c, o)] += 1
            if o != "same" and len(ex[o]) < 6: ex[o].append((c, a1, a2))
    print("sampled S1s", dict(nS1))
    for k, v in sorted(cnt.items()): print(*k, v, f"per_S1={v / nS1[k[0]]:.4f}")
    for o, L in ex.items():
        print("==", o)
        for c, a1, a2 in L: print("  ", c, "|", a1[:60], "||", a2[:60])
