"""Merge rows files: a full base file with some S1 rows replaced by a partial rows file (same header); base order kept.

    python -m ber.k_merge BASE_FULL.tsv PARTIAL.tsv OUT.tsv
"""
import sys, hashlib
base, part, out = sys.argv[1:4]
rep = {}
with open(part, encoding="utf-8") as f:
    head = next(f)
    for line in f:
        rep[line.split("\t", 1)[0]] = line
n = 0
with open(base, encoding="utf-8") as fin, open(out, "w", encoding="utf-8", newline="\n") as fout:
    fout.write(next(fin))
    for line in fin:
        s1 = line.split("\t", 1)[0]
        if s1 in rep:
            line = rep.pop(s1); n += 1
        fout.write(line)
assert not rep, f"{len(rep)} rows of the partial file not in base"
print(out, "replaced", n, hashlib.sha256(open(out, "rb").read()).hexdigest())
