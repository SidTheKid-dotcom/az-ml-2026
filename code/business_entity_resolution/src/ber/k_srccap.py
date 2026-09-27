"""Per-S1 per-source cap (generator bound measured on arena A labels: at most 5 S2 and 6 S3 true records per S1).

Rows files list each S1's links in decreasing model probability (K writers); links beyond the cap for their
source (``S2-``/``S3-`` id prefix) are dropped from the end. Other rows are copied byte for byte.

    python -m ber.k_srccap IN.tsv OUT.tsv [--s2 5 --s3 6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--s2", type=int, default=5)
    p.add_argument("--s3", type=int, default=6)
    a = p.parse_args()
    dropped = rows_changed = 0
    with open(a.inp, encoding="utf-8") as fin, open(a.out + ".tmp", "w", encoding="utf-8", newline="\n") as fout:
        fout.write(next(fin))
        for line in fin:
            s1, ids = line.rstrip("\n").split("\t")
            if ids:
                items = ids.split(",")
                n2 = n3 = 0
                keep = []
                for r in items:
                    if r.startswith("S2"):
                        n2 += 1
                        if n2 > a.s2:
                            continue
                    elif r.startswith("S3"):
                        n3 += 1
                        if n3 > a.s3:
                            continue
                    keep.append(r)
                if len(keep) != len(items):
                    dropped += len(items) - len(keep)
                    rows_changed += 1
                    line = f"{s1}\t{','.join(keep)}\n"
            fout.write(line)
    Path(a.out + ".tmp").replace(a.out)
    rep = {"input": a.inp, "output": a.out, "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
           "cap": {"S2": a.s2, "S3": a.s3}, "links_dropped": dropped, "rows_changed": rows_changed}
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
