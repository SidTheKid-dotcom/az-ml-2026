"""Apply the K3 French veto to any matching/rows file (label-free).

Removes French links whose K3 edit-operation probability is below ``--veto`` (default 0.05), using the
per-candidate scores in ``france_k3_scores.parquet`` (``ber.k3_france_mix --export``). Links without a K3
score (pairs outside the dense test candidates) are kept and counted. Non-French rows are copied byte for
byte. Reports the removed links' operation rates (delta shift, decoy word, legal swap) and probe overlap.

    python -m ber.k3_apply_veto IN.tsv OUT.tsv --scores france_k3_scores.parquet [--veto 0.05]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--scores", required=True)
    p.add_argument("--veto", type=float, default=0.05)
    a = p.parse_args()
    t = pq.read_table(a.scores).to_pandas()
    t = t[t["k3"] < a.veto]
    bad = {(s, r): i for i, (s, r) in enumerate(zip(t["s1_id"], t["record_id"]))}
    french_s1 = set(pq.read_table(a.scores, columns=["s1_id"]).column("s1_id").to_pylist())
    removed = []
    with open(a.inp, encoding="utf-8") as fin, open(a.out + ".tmp", "w", encoding="utf-8", newline="\n") as fout:
        fout.write(next(fin))
        for line in fin:
            s1, ids = line.rstrip("\n").split("\t")
            if s1 in french_s1 and ids:
                keep = []
                for r in ids.split(","):
                    i = bad.get((s1, r))
                    if i is None:
                        keep.append(r)
                    else:
                        removed.append(i)
                line = f"{s1}\t{','.join(keep)}\n"
            fout.write(line)
    Path(a.out + ".tmp").replace(a.out)
    rem = t.iloc[removed]
    h = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    report = {"input": a.inp, "output": a.out, "sha256": h, "veto": a.veto, "removed_links": int(len(rem)),
              "removed_rates": {"hn_delta_shift": float((rem["hn_in_delta"] == 1).mean()) if len(rem) else None,
                                "decoy_word_added": float((rem["decoy_added"] > 0).mean()) if len(rem) else None,
                                "legal_swapped": float((rem["legal_rel"] == 5).mean()) if len(rem) else None,
                                "record_empty_address": float((rem["rec_addr_empty"] == 1).mean()) if len(rem) else None},
              "removed_probe_links": {"T2r": int(rem["t2r_probe"].sum()), "Cprime": int(rem["cprime_probe"].sum())},
              "removed_in_s1_file": int(rem["in_s1_file"].sum())}
    Path(a.out + ".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
