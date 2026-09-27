"""fr_swap: remove French same-address one-word-substitution links (another team's leaderboard-proven rule).

A French link is removed when the S1 and the record share the house number (first digit group) and a
near-identical street (rapidfuzz ratio >= 85 on the digit-free, accent-folded address), and their names (legal
forms stripped, accent-folded) have the same token count with exactly one differing token on each side, where both
tokens are frequent French S1 words (in >= 20 French S1 names), are not a typo (difflib ratio < 0.6), not an
abbreviation (shorter <= 4 chars, same first letter, in-order subsequence) and not copy suffixes
(cie/services/associes/fils/et/france). Non-French rows are copied byte for byte.

    python -m ber.k_frswap IN.tsv OUT.tsv
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq
from rapidfuzz import fuzz

from ber.edit_ops import LEGAL_CLASS, clean, fold, tokens

DATA = Path("/home/ubuntu/k/data")
COPY = {"cie", "services", "service", "associes", "fils", "et", "france"}
_DIG = re.compile(r"[0-9]+")


def name_tokens(raw):
    return [t for t in tokens(fold(clean(raw))) if t not in LEGAL_CLASS]


def addr_parts(raw):
    f = fold(clean(raw))
    groups = _DIG.findall(f)
    return (groups[0] if groups else None), " ".join(t for t in tokens(f) if not t.isdigit() and not any(c.isdigit() for c in t))


def is_abbrev(a, b):
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if len(s) > 4 or not s or s[0] != l[0]:
        return False
    it = iter(l)
    return all(c in it for c in s)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inp")
    p.add_argument("out")
    p.add_argument("--min-freq", type=int, default=20)
    a = p.parse_args()
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "business_name", "business_address",
                                                                  "country"]).to_pandas()
    fr = ref[ref.country == "France"]
    s1 = {e: (n, ad) for e, n, ad in zip(fr.entity_id, fr.business_name, fr.business_address)}
    vocab = Counter()
    for n in fr.business_name:
        vocab.update(set(name_tokens(n)))
    pool = pq.read_table(DATA / "test/pool.parquet", columns=["entity_id", "business_name", "business_address"]).to_pandas()
    extra = pq.read_table(DATA / "u8src/test_pool.parquet", columns=["entity_id", "business_name", "business_address"]).to_pandas()
    recs = {}
    removed, emptied, swaps = 0, 0, Counter()
    needed = set()
    with open(a.inp, encoding="utf-8") as f:
        next(f)
        for line in f:
            e, ids = line.rstrip("\n").split("\t")
            if e in s1 and ids:
                needed.update(ids.split(","))
    for df in (pool, extra):
        m = df.entity_id.isin(needed)
        for e, n, ad in zip(df.entity_id[m], df.business_name[m], df.business_address[m]):
            recs.setdefault(e, (n, ad))
    with open(a.inp, encoding="utf-8") as fin, open(a.out + ".tmp", "w", encoding="utf-8", newline="\n") as fout:
        fout.write(next(fin))
        for line in fin:
            e, ids = line.rstrip("\n").split("\t")
            if e not in s1 or not ids:
                fout.write(line)
                continue
            sn, sa = s1[e]
            s_hn, s_st = addr_parts(sa)
            s_tok = name_tokens(sn)
            keep = []
            for r in ids.split(","):
                rn, ra_ = recs.get(r, ("", ""))
                drop = False
                r_hn, r_st = addr_parts(ra_)
                if s_hn is not None and s_hn == r_hn and s_st and r_st and fuzz.ratio(s_st, r_st) >= 85:
                    r_tok = name_tokens(rn)
                    if len(r_tok) == len(s_tok):
                        cs, cr = Counter(s_tok), Counter(r_tok)
                        ds, dr = list((cs - cr).elements()), list((cr - cs).elements())
                        if len(ds) == 1 and len(dr) == 1:
                            x, y = ds[0], dr[0]
                            if (vocab[x] >= a.min_freq and vocab[y] >= a.min_freq and x not in COPY and y not in COPY
                                    and difflib.SequenceMatcher(None, x, y).ratio() < 0.6 and not is_abbrev(x, y)):
                                drop = True
                                swaps[tuple(sorted((x, y)))] += 1
                if drop:
                    removed += 1
                else:
                    keep.append(r)
            if not keep:
                emptied += 1
            fout.write(f"{e}\t{','.join(keep)}\n")
    Path(a.out + ".tmp").replace(a.out)
    rep = {"input": a.inp, "output": a.out, "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
           "removed": removed, "rows_emptied_now_empty": emptied,
           "top_swaps": [[list(k), v] for k, v in swaps.most_common(25)]}
    Path(a.out + ".json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("sha256", "removed", "rows_emptied_now_empty")}), rep["top_swaps"][:6])


if __name__ == "__main__":
    main()
