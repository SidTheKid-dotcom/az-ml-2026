"""France cross-family vote (label-free): keep a French link when >= 2 families accept it; 1-vote (disputed)
links fall back to the scored S1 file's #5 rule (kept iff in the S1 file); 0-vote links are dropped (or kept if
in the S1 file with ``--zero-vote keep-s1``). One owner (most votes, then S1-file membership), cap 11. Optional
K3 veto (K3 < v) afterwards. Families are rows files (full or France-only TSV, official header); only French S1
rows are read. Diagnostics vs the S1 file: added/removed, decoy-fingerprint rates on changed links (K3 op
features, dense candidates only), probe acceptance (T2r ~64-69% true, C' ~92-96% true: pricing inferences).

    python -m ber.k_france_vote --family c2fr=A.tsv --family qfr=B.tsv --family g=C.tsv --out DIR [--veto 0.05]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.k3_test import DATA, sha256

HEADER = "source1_entity_id\tmatched_entity_ids"


def read_links(path, keep_s1):
    s, r = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            a, ids = line.rstrip("\n").split("\t")
            if ids and a in keep_s1:
                for x in ids.split(","):
                    s.append(a)
                    r.append(x)
    return pd.DataFrame({"s1": s, "rec": r})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--family", action="append", required=True, help="name=rows.tsv")
    p.add_argument("--out", required=True)
    p.add_argument("--s1-file", default=str(DATA / "s1file/matching_results.tsv"))
    p.add_argument("--scores", default="/home/ubuntu/k/out/k3_france_mix/france_k3_scores.parquet")
    p.add_argument("--veto", type=float, default=None)
    p.add_argument("--zero-vote", choices=["drop", "keep-s1"], default="drop")
    p.add_argument("--tag", default="vote")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "country"]).to_pandas()
    fr_ids = ref.loc[ref.country == "France", "entity_id"].tolist()
    fr_set = set(fr_ids)
    base = read_links(a.s1_file, fr_set).assign(in_s1=True)
    fams = dict(x.split("=", 1) for x in a.family)
    votes = None
    agree = {}
    for name, path in fams.items():
        d = read_links(path, fr_set).assign(**{name: True})
        votes = d if votes is None else votes.merge(d, on=["s1", "rec"], how="outer")
        agree[name] = {"links": int(len(d))}
    names = list(fams)
    votes[names] = votes[names].fillna(False).astype(bool)
    allv = votes.merge(base, on=["s1", "rec"], how="outer")
    allv[names] = allv[names].fillna(False).astype(bool)
    allv["in_s1"] = allv["in_s1"].fillna(False).astype(bool)
    allv["votes"] = allv[names].sum(axis=1)
    need = 2 if len(names) >= 3 else len(names)
    keep = allv["votes"] >= need
    disputed = (allv["votes"] >= 1) & ~keep
    keep |= disputed & allv["in_s1"]
    if a.zero_vote == "keep-s1":
        keep |= (allv["votes"] == 0) & allv["in_s1"]
    kept = allv[keep].copy()
    # one owner: most votes, then S1-file membership, then S1 id
    kept = kept.sort_values(["rec", "votes", "in_s1", "s1"], ascending=[True, False, False, True])
    kept = kept.drop_duplicates("rec", keep="first")
    sc = pq.read_table(a.scores).to_pandas().rename(columns={"s1_id": "s1", "record_id": "rec"})
    if a.veto is not None:
        kept = kept.merge(sc[["s1", "rec", "k3"]], on=["s1", "rec"], how="left")
        kept = kept[~(kept["k3"] < a.veto)].drop(columns=["k3"])
    kept = kept.sort_values(["s1", "votes", "in_s1"], ascending=[True, False, False])
    kept["rank"] = kept.groupby("s1").cumcount()
    kept = kept[kept["rank"] < 11]
    lists = kept.groupby("s1")["rec"].apply(list).to_dict()
    path = out / f"france_rows_{a.tag}.tsv"
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for s in fr_ids:
            f.write(f"{s}\t{','.join(lists.get(s, []))}\n")
    # diagnostics
    key = lambda d: d["s1"] + "|" + d["rec"]  # noqa: E731
    kk, bk = set(key(kept)), set(key(base))
    sc["key"] = key(sc)
    sc = sc.set_index("key")

    def rates(keys):
        keys = list(keys)
        f = sc.reindex(keys)
        have = f["k3"].notna()
        f = f[have]
        if not len(f):
            return {"links": len(keys), "with_k3": 0}
        return {"links": len(keys), "with_k3": int(have.sum()),
                "hn_delta_shift": float((f["hn_in_delta"] == 1).mean()),
                "hn_even_shift": float((f["hn_even_shift"] == 1).mean()),
                "decoy_word_added": float((f["decoy_added"] > 0).mean()),
                "legal_swapped": float((f["legal_rel"] == 5).mean()),
                "legal_added_or_dropped": float(f["legal_rel"].isin([3, 4]).mean()),
                "record_empty_address": float((f["rec_addr_empty"] == 1).mean()),
                "mean_k3": float(f["k3"].mean())}

    probes = {}
    for name, fn in (("T2r", "probeA_T2r.tsv"), ("Cprime", "probeCprime.tsv")):
        pr = pd.read_csv(DATA / "probes" / fn, sep="\t", dtype=str)
        pk = set(pr.iloc[:, 0] + "|" + pr.iloc[:, 1])
        probes[name] = {"links": len(pk), "share_kept": len(pk & kk) / max(1, len(pk))}
    for name in names:
        fk = set(key(allv[allv[name]]))
        agree[name].update({"shared_with_s1_file": len(fk & bk), "added_vs_s1_file": len(fk - bk),
                            "removed_vs_s1_file": len(bk - fk)})
    lens = np.array([len(lists.get(s, [])) for s in fr_ids])
    report = {"families": fams, "rule": f">= {need} votes keep; disputed -> S1-file #5 rule; 0 votes {a.zero_vote}",
              "veto": a.veto, "file": {"path": str(path), "sha256": sha256(path)}, "links": int(len(kept)),
              "s1_file_links": int(len(base)), "added": rates(kk - bk), "removed": rates(bk - kk),
              "mean_list": float(lens.mean()), "empty_rows": int((lens == 0).sum()), "probes": probes,
              "vote_histogram": {str(k): int(v) for k, v in allv["votes"].value_counts().sort_index().items()},
              "per_family": agree, "labels_read": False}
    (out / f"report_{a.tag}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
