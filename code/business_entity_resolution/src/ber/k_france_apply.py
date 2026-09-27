"""France primary: the US/India-trained K-robust (5-fold, v1 + C2 40k + Qwen + K3 + density-robust claimant
features; every feature exists for French rows) applied to French dense candidates at a high threshold chosen
label-free from the probe sets (probe A ~0.64-0.69 true, probe C' ~0.92-0.96 true: pricing inferences).

Reports a tau grid (links, probe acceptance, falsifier rates on added and removed links vs the S1 file, empty rows),
then writes French rows at ``--tau`` after the post-rules: drop empty-address links whose record name (accent-folded,
legal forms stripped) equals the names of >= 2 French S1s. (Per-source caps and fr_swap run on the assembled file.)

    python -m ber.k_france_apply grid --out DIR
    python -m ber.k_france_apply write --out DIR --tau 0.9
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import record_assign as ra
from ber.edit_ops import FEATURES as F
from ber.k3_test import DATA, keys, read_links, sha256, write_rows
from ber.k_frswap import name_tokens

T0 = time.time()
MODEL = Path("/home/ubuntu/k/out/k5_robust")


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def score(out):
    import lightgbm as lgb

    path = out / "france_prob_k5robust.npy"
    d = DATA / "test_prep"
    tl = np.load(d / "left.npy").astype(np.int64)
    tr = np.load(d / "right.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")
    rows = np.flatnonzero(code[tl] == 0)
    if path.exists():
        return tl, tr, code, rows, np.load(path)
    xb = np.load(d / "x_base.npy", mmap_mode="r")
    xq = np.load(d / "x_q.npy", mmap_mode="r")
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    k3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
    rec_ids = pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    base = np.asarray(xb[:, ra.BASE_NAMES.index("base")], dtype=np.float32)
    xbr = np.asarray(xb[rows], dtype=np.float32)
    xqr = np.asarray(xq[rows], dtype=np.float32)
    claim = ra.claimant_features(tr, tl, base, k3, np.zeros(len(tl), bool), len(rec_ids), len(code), rows,
                                 rec_s3[tr[rows]], np.nan_to_num(xbr[:, ra.BASE_NAMES.index("c2")], nan=-1.0),
                                 np.nan_to_num(xqr[:, 0], nan=-1.0), tr[rows])
    x = np.hstack([xbr, xqr, np.asarray(feat[rows], dtype=np.float32), claim])
    models = [lgb.Booster(model_file=str(f)) for f in sorted(MODEL.glob("k2_fold*.txt"))]
    p = np.mean([m.predict(x, num_threads=30) for m in models], axis=0).astype(np.float32)
    np.save(path, p)
    log(f"scored {len(rows)} French rows with {len(models)} fold models")
    return tl, tr, code, rows, p


def setup(out):
    tl, tr, code, rows, pf = score(out)
    prob = np.zeros(len(tl), dtype=np.float64)
    prob[rows] = pf
    s1_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                        .to_pylist(), dtype=object)
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    n_rec = len(rec_ids)
    s1_index = {v: i for i, v in enumerate(s1_ids)}
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    ck = keys(tl, tr, n_rec)
    order = np.argsort(ck)
    sk = ck[order]

    def rows_of(k):
        pos = np.minimum(np.searchsorted(sk, k), len(sk) - 1)
        return order[pos[sk[pos] == k]]

    sl, sr = read_links(DATA / "s1file/matching_results.tsv", s1_index, rec_index)
    in_s1 = np.zeros(len(tl), bool)
    in_s1[rows_of(np.unique(keys(sl, sr, n_rec)))] = True
    probe = {}
    for name, fn in (("A", "probeA_T2r.tsv"), ("C", "probeCprime.tsv")):
        pr = pd.read_csv(DATA / "probes" / fn, sep="\t", dtype=str)
        kk = np.asarray([s1_index[s] * n_rec + rec_index[r] for s, r in zip(pr.iloc[:, 0], pr.iloc[:, 1])])
        m = np.zeros(len(tl), bool)
        m[rows_of(np.unique(kk))] = True
        probe[name] = m
    return dict(tl=tl, tr=tr, code=code, rows=rows, prob=prob, s1_ids=s1_ids, rec_ids=rec_ids, in_s1=in_s1,
                probe=probe)


def rates(feat, mask):
    r = np.flatnonzero(mask)
    if not len(r):
        return {"links": 0}
    f = np.asarray(feat[r], dtype=np.float32)
    return {"links": int(len(r)), "delta": round(float(np.mean(f[:, F.index("hn_in_delta")] == 1)), 4),
            "even": round(float(np.mean(f[:, F.index("hn_even_shift")] == 1)), 4),
            "decoy": round(float(np.mean(f[:, F.index("decoy_added")] > 0)), 4),
            "legal_swap": round(float(np.mean(f[:, F.index("legal_rel")] == 5)), 4),
            "EA": round(float(np.mean(f[:, F.index("rec_addr_empty")] == 1)), 3)}


def ea_shared_mask(S, feat):
    """Empty-address links whose record name equals the names of >= 2 French S1s."""
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "business_name", "country"]).to_pandas()
    fr = ref[ref.country == "France"]
    cnt = Counter(" ".join(name_tokens(n)) for n in fr.business_name)
    pool = pq.read_table(DATA / "test/pool.parquet", columns=["business_name"]).column("business_name").to_pylist()
    rows = S["rows"]
    ea = np.asarray(feat[rows, F.index("rec_addr_empty")], dtype=np.float32) == 1
    m = np.zeros(len(S["tl"]), bool)
    idx = rows[ea]
    names = [" ".join(name_tokens(pool[r])) for r in S["tr"][idx]]
    shared = np.asarray([cnt.get(n, 0) >= 2 for n in names], dtype=bool)
    m[idx[shared]] = True
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["grid", "write"])
    p.add_argument("--out", required=True)
    p.add_argument("--tau", type=float, default=0.9)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    S = setup(out)
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    fr = S["code"][S["tl"]] == 0
    n_s1 = len(S["code"])
    base = S["in_s1"] & fr
    shared = ea_shared_mask(S, feat)
    log(f"EA links with a record name shared by >=2 French S1s (candidates): {int(shared.sum())}")
    grid = {}
    taus = [round(t, 3) for t in np.arange(0.6, 0.951, 0.025)] if a.command == "grid" else [a.tau]
    for t in taus:
        acc = ra.decode(S["prob"], S["tl"], S["tr"], ("thr", t), n_s1) & fr
        acc_pr = acc & ~shared
        lens = np.bincount(S["tl"][acc_pr], minlength=n_s1)[S["code"] == 0]
        grid[str(t)] = {"links": int(acc.sum()), "links_after_ea_rule": int(acc_pr.sum()),
                        "ea_rule_removed": int((acc & shared).sum()),
                        "probe_A_accept": round(float(acc_pr[S["probe"]["A"]].mean()), 4),
                        "probe_C_accept": round(float(acc_pr[S["probe"]["C"]].mean()), 4),
                        "added": rates(feat, acc_pr & ~base), "removed": rates(feat, base & ~acc_pr),
                        "empty_rows": int((lens == 0).sum()), "mean_list": round(float(lens.mean()), 4)}
        log(f"tau {t}: {json.dumps(grid[str(t)])}")
        if a.command == "write":
            info = write_rows(out / f"france_rows_k5robust_t{t}.tsv", S["s1_ids"], S["rec_ids"], S["tl"], S["tr"],
                              acc_pr, S["prob"].astype(np.float32), S["code"] == 0)
            grid[str(t)]["file"] = info
    (out / f"{a.command}.json").write_text(json.dumps({"model": str(MODEL), "grid": grid, "s1_file_links": int(base.sum()),
                                                        "labels_read": False}, indent=1) + "\n")


if __name__ == "__main__":
    main()
