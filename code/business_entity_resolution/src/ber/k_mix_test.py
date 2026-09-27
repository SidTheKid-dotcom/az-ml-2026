"""Test rows from a per-source mixture of two k_final runs (same row order: dense, then the sources in order).

Default: K-final F (with C2 v2) for dense and reverse rows, K-final C for route rows (C's test route acceptance
matches B_REST, F's is 1.6x on India). Threshold ``--t`` for all rows, route rows additionally >= ``--route-thr``.
B_REST (same mixture of OOFs): 0.989372 vs C guarded 0.989229, +0.000143 [+0.000097, +0.00019].

    python -m ber.k_mix_test --a DIR_F --b DIR_C --out FILE
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber import record_assign as ra

DATA = ra.DATA
CACHE = Path("/home/ubuntu/k/cache")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True, help="run for dense + reverse rows")
    p.add_argument("--b", required=True, help="run for route rows")
    p.add_argument("--out", required=True)
    p.add_argument("--t", type=float, default=0.8)
    p.add_argument("--route-thr", type=float, default=0.955)
    a = p.parse_args()
    pa, pb = np.load(Path(a.a) / "test_p.npy").astype(np.float64), np.load(Path(a.b) / "test_p.npy").astype(np.float64)
    la, lb = np.load(Path(a.a) / "test_left.npy"), np.load(Path(a.b) / "test_left.npy")
    sa = json.loads((Path(a.a) / "test_src_rows.json").read_text())
    sb = json.loads((Path(a.b) / "test_src_rows.json").read_text())
    if len(pa) != len(pb) or not np.array_equal(la, lb) or sa != sb:
        raise SystemExit("the two runs do not share the test row order")
    r0, r1 = sa["routes"]
    p_mix = pa.copy()
    p_mix[r0:r1] = pb[r0:r1]
    d = DATA / "test_prep"
    tl = np.load(d / "left.npy").astype(np.int64)
    tr = np.load(d / "right.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")
    rows = np.flatnonzero(code[tl] != 0)
    pool_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                          .to_pylist(), dtype=object)
    ref_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    rid = [pool_ids[tr[rows]]]
    for s in ("routes", "rev"):
        z = np.load(CACHE / f"{s}_test.npz", allow_pickle=False)
        rid.append(np.asarray(z["rec_id"]).astype(object))
    rid = np.concatenate(rid)
    left = la.astype(np.int64)
    if len(rid) != len(left) or not np.array_equal(left[:len(rows)], tl[rows]):
        raise SystemExit("row reconstruction mismatch")
    k = pd.Series(ref_ids[left] + "|" + rid)
    order = np.lexsort((-p_mix, k.to_numpy()))
    keep = np.zeros(len(p_mix), dtype=bool)
    keep[order[~k.iloc[order].duplicated().to_numpy()]] = True
    q = np.where(keep, p_mix, 0.0)
    q[r0:r1] = np.where(p_mix[r0:r1] >= a.route_thr, q[r0:r1], 0.0)
    _, rkey = np.unique(rid, return_inverse=True)
    n_s1 = len(code)
    acc = ra.decode(q, left, rkey, ("thr", a.t), n_s1)
    sel = np.flatnonzero(acc)
    sel = sel[np.lexsort((-p_mix[sel], left[sel]))]
    starts = np.searchsorted(left[sel], np.arange(n_s1), side="left")
    stops = np.searchsorted(left[sel], np.arange(n_s1), side="right")
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for i in np.flatnonzero(code != 0):
            f.write(f"{ref_ids[i]}\t{','.join(rid[sel[starts[i]:stops[i]]])}\n")
    per = {}
    for s, (x0, x1) in sa.items():
        cc = code[left[x0:x1]]
        per[s] = {"accepted": int(acc[x0:x1].sum()), "per_1k_India": round(1000 * float(acc[x0:x1][cc == 1].mean()), 2),
                  "per_1k_US": round(1000 * float(acc[x0:x1][cc == 2].mean()), 2)}
    lengths = (stops - starts)[code != 0]
    print(json.dumps({"file": a.out, "sha256": hashlib.sha256(Path(a.out).read_bytes()).hexdigest(),
                      "links": int(lengths.sum()), "longest": int(lengths.max()), "per_source": per,
                      "t": a.t, "route_thr": a.route_thr}))


if __name__ == "__main__":
    main()
