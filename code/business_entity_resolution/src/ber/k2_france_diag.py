"""Label-free French diagnostics for a K pair model (binary, record_assign features) applied to French rows.

Scores every French test candidate with the two half models of ``--model-dir`` (mean probability), decodes
at the model's rule (one owner + cap 11), writes French rows and compares them with the scored S1 file's
French rows: links added/removed, decoy-fingerprint rates (delta shift, decoy word, legal swap, even shift),
K3 of the changed links, probe acceptance (T2r ~64-69% true, C' ~92-96% true: pricing-model inferences).
Nothing here measures French accuracy; the model is fit on US/India B_REST only.

    python -m ber.k2_france_diag --model-dir DIR --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber import record_assign as ra
from ber.edit_ops import FEATURES
from ber.k3_test import DATA, keys, read_links, write_rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/home/ubuntu/k/out/k2b_robust")
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float)
    a = p.parse_args()
    import lightgbm as lgb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rep = json.loads((Path(a.model_dir) / "report.json").read_text())
    kind, val = rep["full_pick_for_test"]
    thr = float(a.threshold if a.threshold is not None else val)
    models = [lgb.Booster(model_file=str(Path(a.model_dir) / f"k2_half{h}.txt")) for h in (0, 1)]
    d = DATA / "test_prep"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    code = np.load(d / "country_code.npy")
    xb = np.asarray(np.load(d / "x_base.npy", mmap_mode="r"), dtype=np.float32)
    xq = np.asarray(np.load(d / "x_q.npy", mmap_mode="r"), dtype=np.float32)
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    k3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
    s1_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                        .to_pylist(), dtype=object)
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    n_s1, n_rec = len(s1_ids), len(rec_ids)
    rows = np.flatnonzero(code[left] == 0)
    claim = ra.claimant_features(right, left, xb[:, ra.BASE_NAMES.index("base")], k3, np.zeros(len(left), bool), n_rec,
                                 n_s1, rows, rec_s3[right[rows]],
                                 np.nan_to_num(xb[rows, ra.BASE_NAMES.index("c2")], nan=-1.0),
                                 np.nan_to_num(xq[rows, 0], nan=-1.0), right[rows])
    x = np.hstack([xb[rows], xq[rows], np.asarray(feat[rows], dtype=np.float32), claim])
    prob = np.zeros(len(left), dtype=np.float64)
    prob[rows] = 0.5 * (models[0].predict(x, num_threads=30) + models[1].predict(x, num_threads=30))
    del x
    acc = ra.decode(prob, left, right, ("thr", thr), n_s1)
    fr = code[left] == 0
    acc &= fr
    s1_index = {v: i for i, v in enumerate(s1_ids)}
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    ck = keys(left, right, n_rec)
    order = np.argsort(ck)
    sk = ck[order]

    def rows_of(k):
        pos = np.minimum(np.searchsorted(sk, k), len(sk) - 1)
        return order[pos[sk[pos] == k]]

    sl, sr = read_links(DATA / "s1file/matching_results.tsv", s1_index, rec_index)
    s1_keys = np.unique(keys(sl, sr, n_rec))
    in_s1 = np.zeros(len(left), dtype=bool)
    in_s1[rows_of(s1_keys)] = True
    tl, tr = read_links(DATA / "probes/drop_T2r.tsv", s1_index, rec_index)
    t2r = np.zeros(len(left), dtype=bool)
    t2r[rows_of(np.setdiff1d(s1_keys, np.unique(keys(tl, tr, n_rec))))] = True
    cp_pairs = json.loads((DATA / "probes/drop_Cprime_dropped.json").read_text())
    cp = np.zeros(len(left), dtype=bool)
    cp[rows_of(np.unique(np.asarray([s1_index[s] * n_rec + rec_index[r] for s, r in cp_pairs], dtype=np.int64)))] = True
    idx = {n: FEATURES.index(n) for n in ("hn_in_delta", "hn_even_shift", "decoy_added", "legal_rel", "rec_addr_empty")}

    def rates(mask):
        r = np.flatnonzero(mask)
        if not len(r):
            return {"links": 0}
        f = np.asarray(feat[r], dtype=np.float32)
        return {"links": int(len(r)), "hn_delta_shift": float(np.mean(f[:, idx["hn_in_delta"]] == 1)),
                "hn_even_shift": float(np.mean(f[:, idx["hn_even_shift"]] == 1)),
                "decoy_word_added": float(np.mean(f[:, idx["decoy_added"]] > 0)),
                "legal_swapped": float(np.mean(f[:, idx["legal_rel"]] == 5)),
                "legal_added_or_dropped": float(np.mean(np.isin(f[:, idx["legal_rel"]], (3, 4)))),
                "record_empty_address": float(np.mean(f[:, idx["rec_addr_empty"]] == 1)),
                "mean_k3": float(k3[r].mean()), "mean_model_p": float(prob[r].mean())}

    base = in_s1 & fr
    k_model = np.bincount(left[acc], minlength=n_s1)[code == 0]
    probes = {name: {"links": int(m.sum()), "share_accepted": float(acc[m].mean()), "mean_p": float(prob[m].mean())}
              for name, m in (("T2r", t2r), ("Cprime", cp))}
    result = {"model_dir": a.model_dir, "threshold": thr, "links": int(acc.sum()), "s1_file_links": int(base.sum()),
              "added": rates(acc & ~base), "removed": rates(base & ~acc), "kept": rates(acc & base),
              "mean_list": float(k_model.mean()), "empty_rows": int((k_model == 0).sum()), "probes": probes,
              "file": write_rows(out / "france_rows.tsv", s1_ids, rec_ids, left, right, acc, prob.astype(np.float32),
                                 code == 0), "labels_read": False}
    np.save(out / "france_prob.npy", prob[rows].astype(np.float32))
    (out / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    print(json.dumps(result, default=float))


if __name__ == "__main__":
    main()
