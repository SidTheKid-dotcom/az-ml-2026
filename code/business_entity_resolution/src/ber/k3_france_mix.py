"""K3 on French rows as a correction of the scored S1 file (label-free; nothing here measures French accuracy).

Variants (French rows only; US/India rows untouched):
* ``veto_v``: S1-file French links minus links whose K3 probability < v;
* ``add_a``: S1-file French links plus K3 >= a links to records the S1 file leaves unassigned
  (one owner by K3, cap 11);
* ``veto_v_add_a``: both.
Each variant reports links removed/added, the house-number delta-shift / even-shift / decoy-word /
legal-swap rates of the changed links, and how many probe links (T2r ~64-69% true, C' ~92-96% true,
pricing-model inferences) it removes. Rows files are written for France only, in reference order.

    python -m ber.k3_france_mix --out DIR
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.edit_ops import FEATURES
from ber.k3_test import DATA, keys, read_links, sha256, write_rows

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--k3-test", default="/home/ubuntu/k/out/k3_test")
    p.add_argument("--vetos", default="0.02,0.05,0.1,0.2")
    p.add_argument("--adds", default="0.9,0.95,0.98")
    p.add_argument("--write", default="", help="comma list of variant names to write as France rows files")
    p.add_argument("--export", action="store_true", help="write every French candidate row's K3 score (parquet)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    left = np.load(DATA / "test_prep/left.npy").astype(np.int64)
    right = np.load(DATA / "test_prep/right.npy").astype(np.int64)
    code = np.load(DATA / "test_prep/country_code.npy")
    prob = np.load(Path(a.k3_test) / "k3_prob_test_cand.npy")
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    s1_ids = np.asarray(pq.read_table(DATA / "test/reference.parquet", columns=["entity_id"]).column("entity_id")
                        .to_pylist(), dtype=object)
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    n_rec = len(rec_ids)
    s1_index = {v: i for i, v in enumerate(s1_ids)}
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    cand_keys = keys(left, right, n_rec)
    order = np.argsort(cand_keys)
    sk = cand_keys[order]

    def rows_of(k):
        pos = np.minimum(np.searchsorted(sk, k), len(sk) - 1)
        ok = sk[pos] == k
        return order[pos[ok]]

    sl, sr = read_links(DATA / "s1file/matching_results.tsv", s1_index, rec_index)
    s1_keys = np.unique(keys(sl, sr, n_rec))
    in_s1 = np.zeros(len(left), dtype=bool)
    in_s1[rows_of(s1_keys)] = True
    fr = code[left] == 0
    tl, tr = read_links(DATA / "probes/drop_T2r.tsv", s1_index, rec_index)
    t2r = np.zeros(len(left), dtype=bool)
    t2r[rows_of(np.setdiff1d(s1_keys, np.unique(keys(tl, tr, n_rec))))] = True
    cp_pairs = json.loads((DATA / "probes/drop_Cprime_dropped.json").read_text())
    cp = np.zeros(len(left), dtype=bool)
    cp[rows_of(np.unique(np.asarray([s1_index[s] * n_rec + rec_index[r] for s, r in cp_pairs], dtype=np.int64)))] = True
    idx = {n: FEATURES.index(n) for n in ("hn_in_delta", "hn_even_shift", "decoy_added", "legal_rel", "rec_addr_empty",
                                           "core_equal")}

    def rates(mask):
        rows = np.flatnonzero(mask)
        if not len(rows):
            return {"links": 0}
        f = np.asarray(feat[rows], dtype=np.float32)
        return {"links": int(len(rows)), "hn_delta_shift": float(np.mean(f[:, idx["hn_in_delta"]] == 1)),
                "hn_even_shift": float(np.mean(f[:, idx["hn_even_shift"]] == 1)),
                "decoy_word_added": float(np.mean(f[:, idx["decoy_added"]] > 0)),
                "legal_swapped": float(np.mean(f[:, idx["legal_rel"]] == 5)),
                "legal_added_or_dropped": float(np.mean(np.isin(f[:, idx["legal_rel"]], (3, 4)))),
                "record_empty_address": float(np.mean(f[:, idx["rec_addr_empty"]] == 1)),
                "core_name_equal": float(np.mean(f[:, idx["core_equal"]] == 1)),
                "mean_k3": float(prob[rows].mean()), "t2r_probe_links": int(t2r[rows].sum()),
                "cprime_probe_links": int(cp[rows].sum())}

    base = in_s1 & fr
    claimed = np.zeros(n_rec, dtype=bool)
    claimed[right[in_s1]] = True
    list_len = np.bincount(left[base], minlength=len(s1_ids))
    report = {"s1_file_french_links": int(base.sum()), "s1_file_french_rates": rates(base),
              "probe_links_in_s1_file": {"T2r": int((t2r & base).sum()), "Cprime": int((cp & base).sum())},
              "variants": {}}
    variants = {}

    def additions(thr):
        cand = np.flatnonzero(fr & ~in_s1 & (prob >= thr) & ~claimed[right])
        cand = cand[np.lexsort((left[cand], -prob[cand], right[cand]))]  # best claimant per record first
        first = np.ones(len(cand), dtype=bool)
        first[1:] = right[cand][1:] != right[cand][:-1]
        cand = cand[first]
        cand = cand[np.lexsort((-prob[cand], left[cand]))]
        room = 11 - list_len[left[cand]]
        start = np.searchsorted(left[cand], left[cand], side="left")
        rank = np.arange(len(cand)) - start
        add = np.zeros(len(left), dtype=bool)
        add[cand[rank < room]] = True
        return add

    for v in [float(x) for x in a.vetos.split(",") if x]:
        variants[f"veto_{v}"] = base & ~(prob < v)
    for t in [float(x) for x in a.adds.split(",") if x]:
        variants[f"add_{t}"] = base | additions(t)
    for v in [float(x) for x in a.vetos.split(",") if x][:2]:
        for t in [float(x) for x in a.adds.split(",") if x][:2]:
            variants[f"veto_{v}_add_{t}"] = (base & ~(prob < v)) | additions(t)
    for name, acc in variants.items():
        k = np.bincount(left[acc], minlength=len(s1_ids))[code == 0]
        report["variants"][name] = {"links": int(acc.sum()), "removed": rates(base & ~acc), "added": rates(acc & ~base),
                                    "mean_list": float(k.mean()), "empty_rows": int((k == 0).sum()),
                                    "max_list": int(k.max())}
        log(f"{name}: {json.dumps(report['variants'][name], default=float)}")
    report["s1_file_mean_list"] = float(list_len[code == 0].mean())
    report["s1_file_empty_rows"] = int((list_len[code == 0] == 0).sum())
    files = {}
    for name in [x for x in a.write.split(",") if x]:
        acc = variants[name]
        score = np.where(in_s1, 2.0, prob).astype(np.float32)  # S1-file links first, then by K3
        files[name] = write_rows(out / f"france_rows_{name}.tsv", s1_ids, rec_ids, left, right, acc, score, code == 0)
    if a.export:
        import pyarrow as pa

        rows = np.flatnonzero(fr)
        f = np.asarray(feat[rows], dtype=np.float32)
        cols = {"s1_id": pa.array(s1_ids[left[rows]].tolist(), pa.large_string()),
                "record_id": pa.array(rec_ids[right[rows]].tolist(), pa.large_string()),
                "k3": pa.array(prob[rows]), "in_s1_file": pa.array(in_s1[rows]),
                "t2r_probe": pa.array(t2r[rows]), "cprime_probe": pa.array(cp[rows])}
        for n in ("legal_rel", "hn_rel", "hn_diff", "hn_in_delta", "hn_even_shift", "decoy_added", "rec_addr_empty",
                  "core_equal"):
            cols[n] = pa.array(f[:, FEATURES.index(n)])
        path = out / "france_k3_scores.parquet"
        pq.write_table(pa.table(cols), path)
        files["france_k3_scores"] = {"path": str(path), "sha256": sha256(path), "rows": int(len(rows))}
    report["files"] = files
    report["labels_read"] = False
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log("done")


if __name__ == "__main__":
    main()
