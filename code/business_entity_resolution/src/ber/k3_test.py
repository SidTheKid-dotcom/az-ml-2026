"""K3 on the test candidates: US/India and French row files plus label-free French diagnostics.

Scores every test candidate row (U prep test rows) with the mean of the two A-half K3 models, decodes
at the A out-of-sample threshold (one owner + cap 11), writes rows files in reference order (header
``source1_entity_id\tmatched_entity_ids``) and compares French rows with the scored S1 file
(0.978287, SHA 559f37b9...): links added/removed, list length, house-number-shift and decoy-word rates on
added links, and the probe sets (T2r = S1 file minus drop_T2r, C' = drop_Cprime dropped.json).
No test labels exist; nothing here is a French accuracy.

    python -m ber.k3_test --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import one_owner
from ber.edit_ops import FEATURES

T0 = time.time()
DATA = Path("/home/ubuntu/k/data")
K3DIR = Path("/home/ubuntu/k/out/k3")
HEADER = "source1_entity_id\tmatched_entity_ids"
COUNTRY = {0: "France", 1: "India", 2: "US"}


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rank_within(group, score):
    order = np.lexsort((np.arange(len(score)), -score, group))
    g = group[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    idx = np.arange(len(g))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    rank = np.empty(len(g), dtype=np.int32)
    rank[order] = (idx - first).astype(np.int32)
    return rank


def decide(prob, left, right, threshold, cap=11):
    accept = one_owner(prob >= threshold, left, right, prob)
    rows = np.flatnonzero(accept)
    if rows.size:
        accept[rows[rank_within(left[rows], prob[rows]) >= cap]] = False
    return accept


def write_rows(path, s1_ids, rec_ids, left, right, accept, prob, keep_s1):
    """Rows file for the S1s in ``keep_s1`` (reference order), links sorted by probability."""
    rows = np.flatnonzero(accept)
    rows = rows[np.lexsort((-prob[rows], left[rows]))]
    starts = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="left")
    stops = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(HEADER + "\n")
        for i in np.flatnonzero(keep_s1):
            f.write(f"{s1_ids[i]}\t{','.join(rec_ids[right[rows[starts[i]:stops[i]]]])}\n")
    tmp.replace(path)
    return {"path": str(path), "sha256": sha256(path), "rows": int(keep_s1.sum()),
            "links": int(accept[keep_s1[left]].sum())}


def read_links(path, s1_index, rec_index):
    ls, rs = [], []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s1, ids = line.rstrip("\n").split("\t")
            if ids:
                i = s1_index[s1]
                for r in ids.split(","):
                    ls.append(i)
                    rs.append(rec_index[r])
    return np.asarray(ls, dtype=np.int64), np.asarray(rs, dtype=np.int64)


def keys(l, r, n_rec):
    return l * n_rec + r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--threshold", type=float, help="default: K3 report A out-of-sample threshold")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import lightgbm as lgb

    rep = json.loads((K3DIR / "report.json").read_text())
    thr = float(a.threshold if a.threshold is not None else rep["a_oof_threshold"])
    models = [lgb.Booster(model_file=str(K3DIR / f"k3_half{h}.txt")) for h in (0, 1)]
    left = np.load(DATA / "test_prep/left.npy").astype(np.int64)
    right = np.load(DATA / "test_prep/right.npy").astype(np.int64)
    code = np.load(DATA / "test_prep/country_code.npy")
    ref = pq.read_table(DATA / "test/reference.parquet", columns=["entity_id", "country"])
    s1_ids = np.asarray(ref.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref.column("country").to_pylist(), dtype=object)
    if not all(np.all(country[code == c] == n) for c, n in COUNTRY.items()):
        raise SystemExit("country codes do not match the reference")
    rec_ids = np.asarray(pq.read_table(DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id")
                         .to_pylist(), dtype=object)
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids))
    n_rec = len(rec_ids)
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    prob_path = out / "k3_prob_test_cand.npy"
    if prob_path.exists():
        prob = np.load(prob_path)
    else:
        prob = np.empty(len(left), dtype=np.float32)
        for s in range(0, len(left), 2_000_000):
            x = np.hstack([np.asarray(feat[s:s + 2_000_000], dtype=np.float32),
                           rec_s3[right[s:s + 2_000_000]].astype(np.float32)[:, None]])
            prob[s:s + 2_000_000] = 0.5 * (models[0].predict(x, num_threads=30) + models[1].predict(x, num_threads=30))
        np.save(prob_path, prob)
    log(f"scored {len(prob)} test candidate rows; threshold {thr}")
    accept = decide(prob, left, right, thr)
    fr_s1 = code == 0
    files = {"us_india": write_rows(out / "k3_us_india_rows.tsv", s1_ids, rec_ids, left, right, accept, prob, ~fr_s1),
             "france": write_rows(out / "k3_france_rows.tsv", s1_ids, rec_ids, left, right, accept, prob, fr_s1)}
    log(json.dumps(files))

    # ---- French diagnostics vs the scored S1 file
    s1_index = {v: i for i, v in enumerate(s1_ids)}
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    sl, sr = read_links(DATA / "s1file/matching_results.tsv", s1_index, rec_index)
    s1_keys = np.unique(keys(sl, sr, n_rec))
    cand_keys = keys(left, right, n_rec)
    order = np.argsort(cand_keys)
    sorted_keys = cand_keys[order]

    def rows_of(k):
        pos = np.searchsorted(sorted_keys, k)
        pos = np.minimum(pos, len(sorted_keys) - 1)
        ok = sorted_keys[pos] == k
        return order[pos[ok]], int((~ok).sum())

    s1_rows, s1_missing = rows_of(s1_keys)
    in_s1 = np.zeros(len(left), dtype=bool)
    in_s1[s1_rows] = True
    fr_row = fr_s1[left]
    hn = FEATURES.index("hn_in_delta")
    ev = FEATURES.index("hn_even_shift")
    dec = FEATURES.index("decoy_added")
    legal = FEATURES.index("legal_rel")

    def rates(rows):
        if len(rows) == 0:
            return {"links": 0}
        f = np.asarray(feat[np.sort(rows)], dtype=np.float32)
        return {"links": int(len(rows)), "hn_delta_shift_rate": float(np.nanmean(f[:, hn] == 1)),
                "hn_even_shift_rate": float(np.nanmean(f[:, ev] == 1)),
                "decoy_word_added_rate": float(np.mean(f[:, dec] > 0)),
                "legal_swapped_rate": float(np.mean(f[:, legal] == 5)),
                "legal_swap_and_delta_shift_rate": float(np.mean((f[:, legal] == 5) & (f[:, hn] == 1))),
                "mean_k3": float(prob[rows].mean())}

    added = np.flatnonzero(accept & ~in_s1 & fr_row)
    removed = np.flatnonzero(~accept & in_s1 & fr_row)
    kept = np.flatnonzero(accept & in_s1 & fr_row)
    k_fr_k3 = np.bincount(left[accept & fr_row], minlength=len(s1_ids))[fr_s1]
    k_fr_s1 = np.bincount(left[in_s1 & fr_row], minlength=len(s1_ids))[fr_s1]
    diag = {"s1_file_links_not_in_candidates": s1_missing,
            "france_s1": int(fr_s1.sum()),
            "k3_links": int((accept & fr_row).sum()), "s1_file_links": int((in_s1 & fr_row).sum()),
            "added": rates(added), "removed": rates(removed), "kept": rates(kept),
            "s1_file_all_french_links": rates(np.flatnonzero(in_s1 & fr_row)),
            "mean_list_k3": float(k_fr_k3.mean()), "mean_list_s1_file": float(k_fr_s1.mean()),
            "empty_rows_k3": int((k_fr_k3 == 0).sum()), "empty_rows_s1_file": int((k_fr_s1 == 0).sum())}
    # probes
    tl, tr = read_links(DATA / "probes/drop_T2r.tsv", s1_index, rec_index)
    t2r_keys = np.setdiff1d(s1_keys, np.unique(keys(tl, tr, n_rec)))
    cp = json.loads((DATA / "probes/drop_Cprime_dropped.json").read_text())
    cp_keys = np.unique(np.asarray([s1_index[s] * n_rec + rec_index[r] for s, r in cp], dtype=np.int64))
    probes = {}
    for name, k, truth in (("T2r", t2r_keys, "~64-69% true (pricing-model inference)"),
                           ("Cprime", cp_keys, "~92-96% true (pricing-model inference)")):
        rows, miss = rows_of(k)
        probes[name] = {"links": int(len(k)), "not_in_candidates": miss, "prior_truth": truth,
                        "mean_k3": float(prob[rows].mean()), "share_accepted_by_k3": float(accept[rows].mean()),
                        "share_k3_ge_0_5": float((prob[rows] >= 0.5).mean())}
    diag["probes"] = probes
    # US/India vs the S1 file (pair tree rows), label-free agreement only
    ui_row = ~fr_row
    diag["us_india_vs_s1_file"] = {"k3_links": int((accept & ui_row).sum()),
                                   "s1_file_links": int((in_s1 & ui_row).sum()),
                                   "agree": int((accept & in_s1 & ui_row).sum())}
    result = {"threshold": thr, "k3_models": [str(K3DIR / f"k3_half{h}.txt") for h in (0, 1)], "files": files,
              "french_diagnostics": diag, "labels_read": False, "elapsed_seconds": round(time.time() - T0, 1)}
    (out / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    log(json.dumps(diag, default=float))


if __name__ == "__main__":
    main()
