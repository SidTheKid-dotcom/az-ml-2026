"""Export K features aligned to U's prep rows (B_REST: brest_prep order; test: test_prep order).

Columns: K3 probability (A-trained, out-of-sample on B_REST and test), the density-robust claimant-set
features (``record_assign`` CLAIM_NAMES minus the ``dense`` drop group) and the top K3 edit-op features by
gain in the K-robust model. float32 .npy plus a JSON with the column names.

    python -m ber.k_export --model-dir DIR --out DIR [--top 30]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber import record_assign as ra

DENSE = {"rec_out_best_base", "rec_out_best_k3", "rec_n_claim_grp", "rec_n_claim_all", "rec_best_c2_grp",
         "rec_best_q_grp", "rec_k3_mass", "rank_base_in_rec", "rank_k3_in_rec", "rec_base_ge_0_5", "rec_k3_ge_0_5",
         "is_none"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="/home/ubuntu/k/out/k2b_robust")
    p.add_argument("--out", required=True)
    p.add_argument("--top", type=int, default=30)
    a = p.parse_args()
    import lightgbm as lgb

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gains = np.zeros(len(ra.NAMES))
    for h in (0, 1):
        m = lgb.Booster(model_file=str(Path(a.model_dir) / f"k2_half{h}.txt"))
        gains += m.feature_importance("gain")
    gain = dict(zip(ra.NAMES, gains))
    top_k3 = [n for n in sorted(ra.K3F, key=lambda n: -gain[n])][:a.top]
    claim = [n for n in ra.CLAIM_NAMES if n not in DENSE]
    cols = claim + top_k3
    # B_REST (brest_prep order)
    B, x, *_ = ra.build_brest()
    np.save(out / "brest_k_features.npy", x[:, [ra.NAMES.index(n) for n in cols]].astype(np.float32))
    del x
    # test (test_prep order, every row)
    d = ra.DATA / "test_prep"
    left = np.load(d / "left.npy").astype(np.int64)
    right = np.load(d / "right.npy").astype(np.int64)
    xb = np.load(d / "x_base.npy", mmap_mode="r")
    xq = np.load(d / "x_q.npy", mmap_mode="r")
    k3 = np.load(Path("/home/ubuntu/k/out/k3_test/k3_prob_test_cand.npy"))
    rec_ids = pq.read_table(ra.DATA / "test/pool.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    rec_s3 = np.fromiter((i.startswith("S3") for i in rec_ids), dtype=bool, count=len(rec_ids)).astype(np.float32)
    n_s1 = len(np.load(d / "country_code.npy"))
    rows = np.arange(len(left))
    cf = ra.claimant_features(right, left, np.asarray(xb[:, ra.BASE_NAMES.index("base")], dtype=np.float32), k3,
                              np.zeros(len(left), bool), len(rec_ids), n_s1, rows, rec_s3[right],
                              np.nan_to_num(np.asarray(xb[:, ra.BASE_NAMES.index("c2")], dtype=np.float32), nan=-1.0),
                              np.nan_to_num(np.asarray(xq[:, 0], dtype=np.float32), nan=-1.0), right)
    feat = np.load(Path("/home/ubuntu/k/out/k3feat_test/feat.npy"), mmap_mode="r")
    parts = []
    for n in cols:
        if n in ra.CLAIM_NAMES:
            parts.append(cf[:, ra.CLAIM_NAMES.index(n)])
        else:
            parts.append(np.asarray(feat[:, ra.K3F.index(n)], dtype=np.float32))
    np.save(out / "test_k_features.npy", np.column_stack(parts).astype(np.float32))
    (out / "columns.json").write_text(json.dumps({
        "columns": cols, "brest_rows": "U brest_prep row order (3,895,977)", "test_rows": "U test_prep row order (14,513,431)",
        "k3": "A-trained K3 (mean of two A-half models); out-of-sample on B_REST and test",
        "claimant_features": "computed over every candidate row of the record/S1 (arena: all partitions; test: all rows)",
        "dropped_density_sensitive": sorted(DENSE)}, indent=2) + "\n")
    print("exported", len(cols), "columns")


if __name__ == "__main__":
    main()
