"""Score a completed full-pool retrieval screen against complete raw truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--search", type=Path, required=True)
    p.add_argument("--root", type=Path, default=Path("/home/ubuntu/takeover_20260927_1000"))
    p.add_argument("--audit", type=Path)
    p.add_argument("--u", type=Path, default=Path("/home/ubuntu/run27/U"))
    p.add_argument("--r", type=Path, default=Path("/home/ubuntu/run27/R"))
    p.add_argument("--arena", type=Path, default=Path("/opt/az-ml-new/artifacts/arena"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    audit = a.audit or a.root / "raw_audit"
    assert (a.search / "complete.json").is_file(), "Search is not complete"
    truth = pd.read_parquet(audit / "screen_truth.parquet")
    population = pd.read_parquet(audit / "screen_true_count.parquet")
    queries = np.load(a.search / "query_indices.npy")
    right = np.load(a.search / "pool_indices.npy")
    assert np.array_equal(np.sort(queries), np.sort(population.left_idx))
    assert right.shape[0] == len(queries)
    n_pool = 10_320_219
    assert np.all((right >= -1) & (right < n_pool))
    tl, tr = truth.left_idx.to_numpy(dtype=np.int64), truth.right_idx.to_numpy(dtype=np.int64)
    truth_keys = tl * n_pool + tr
    new_keys = np.repeat(queries.astype(np.int64), right.shape[1]) * n_pool + right.ravel()
    new_keys = np.unique(new_keys[right.ravel() >= 0])
    w = np.load(a.root / "data/brest_world.npz")
    screen = population.left_idx.to_numpy(dtype=np.int64)
    smask = np.zeros(len(w["true_count"]), bool)
    smask[screen] = True
    tc = w["true_count"]
    assert np.array_equal(tc[screen], population.true_count)
    old_map = pq.read_table(a.arena / "v1/pool.parquet", columns=["raw_row_idx"])["raw_row_idx"].to_numpy()
    raw_ids = pd.Index(pq.read_table(a.r / "texts/train_pool.parquet", columns=["entity_id"])["entity_id"].to_numpy())
    extras = json.loads((a.u / "prep/brest_u8t/extra_rec_ids.json").read_text())
    extra_map = raw_ids.get_indexer(extras)
    assert np.all(extra_map >= 0)
    modern_map = np.r_[old_map, extra_map]
    del raw_ids
    m = smask[w["left"]]
    current_keys = np.unique(w["left"][m].astype(np.int64) * n_pool + modern_map[w["right"][m]])
    dense = []
    for batch in pq.ParquetFile(a.arena / "v1/pairs.parquet").iter_batches(
            batch_size=1_000_000, columns=["left_idx", "right_idx"]):
        l, r = batch.column(0).to_numpy(), batch.column(1).to_numpy()
        m = smask[l]
        dense.append(l[m].astype(np.int64) * n_pool + old_map[r[m]])
    dense = np.unique(np.concatenate(dense))
    hit_current, hit_dense, hit_new = (np.isin(truth_keys, k) for k in (current_keys, dense, new_keys))

    def measure(hit):
        tp = np.bincount(tl[hit], minlength=len(tc))[screen]
        den = 4 * tp + tc[screen]
        f = np.divide(5 * tp, den, out=np.ones(len(screen), float), where=den > 0)
        return {"s1": len(screen), "true_links": int(tc[screen].sum()), "found_true": int(tp.sum()),
                "missed_true": int(tc[screen].sum() - tp.sum()),
                "link_recall": float(tp.sum() / tc[screen].sum()), "oracle_macro_f05": float(f.mean())}

    report = {"scope": "fixed 30000 B_REST S1s, all 10320219 supplied targets, complete raw truth",
              "model_search": str(a.search), "current_pruned_union": measure(hit_current),
              "legacy_dense_top50": measure(hit_dense), "new_dense": measure(hit_new),
              "current_plus_new": measure(hit_current | hit_new),
              "unpruned_existing_union": measure(hit_current | hit_dense),
              "unpruned_existing_plus_new": measure(hit_current | hit_dense | hit_new),
              "new_true_beyond_current": int((hit_new & ~hit_current).sum()),
              "new_true_beyond_unpruned_existing": int((hit_new & ~hit_current & ~hit_dense).sum()),
              "new_candidates": len(new_keys), "labels_used_in_retrieval": False,
              "holdout2_rows_read": 0}
    # Candidate export is independent of truth: keep ALL newly retrieved pairs.
    extra = np.setdiff1d(new_keys, current_keys)
    a.out.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"left_idx": (extra // n_pool).astype(np.int32),
                             "raw_right_idx": (extra % n_pool).astype(np.int32)}), a.out / "extra_pairs.parquet")
    (a.out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
