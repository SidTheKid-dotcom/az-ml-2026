"""Export string-feature pair tables (ids + 52 features + stage-1/stage-2 scores) as parquet parts.

    python -m ber.string_gbdt_export arena --feat O/arena --data D --work W --out DIR   # B_REST rows only
    python -m ber.string_gbdt_export test --feat O/test --work W --out DIR

Arena B_REST p1/prob are the training run's saved arrays (stage-1 mean of both folds; stage-2
on stage-2 candidate rows, 0 elsewhere). Test scores are recomputed with ber.string_gbdt_apply.score.
Columns: s1_id, rec_id, route (eureka-route pair), [label for arena], FEATURES..., sg_p1, sg_prob.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.string_gbdt import FEATURES, log
from ber.string_gbdt_train import load_feat


def write_parts(out, rows, s1_ids, rec_ids, left, right, route, X, p1, prob, label=None, part_rows=5_000_000):
    out.mkdir(parents=True, exist_ok=True)
    for k, i in enumerate(range(0, len(rows), part_rows)):
        rr = rows[i:i + part_rows]
        x = X[rr]
        cols = {"s1_id": pa.array(s1_ids[left[rr]]), "rec_id": pa.array(rec_ids[right[rr]]),
                "route": pa.array(route[rr])}
        if label is not None:
            cols["label"] = pa.array(label[rr])
        for j, name in enumerate(FEATURES):
            cols[name] = pa.array(x[:, j])
        cols["sg_p1"] = pa.array(p1[rr])
        cols["sg_prob"] = pa.array(prob[rr])
        pq.write_table(pa.table(cols), out / f"part-{k:03d}.parquet", compression="zstd")
        log(f"wrote part {k} ({len(rr)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["arena", "test"])
    ap.add_argument("--feat", required=True)
    ap.add_argument("--data")
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.what == "arena":
        pairs, X, s1, rec = load_feat(args.feat)
        left, right = pairs["left"], pairs["right"]
        part = np.load(Path(args.data) / "arena/finalize_v1/s1_partition.npy")
        rows = np.flatnonzero(part[left] == 1)
        p1 = np.load(Path(args.work) / "p1.npy")
        prob = np.load(Path(args.work) / "prob.npy")
        write_parts(Path(args.out), rows, s1.entity_id.to_numpy(), rec.entity_id.to_numpy(), left, right,
                    pairs["route"], X, p1, prob, label=pairs["label"])
    else:
        from ber.string_gbdt_apply import score
        pairs, X, s1, rec, left, right, p1, prob, _ = score(args.feat, args.work)
        rows = np.arange(len(left))
        write_parts(Path(args.out), rows, s1.entity_id.to_numpy(), rec.entity_id.to_numpy(), left, right,
                    pairs["route"], X, p1, prob)


if __name__ == "__main__":
    main()
