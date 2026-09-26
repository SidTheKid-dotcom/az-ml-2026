"""Learned ranker for union pairs (beyond dense top-50), fitted on arena partition A only.

`fit`: label the A union pairs with training truth for A S1 IDs only, fit a
CatBoost classifier on label-free pair features (text similarities, branch
flags, posting sizes, per-S1 union size, text rank), save the model.
`apply`: score a full union (B_REST or test), re-rank per S1 by model
probability and write a mined directory (top ``--keep`` per S1) that
`ber.union_mine evaluate/export` read unchanged. B_REST labels are never used
for fitting; SLICE and HOLDOUT-2 are not read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from ber.union_mine import BRANCHES, b_rest_truth, log

PAIR_COLUMNS = ["score", "name_sim", "address_sim", "name_ratio", "name_tset", "name_partial", "address_ratio",
                "address_tsort", "s1_name_len", "rec_name_len", "s1_addr_len", "rec_addr_len"]


def features(mined: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairs = pd.read_parquet(mined / "new_pairs.parquet")
    reference = pd.read_parquet(mined / "reference.parquet", columns=["entity_id", "country"])
    postings = np.load(mined / "s1_postings.npy")
    s1 = pairs.s1.to_numpy()
    x = pairs[PAIR_COLUMNS].astype(np.float32).copy()
    bits = pairs.bits.to_numpy()
    for i, b in enumerate(BRANCHES):
        x[f"hit_{b}"] = ((bits >> i) & 1).astype(np.float32)
        x[f"post_{b}"] = np.log1p(postings[s1, i]).astype(np.float32)
    x["hits"] = x[[f"hit_{b}" for b in BRANCHES]].sum(axis=1)
    x["text_rank"] = pairs["rank"].to_numpy().astype(np.float32)
    x["s1_new_pairs"] = np.bincount(s1, minlength=len(reference))[s1].astype(np.float32)
    x["is_india"] = (reference.country.to_numpy()[s1] == "India").astype(np.float32)
    x["name_minus_addr"] = x.name_sim - x.address_sim
    return pairs, x


def pool_ids(raw: Path) -> pd.Series:
    return pd.concat([pd.read_csv(raw / f"train_source{s}.tsv", sep="\t", dtype=str, keep_default_na=False,
                                  usecols=["entity_id"]) for s in (2, 3)], ignore_index=True).entity_id


def fit_cmd(args) -> dict:
    from catboost import CatBoostClassifier
    started = time.time()
    mined, out = Path(args.mined), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    reference = pd.read_parquet(mined / "reference.parquet", columns=["entity_id", "arena_idx"])
    partition = np.load(Path("/workspace/az-ml-new/arena/finalize_v1/s1_partition.npy"))
    if not np.all(partition[reference.arena_idx.to_numpy()] == 0):
        raise SystemExit("ranker must be fitted on partition A only")
    pairs, x = features(mined)
    owners = b_rest_truth(Path(args.raw), reference)  # truth for these (A) S1 IDs only
    owner_raw = pool_ids(Path(args.raw)).map(owners).fillna(-1).to_numpy().astype(np.int64)
    y = owner_raw[pairs.raw.to_numpy()] == pairs.s1.to_numpy()
    log(f"A union pairs {len(y):,}, positives {int(y.sum()):,}")
    rng = np.random.default_rng(20260926)
    neg = np.flatnonzero(~y)
    take = np.concatenate([np.flatnonzero(y), rng.choice(neg, size=min(len(neg), args.negatives), replace=False)])
    groups = pairs.s1.to_numpy()[take]
    valid = (groups % 10) == 0  # S1-disjoint early-stopping split inside A
    model = CatBoostClassifier(iterations=args.iterations, depth=8, learning_rate=0.08, loss_function="Logloss",
                               thread_count=args.threads, random_seed=20260926, verbose=100,
                               od_type="Iter", od_wait=50)
    model.fit(x.iloc[take[~valid]], y[take[~valid]], eval_set=(x.iloc[take[valid]], y[take[valid]]))
    model.save_model(str(out / "ranker.cbm"))
    importance = dict(zip(x.columns, map(float, model.get_feature_importance())))
    report = {"fit_on": "arena partition A union pairs", "pairs": int(len(y)), "positives": int(y.sum()),
              "rows_used": int(len(take)), "negatives_sampled": int(min(len(neg), args.negatives)),
              "best_iteration": int(model.get_best_iteration()), "features": list(x.columns),
              "importance": importance, "seconds": time.time() - started}
    (out / "fit.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({k: report[k] for k in ("pairs", "positives", "best_iteration", "seconds")}))
    return report


def apply_cmd(args) -> dict:
    from catboost import CatBoostClassifier
    mined, out = Path(args.mined), Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model = CatBoostClassifier()
    model.load_model(str(Path(args.model) / "ranker.cbm"))
    pairs, x = features(mined)
    prob = model.predict_proba(x, thread_count=args.threads)[:, 1].astype(np.float32)
    s1, raw = pairs.s1.to_numpy(), pairs.raw.to_numpy()
    order = np.lexsort((raw, -prob, s1))
    first = np.ones(len(order), dtype=bool)
    first[1:] = s1[order][1:] != s1[order][:-1]
    group_start = np.maximum.accumulate(np.where(first, np.arange(len(order)), 0))
    rank = (np.arange(len(order)) - group_start).astype(np.int32)
    keep = rank < args.keep
    frame = pairs.iloc[order[keep]].reset_index(drop=True)
    frame["text_rank"] = frame["rank"]
    frame["rank"] = rank[keep]
    frame["text_score"] = frame["score"]
    frame["score"] = prob[order[keep]]
    frame.to_parquet(out / "new_pairs.parquet", index=False)
    shutil.copyfile(mined / "reference.parquet", out / "reference.parquet")
    shutil.copyfile(mined / "s1_postings.npy", out / "s1_postings.npy")
    per_k = {k: int((frame["rank"] < k).sum()) for k in (1, 2, 3, 5, 10) if k <= args.keep}
    report = {"source": str(mined), "model": str(args.model), "pairs_scored": int(len(prob)),
              "kept": int(len(frame)), "keep": args.keep, "new_pairs_by_k": per_k,
              "s1_with_new": int(frame.s1.nunique()), "truth_read": False}
    (out / "mine.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--mined", required=True, help="union_mine --set a --features --keep 0 output")
    f.add_argument("--raw", required=True)
    f.add_argument("--negatives", type=int, default=4_000_000)
    f.add_argument("--iterations", type=int, default=600)
    f.add_argument("--threads", type=int, default=8)
    f.add_argument("--output", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--mined", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--keep", type=int, default=10)
    a.add_argument("--threads", type=int, default=8)
    a.add_argument("--output", required=True)
    args = parser.parse_args()
    (fit_cmd if args.command == "fit" else apply_cmd)(args)


if __name__ == "__main__":
    main()
