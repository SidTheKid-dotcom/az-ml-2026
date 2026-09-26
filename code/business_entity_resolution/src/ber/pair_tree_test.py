"""Apply the saved partition-A pair CatBoost to the test candidates.

Commands::

    python -m ber.pair_tree_test parity   # rebuild arena features, compare to saved probabilities
    python -m ber.pair_tree_test test --scores DIR --out DIR

``--scores`` holds ``index/left.npy``, ``index/right.npy``, ``probabilities.npy``,
``base_probabilities.npy`` and ``dense_test/{reference,pool}.parquet`` from
``artifacts/v1_test_scores``. Threshold and variant come from the arena report.
No labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostClassifier

from ber.arena_decision import name_only_rule
from ber.forensic_features import EMPTY
from ber.pair_tree_arena import ART, INDEX, OUT, build_pairs, decide, load_scores, log, predict

COLUMNS = ["business_name", "business_address", "country"]
HEADER = "source1_entity_id\tmatched_entity_ids"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model():
    model = CatBoostClassifier()
    model.load_model(str(OUT / "model.cbm"))
    return model


def parity() -> None:
    ref = pq.read_table(ART / "v1" / "reference.parquet", columns=COLUMNS).to_pandas()
    pool = pq.read_table(ART / "v1" / "pool.parquet", columns=COLUMNS).to_pandas()
    left = np.asarray(np.load(INDEX / "left.npy", mmap_mode="r")).astype(np.int64)
    right = np.asarray(np.load(INDEX / "right.npy", mmap_mode="r")).astype(np.int64)
    pairs = build_pairs(ref, pool, left, right, load_scores(ART / "v1" / "probabilities.npy"),
                        load_scores(ART / "v1" / "base_probabilities.npy"))
    prob = predict(load_model(), pairs, len(left))
    saved = np.load(OUT / "probabilities.npy")
    diff = float(np.max(np.abs(prob - saved)))
    result = {"max_abs_diff": diff, "pairs": int(len(prob)), "passed": diff < 1e-6}
    (OUT / "parity.json").write_text(json.dumps(result, indent=2) + "\n")
    log(result)
    if not result["passed"]:
        raise SystemExit("feature parity failed")


def test(scores: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    arena = json.loads((OUT / "report.json").read_text())
    chosen = arena["chosen_on_a_tune"]
    if not arena["passes_promotion_gate"]:
        raise SystemExit("arena model did not pass the promotion gate")
    ref_table = pq.read_table(scores / "dense_test" / "reference.parquet", columns=["entity_id"] + COLUMNS)
    pool_table = pq.read_table(scores / "dense_test" / "pool.parquet", columns=["entity_id"] + COLUMNS)
    s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object)
    rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref_table.column("country").to_pylist(), dtype=object)
    ref = ref_table.select(COLUMNS).to_pandas()
    pool = pool_table.select(COLUMNS).to_pandas()
    del ref_table, pool_table
    left = np.asarray(np.load(scores / "index" / "left.npy", mmap_mode="r")).astype(np.int64)
    right = np.asarray(np.load(scores / "index" / "right.npy", mmap_mode="r")).astype(np.int64)
    blend = load_scores(scores / "probabilities.npy")
    base = load_scores(scores / "base_probabilities.npy")
    if not (len(left) == len(right) == len(blend) == len(base)):
        raise SystemExit("test score arrays are not aligned")
    pairs = build_pairs(ref, pool, left, right, blend, base)
    del ref, pool
    prob = predict(load_model(), pairs, len(left))
    np.save(out / "probabilities.npy", prob)
    accept = decide(prob, left, right, chosen["threshold"], len(s1_ids))
    if chosen["variant"] == "model_then_name_only":
        accept, _ = name_only_rule(accept, left, right, pairs.rec_kind == EMPTY, pairs.name_exact,
                                   pairs.s1_mult[left])
    rows = np.flatnonzero(accept)
    rows = rows[np.lexsort((-prob[rows], left[rows]))]
    starts = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="left")
    stops = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="right")
    path = out / "matching_results.tsv"
    tmp = out / "matching_results.tsv.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for i, s1 in enumerate(s1_ids):
            handle.write(f"{s1}\t{','.join(rec_ids[right[rows[starts[i]:stops[i]]]])}\n")
    tmp.replace(path)
    lengths = stops - starts
    by_country = {}
    for code in np.unique(country):
        m = country == code
        by_country[str(code)] = {"s1": int(m.sum()), "links": int(lengths[m].sum()),
                                 "empty_rows": int((m & (lengths == 0)).sum()),
                                 "longest_list": int(lengths[m].max())}
    report = {
        "model": str(OUT / "model.cbm"), "model_sha256": sha256(OUT / "model.cbm"),
        "threshold": chosen["threshold"], "variant": chosen["variant"],
        "inputs": {name: sha256(scores / name) for name in
                   ("index/left.npy", "index/right.npy", "probabilities.npy", "base_probabilities.npy",
                    "dense_test/reference.parquet", "dense_test/pool.parquet")},
        "rows": int(len(s1_ids)), "links": int(len(rows)), "candidate_pairs_scored": int(len(pairs.cand_rows)),
        "contested_records": int((np.bincount(right[rows], minlength=len(rec_ids)) > 1).sum()),
        "longest_list": int(lengths.max()), "by_country": by_country,
        "output": str(path), "output_sha256": sha256(path), "labels_read": False,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("parity")
    t = sub.add_parser("test")
    t.add_argument("--scores", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "parity":
        parity()
    else:
        test(args.scores, args.out)


if __name__ == "__main__":
    main()
