"""C2 reranker scores → arena B_REST decision and a test matching file.

`evaluate`: blend p = w·C2 + (1−w)·tree on tree-gated pairs, one owner per
record, lists capped at 11; choose (w, threshold) on one half of B_REST
components and score the other half (both directions), compare with v1 blend
≥ 0.775 + one owner on the same S1s (component bootstrap), and report the chosen
rule with the empty-address name-only rule on top. SLICE and HOLDOUT-2 are not
read. `test`: apply a chosen (w, threshold) to test scores and write an LF
matching file with one owner, cap 11 and the name-only rule. No test labels exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule
from ber.forensic_features import EMPTY
from ber.pair_tree_arena import build_pairs, decide, false_merge, load_scores

COLUMNS = ["business_name", "business_address", "country"]
HEADER = "source1_entity_id\tmatched_entity_ids"
WEIGHTS = (0.5, 0.65, 0.75, 0.85, 1.0)
THRESHOLDS = tuple(np.round(np.arange(0.40, 0.96, 0.025), 3))


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def blend(c2, tree, gate, weight):
    gated = tree >= gate
    return np.where(gated, weight * c2 + (1 - weight) * tree, 0).astype(np.float32)


def evaluate_arena(args) -> dict:
    arena, labels, final = Path(args.arena), Path(args.labels), Path(args.finalize)
    summary = json.loads((labels / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("labels include HOLDOUT-2 rows")
    table = pq.read_table(arena / "pairs.parquet", columns=["left_idx", "right_idx"])
    left_all = table.column("left_idx").to_numpy().astype(np.int64)
    right_all = table.column("right_idx").to_numpy().astype(np.int64)
    del table
    partition = np.load(final / "s1_partition.npy")
    components = np.load(final / "component_ids.npy")
    population = partition == 1  # B_REST only
    rows = np.flatnonzero(population[left_all])
    left, right = left_all[rows], right_all[rows]
    label = np.load(labels / "label.npy", mmap_mode="r")[rows]
    true_count = np.load(labels / "true_count.npy")
    tree = load_scores(arena / "base_probabilities.npy")[rows]
    v1 = load_scores(arena / "probabilities.npy")[rows]
    c2 = load_scores(Path(args.scores) / "scores.npy")[rows]
    mask = np.load(Path(args.scores) / "scored_mask.npy")[rows]
    if np.any((tree >= args.gate) & ~mask):
        raise SystemExit("C2 scores miss gated B_REST pairs")
    country = pq.read_table(arena / "reference.parquet", columns=["country"]).column("country").to_numpy(zero_copy_only=False)
    n_s1 = len(true_count)
    log(f"B_REST: {int(population.sum())} S1s, {len(rows)} pairs, gated {int(mask.sum())}")

    def run(prob, threshold):
        accept = decide(prob, left, right, threshold, n_s1)
        result = evaluate(accept, left, label, true_count, population=population, country=country,
                          return_per_s1=True)
        return accept, result

    base_accept, base = run(v1, 0.775)
    half = (components % 2).astype(np.int8)
    grid = {}
    for w in WEIGHTS:
        prob = blend(c2, tree, args.gate, w)
        for t in THRESHOLDS:
            _, result = run(prob, t)
            per = np.asarray(result["per_s1"])
            grid[(w, float(t))] = [float(per[population & (half == h)].mean()) for h in (0, 1)]
        log(f"w={w} best half0 {max(grid[(w, float(t))][0] for t in THRESHOLDS):.6f}")
    picks = [max(grid, key=lambda key: grid[key][h]) for h in (0, 1)]
    per_cross = np.zeros(n_s1)
    for h, pick in zip((0, 1), picks):
        other = population & (half == 1 - h)
        _, result = run(blend(c2, tree, args.gate, pick[0]), pick[1])
        per_cross[other] = np.asarray(result["per_s1"])[other]
    per_base = np.asarray(base["per_s1"])
    cross = component_bootstrap(per_base, per_cross, components, population=population, seed=20260926)
    full_pick = max(grid, key=lambda key: sum(grid[key]))
    prob = blend(c2, tree, args.gate, full_pick[0])
    accept, chosen = run(prob, full_pick[1])
    report = {"scope": "arena B_REST (SLICE, HOLDOUT-2 unread)", "gate": args.gate,
              "baseline_v1_one_owner": {k: v for k, v in base.items() if k != "per_s1"},
              "baseline_false_merge": false_merge(base_accept, left, population, true_count),
              "crossfit_picks": [list(p) for p in picks],
              "crossfit_macro_f05": float(per_cross[population].mean()),
              "crossfit_vs_v1_bootstrap": cross,
              "chosen_on_full_b_rest": {"weight": full_pick[0], "threshold": full_pick[1]},
              "chosen_result": {k: v for k, v in chosen.items() if k != "per_s1"},
              "chosen_false_merge": false_merge(accept, left, population, true_count),
              "grid": {f"{w}:{t}": v for (w, t), v in grid.items()}}
    if args.name_only:
        ref = pq.read_table(arena / "reference.parquet", columns=COLUMNS).to_pandas()
        pool = pq.read_table(arena / "pool.parquet", columns=COLUMNS).to_pandas()
        pairs = build_pairs(ref, pool, left_all, right_all, load_scores(arena / "probabilities.npy"),
                            load_scores(arena / "base_probabilities.npy"))
        rec_empty = (pairs.rec_kind == EMPTY)[rows]
        exact, mult = pairs.name_exact[rows], pairs.s1_mult[left]
        for name, acc in (("v1", base_accept), ("c2", accept)):
            ruled, _ = name_only_rule(acc, left, right, rec_empty, exact, mult)
            result = evaluate(ruled, left, label, true_count, population=population, country=country,
                              return_per_s1=True)
            report[f"{name}_plus_name_only"] = {k: v for k, v in result.items() if k != "per_s1"}
            report[f"{name}_plus_name_only_false_merge"] = false_merge(ruled, left, population, true_count)
            report[f"_{name}_per"] = np.asarray(result["per_s1"])
        report["name_only_c2_vs_v1_bootstrap"] = component_bootstrap(
            report.pop("_v1_per"), report.pop("_c2_per"), components, population=population, seed=20260926)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps({k: report[k] for k in ("crossfit_macro_f05", "crossfit_vs_v1_bootstrap",
                                           "chosen_on_full_b_rest")}, default=float))
    return report


def build_test(args) -> dict:
    scores = Path(args.test_inputs)
    ref_table = pq.read_table(scores / "dense_test" / "reference.parquet", columns=["entity_id"] + COLUMNS)
    pool_table = pq.read_table(scores / "dense_test" / "pool.parquet", columns=["entity_id"] + COLUMNS)
    s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object)
    rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref_table.column("country").to_pylist(), dtype=object)
    ref, pool = ref_table.select(COLUMNS).to_pandas(), pool_table.select(COLUMNS).to_pandas()
    left = np.asarray(np.load(scores / "index" / "left.npy", mmap_mode="r")).astype(np.int64)
    right = np.asarray(np.load(scores / "index" / "right.npy", mmap_mode="r")).astype(np.int64)
    v1 = load_scores(scores / "probabilities.npy")
    tree = load_scores(scores / "base_probabilities.npy")
    c2 = load_scores(Path(args.scores) / "scores.npy")
    mask = np.load(Path(args.scores) / "scored_mask.npy")
    if not (len(left) == len(right) == len(tree) == len(c2)) or np.any((tree >= args.gate) & ~mask):
        raise SystemExit("test arrays misaligned or gated pairs unscored")
    prob = blend(c2, tree, args.gate, args.weight)
    accept = decide(prob, left, right, args.threshold, len(s1_ids))
    pairs = build_pairs(ref, pool, left, right, v1, tree)
    accept, info = name_only_rule(accept, left, right, pairs.rec_kind == EMPTY, pairs.name_exact, pairs.s1_mult[left])
    rows = np.flatnonzero(accept)
    rows = rows[np.lexsort((-prob[rows], left[rows]))]
    starts = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="left")
    stops = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="right")
    lengths = stops - starts
    if lengths.max() > 11:  # name-only additions can exceed the cap; keep the 11 best
        keep = np.ones(len(rows), dtype=bool)
        for i in np.flatnonzero(lengths > 11):
            keep[starts[i] + 11:stops[i]] = False
        rows = rows[keep]
        starts = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="left")
        stops = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="right")
        lengths = stops - starts
    contested = int((np.bincount(right[rows], minlength=len(rec_ids)) > 1).sum())
    if contested:
        raise SystemExit(f"{contested} contested records")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    tmp, path = out / "matching_results.tsv.tmp", out / "matching_results.tsv"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for i, s1 in enumerate(s1_ids):
            handle.write(f"{s1}\t{','.join(rec_ids[right[rows[starts[i]:stops[i]]]])}\n")
    tmp.replace(path)
    by_country = {str(c): {"s1": int((country == c).sum()), "links": int(lengths[country == c].sum()),
                           "empty_rows": int(((country == c) & (lengths == 0)).sum()),
                           "mean_list": float(lengths[country == c].mean())} for c in np.unique(country)}
    report = {"weight": args.weight, "threshold": args.threshold, "gate": args.gate,
              "c2_scores": str(args.scores), "rows": int(len(s1_ids)), "links": int(len(rows)),
              "contested_records": contested, "longest_list": int(lengths.max()),
              "by_country": by_country, "name_only": {k: v for k, v in info.items() if not hasattr(v, "shape")},
              "labels_read": False, "output": str(path)}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps(report, default=float))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--arena", required=True, help="Dir with pairs/reference/pool parquet and v1 probabilities")
    e.add_argument("--labels", required=True)
    e.add_argument("--finalize", required=True, help="Dir with s1_partition.npy and component_ids.npy")
    e.add_argument("--scores", required=True, help="score_external output on arena B_REST")
    e.add_argument("--gate", type=float, default=.01)
    e.add_argument("--name-only", action="store_true")
    e.add_argument("--output", required=True)
    t = sub.add_parser("test")
    t.add_argument("--test-inputs", required=True)
    t.add_argument("--scores", required=True, help="score_external output on test pairs")
    t.add_argument("--weight", type=float, required=True)
    t.add_argument("--threshold", type=float, required=True)
    t.add_argument("--gate", type=float, default=.01)
    t.add_argument("--output", required=True)
    args = parser.parse_args()
    (evaluate_arena if args.command == "evaluate" else build_test)(args)


if __name__ == "__main__":
    main()
