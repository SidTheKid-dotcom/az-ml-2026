"""Merge C2-scored union pairs (beyond the dense top-50) into an existing dense decision.

Rule (both modes): the dense decision is fixed. A new pair is accepted when its
C2 probability ≥ t, its record is not already accepted by any dense link, it
wins its record among new claimants (highest C2; ties lower S1), and the S1's
list stays ≤ 11 (dense links first, then new links by C2). Optional ``--k``
keeps only new pairs with union rank < k.

`arena`: B_REST only (SLICE and HOLDOUT-2 unread). The dense decision is the
blend w·C2 + (1−w)·tree on tree-gated pairs ≥ threshold, one owner, cap 11
(``--dense-scores``, default step-10k), or a supplied accept mask
(``--dense-accept``, aligned to arena pairs or to ``--dense-rows``). t is
chosen on one half of the B_REST claim components and scored on the other
(both directions); the cross-fitted merge is compared with dense-only through a
component bootstrap. Every true link counts through ``true_count``, so misses
lower the score.

`test`: reads the dense matching file (test order), adds new links with a
fixed t, and writes matching_results.tsv plus candidate_pairs.tsv = dense ∪ new
(LF, official headers), then re-reads both and checks row order, duplicates,
one owner, list cap and matches ⊆ candidates. No test labels exist.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ARENA = Path("/workspace/az-ml-new/arena")
MATCH_HEADER = "source1_entity_id\tmatched_entity_ids"
CAND_HEADER = "source1_entity_id\tcandidate_entity_ids"
CAP = 11
THRESHOLDS = tuple(np.round(np.r_[np.arange(0.05, 0.90, 0.05), 0.9, 0.925, 0.95, 0.97, 0.98, 0.99, 0.995, 2.0], 4))


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def rank_within(group, score):
    order = np.lexsort((np.arange(len(score)), -score, group))
    g = group[order]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    idx = np.arange(len(g))
    first = np.maximum.accumulate(np.where(start, idx, 0))
    rank = np.empty(len(g), dtype=np.int64)
    rank[order] = idx - first
    return rank


def add_new(s1, rec_key, c2, eligible, owned, dense_count, threshold):
    """Accept mask over new pairs: C2 ≥ t, record not dense-owned, one owner by C2, list cap."""
    cand = eligible & (c2 >= threshold) & ~owned
    rows = np.flatnonzero(cand)
    accept = np.zeros(len(s1), dtype=bool)
    if not rows.size:
        return accept
    # One owner among new claimants: highest C2, ties lower S1.
    order = np.lexsort((s1[rows], -c2[rows], rec_key[rows]))
    r = rec_key[rows][order]
    first = np.ones(len(r), dtype=bool)
    first[1:] = r[1:] != r[:-1]
    rows = rows[order[first]]
    rank = rank_within(s1[rows], c2[rows])
    rows = rows[dense_count[s1[rows]] + rank < CAP]
    accept[rows] = True
    return accept


def arena_cmd(args) -> dict:
    from ber.arena_decision import component_bootstrap, entity_f05
    from ber.pair_tree_arena import decide, load_scores
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    partition = np.load(ARENA / "finalize_v1" / "s1_partition.npy")
    population = partition == 1
    components = np.load(ARENA / "finalize_v1" / "component_ids.npy")
    true_count = np.load(ARENA / "labels" / "true_count.npy").astype(np.int64)
    n_s1 = len(true_count)
    table = pq.read_table(ARENA / "v1" / "pairs.parquet", columns=["left_idx", "right_raw_idx"])
    left_all = table.column("left_idx").to_numpy().astype(np.int64)
    rows = np.flatnonzero(population[left_all])
    left = left_all[rows]
    rraw = table.column("right_raw_idx").to_numpy().astype(np.int64)[rows]
    del table, left_all
    label = np.asarray(np.load(ARENA / "labels" / "label.npy", mmap_mode="r")[rows])
    if args.dense_accept or args.dense_prob:
        acc = np.load(args.dense_accept or args.dense_prob)
        if args.dense_rows:
            full = np.zeros(int(np.load(ARENA / "labels" / "label.npy", mmap_mode="r").shape[0]), dtype=acc.dtype)
            full[np.load(args.dense_rows)] = acc
            acc = full
        acc = np.asarray(acc[rows])
        if args.dense_accept:
            dense_accept = acc.astype(bool)
            dense_rule = {"accept_mask": args.dense_accept, "rows": args.dense_rows}
        else:  # probability (e.g. pair-tree OOF): threshold, one owner, cap 11
            dense_accept = decide(acc.astype(np.float32), left, rraw, args.threshold, n_s1)
            dense_rule = {"prob": args.dense_prob, "rows": args.dense_rows, "threshold": args.threshold}
    else:
        tree = load_scores(ARENA / "v1" / "base_probabilities.npy")[rows]
        c2d = load_scores(Path(args.dense_scores) / "scores.npy")[rows]
        mask = np.load(Path(args.dense_scores) / "scored_mask.npy")[rows]
        if np.any((tree >= args.gate) & ~mask):
            raise SystemExit("dense C2 scores miss gated pairs")
        dense_prob = np.where(tree >= args.gate, args.weight * c2d + (1 - args.weight) * tree, 0).astype(np.float32)
        dense_accept = decide(dense_prob, left, rraw, args.threshold, n_s1)
        dense_rule = {"scores": args.dense_scores, "weight": args.weight, "threshold": args.threshold, "gate": args.gate}
    k_dense = np.bincount(left[dense_accept], minlength=n_s1)
    tp_dense = np.bincount(left[dense_accept & label], minlength=n_s1)
    owned_records = np.unique(rraw[dense_accept])
    per_dense = entity_f05(tp_dense, k_dense, true_count)
    log(f"dense-only B_REST macro F0.5 {per_dense[population].mean():.6f}")

    setdir = Path(args.set)
    ref = pd.read_parquet(setdir / "reference.parquet", columns=["arena_idx", "country"])
    pairs = pd.read_parquet(setdir / "pairs.parquet", columns=["s1", "raw", "rank"])
    new_label = np.load(setdir / "eval_label.npy")
    c2 = np.load(Path(args.new_scores) / "scores.npy").astype(np.float32)
    if not (len(c2) == len(pairs) == len(new_label)):
        raise SystemExit("new-pair scores/labels are not aligned to the exported set")
    s1 = ref.arena_idx.to_numpy()[pairs.s1.to_numpy()]
    if not population[s1].all():
        raise SystemExit("new pairs outside B_REST")
    nraw = pairs.raw.to_numpy().astype(np.int64)
    owned = np.isin(nraw, owned_records)
    half = (components % 2).astype(np.int8)

    def merged(threshold, k):
        acc = add_new(s1, nraw, c2, pairs["rank"].to_numpy() < k, owned, k_dense, threshold)
        kk = k_dense + np.bincount(s1[acc], minlength=n_s1)
        tp = tp_dense + np.bincount(s1[acc & new_label], minlength=n_s1)
        return acc, entity_f05(tp, kk, true_count)

    ref_country = pd.read_parquet(ARENA / "v1" / "reference.parquet", columns=["country"]).country.to_numpy()
    ks = sorted(set(args.ks))
    report = {"scope": "arena B_REST (SLICE, HOLDOUT-2 unread)", "dense_rule": dense_rule,
              "dense_only_macro_f05": float(per_dense[population].mean()),
              "dense_only_links": int(k_dense[population].sum()), "new_pairs": int(len(c2)),
              "new_pairs_owned_by_dense": int(owned.sum()), "by_k": {}}
    for k in ks:
        grid = {}
        for t in THRESHOLDS:
            _, per = merged(t, k)
            grid[float(t)] = [float(per[population & (half == h)].mean()) for h in (0, 1)]
        picks = [max(grid, key=lambda t: (grid[t][h], t)) for h in (0, 1)]
        per_cross = per_dense.copy()
        for h, t in zip((0, 1), picks):
            _, per = merged(t, k)
            other = population & (half == 1 - h)
            per_cross[other] = per[other]
        boot = component_bootstrap(per_dense, per_cross, components, n=args.bootstrap, population=population,
                                   seed=20260926)
        full_t = max(grid, key=lambda t: (sum(grid[t]), t))
        acc, per_full = merged(full_t, k)
        by_country = {}
        for c in np.unique(ref_country[s1]):
            m = acc & (ref_country[s1] == c)
            by_country[str(c)] = {"new_links": int(m.sum()), "new_true": int((m & new_label).sum())}
        report["by_k"][str(k)] = {
            "crossfit_picks": picks, "crossfit_macro_f05": float(per_cross[population].mean()),
            "crossfit_vs_dense": boot, "full_b_rest_threshold": full_t,
            "full_threshold_macro_f05": float(per_full[population].mean()),
            "full_threshold_new_links": int(acc.sum()), "full_threshold_new_true": int((acc & new_label).sum()),
            "full_threshold_new_precision": float((acc & new_label).sum() / max(1, acc.sum())),
            "new_links_to_singletons": int((acc & (true_count[s1] == 0)).sum()),
            "by_country": by_country,
            "grid": {str(t): v for t, v in grid.items()}}
        log(f"k={k} picks {picks} crossfit {per_cross[population].mean():.6f} delta {boot['delta']:+.6f} "
            f"CI [{boot['ci_low']:+.6f}, {boot['ci_high']:+.6f}]; full t={full_t} new {int(acc.sum())} "
            f"true {int((acc & new_label).sum())}")
    report["seconds"] = time.time() - started
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    return report


def _read_rows(path: Path, header: str, s1_ids: np.ndarray) -> list[list[str]]:
    raw = path.read_bytes()
    if b"\r" in raw:
        raise SystemExit(f"{path}: CR bytes present")
    lines = raw.decode("utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if lines[0] != header:
        raise SystemExit(f"{path}: bad header")
    body = lines[1:]
    if len(body) != len(s1_ids):
        raise SystemExit(f"{path}: {len(body)} rows, expected {len(s1_ids)}")
    result = []
    for i, line in enumerate(body):
        sid, tab, payload = line.partition("\t")
        if not tab or sid != s1_ids[i]:
            raise SystemExit(f"{path}: row {i} malformed or out of order")
        result.append(payload.split(",") if payload else [])
    return result


def test_cmd(args) -> dict:
    started = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ti = Path(args.test_inputs)
    s1_ids = np.asarray(pq.read_table(ti / "dense_test" / "reference.parquet", columns=["entity_id"])
                        .column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(pq.read_table(ti / "dense_test" / "reference.parquet", columns=["country"])
                         .column("country").to_pylist(), dtype=object)
    n = len(s1_ids)
    dense_rows = _read_rows(Path(args.dense_matching), MATCH_HEADER, s1_ids)
    dense_count = np.fromiter((len(r) for r in dense_rows), dtype=np.int64, count=n)
    owned_ids = pd.Index([x for r in dense_rows for x in r])
    if owned_ids.has_duplicates:
        raise SystemExit("dense matching file has contested records")
    log(f"dense matching: {int(dense_count.sum()):,} links")

    setdir = Path(args.set)
    set_ref = pq.read_table(setdir / "reference.parquet", columns=["entity_id"]).column("entity_id").to_pylist()
    if len(set_ref) != n or any(a != b for a, b in zip(set_ref, s1_ids)):
        raise SystemExit("exported test set reference is not the test S1 order")
    new_pool_ids = np.asarray(pq.read_table(setdir / "pool.parquet", columns=["entity_id"])
                              .column("entity_id").to_pylist(), dtype=object)
    left = np.load(setdir / "left.npy").astype(np.int64)
    right = np.load(setdir / "right.npy").astype(np.int64)
    rank = pd.read_parquet(setdir / "pairs.parquet", columns=["rank"])["rank"].to_numpy()
    c2 = np.load(Path(args.new_scores) / "scores.npy").astype(np.float32)
    if not (len(left) == len(right) == len(rank) == len(c2)):
        raise SystemExit("new-pair arrays are not aligned")
    eligible = rank < args.k
    owned = pd.Index(new_pool_ids).isin(owned_ids)[right]
    acc = add_new(left, right, c2, eligible, owned, dense_count, args.threshold)
    log(f"new links accepted: {int(acc.sum()):,} of {int(eligible.sum()):,} eligible new pairs")

    # matching_results.tsv: dense list, then new links by C2.
    rows = np.flatnonzero(acc)
    rows = rows[np.lexsort((-c2[rows], left[rows]))]
    starts = np.searchsorted(left[rows], np.arange(n), side="left")
    stops = np.searchsorted(left[rows], np.arange(n), side="right")
    match_path, tmp = out / "matching_results.tsv", out / "matching_results.tsv.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(MATCH_HEADER + "\n")
        for i in range(n):
            ids = dense_rows[i] + list(new_pool_ids[right[rows[starts[i]:stops[i]]]])
            handle.write(f"{s1_ids[i]}\t{','.join(ids)}\n")
    tmp.replace(match_path)

    # candidate_pairs.tsv = dense candidates ∪ eligible new pairs, sorted IDs per S1.
    dense_pool_ids = np.asarray(pq.read_table(ti / "dense_test" / "pool.parquet", columns=["entity_id"])
                                .column("entity_id").to_pylist(), dtype=object)
    dl = np.load(ti / "index" / "left.npy").astype(np.int64)
    dr = np.load(ti / "index" / "right.npy").astype(np.int64)
    all_ids = pd.Index(np.concatenate([dense_pool_ids, new_pool_ids])).unique().sort_values()
    code_dense = all_ids.get_indexer(dense_pool_ids)
    code_new = all_ids.get_indexer(new_pool_ids)
    cand_s1 = np.concatenate([dl, left[eligible]])
    cand_code = np.concatenate([code_dense[dr], code_new[right[eligible]]])
    del dl, dr
    order = np.lexsort((cand_code, cand_s1))
    cand_s1, cand_code = cand_s1[order], cand_code[order]
    key = cand_s1 * len(all_ids) + cand_code
    if np.any(key[1:] == key[:-1]):
        raise SystemExit("duplicate candidate pairs after union")
    id_array = np.asarray(all_ids, dtype=object)
    cstarts = np.searchsorted(cand_s1, np.arange(n), side="left")
    cstops = np.searchsorted(cand_s1, np.arange(n), side="right")
    cand_path, tmp = out / "candidate_pairs.tsv", out / "candidate_pairs.tsv.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(CAND_HEADER + "\n")
        for i in range(n):
            handle.write(f"{s1_ids[i]}\t{','.join(id_array[cand_code[cstarts[i]:cstops[i]]])}\n")
    tmp.replace(cand_path)
    log("files written; validating")

    # Validation by re-reading both files.
    match_rows = _read_rows(match_path, MATCH_HEADER, s1_ids)
    cand_rows = _read_rows(cand_path, CAND_HEADER, s1_ids)
    lengths = np.fromiter((len(r) for r in match_rows), dtype=np.int64, count=n)
    problems = []
    flat = [x for r in match_rows for x in r]
    if len(flat) != len(set(flat)):
        problems.append("contested or duplicate record IDs in matches")
    if lengths.max() > CAP:
        problems.append(f"longest list {int(lengths.max())} > {CAP}")
    outside = sum(1 for m, c in zip(match_rows, cand_rows) if m and not set(m) <= set(c))
    if outside:
        problems.append(f"{outside} S1 rows with matches outside candidates")
    dup_cand = sum(1 for c in cand_rows if len(c) != len(set(c)))
    if dup_cand:
        problems.append(f"{dup_cand} candidate rows with duplicate IDs")
    by_country = {str(c): {"s1": int((country == c).sum()), "links": int(lengths[country == c].sum()),
                           "new_links": int(np.bincount(left[acc], minlength=n)[country == c].sum()),
                           "empty_rows": int(((country == c) & (lengths == 0)).sum())}
                  for c in np.unique(country)}
    import hashlib
    report = {"threshold": args.threshold, "k": args.k, "dense_matching": str(args.dense_matching),
              "new_scores": str(args.new_scores), "rows": n, "dense_links": int(dense_count.sum()),
              "new_links": int(acc.sum()), "links": int(lengths.sum()), "longest_list": int(lengths.max()),
              "candidate_pairs": int(len(cand_s1)), "new_candidate_pairs": int(eligible.sum()),
              "by_country": by_country, "problems": problems, "passed": not problems, "labels_read": False,
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (match_path, cand_path)},
              "seconds": time.time() - started}
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    log(json.dumps({k: report[k] for k in ("new_links", "links", "longest_list", "candidate_pairs", "passed",
                                           "problems")}))
    if problems:
        raise SystemExit(f"validation failed: {problems}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("arena")
    a.add_argument("--set", required=True, help="union_mine export dir for B_REST (with eval_label.npy)")
    a.add_argument("--new-scores", required=True, help="score_external output on the exported B_REST set")
    a.add_argument("--dense-scores", default="/opt/az-ml-new/runs/c2_step010000/b_rest_scores")
    a.add_argument("--weight", type=float, default=0.9)
    a.add_argument("--threshold", type=float, default=0.10)
    a.add_argument("--gate", type=float, default=0.01)
    a.add_argument("--dense-accept", help="bool .npy dense accept mask (arena pair rows or --dense-rows)")
    a.add_argument("--dense-prob", help="float .npy dense probability (e.g. pair-tree OOF); uses --threshold")
    a.add_argument("--dense-rows", help="arena pair row indices of --dense-accept/--dense-prob")
    a.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 5, 10])
    a.add_argument("--bootstrap", type=int, default=2000)
    a.add_argument("--output", required=True)
    t = sub.add_parser("test")
    t.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
    t.add_argument("--dense-matching", required=True, help="dense-decision matching_results.tsv (test order)")
    t.add_argument("--set", required=True, help="union_mine export dir for test")
    t.add_argument("--new-scores", required=True)
    t.add_argument("--threshold", type=float, required=True)
    t.add_argument("--k", type=int, default=10 ** 6)
    t.add_argument("--output", required=True)
    args = parser.parse_args()
    (arena_cmd if args.command == "arena" else test_cmd)(args)


if __name__ == "__main__":
    main()
