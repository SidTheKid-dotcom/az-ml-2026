"""Pair CatBoost of :mod:`ber.pair_tree_c2` plus :mod:`ber.collective_features` (arena B_REST cross-fit).

Commands::

    python -m ber.collective_pair_tree evaluate --work DIR
    python -m ber.collective_pair_tree test --work DIR --out DIR

``evaluate`` reuses the baseline work dir's candidate rows, splits and saved OOF (variant picks
from its ``crossfit_state.json``) as the paired baseline, adds the collective features, redoes the
same 2-fold component cross-fit (inner-OOF thresholds), reports signal truth-rate tables, and a
component bootstrap against the baseline. ``test`` fits on every B_REST candidate row and writes
the S1 composition: this model for US/India, France rows from the plain C2 blend + name-only (as
the 0.978287 upload). B_REST labels only; SLICE and HOLDOUT-2 are never read.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.collective_features import FEATURES as COLLECTIVE, build_tables, pair_features
from ber.pair_tree_c2 import (COLUMNS, GATE, VARIANTS, accept_per_row, by_country, c2_blend, c2_features,
                              c2_on_rows, crossfit, decode, feature_names, fit_catboost, load_arena, log, proba,
                              sha256, slim, v1_cache, validate_tsv, write_tsv)
from ber.arena_decision import component_bootstrap

BASE_F05 = 0.9855111442596571


def _base_args(args):
    return argparse.Namespace(arena=args.arena, labels=args.labels, finalize=args.finalize, work=args.base_work,
                              cache=args.cache, scores=args.scores, extended=bool(getattr(args, "extended", False)))


def _collective(path, ref_dir, left, right):
    path = Path(path)
    if path.exists():
        return np.load(path)
    ref = pq.read_table(Path(ref_dir) / "reference.parquet", columns=["business_name", "business_address"])
    pool = pq.read_table(Path(ref_dir) / "pool.parquet", columns=["business_name", "business_address"])
    tables = build_tables(ref.column("business_name"), ref.column("business_address"),
                          pool.column("business_name"), pool.column("business_address"), log=log)
    del ref, pool
    log(f"collective pair features for {len(left)} rows")
    xc = pair_features(tables, left, right)
    np.save(path, xc)
    return xc


def _bucket(mask, y, base_acc):
    n = int(mask.sum())
    rej = mask & ~base_acc
    return {"n": n, "true_rate": round(float(y[mask].mean()), 4) if n else None,
            "base_accept_rate": round(float(base_acc[mask].mean()), 4) if n else None,
            "n_base_rejected": int(rej.sum()), "true_rate_base_rejected": round(float(y[rej].mean()), 4) if rej.any() else None,
            "true_rate_base_accepted": round(float(y[mask & base_acc].mean()), 4) if (mask & base_acc).any() else None}


def signals(xc, y, rec_empty, base_acc):
    f = {name: xc[:, i] for i, name in enumerate(COLLECTIVE)}
    out = {"empty_address": {}, "name_disjoint_at_key": {}, "bridges": {}}
    e = rec_empty
    out["empty_address"]["all"] = _bucket(e, y, base_acc)
    for ce in (0, 1):
        for at in (0, 1):
            for off in (0, 1):
                for oth in (0, 1):
                    m = e & ((f["core_eq"] > 0) == bool(ce)) & ((f["sib_at_s1_key"] > 0) == bool(at)) \
                        & (((f["sib_off_s1_key"] > 0) | (f["orphan_keys"] > 0)) == bool(off)) \
                        & ((f["s1_core_other"] > 0) == bool(oth))
                    out["empty_address"][f"core_eq={ce}|sib_at={at}|off_or_orphan={off}|other_s1_core={oth}"] = _bucket(m, y, base_acc)
    d = (f["core_jaccard"] == 0) & (f["core_empty"] == 0) & (f["key_eq"] > 0)
    out["name_disjoint_at_key"]["all"] = _bucket(d, y, base_acc)
    for lo, hi in ((0, 1), (1, 2), (2, 4), (4, 1e9)):
        n_name = np.expm1(f["log1p_pool_name"])
        m = d & (n_name > lo) & (n_name <= hi)
        out["name_disjoint_at_key"][f"pool_same_name in ({lo},{hi}]"] = _bucket(m, y, base_acc)
    n_core = np.expm1(f["log1p_pool_core"])
    for lo, hi in ((0, 1), (1, 2), (2, 1e9)):
        m = d & (n_core > lo) & (n_core <= hi)
        out["name_disjoint_at_key"][f"pool_same_core in ({lo},{hi}]"] = _bucket(m, y, base_acc)
    for k in (0, 1):
        out["name_disjoint_at_key"][f"alias_bridged={k}"] = _bucket(d & ((f["alias_bridged_to_s1"] > 0) == bool(k)), y, base_acc)
    out["bridges"]["rec_is_bridge"] = _bucket(f["rec_is_bridge"] > 0, y, base_acc)
    out["bridges"]["rec_bridge_s1"] = _bucket(f["rec_bridge_s1"] > 0, y, base_acc)
    out["bridges"]["alias_bridged_to_s1"] = _bucket(f["alias_bridged_to_s1"] > 0, y, base_acc)
    return out


def evaluate_cmd(args) -> dict:
    work, base_work = Path(args.work), Path(args.base_work)
    work.mkdir(parents=True, exist_ok=True)
    data = load_arena(_base_args(args))
    arena, x, y = data["arena"], data["x"], data["label"]
    if not np.array_equal(np.load(base_work / "b_rest_pair_rows.npy"), data["pair_rows"]):
        raise SystemExit("candidate rows differ from the baseline work dir")
    component, population = data["component"], arena.population
    comp_row = component[arena.left]
    half_row, inner_row = (comp_row % 2).astype(np.int8), ((comp_row // 2) % 2).astype(np.int8)
    s1_half = (component % 2).astype(np.int8)

    state = json.loads((base_work / "crossfit_state.json").read_text())
    oof_b = np.load(base_work / "oof.npy")
    per_base, base_results, base_acc = {}, {}, {}
    for v in VARIANTS:
        thr = np.where(half_row == 1, state["picks"]["0"][v]["threshold"], state["picks"]["1"][v]["threshold"])
        base_acc[v] = accept_per_row(arena, oof_b, thr, v)
        result = arena.score(base_acc[v], full=True)
        per_base[v] = result.pop("per_s1")
        base_results[v] = slim(result)
    log(f"baseline model {base_results['model']['macro_f05']:.6f}")
    if abs(base_results["model"]["macro_f05"] - BASE_F05) > 1e-9:
        raise SystemExit("baseline parity failed")

    xc = _collective(work / "b_rest_collective.npy", args.arena, arena.left, arena.right)
    sig = signals(xc, y, arena.rec_empty, base_acc["model"])
    (work / "signals.json").write_text(json.dumps(sig, indent=1) + "\n")
    log("signals written")

    xn = np.hstack([x, xc]).astype(np.float32)
    names = feature_names(args.extended) + COLLECTIVE
    fit = lambda xs, ys: fit_catboost(xs, ys, args.threads)  # noqa: E731
    oof, picks, models, info = crossfit(xn, y, arena, half_row, inner_row, s1_half, fit)
    for h, model in models.items():
        model.save_model(str(work / f"model_half{h}.cbm"))
    np.save(work / "oof.npy", oof)
    results, per, boot = {}, {}, {}
    for v in VARIANTS:
        thr = np.where(half_row == 1, picks[0][v]["threshold"], picks[1][v]["threshold"])
        result = arena.score(accept_per_row(arena, oof, thr, v), full=True)
        per[v] = result.pop("per_s1")
        results[v] = slim(result)
        log(f"cross-fit {v}: {result['macro_f05']:.6f}")
        for bv in VARIANTS:
            boot[f"{v}_minus_base_{bv}"] = component_bootstrap(per_base[bv], per[v], component, n=args.bootstrap,
                                                               seed=20260926, population=population)
    grid_full, best_full = arena.sweep(oof, None, population)
    importance = {str(h): dict(zip(names, map(float, m.get_feature_importance()))) for h, m in models.items()}
    report = {"question": "pair_tree_c2 features + collective features vs pair_tree_c2 (paired, same splits)",
              "scope": "arena B_REST only; SLICE and HOLDOUT-2 unread", "features": names,
              "base_work": str(base_work), "baseline": base_results, "crossfit_picks": {str(h): picks[h] for h in picks},
              "crossfit": results, "bootstrap": boot,
              "preferred_variant": max(VARIANTS, key=lambda v: results[v]["macro_f05"]),
              "full_oof_threshold": {v: best_full[v] for v in VARIANTS},
              "full_oof_f05_at_threshold": {v: grid_full[v][best_full[v]] for v in VARIANTS},
              "fold_info": {str(h): info[h] for h in info}, "feature_importance": importance,
              "extended": bool(args.extended), "c2_scores": str(Path(args.scores).resolve()),
              "holdout2_rows_read": 0, "slice_read": False}
    (work / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps({k: report[k] for k in ("baseline", "crossfit", "bootstrap")}, default=float))
    return report


def test_cmd(args) -> dict:
    work, out = Path(args.work), Path(args.out)
    report = json.loads((work / "report.json").read_text())
    variant = args.variant or "model"
    threshold = float(args.threshold if args.threshold is not None else report["full_oof_threshold"][variant])
    out.mkdir(parents=True, exist_ok=True)
    model_path = work / "final_model.cbm"
    from catboost import CatBoostClassifier
    if model_path.exists():
        model = CatBoostClassifier()
        model.load_model(str(model_path))
    else:
        data = load_arena(_base_args(args))
        xc = _collective(work / "b_rest_collective.npy", args.arena, data["arena"].left, data["arena"].right)
        model, fit_rows = fit_catboost(np.hstack([data["x"], xc]).astype(np.float32), data["label"], args.threads)
        model.save_model(str(model_path))
        del data, xc
        log(f"final model fit on {fit_rows} B_REST rows")
    base_report = json.loads((Path(args.base_run) / "report.json").read_text())
    plain_w = float(base_report["plain_c2_name_only"]["weight"])
    plain_t = float(base_report["plain_c2_name_only"]["threshold"])

    scores = Path(args.test_inputs)
    ref_table = pq.read_table(scores / "dense_test" / "reference.parquet", columns=["entity_id"] + COLUMNS)
    pool_table = pq.read_table(scores / "dense_test" / "pool.parquet", columns=["entity_id"] + COLUMNS)
    s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object)
    rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref_table.column("country").to_pylist(), dtype=object)
    n_s1, n_rec = len(s1_ids), len(rec_ids)
    del ref_table, pool_table
    cache = v1_cache(None, None, None, None, None, None, Path(args.cache) / "test_v1_cache.npz")
    left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
    n_pairs = int(cache["n"][2])
    c2 = c2_on_rows(args.test_scores, cache["cand_rows"], cache["base"], must_score=np.ones(len(left), bool),
                    n_pairs=n_pairs)
    x2 = c2_features(left, right, c2, cache["base"], n_s1, n_rec, extended=bool(report.get("extended", False)))
    xc = _collective(out / "test_collective.npy", scores / "dense_test", left, right)
    prob = proba(model, np.hstack([cache["v1x"], x2, xc]).astype(np.float32))
    np.save(out / "probabilities_candidates.npy", prob)
    mult = cache["s1_mult"][left]
    accept = decode(prob, threshold, variant, left, right, cache["rec_empty"], cache["name_exact"], mult)
    plain_prob = c2_blend(x2[:, 0], cache["base"], plain_w)
    plain = decode(plain_prob, plain_t, "model_then_name_only", left, right, cache["rec_empty"], cache["name_exact"],
                   mult)
    france = country[left] == "France"
    mixed, mixed_prob = np.where(france, plain, accept), np.where(france, plain_prob, prob)
    s1_prob = np.load(Path(args.base_run) / "probabilities_candidates.npy")
    s1_accept = decode(s1_prob, float(base_report["threshold"]), base_report["variant"], left, right,
                       cache["rec_empty"], cache["name_exact"], mult)
    s1_mixed = np.where(france, plain, s1_accept)
    la = np.asarray(np.load(scores / "index" / "left.npy", mmap_mode="r")).astype(np.int64)
    ra = np.asarray(np.load(scores / "index" / "right.npy", mmap_mode="r")).astype(np.int64)
    pair_keys = np.sort(la * n_rec + ra)
    del la, ra
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    path = out / "matching_results.tsv"
    lengths = write_tsv(path, s1_ids, rec_ids, left, right, mixed, mixed_prob)
    check = validate_tsv(path, s1_ids, rec_index, pair_keys, n_rec)
    diff = {str(c): {"added_vs_s1": int((mixed & ~s1_mixed & (country[left] == c)).sum()),
                     "removed_vs_s1": int((s1_mixed & ~mixed & (country[left] == c)).sum())} for c in np.unique(country)}
    result = {"variant": variant, "threshold": threshold, "model": str(model_path), "model_sha256": sha256(model_path),
              "file": str(path), "sha256": sha256(path), "validation": check, "by_country": by_country(country, lengths),
              "vs_s1_upload_composition": diff, "france_rows": "plain C2 blend + name-only (unchanged from S1)",
              "labels_read": False}
    (out / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    log(json.dumps(result, default=float))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "test"):
        p = sub.add_parser(name)
        p.add_argument("--arena", default="/workspace/az-ml-new/arena/v1")
        p.add_argument("--labels", default="/workspace/az-ml-new/arena/labels")
        p.add_argument("--finalize", default="/workspace/az-ml-new/arena/finalize_v1")
        p.add_argument("--scores", default="/opt/az-ml-new/runs/c2_step040000/b_rest_scores")
        p.add_argument("--base-work", default="/opt/az-ml-new/agents/pair_tree_c2/step040000")
        p.add_argument("--cache", default="/opt/az-ml-new/agents/pair_tree_c2/step005000")
        p.add_argument("--work", required=True)
        p.add_argument("--threads", type=int, default=6)
        p.add_argument("--bootstrap", type=int, default=1000)
        p.add_argument("--extended", action="store_true", help="Also use pair_tree_c2 C2_EXTRA features")
        if name == "test":
            p.add_argument("--out", required=True)
            p.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
            p.add_argument("--test-scores", default="/opt/az-ml-new/runs/c2_step040000/test_scores")
            p.add_argument("--base-run", default="/opt/az-ml-new/runs/s1_pair_tree_40k")
            p.add_argument("--variant", choices=VARIANTS)
            p.add_argument("--threshold", type=float)
    args = parser.parse_args()
    (evaluate_cmd if args.command == "evaluate" else test_cmd)(args)


if __name__ == "__main__":
    main()
