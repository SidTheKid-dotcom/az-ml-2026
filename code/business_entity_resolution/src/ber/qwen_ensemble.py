"""C2 + Qwen3 ensemble on arena B_REST (component-half cross-fit) and a test file builder.

Commands::

    python -m ber.qwen_ensemble evaluate --scores C2_BREST --qwen-scores Q_BREST --work DIR \
        [--baseline-work PAIR_TREE_C2_WORK] [--cache /opt/az-ml-new/agents/pair_tree_c2/step005000]
    python -m ber.qwen_ensemble test --scores C2_BREST --qwen-scores Q_BREST --work DIR \
        --test-scores C2_TEST --qwen-test-scores Q_TEST --out DIR

Two ensembles, both fitted on B_REST only (C2 and Qwen trained on partition A):

* ``tree_q`` -- the ``ber.pair_tree_c2`` pair CatBoost (12 v1 + 8 C2 features) plus
  the same 8 rival/rank features computed on Qwen scores and 3 C2-Qwen agreement
  features. Same 2-fold component cross-fit and inner-OOF threshold choice as
  ``pair_tree_c2`` (half = component % 2, inner = (component // 2) % 2).
* ``blend3`` -- ``a*C2 + b*Qwen + (1-a-b)*tree`` on tree-gated rows, then one owner
  + cap 11 (optionally name-only + cap); (a, b, t) picked on one component half
  and applied to the other.

Baselines: v1, the C2 blend grid (cross-fit) and the C2-only pair tree (reused
from ``--baseline-work`` when it was built from the same C2 scores, else refit).
Paired component-bootstrap CIs are reported. SLICE and HOLDOUT-2 labels are never
read; C2/Qwen scores on partition A are never used.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule
from ber.pair_tree_c2 import (C2_GRID_WEIGHTS, COLUMNS, GATE, THRESHOLDS, V1_THRESHOLD, VARIANTS,
                              accept_per_row, by_country, c2_features, c2_on_rows, cap, crossfit, decode,
                              feature_names, fit_catboost, proba, rival_rates, sha256, slim, v1_cache,
                              validate_tsv, write_tsv)
from ber.pair_tree_arena import load_scores

Q_FEATURES = ["q", "q_blend", "q_rank_in_s1", "q_gap_to_s1_top", "s1_q_ge_0_5", "rival_q",
              "record_claimants_q_ge_0_5", "q_minus_rival", "c2q_mean", "c2q_min", "c2_minus_q"]
BLEND_C2 = (0.3, 0.45, 0.6, 0.75, 0.9)
BLEND_Q = (0.0, 0.1, 0.25, 0.4, 0.6)
BLEND_THRESHOLDS = (0.02, 0.03, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6)
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def q_features(left, right, c2, q, base, n_s1, n_rec):
    """Qwen rival/rank block (same definitions as the C2 block) plus C2-Qwen agreement columns."""
    block = c2_features(left, right, q, base, n_s1, n_rec, extended=False)
    s_c2 = np.where(np.isnan(c2), -1.0, c2).astype(np.float32)
    s_q = block[:, 0]
    agree = np.column_stack([(s_c2 + s_q) / 2, np.minimum(s_c2, s_q), s_c2 - s_q]).astype(np.float32)
    return np.hstack([block, agree])


def blend3(c2, q, base, a, b, gate=GATE):
    s_c2 = np.maximum(np.nan_to_num(c2, nan=0.0), 0)
    s_q = np.maximum(np.nan_to_num(q, nan=0.0), 0)
    return np.where(base >= gate, a * s_c2 + b * s_q + (1 - a - b) * base, 0).astype(np.float32)


def blend_weights():
    return [(a, b) for a in BLEND_C2 for b in BLEND_Q if a + b <= 1.0 + 1e-9]


# ----------------------------------------------------------------------------- arena


def q_on_rows(q_dir, c2_dir, cand_rows, base, must_score, n_pairs):
    """Qwen scores on candidate rows (NaN where unscored), honoring a declared ``--min-c2`` filter.

    A filtered Qwen score dir must name the same C2 score dir this run uses; rows the
    filter skipped (C2 < min_c2) are left unscored (feature value -1), identically
    on B_REST and test.
    """
    complete = json.loads((Path(q_dir) / "complete.json").read_text())
    flt = complete.get("c2_filter")
    if flt is not None:
        if flt["c2_scores"] != str(Path(c2_dir).resolve()):
            raise SystemExit(f"{q_dir} was filtered with C2 scores {flt['c2_scores']}, not {c2_dir}")
        c2_full = np.load(Path(c2_dir) / "scores.npy", mmap_mode="r")
        must_score = must_score & (np.asarray(c2_full[cand_rows]) >= flt["min_c2"])
    return c2_on_rows(q_dir, cand_rows, base, must_score=must_score, n_pairs=n_pairs), flt


def load_arena_q(args) -> dict:
    """``pair_tree_c2.load_arena`` plus Qwen features on the same candidate rows."""
    from ber.pair_tree_c2 import load_arena

    data = load_arena(args)  # v1 + C2 features on B_REST candidate rows (cache is only read)
    cache_path = Path(args.cache or args.work) / "arena_v1_cache.npz"
    with np.load(cache_path) as cache:
        cand_rows, base_all = cache["cand_rows"], cache["base"]
        left_c, right_c = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
        n_pairs = int(cache["n"][2])
    partition = data["partition"]
    brest_c = partition[left_c] == 1
    c2 = c2_on_rows(args.scores, cand_rows, base_all, must_score=brest_c, n_pairs=n_pairs)
    q, q_filter = q_on_rows(args.qwen_scores, args.scores, cand_rows, base_all, brest_c, n_pairs)
    c2[~brest_c] = np.nan
    q[~brest_c] = np.nan
    rows = np.flatnonzero(brest_c)
    if not np.array_equal(cand_rows[rows], data["pair_rows"]):
        raise SystemExit("candidate rows differ from pair_tree_c2.load_arena")
    xq = q_features(left_c, right_c, c2, q, base_all, data["n_s1"], data["n_rec"])[rows]
    data["xq"] = xq
    data["q"] = xq[:, 0]
    data["q_shift"] = rival_rates(xq)
    data["q_filter"] = q_filter
    log(f"Qwen scored on {int((~np.isnan(q)).sum())} of {len(q)} arena candidate rows (B_REST only)")
    return data


def per_s1(arena, accept):
    return arena.score(accept, full=True)


_GRID = {}


def _grid_config(key):
    """Half macros for one (c2 weight, qwen weight, threshold) config, both variants (fork worker)."""
    arena, a, b, t = _GRID["arena"], *key
    p = blend3(_GRID["c2"], _GRID["q"], _GRID["base"], a, b)
    accept = decode(p, t, "model", arena.left, arena.right, None, None, None)
    ruled, _ = name_only_rule(accept, arena.left, arena.right, arena.rec_empty, arena.name_exact, arena.mult)
    out = {}
    for v, acc in (("model", accept), ("model_then_name_only", cap(ruled, p, arena.left))):
        per = evaluate(acc, arena.left, arena.label, arena.true_count, return_per_s1=True)["per_s1"]
        out[v] = [float(per[m].mean()) for m in _GRID["halves"]]
    return key, out


def blend_grid(arena, c2, q, base, s1_half, weights, thresholds, workers, name):
    """(a, b, t) per variant picked on one component half and applied to the other.

    ``p = a*C2 + b*Qwen + (1-a-b)*tree`` on tree-gated rows. With ``b = 0`` this is
    exactly ``pair_tree_c2.c2_grid``. Returns the report (picks, full-B_REST pick,
    grid) and the cross-fit accept mask per variant.
    """
    import multiprocessing as mp

    _GRID.update(arena=arena, c2=c2, q=q, base=base,
                 halves=[arena.population & (s1_half == h) for h in (0, 1)])
    keys = [(a, b, t) for a, b in weights for t in thresholds]
    grid = {v: {} for v in VARIANTS}
    if workers > 1:
        with mp.get_context("fork").Pool(workers) as pool:
            done = pool.imap_unordered(_grid_config, keys, chunksize=2)
            for key, out in done:
                for v in VARIANTS:
                    grid[v][key] = out[v]
    else:
        for key in keys:
            for v, halves in _grid_config(key)[1].items():
                grid[v][key] = halves
    _GRID.clear()
    picks = {h: {v: max(grid[v], key=lambda k: (grid[v][k][h], -abs(k[2] - .1))) for v in VARIANTS} for h in (0, 1)}
    full = {v: max(grid[v], key=lambda k: sum(grid[v][k])) for v in VARIANTS}
    accepts = {}
    other = s1_half[arena.left]
    for v in VARIANTS:
        combined = np.zeros(len(arena.left), dtype=bool)
        for h in (0, 1):
            a, b, t = picks[h][v]
            combined |= arena.accept(blend3(c2, q, base, a, b), t, v) & (other == 1 - h)
        accepts[v] = combined
    as_dict = lambda k: {"c2": k[0], "qwen": k[1], "tree": round(1 - k[0] - k[1], 6), "threshold": k[2]}  # noqa: E731
    report = {"picks": {str(h): {v: {**as_dict(picks[h][v]), "half_f05": grid[v][picks[h][v]][h]} for v in VARIANTS}
                        for h in (0, 1)},
              "full_b_rest_pick": {v: as_dict(full[v]) for v in VARIANTS},
              "grid": {v: {f"{k[0]}:{k[1]}:{k[2]}": s for k, s in grid[v].items()} for v in VARIANTS}}
    log(f"{name}: {len(keys)} configs; picks {report['picks']}")
    return report, accepts


def c2_tree_baseline(args, data, x_base, arena, half_row, inner_row, s1_half):
    """C2-only pair-tree cross-fit accepts: reuse --baseline-work when it matches, else refit."""
    work = Path(args.baseline_work) if args.baseline_work else None
    if work is not None and (work / "crossfit_state.json").exists():
        state = json.loads((work / "crossfit_state.json").read_text())
        rows = np.load(work / "b_rest_pair_rows.npy")
        if (state["c2_scores"] == str(Path(args.scores).resolve()) and state["features"] == feature_names(args.extended)
                and np.array_equal(rows, data["pair_rows"])):
            oof = np.load(work / "oof.npy")
            picks = {int(h): v for h, v in state["picks"].items()}
            log(f"reused C2-only pair-tree cross-fit from {work}")
            return oof, picks, str(work)
        log(f"--baseline-work {work} was built from other scores/features/rows; refitting")
    fit = lambda xs, ys: fit_catboost(xs, ys, args.threads)  # noqa: E731
    oof, picks, _, _ = crossfit(x_base, data["label"], arena, half_row, inner_row, s1_half, fit)
    np.save(Path(args.work) / "baseline_c2_tree_oof.npy", oof)
    return oof, picks, "refit"


def evaluate_cmd(args) -> dict:
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    data = load_arena_q(args)
    arena, y = data["arena"], data["label"]
    x_base = data["x"]
    x = np.hstack([x_base, data["xq"]]).astype(np.float32)
    names = feature_names(args.extended) + Q_FEATURES
    component, population = data["component"], arena.population
    comp_row = component[arena.left]
    half_row, inner_row = (comp_row % 2).astype(np.int8), ((comp_row // 2) % 2).astype(np.int8)
    s1_half = (component % 2).astype(np.int8)
    log(f"B_REST {int(population.sum())} S1s, {len(y)} candidate rows, {x.shape[1]} features")

    results, per = {}, {}

    def record(name, accept):
        result = per_s1(arena, accept)
        per[name] = result.pop("per_s1")
        results[name] = slim(result)
        log(f"{name}: {results[name]['macro_f05']:.6f} P {results[name]['precision']:.5f} "
            f"R {results[name]['recall']:.5f} FL {results[name]['false_links']}")

    record("v1_one_owner", arena.accept(data["blend"], V1_THRESHOLD, "model"))
    c2w, grids = C2_GRID_WEIGHTS, {}
    for name, weights, thresholds in (("c2_grid", [(w, 0.0) for w in c2w], THRESHOLDS),
                                      ("qwen_grid", [(0.0, w) for w in c2w], THRESHOLDS),
                                      ("blend3", blend_weights(), BLEND_THRESHOLDS)):
        grids[name], accepts = blend_grid(arena, data["c2"], data["q"], data["base"], s1_half, weights,
                                          thresholds, args.grid_workers, name)
        for v, acc in accepts.items():
            record(f"{name}_{v}", acc)

    base_oof, base_picks, base_source = c2_tree_baseline(args, data, x_base, arena, half_row, inner_row, s1_half)
    for v in VARIANTS:
        thr = np.where(half_row == 1, base_picks[0][v]["threshold"], base_picks[1][v]["threshold"])
        record(f"c2_tree_{v}", accept_per_row(arena, base_oof, thr, v))

    state_path = work / "crossfit_state.json"
    key = {"c2_scores": str(Path(args.scores).resolve()), "qwen_scores": str(Path(args.qwen_scores).resolve()),
           "features": names}
    if args.reuse_fits and state_path.exists() and all(json.loads(state_path.read_text()).get(k) == v
                                                       for k, v in key.items()):
        state = json.loads(state_path.read_text())
        oof = np.load(work / "oof.npy")
        picks = {int(h): v for h, v in state["picks"].items()}
        importance = state["importance"]
        log("reused saved Qwen-tree cross-fit")
    else:
        fit = lambda xs, ys: fit_catboost(xs, ys, args.threads)  # noqa: E731
        oof, picks, models, _ = crossfit(x, y, arena, half_row, inner_row, s1_half, fit)
        for h, model in models.items():
            model.save_model(str(work / f"model_half{h}.cbm"))
        np.save(work / "oof.npy", oof)
        np.save(work / "b_rest_pair_rows.npy", data["pair_rows"])
        importance = {str(h): dict(zip(names, map(float, m.get_feature_importance()))) for h, m in models.items()}
        state_path.write_text(json.dumps({**key, "picks": {str(h): v for h, v in picks.items()},
                                          "importance": importance}, default=float) + "\n")
    for v in VARIANTS:
        thr = np.where(half_row == 1, picks[0][v]["threshold"], picks[1][v]["threshold"])
        record(f"tree_q_{v}", accept_per_row(arena, oof, thr, v))
    grid_full, best_full = arena.sweep(oof, None, population)

    boot = {}
    pairs = [(f"tree_q_{v}", f"c2_tree_{v}") for v in VARIANTS]
    pairs += [(f"blend3_{v}", f"c2_grid_{v}") for v in VARIANTS]
    pairs += [(f"tree_q_{v}", f"blend3_{v}") for v in VARIANTS]
    pairs += [(f"qwen_grid_{v}", f"c2_grid_{v}") for v in VARIANTS]
    for new, old in pairs:
        boot[f"{new}_minus_{old}"] = component_bootstrap(per[old], per[new], component, n=args.bootstrap,
                                                         seed=20260926, population=population)
        log(f"{new} - {old}: {boot[f'{new}_minus_{old}']['delta']:+.6f} "
            f"[{boot[f'{new}_minus_{old}']['ci_low']:+.6f}, {boot[f'{new}_minus_{old}']['ci_high']:+.6f}]")
    preferred = max(VARIANTS, key=lambda v: results[f"tree_q_{v}"]["macro_f05"])
    report = {
        "question": "Does a Qwen3-Reranker-0.6B score add to C2 in the B_REST decision layer?",
        "scope": "arena B_REST only; SLICE and HOLDOUT-2 labels unread; partition-A scores unused",
        **key, "baseline_c2_tree": base_source, "extended": bool(args.extended),
        "population": {"b_rest_s1": int(population.sum()), "candidate_rows": int(len(y)),
                       "candidate_true_rows": int(y.sum())},
        "split": "half = component_id % 2, inner = (component_id // 2) % 2 (as pair_tree_c2)",
        "results": results, "bootstrap": boot, "tree_q_picks": {str(h): picks[h] for h in picks},
        "tree_q_preferred_variant": preferred,
        "tree_q_full_oof_threshold": {v: best_full[v] for v in VARIANTS},
        "tree_q_full_oof_f05": {v: grid_full[v][best_full[v]] for v in VARIANTS},
        **{name: {k: grids[name][k] for k in ("picks", "full_b_rest_pick")} for name in grids},
        "grids": {name: grids[name]["grid"] for name in grids}, "feature_importance": importance,
        "rival_rates": {"c2": data["shift"]["rival_rates"], "qwen": data["q_shift"]},
        "qwen_c2_filter": data["q_filter"],
        "holdout2_rows_read": 0, "slice_read": False, "elapsed_seconds": round(time.time() - T0, 1)}
    (work / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps({"results": {k: v["macro_f05"] for k, v in results.items()},
                    "bootstrap": {k: [v["delta"], v["ci_low"], v["ci_high"]] for k, v in boot.items()}}))
    return report


# ----------------------------------------------------------------------------- test


def test_cmd(args) -> dict:
    work, out = Path(args.work), Path(args.out)
    report = json.loads((work / "report.json").read_text())
    if (report["c2_scores"] != str(Path(args.scores).resolve())
            or report["qwen_scores"] != str(Path(args.qwen_scores).resolve())):
        raise SystemExit("work report was built from different arena C2/Qwen scores")
    args.extended = bool(report.get("extended", False))
    variant = args.variant or report["tree_q_preferred_variant"]
    threshold = float(args.threshold if args.threshold is not None else report["tree_q_full_oof_threshold"][variant])
    out.mkdir(parents=True, exist_ok=True)
    model_path = work / "final_model.cbm"
    if model_path.exists():
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(model_path))
        fit_rows = None
        log(f"loaded {model_path}")
    else:
        data = load_arena_q(args)
        x = np.hstack([data["x"], data["xq"]]).astype(np.float32)
        model, fit_rows = fit_catboost(x, data["label"], args.threads)
        model.save_model(str(model_path))
        del data, x
        log(f"final Qwen-tree model fit on {fit_rows} B_REST rows")

    inputs = Path(args.test_inputs)
    ref_table = pq.read_table(inputs / "dense_test" / "reference.parquet", columns=["entity_id"] + COLUMNS)
    pool_table = pq.read_table(inputs / "dense_test" / "pool.parquet", columns=["entity_id"] + COLUMNS)
    s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object)
    rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref_table.column("country").to_pylist(), dtype=object)
    n_s1, n_rec = len(s1_ids), len(rec_ids)
    cache_path = Path(args.cache or work) / "test_v1_cache.npz"
    left_all = np.asarray(np.load(inputs / "index" / "left.npy", mmap_mode="r")).astype(np.int64)
    right_all = np.asarray(np.load(inputs / "index" / "right.npy", mmap_mode="r")).astype(np.int64)
    if cache_path.exists():
        cache = v1_cache(None, None, None, None, None, None, cache_path)
    else:
        cache = v1_cache(ref_table.select(COLUMNS).to_pandas(), pool_table.select(COLUMNS).to_pandas(), left_all,
                         right_all, load_scores(inputs / "probabilities.npy"),
                         load_scores(inputs / "base_probabilities.npy"), Path(args.work) / "test_v1_cache.npz")
    del ref_table, pool_table
    left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
    n_pairs = int(cache["n"][2])
    if n_pairs != len(left_all):
        raise SystemExit("test v1 cache does not match the test pair index")
    everything = np.ones(len(left), bool)
    c2 = c2_on_rows(args.test_scores, cache["cand_rows"], cache["base"], must_score=everything, n_pairs=n_pairs)
    q, q_filter = q_on_rows(args.qwen_test_scores, args.test_scores, cache["cand_rows"], cache["base"], everything,
                            n_pairs)
    arena_filter = report.get("qwen_c2_filter")
    if (q_filter is None) != (arena_filter is None) or (q_filter and q_filter["min_c2"] != arena_filter["min_c2"]):
        raise SystemExit("Qwen test and B_REST scores used different C2 filters")
    x2 = c2_features(left, right, c2, cache["base"], n_s1, n_rec, extended=args.extended)
    xq = q_features(left, right, c2, q, cache["base"], n_s1, n_rec)
    prob = proba(model, np.hstack([cache["v1x"], x2, xq]).astype(np.float32))
    np.save(out / "probabilities_candidates.npy", prob)
    mult = cache["s1_mult"][left]
    rec_empty, name_exact = cache["rec_empty"], cache["name_exact"]
    accept = decode(prob, threshold, variant, left, right, rec_empty, name_exact, mult)
    pick = report["c2_grid"]["full_b_rest_pick"]["model_then_name_only"]
    plain_prob = np.where(cache["base"] >= GATE, pick["c2"] * np.maximum(x2[:, 0], 0)
                          + (1 - pick["c2"]) * cache["base"], 0).astype(np.float32)
    plain = decode(plain_prob, pick["threshold"], "model_then_name_only", left, right, rec_empty, name_exact, mult)
    b3 = report["blend3"]["full_b_rest_pick"]["model_then_name_only"]
    b3_prob = blend3(c2, q, cache["base"], b3["c2"], b3["qwen"])
    b3_accept = decode(b3_prob, b3["threshold"], "model_then_name_only", left, right, rec_empty, name_exact, mult)
    france = country[left] == "France"
    if not np.any(country == "France"):
        raise SystemExit("no France rows in the test reference; check country labels")
    outputs = [("model", accept, prob),
               ("model_us_india_c2_france", np.where(france, plain, accept), np.where(france, plain_prob, prob)),
               ("blend3_name_only", b3_accept, b3_prob),
               ("blend3_us_india_c2_france", np.where(france, plain, b3_accept), np.where(france, plain_prob, b3_prob))]
    rec_index = {v: i for i, v in enumerate(rec_ids)}
    pair_keys = np.sort(left_all * n_rec + right_all)
    del left_all, right_all
    files = {}
    for name, acc, p in outputs:
        path = out / ("matching_results.tsv" if name == "model" else f"{name}/matching_results.tsv")
        lengths = write_tsv(path, s1_ids, rec_ids, left, right, acc, p)
        check = validate_tsv(path, s1_ids, rec_index, pair_keys, n_rec)
        files[name] = {"path": str(path), "sha256": sha256(path), "validation": check,
                       "by_country": by_country(country, lengths)}
        log(f"{name}: links {check['links']} passed {check['passed']} {check['problems']}")
    compare = {}
    if args.compare_file:
        other = {}
        for line in Path(args.compare_file).read_text(encoding="utf-8").split("\n")[1:]:
            if line:
                s1, ids = line.split("\t")
                other[s1] = set(filter(None, ids.split(",")))
        for name, acc, _ in outputs:
            mine = {}
            for row in np.flatnonzero(acc):
                mine.setdefault(s1_ids[left[row]], set()).add(rec_ids[right[row]])
            diff = {}
            for code in np.unique(country):
                keys = s1_ids[country == code]
                added = sum(len(mine.get(k, set()) - other.get(k, set())) for k in keys)
                removed = sum(len(other.get(k, set()) - mine.get(k, set())) for k in keys)
                diff[str(code)] = {"links_only_here": added, "links_only_in_compare": removed}
            compare[name] = diff
    result = {"variant": variant, "threshold": threshold, "model": str(model_path), "model_sha256": sha256(model_path),
              "final_fit_rows": fit_rows, "plain_c2_france": pick, "blend3_pick": b3,
              "arena_report": str(work / "report.json"),
              "arena_tree_q": report["results"][f"tree_q_{variant}"],
              "test_c2_scores": str(Path(args.test_scores).resolve()),
              "test_qwen_scores": str(Path(args.qwen_test_scores).resolve()),
              "test_qwen_complete": json.loads((Path(args.qwen_test_scores) / "complete.json").read_text()).get("checkpoint"),
              "rival_rates": {"test_c2": rival_rates(x2), "test_qwen": rival_rates(xq), "arena": report["rival_rates"]},
              "candidate_rows": int(len(left)), "files": files, "compare_file": args.compare_file,
              "compare_by_country": compare, "labels_read": False, "elapsed_seconds": round(time.time() - T0, 1)}
    (out / "report.json").write_text(json.dumps(result, indent=2, default=float) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("evaluate", "test"):
        p = sub.add_parser(name)
        p.add_argument("--arena", default="/workspace/az-ml-new/arena/v1")
        p.add_argument("--labels", default="/workspace/az-ml-new/arena/labels")
        p.add_argument("--finalize", default="/workspace/az-ml-new/arena/finalize_v1")
        p.add_argument("--scores", required=True, help="C2 arena B_REST score dir")
        p.add_argument("--qwen-scores", required=True, help="Qwen arena B_REST score dir (ber.qwen_train score)")
        p.add_argument("--work", required=True)
        p.add_argument("--cache", default="/opt/az-ml-new/agents/pair_tree_c2/step005000",
                       help="Dir holding arena_v1_cache.npz / test_v1_cache.npz (read only)")
        p.add_argument("--extended", action="store_true", help="C2_EXTRA features in the C2 block")
        p.add_argument("--threads", type=int, default=int(os.environ.get("TREE_THREADS", "20")))
        if name == "evaluate":
            p.add_argument("--baseline-work", help="pair_tree_c2 work dir with the C2-only cross-fit to reuse")
            p.add_argument("--bootstrap", type=int, default=1000)
            p.add_argument("--grid-workers", type=int, default=12, help="Forked processes for the blend grids")
            p.add_argument("--reuse-fits", action="store_true")
        else:
            p.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
            p.add_argument("--test-scores", required=True, help="C2 test score dir")
            p.add_argument("--qwen-test-scores", required=True, help="Qwen test score dir")
            p.add_argument("--variant", choices=VARIANTS)
            p.add_argument("--threshold", type=float)
            p.add_argument("--compare-file", help="Existing matching_results.tsv to diff against, by country")
            p.add_argument("--out", required=True)
    args = parser.parse_args()
    (evaluate_cmd if args.command == "evaluate" else test_cmd)(args)

if __name__ == "__main__":
    main()
