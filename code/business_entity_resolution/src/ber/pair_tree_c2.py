"""C2-aware pair CatBoost: arena B_REST 2-fold component cross-fit, then a test matching file.

Commands::

    python -m ber.pair_tree_c2 evaluate --scores RUN/b_rest_scores --work DIR
    python -m ber.pair_tree_c2 test --scores RUN/b_rest_scores --test-scores RUN/test_scores --work DIR --out DIR

C2 was trained on partition A, so only its B_REST scores are used and the model
is fit on B_REST alone. B_REST S1s are split by claim-graph component parity
(``component_ids % 2``); a CatBoost is fit on one half and predicts the other,
both directions. Each half's threshold (per variant) is chosen on inner
out-of-fold predictions inside that half (components split by ``(id // 2) % 2``),
so no in-sample probability picks a threshold. Variants: one owner + cap 11
(``model``), and the same followed by the empty-address name-only rule and the
cap again (``model_then_name_only``). Labels are indexed only on B_REST rows;
SLICE and HOLDOUT-2 labels are never read.

Features: the 12 v1 features of :func:`ber.pair_tree_arena.build_pairs` plus C2
score, C2 blend (0.5 C2 + 0.5 tree on tree-gated pairs), C2 rank and gap to top
in the S1 list, S1 count of C2 >= 0.5, best rival C2 for the record, record
claimants with C2 >= 0.5, and C2 minus rival. Arena C2 exists only for B_REST
pairs, so C2 rival features on the arena see B_REST claimants only; on test
every claimant is scored (reported as a known shift).

``test`` fits the final model on every B_REST candidate row with the threshold
chosen on the full cross-fit out-of-fold predictions, and writes validated
files: the model on every country (``matching_results.tsv``), the model for
US/India with France rows from the plain C2 blend + name-only file, that plain
file itself (weight/threshold picked on B_REST), and, with ``--france-scores``,
the model for US/India with France rows from a France fine-tuned C2 (plain
blend + name-only). France has no labels; no French score is measured here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, name_only_rule, one_owner
from ber.forensic_features import EMPTY
from ber.pair_tree_arena import FEATURES as V1_FEATURES
from ber.pair_tree_arena import build_pairs, decide, false_merge, load_scores, rank_within, top_two

C2_FEATURES = ["c2", "c2_blend", "c2_rank_in_s1", "c2_gap_to_s1_top", "s1_c2_ge_0_5", "rival_c2",
               "record_claimants_c2_ge_0_5", "c2_minus_rival"]
C2_EXTRA = ["s1_c2_ge_0_1", "record_claimants_c2_ge_0_1", "c2_gap_to_s1_next", "s1_c2_sum"]
FEATURES = list(V1_FEATURES) + C2_FEATURES


def feature_names(extended=False):
    return FEATURES + (C2_EXTRA if extended else [])

VARIANTS = ("model", "model_then_name_only")
THRESHOLDS = tuple(float(t) for t in np.round(np.arange(0.05, 0.951, 0.025), 3))
GATE, C2_WEIGHT, C2_THRESHOLD, V1_THRESHOLD, CAP = 0.01, 0.5, 0.5, 0.775, 11
C2_GRID_WEIGHTS = (0.5, 0.75, 0.9, 1.0)
FIT_ROWS = 3_000_000
CATBOOST = {"iterations": 800, "depth": 8, "learning_rate": 0.08, "loss_function": "Logloss"}
SEED = 0
COLUMNS = ["business_name", "business_address", "country"]
HEADER = "source1_entity_id\tmatched_entity_ids"
T0 = time.time()


def log(message):
    print(f"[{time.time() - T0:7.1f}s] {message}", flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ----------------------------------------------------------------------------- features

def c2_features(left, right, c2, base, n_s1, n_rec, gate=GATE, extended=False):
    """C2 feature matrix for candidate rows (``left``/``right``/``c2``/``base`` already on those rows).

    ``c2`` is the reranker score, NaN where it was not scored; unscored rows get
    -1 and never count as a rival or a strong claimant. ``extended`` appends
    ``C2_EXTRA``: counts at C2 >= 0.1, the margin to the next candidate of the
    S1 list, and the S1's summed C2.
    """
    s = np.where(np.isnan(c2), -1.0, c2).astype(np.float32)
    p = np.where(base >= gate, C2_WEIGHT * np.maximum(s, 0) + (1 - C2_WEIGHT) * base, 0).astype(np.float32)
    rank = np.minimum(rank_within(left, s), 99).astype(np.float32)
    s1_top, _ = top_two(left, s, n_s1)
    strong = s >= 0.5
    s1_strong = np.bincount(left[strong], minlength=n_s1).astype(np.float32)
    rec_top1, rec_top2 = top_two(right, s, n_rec)
    rival = np.where(s >= rec_top1[right], rec_top2[right], rec_top1[right]).astype(np.float32)
    rec_strong = np.bincount(right[strong], minlength=n_rec).astype(np.float32)
    columns = [s, p, rank, s - s1_top[left], s1_strong[left], rival, rec_strong[right], s - rival]
    if extended:
        weak = s >= 0.1
        order = np.lexsort((np.arange(len(s)), -s, left))
        s_sorted, l_sorted = s[order], left[order]
        following = np.full(len(s), -1.0, dtype=np.float32)
        following[:-1] = np.where(l_sorted[1:] == l_sorted[:-1], s_sorted[1:], -1.0)
        gap_next = np.empty(len(s), dtype=np.float32)
        gap_next[order] = s_sorted - following
        columns += [np.bincount(left[weak], minlength=n_s1)[left], np.bincount(right[weak], minlength=n_rec)[right],
                    gap_next, np.bincount(left, weights=np.maximum(s, 0), minlength=n_s1)[left]]
    return np.column_stack(columns).astype(np.float32)


def rival_rates(x2):
    """Share of rows at C2 >= level whose best rival C2 is also >= level (arena vs test shift check)."""
    s, rival = x2[:, 0], x2[:, 5]
    return {str(level): float((rival[s >= level] >= level).mean()) if np.any(s >= level) else None
            for level in (0.1, 0.5)}


def c2_blend(c2, base, weight, gate=GATE):
    """Plain blend ``weight * C2 + (1 - weight) * tree`` on tree-gated rows (as ``ber.c2_stack.blend``); 0 elsewhere."""
    return np.where(base >= gate, weight * np.maximum(c2, 0) + (1 - weight) * base, 0).astype(np.float32)


def v1_cache(ref, pool, left, right, blend, base, path):
    """Candidate rows and their v1 features / name-only inputs, cached as one npz."""
    if path.exists():
        log(f"load v1 feature cache {path}")
        with np.load(path) as data:
            return {key: data[key] for key in data.files}
    pairs = build_pairs(ref, pool, left, right, blend, base)
    rows = pairs.cand_rows
    cache = {"cand_rows": rows, "left": left[rows].astype(np.int32), "right": right[rows].astype(np.int32),
             "v1x": pairs.features(rows), "name_exact": pairs.name_exact[rows],
             "rec_empty": pairs.rec_kind[rows] == EMPTY, "s1_mult": np.asarray(pairs.s1_mult, dtype=np.int64),
             "blend": blend[rows], "base": base[rows], "n": np.array([len(ref), len(pool), len(left)])}
    tmp = path.with_suffix(".tmp.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tmp, **cache)
    tmp.replace(path)
    log(f"saved v1 feature cache {path} ({len(rows)} candidate rows)")
    return cache


def c2_on_rows(scores_dir, cand_rows, base_c, must_score, n_pairs=None):
    """C2 scores on candidate rows (NaN where unscored); every gated row in ``must_score`` must be scored."""
    scores_dir = Path(scores_dir)
    if not (scores_dir / "complete.json").exists():
        raise SystemExit(f"{scores_dir} has no complete.json")
    full_scores = np.load(scores_dir / "scores.npy", mmap_mode="r")
    full_mask = np.load(scores_dir / "scored_mask.npy", mmap_mode="r")
    if n_pairs is not None and not (len(full_scores) == len(full_mask) == n_pairs):
        raise SystemExit(f"{scores_dir} arrays have {len(full_scores)}/{len(full_mask)} rows, expected {n_pairs}")
    c2 = np.asarray(full_scores[cand_rows], dtype=np.float32)
    scored = np.asarray(full_mask[cand_rows], dtype=bool)
    missing = must_score & (base_c >= GATE) & ~scored
    if missing.any():
        raise SystemExit(f"{int(missing.sum())} gated candidate rows have no C2 score")
    c2 = np.nan_to_num(c2, nan=0.0)
    c2[~scored] = np.nan
    return c2


# ----------------------------------------------------------------------------- decoding

def cap(accept, prob, left, k=CAP):
    """Keep at most ``k`` accepted links per S1, highest probability first (ties: row order)."""
    accept = accept.copy()
    rows = np.flatnonzero(accept)
    if rows.size:
        accept[rows[rank_within(left[rows], prob[rows]) >= k]] = False
    return accept


def decode(prob, threshold, variant, left, right, rec_empty, name_exact, mult):
    """One owner + cap 11 at ``threshold``; ``*_name_only`` variants add the name-only rule, then cap again."""
    accept = decide(prob, left, right, threshold, None)
    if variant.endswith("name_only"):
        accept, _ = name_only_rule(accept, left, right, rec_empty, name_exact, mult)
        accept = cap(accept, prob, left)
    return accept


class Arena:
    """B_REST candidate rows with labels, groups and name-only inputs; decode/score on any row subset."""

    def __init__(self, left, right, label, true_count, rec_empty, name_exact, mult, population, country=None):
        self.left, self.right, self.label = left, right, label
        self.true_count, self.rec_empty, self.name_exact, self.mult = true_count, rec_empty, name_exact, mult
        self.population, self.country = population, country

    def accept(self, prob, threshold, variant, rows=None):
        if rows is None:
            return decode(prob, threshold, variant, self.left, self.right, self.rec_empty, self.name_exact, self.mult)
        out = np.zeros(len(self.left), dtype=bool)
        out[rows] = decode(prob[rows], threshold, variant, self.left[rows], self.right[rows], self.rec_empty[rows],
                           self.name_exact[rows], self.mult[rows])
        return out

    def score(self, accept, population=None, full=False):
        pop = self.population if population is None else population
        result = evaluate(accept, self.left, self.label, self.true_count, population=pop,
                          country=self.country if full else None, return_per_s1=full)
        if full:
            result["false_merge"] = false_merge(accept, self.left, pop, self.true_count)
        return result

    def sweep(self, prob, rows, population, thresholds=THRESHOLDS):
        """Macro F0.5 per (variant, threshold) on ``rows``; returns the grid and the best threshold per variant."""
        grid = {v: {} for v in VARIANTS}
        for t in thresholds:
            for v in VARIANTS:
                grid[v][t] = self.score(self.accept(prob, t, v, rows), population)["macro_f05"]
        best = {v: max(grid[v], key=lambda t: (grid[v][t], -abs(t - 0.5))) for v in VARIANTS}
        return grid, best


def fit_catboost(x, y, threads, seed=SEED, params=None, max_rows=FIT_ROWS):
    from catboost import CatBoostClassifier

    rows = np.arange(len(y))
    if len(rows) > max_rows:
        rows = np.sort(np.random.default_rng(seed).choice(rows, size=max_rows, replace=False))
    model = CatBoostClassifier(**(params or CATBOOST), thread_count=threads, random_seed=seed, verbose=200,
                               allow_writing_files=False)
    model.fit(x[rows], y[rows].astype(np.int8))
    return model, int(len(rows))


def proba(model, x, chunk=4_000_000):
    out = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), chunk):
        out[start:start + chunk] = model.predict_proba(x[start:start + chunk])[:, 1]
    return out


def crossfit(x, y, arena, half_row, inner_row, s1_half, fit):
    """2-fold cross-fit with inner-OOF threshold choice. Returns OOF probabilities, per-half picks and models.

    ``fit(x, y) -> (model, rows)``; ``half_row``/``inner_row`` are per-row groups and
    ``s1_half`` the outer group per S1. Rows of half 1-h are predicted by the model fit on
    half h and later thresholded at half h's pick (see :func:`accept_per_row`).
    """
    oof = np.zeros(len(y), dtype=np.float32)
    picks, models, info = {}, {}, {}
    for h in (0, 1):
        in_half = half_row == h
        inner_pred = np.zeros(len(y), dtype=np.float32)
        for i in (0, 1):
            train, held = in_half & (inner_row == i), in_half & (inner_row != i)
            model, _ = fit(x[train], y[train])
            inner_pred[held] = proba(model, x[held])
            log(f"half {h} inner {i}: fit {int(train.sum())} rows, predicted {int(held.sum())}")
        grid, best = arena.sweep(inner_pred, np.flatnonzero(in_half), arena.population & (s1_half == h))
        model, fit_rows = fit(x[in_half], y[in_half])
        other = ~in_half
        oof[other] = proba(model, x[other])
        picks[h] = {v: {"threshold": best[v], "inner_oof_f05": grid[v][best[v]]} for v in VARIANTS}
        info[h] = {"fit_rows": fit_rows, "positives": int(y[in_half].sum()),
                   "inner_grid": {v: {str(t): s for t, s in grid[v].items()} for v in VARIANTS}}
        models[h] = model
        log(f"half {h}: picks {picks[h]}")
    return oof, picks, models, info


def accept_per_row(arena, prob, thr, variant):
    """Like :func:`decode` with a per-row threshold ``thr``: ownership and the cap rank by ``prob``."""
    accept = cap(one_owner(prob >= thr, arena.left, arena.right, prob), prob, arena.left)
    if variant.endswith("name_only"):
        accept, _ = name_only_rule(accept, arena.left, arena.right, arena.rec_empty, arena.name_exact, arena.mult)
        accept = cap(accept, prob, arena.left)
    return accept


def c2_grid(arena, c2, base, s1_half, weights=C2_GRID_WEIGHTS, thresholds=THRESHOLDS):
    """Plain C2 blend baseline: (weight, threshold) per variant chosen on one half, applied to the other.

    Every config is decoded on all B_REST rows (one owner + cap 11, optionally
    name-only + cap); half macros come from the per-S1 scores. Returns picks,
    the cross-fit accept mask per variant and the full-B_REST best config.
    """
    halves = [arena.population & (s1_half == h) for h in (0, 1)]
    grid = {v: {} for v in VARIANTS}
    for w in weights:
        p = c2_blend(c2, base, w)
        for t in thresholds:
            accept = decode(p, t, "model", arena.left, arena.right, None, None, None)
            ruled, _ = name_only_rule(accept, arena.left, arena.right, arena.rec_empty, arena.name_exact, arena.mult)
            for v, acc in (("model", accept), ("model_then_name_only", cap(ruled, p, arena.left))):
                per = evaluate(acc, arena.left, arena.label, arena.true_count, return_per_s1=True)["per_s1"]
                grid[v][(w, t)] = [float(per[m].mean()) for m in halves]
        log(f"c2 grid w={w}: best half-0 {max(grid['model_then_name_only'][(w, t)][0] for t in thresholds):.6f}")
    picks = {h: {v: max(grid[v], key=lambda k: grid[v][k][h]) for v in VARIANTS} for h in (0, 1)}
    full = {v: max(grid[v], key=lambda k: sum(grid[v][k])) for v in VARIANTS}
    accepts = {}
    s1_other_half = s1_half[arena.left]
    for v in VARIANTS:
        combined = np.zeros(len(arena.left), dtype=bool)
        for h in (0, 1):
            w, t = picks[h][v]
            acc = arena.accept(c2_blend(c2, base, w), t, v)
            combined |= acc & (s1_other_half == 1 - h)
        accepts[v] = combined
    return {"picks": {str(h): {v: {"weight": picks[h][v][0], "threshold": picks[h][v][1],
                                   "half_f05": grid[v][picks[h][v]][h]} for v in VARIANTS} for h in (0, 1)},
            "full_b_rest_pick": {v: {"weight": full[v][0], "threshold": full[v][1]} for v in VARIANTS},
            "grid": {v: {f"{w}:{t}": vals for (w, t), vals in grid[v].items()} for v in VARIANTS}}, accepts


def slim(result):
    keys = ("s1", "macro_f05", "precision", "recall", "predicted_links", "false_links", "singleton_accuracy",
            "empty_predictions", "false_merge")
    out = {k: result[k] for k in keys if k in result}
    if "by_country" in result:
        out["by_country"] = {c: {k: r[k] for k in ("s1", "macro_f05", "precision", "recall", "false_links")}
                             for c, r in result["by_country"].items()}
    return out


# ----------------------------------------------------------------------------- arena

def load_arena(args):
    arena_dir, labels_dir, final = Path(args.arena), Path(args.labels), Path(args.finalize)
    summary = json.loads((labels_dir / "labels_summary.json").read_text())
    if summary.get("holdout2_rows_read") != 0:
        raise SystemExit("labels include HOLDOUT-2 rows")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.cache or work) / "arena_v1_cache.npz"
    if cache_path.exists():
        cache = v1_cache(None, None, None, None, None, None, cache_path)
    else:
        table = pq.read_table(arena_dir / "pairs.parquet", columns=["left_idx", "right_idx"])
        left = table.column("left_idx").to_numpy().astype(np.int64)
        right = table.column("right_idx").to_numpy().astype(np.int64)
        del table
        ref = pq.read_table(arena_dir / "reference.parquet", columns=COLUMNS).to_pandas()
        pool = pq.read_table(arena_dir / "pool.parquet", columns=COLUMNS).to_pandas()
        cache = v1_cache(ref, pool, left, right, load_scores(arena_dir / "probabilities.npy"),
                         load_scores(arena_dir / "base_probabilities.npy"), cache_path)
        del ref, pool, left, right
    n_s1, n_rec, _ = (int(v) for v in cache["n"])
    partition = np.load(final / "s1_partition.npy")
    component = np.load(final / "component_ids.npy")
    if len(partition) != n_s1 or len(component) != n_s1:
        raise SystemExit("partition/component arrays do not match the arena S1 count")
    left_c, right_c = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
    brest_c = partition[left_c] == 1
    c2 = c2_on_rows(args.scores, cache["cand_rows"], cache["base"], must_score=brest_c, n_pairs=int(cache["n"][2]))
    c2[~brest_c] = np.nan  # C2 on A is in-sample and SLICE is sealed: never use either
    log(f"C2 scored on {int((~np.isnan(c2)).sum())} of {len(c2)} arena candidate rows (B_REST only)")
    x2 = c2_features(left_c, right_c, c2, cache["base"], n_s1, n_rec, extended=args.extended)
    rows = np.flatnonzero(brest_c)
    pair_rows = cache["cand_rows"][rows]
    label = np.asarray(np.load(labels_dir / "label.npy", mmap_mode="r")[pair_rows], dtype=bool)
    true_count = np.load(labels_dir / "true_count.npy")
    country = pq.read_table(arena_dir / "reference.parquet", columns=["country"]).column("country") \
        .to_numpy(zero_copy_only=False)
    population = partition == 1
    left, right = left_c[rows], right_c[rows]
    arena = Arena(left, right, label, true_count, cache["rec_empty"][rows], cache["name_exact"][rows],
                  cache["s1_mult"][left], population, country)
    x = np.hstack([cache["v1x"][rows], x2[rows]]).astype(np.float32)
    # Known shift: B_REST pairs whose record has a gated claimant outside B_REST (C2 rival unseen here).
    outside = np.zeros(n_rec, dtype=bool)
    outside[right_c[~brest_c & (cache["base"] >= GATE)]] = True
    shift = {"b_rest_gated_rows": int((cache["base"][rows] >= GATE).sum()),
             "with_outside_gated_claimant": int(((cache["base"][rows] >= GATE) & outside[right]).sum()),
             "rival_rates": rival_rates(x2[rows])}
    return {"arena": arena, "x": x, "label": label, "pair_rows": pair_rows, "component": component,
            "partition": partition, "blend": cache["blend"][rows], "c2_blend": x2[rows, 1], "c2": x2[rows, 0],
            "base": cache["base"][rows], "n_s1": n_s1,
            "n_rec": n_rec, "shift": shift}


def evaluate_cmd(args) -> dict:
    work = Path(args.work)
    data = load_arena(args)
    arena, x, y = data["arena"], data["x"], data["label"]
    component, population = data["component"], arena.population
    comp_row = component[arena.left]
    half_row, inner_row = (comp_row % 2).astype(np.int8), ((comp_row // 2) % 2).astype(np.int8)
    s1_half = (component % 2).astype(np.int8)
    log(f"B_REST {int(population.sum())} S1s, {len(y)} candidate rows ({int(y.sum())} true); "
        f"half-0 rows {int((half_row == 0).sum())}, half-1 rows {int((half_row == 1).sum())}")

    baselines = {}
    per = {}
    for name, prob, threshold, variant in (("v1_one_owner", data["blend"], V1_THRESHOLD, "model"),
                                           ("c2_blend", data["c2_blend"], C2_THRESHOLD, "model"),
                                           ("c2_blend_name_only", data["c2_blend"], C2_THRESHOLD,
                                            "model_then_name_only")):
        result = arena.score(arena.accept(prob, threshold, variant), full=True)
        per[name] = result.pop("per_s1")
        baselines[name] = slim(result)
        log(f"{name}: {result['macro_f05']:.6f}")
    untrimmed, _ = name_only_rule(arena.accept(data["c2_blend"], C2_THRESHOLD, "model"), arena.left, arena.right,
                                  arena.rec_empty, arena.name_exact, arena.mult)
    baselines["c2_blend_name_only_untrimmed_f05"] = arena.score(untrimmed)["macro_f05"]

    grid_report, grid_accepts = c2_grid(arena, data["c2"], data["base"], s1_half)
    for v, acc in grid_accepts.items():
        result = arena.score(acc, full=True)
        per[f"c2_grid_{v}"] = result.pop("per_s1")
        baselines[f"c2_grid_crossfit_{v}"] = slim(result)
        log(f"c2 grid cross-fit {v}: {result['macro_f05']:.6f}")
    baselines["c2_grid"] = grid_report

    state_path = work / "crossfit_state.json"
    if args.reuse_fits and not state_path.exists() and (work / "report.json").exists():
        old = json.loads((work / "report.json").read_text())
        state_path.write_text(json.dumps({"c2_scores": old["c2_scores"], "features": old["features"],
                                          "picks": old["crossfit_picks"], "info": old["fold_info"],
                                          "importance": old["feature_importance"]}) + "\n")
    if args.reuse_fits and state_path.exists():
        state = json.loads(state_path.read_text())
        if state["c2_scores"] != str(Path(args.scores).resolve()) or state["features"] != feature_names(args.extended):
            raise SystemExit("saved cross-fit state was built from other scores or features")
        oof = np.load(work / "oof.npy")
        picks = {int(h): v for h, v in state["picks"].items()}
        info = {int(h): v for h, v in state["info"].items()}
        importance = state["importance"]
        log("reused saved cross-fit models, OOF and picks")
    else:
        fit = lambda xs, ys: fit_catboost(xs, ys, args.threads)  # noqa: E731
        oof, picks, models, info = crossfit(x, y, arena, half_row, inner_row, s1_half, fit)
        for h, model in models.items():
            model.save_model(str(work / f"model_half{h}.cbm"))
        np.save(work / "oof.npy", oof)
        np.save(work / "b_rest_pair_rows.npy", data["pair_rows"])
        names = feature_names(args.extended)
        importance = {str(h): dict(zip(names, map(float, m.get_feature_importance()))) for h, m in models.items()}
        state_path.write_text(json.dumps({"c2_scores": str(Path(args.scores).resolve()), "features": names,
                                          "picks": {str(h): v for h, v in picks.items()},
                                          "info": {str(h): v for h, v in info.items()},
                                          "importance": importance}, default=float) + "\n")

    results = {}
    for v in VARIANTS:
        thr = np.where(half_row == 1, picks[0][v]["threshold"], picks[1][v]["threshold"])
        result = arena.score(accept_per_row(arena, oof, thr, v), full=True)
        per[v] = result.pop("per_s1")
        results[v] = slim(result)
        log(f"cross-fit {v}: {result['macro_f05']:.6f}")
    boot = {}
    for v in VARIANTS:
        for base in ("v1_one_owner", "c2_blend", "c2_blend_name_only", "c2_grid_model",
                     "c2_grid_model_then_name_only"):
            boot[f"{v}_minus_{base}"] = component_bootstrap(per[base], per[v], component, n=args.bootstrap,
                                                            seed=20260926, population=population)
    boot["model_then_name_only_minus_model"] = component_bootstrap(per["model"], per["model_then_name_only"],
                                                                   component, n=args.bootstrap, seed=20260926,
                                                                   population=population)
    preferred = max(VARIANTS, key=lambda v: results[v]["macro_f05"])
    grid_full, best_full = arena.sweep(oof, None, population)
    report = {
        "question": "C2-aware pair CatBoost (B_REST 2-fold component cross-fit) vs v1 / C2 blend / C2 + name-only",
        "scope": "arena B_REST only; SLICE and HOLDOUT-2 labels unread; C2 A-partition scores unused",
        "c2_scores": str(Path(args.scores).resolve()),
        "c2_complete": json.loads((Path(args.scores) / "complete.json").read_text()).get("checkpoint"),
        "features": feature_names(args.extended), "extended": bool(args.extended),
        "catboost": CATBOOST, "fit_rows_cap": FIT_ROWS, "seed": SEED, "threads": args.threads,
        "candidate_rule": "tree >= 0.01 or exact normalized name or exact-key self (pair_tree_arena.build_pairs)",
        "split": "half = component_id % 2 (outer), inner = (component_id // 2) % 2 for threshold choice",
        "population": {"b_rest_s1": int(population.sum()), "candidate_rows": int(len(y)),
                       "candidate_true_rows": int(y.sum()), "true_links": int(arena.true_count[population].sum()),
                       "half_s1": [int((population & (s1_half == h)).sum()) for h in (0, 1)]},
        "c2_rival_shift": data["shift"],
        "baselines": baselines,
        "crossfit_picks": {str(h): picks[h] for h in picks},
        "crossfit": results,
        "bootstrap": boot,
        "preferred_variant": preferred,
        "full_oof_threshold": {v: best_full[v] for v in VARIANTS},
        "full_oof_f05_at_threshold": {v: grid_full[v][best_full[v]] for v in VARIANTS},
        "full_oof_grid": {v: {str(t): s for t, s in grid_full[v].items()} for v in VARIANTS},
        "fold_info": {str(h): info[h] for h in info},
        "feature_importance": importance,
        "holdout2_rows_read": 0, "slice_read": False,
        "elapsed_seconds": round(time.time() - T0, 1),
    }
    (work / "report.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    log(json.dumps({k: report[k] for k in ("baselines", "crossfit", "crossfit_picks", "preferred_variant",
                                           "full_oof_threshold")}, default=float))
    return report


# ----------------------------------------------------------------------------- test

def write_tsv(path, s1_ids, rec_ids, left, right, accept, prob):
    rows = np.flatnonzero(accept)
    rows = rows[np.lexsort((-prob[rows], left[rows]))]
    starts = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="left")
    stops = np.searchsorted(left[rows], np.arange(len(s1_ids)), side="right")
    tmp = path.with_suffix(".tsv.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(HEADER + "\n")
        for i, s1 in enumerate(s1_ids):
            handle.write(f"{s1}\t{','.join(rec_ids[right[rows[starts[i]:stops[i]]]])}\n")
    tmp.replace(path)
    return stops - starts


def validate_tsv(path, s1_ids, rec_index, pair_keys, n_rec, max_list=CAP):
    """Re-read a matching file: header, S1 order, LF only, list <= cap, no duplicates, one owner, subset of pairs."""
    raw = Path(path).read_bytes()
    problems = []
    if b"\r" in raw:
        problems.append("CR bytes present")
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    if lines[0] != HEADER:
        problems.append("bad header")
    body = lines[1:]
    if len(body) != len(s1_ids):
        problems.append(f"{len(body)} rows, expected {len(s1_ids)}")
    s1_pos, rec_pos, longest = [], [], 0
    for i, line in enumerate(body):
        parts = line.split("\t")
        if len(parts) != 2 or (i < len(s1_ids) and parts[0] != s1_ids[i]):
            problems.append(f"row {i} malformed or out of order")
            break
        ids = [v for v in parts[1].split(",") if v] if parts[1] else []
        if len(ids) != len(set(ids)):
            problems.append(f"row {i} has duplicate IDs")
        longest = max(longest, len(ids))
        for v in ids:
            s1_pos.append(i)
            rec_pos.append(rec_index.get(v, -1))
    rec_pos = np.asarray(rec_pos, dtype=np.int64)
    s1_pos = np.asarray(s1_pos, dtype=np.int64)
    if np.any(rec_pos < 0):
        problems.append(f"{int((rec_pos < 0).sum())} unknown record IDs")
    contested = int((np.bincount(rec_pos[rec_pos >= 0], minlength=n_rec) > 1).sum())
    if contested:
        problems.append(f"{contested} contested record IDs")
    if longest > max_list:
        problems.append(f"longest list {longest} > {max_list}")
    keys = s1_pos * n_rec + rec_pos
    pos = np.minimum(np.searchsorted(pair_keys, keys), len(pair_keys) - 1)
    outside = int((pair_keys[pos] != keys).sum()) if len(keys) else 0
    if outside:
        problems.append(f"{outside} links outside the candidate pairs")
    return {"rows": len(body), "links": int(len(rec_pos)), "longest_list": longest, "contested": contested,
            "outside_candidates": outside, "passed": not problems, "problems": problems}


def by_country(country, lengths):
    return {str(c): {"s1": int((country == c).sum()), "links": int(lengths[country == c].sum()),
                     "empty_rows": int(((country == c) & (lengths == 0)).sum()),
                     "mean_list": float(lengths[country == c].mean()), "longest_list": int(lengths[country == c].max())}
            for c in np.unique(country)}


def test_cmd(args) -> dict:
    work, out = Path(args.work), Path(args.out)
    report = json.loads((work / "report.json").read_text())
    if report["c2_scores"] != str(Path(args.scores).resolve()):
        raise SystemExit("work report was built from different arena C2 scores")
    args.extended = bool(report.get("extended", False))
    variant = args.variant or report["preferred_variant"]
    threshold = float(args.threshold if args.threshold is not None else report["full_oof_threshold"][variant])
    out.mkdir(parents=True, exist_ok=True)

    model_path = work / "final_model.cbm"
    if model_path.exists():
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(model_path))
        fit_rows = None
        log(f"loaded {model_path}")
    else:
        data = load_arena(args)
        model, fit_rows = fit_catboost(data["x"], data["label"], args.threads)
        model.save_model(str(model_path))
        del data
        log(f"final model fit on {fit_rows} B_REST rows")

    scores = Path(args.test_inputs)
    ref_table = pq.read_table(scores / "dense_test" / "reference.parquet", columns=["entity_id"] + COLUMNS)
    pool_table = pq.read_table(scores / "dense_test" / "pool.parquet", columns=["entity_id"] + COLUMNS)
    s1_ids = np.asarray(ref_table.column("entity_id").to_pylist(), dtype=object)
    rec_ids = np.asarray(pool_table.column("entity_id").to_pylist(), dtype=object)
    country = np.asarray(ref_table.column("country").to_pylist(), dtype=object)
    n_s1, n_rec = len(s1_ids), len(rec_ids)
    cache_path = Path(args.cache or work) / "test_v1_cache.npz"
    left_all = np.asarray(np.load(scores / "index" / "left.npy", mmap_mode="r")).astype(np.int64)
    right_all = np.asarray(np.load(scores / "index" / "right.npy", mmap_mode="r")).astype(np.int64)
    if cache_path.exists():
        cache = v1_cache(None, None, None, None, None, None, cache_path)
    else:
        blend_all, base_all = load_scores(scores / "probabilities.npy"), load_scores(scores / "base_probabilities.npy")
        if not (len(left_all) == len(right_all) == len(blend_all) == len(base_all)):
            raise SystemExit("test score arrays are not aligned")
        cache = v1_cache(ref_table.select(COLUMNS).to_pandas(), pool_table.select(COLUMNS).to_pandas(), left_all,
                         right_all, blend_all, base_all, cache_path)
        del blend_all, base_all
    del ref_table, pool_table
    left, right = cache["left"].astype(np.int64), cache["right"].astype(np.int64)
    n_pairs = int(cache["n"][2])
    if n_pairs != len(left_all):
        raise SystemExit("test v1 cache does not match the test pair index")
    c2 = c2_on_rows(args.test_scores, cache["cand_rows"], cache["base"], must_score=np.ones(len(left), bool),
                    n_pairs=n_pairs)
    x2 = c2_features(left, right, c2, cache["base"], n_s1, n_rec, extended=args.extended)
    prob = proba(model, np.hstack([cache["v1x"], x2]).astype(np.float32))
    np.save(out / "probabilities_candidates.npy", prob)
    mult = cache["s1_mult"][left]
    accept = decode(prob, threshold, variant, left, right, cache["rec_empty"], cache["name_exact"], mult)
    pick = report["baselines"]["c2_grid"]["full_b_rest_pick"]["model_then_name_only"]
    plain_w = float(args.plain_weight if args.plain_weight is not None else pick["weight"])
    plain_t = float(args.plain_threshold if args.plain_threshold is not None else pick["threshold"])
    plain_prob = c2_blend(x2[:, 0], cache["base"], plain_w)
    plain = decode(plain_prob, plain_t, "model_then_name_only", left, right, cache["rec_empty"],
                   cache["name_exact"], mult)
    france = country[left] == "France"
    if not np.any(country == "France"):
        raise SystemExit("no France rows in the test reference; check country labels")
    mixed = np.where(france, plain, accept)
    mixed_prob = np.where(france, plain_prob, prob)
    outputs = [("model", accept, prob), ("model_us_india_c2_france", mixed, mixed_prob),
               ("c2_blend_name_only", plain, plain_prob)]
    france_ft = None
    if args.france_scores:
        # French rows from a separately fine-tuned C2: plain blend + name-only on French rows only (no labels).
        fr_rows = np.flatnonzero(france)
        c2_fr = c2_on_rows(args.france_scores, cache["cand_rows"], cache["base"], must_score=france, n_pairs=n_pairs)
        fr_w = float(args.france_weight if args.france_weight is not None else plain_w)
        fr_t = float(args.france_threshold if args.france_threshold is not None else plain_t)
        fr_prob = np.zeros(len(left), dtype=np.float32)
        fr_prob[fr_rows] = c2_blend(np.where(np.isnan(c2_fr[fr_rows]), -1.0, c2_fr[fr_rows]), cache["base"][fr_rows],
                                    fr_w)
        fr_accept = np.zeros(len(left), dtype=bool)
        fr_accept[fr_rows] = decode(fr_prob[fr_rows], fr_t, "model_then_name_only", left[fr_rows], right[fr_rows],
                                    cache["rec_empty"][fr_rows], cache["name_exact"][fr_rows], mult[fr_rows])
        outputs.append(("model_us_india_c2france_ft", np.where(france, fr_accept, accept),
                        np.where(france, fr_prob, prob)))
        france_ft = {"scores": str(Path(args.france_scores).resolve()), "weight": fr_w, "threshold": fr_t,
                     "complete": json.loads((Path(args.france_scores) / "complete.json").read_text()),
                     "france_links": int(fr_accept.sum()), "france_links_plain_c2": int((plain & france).sum()),
                     "france_links_model": int((accept & france).sum()),
                     "overlap_with_plain_c2": int((fr_accept & plain & france).sum())}
        log(f"France fine-tune rows: {france_ft}")

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
        log(f"{name}: {json.dumps(files[name], default=float)}")
    diff = {}
    for code in np.unique(country):
        m = country[left] == code
        diff[str(code)] = {"model_only_links": int((accept & ~plain & m).sum()),
                           "c2_name_only_only_links": int((plain & ~accept & m).sum())}
    arena_shift = report.get("c2_rival_shift")
    result = {
        "variant": variant, "threshold": threshold, "model": str(model_path), "model_sha256": sha256(model_path),
        "plain_c2_name_only": {"weight": plain_w, "threshold": plain_t, "gate": GATE},
        "france_fine_tune": france_ft,
        "final_fit_rows": fit_rows, "arena_report": str(work / "report.json"),
        "arena_crossfit": report["crossfit"][variant], "arena_c2_scores": report["c2_scores"],
        "test_c2_scores": str(Path(args.test_scores).resolve()),
        "test_c2_complete": json.loads((Path(args.test_scores) / "complete.json").read_text()).get("checkpoint"),
        "inputs": {name: sha256(scores / name) for name in
                   ("index/left.npy", "index/right.npy", "probabilities.npy", "base_probabilities.npy",
                    "dense_test/reference.parquet", "dense_test/pool.parquet")},
        "candidate_rows": int(len(left)),
        "c2_rival_rates": {"test": rival_rates(x2), "arena_b_rest": (arena_shift or {}).get("rival_rates")},
        "arena_c2_rival_shift": arena_shift,
        "files": files, "model_vs_c2_name_only_by_country": diff, "labels_read": False,
        "elapsed_seconds": round(time.time() - T0, 1),
    }
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
        p.add_argument("--scores", required=True, help="C2 arena B_REST score dir (scores.npy, scored_mask.npy)")
        p.add_argument("--work", required=True, help="Fold models, OOF and report.json (one dir per checkpoint)")
        p.add_argument("--cache", help="Dir holding the C2-independent v1 feature caches (default: --work)")
        p.add_argument("--threads", type=int, default=int(os.environ.get("TREE_THREADS", "24")))
        if name == "evaluate":
            p.add_argument("--extended", action="store_true", help="Append C2_EXTRA features")
            p.add_argument("--bootstrap", type=int, default=1000)
            p.add_argument("--reuse-fits", action="store_true", help="Reuse fold models/OOF saved in --work")
        else:
            p.add_argument("--test-inputs", default="/opt/az-ml-new/test_inputs")
            p.add_argument("--test-scores", required=True, help="C2 test score dir (needs complete.json)")
            p.add_argument("--variant", choices=VARIANTS)
            p.add_argument("--threshold", type=float)
            p.add_argument("--plain-weight", type=float, help="C2 weight of the plain file (default: arena pick)")
            p.add_argument("--plain-threshold", type=float, help="Threshold of the plain file (default: arena pick)")
            p.add_argument("--france-scores", help="Optional C2-France test score dir for a French-rows variant")
            p.add_argument("--france-weight", type=float, help="C2-France blend weight (default: plain weight)")
            p.add_argument("--france-threshold", type=float, help="C2-France threshold (default: plain threshold)")
            p.add_argument("--out", required=True)
    args = parser.parse_args()
    (evaluate_cmd if args.command == "evaluate" else test_cmd)(args)


if __name__ == "__main__":
    main()
