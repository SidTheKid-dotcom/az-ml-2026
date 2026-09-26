"""Streaming resolver on arena artifacts: a bounded pair model plus the arena decision rules.

Pairs are streamed from parquet in batches and must be strictly increasing by
(left_idx, right_idx); this rejects duplicate pairs and any order that could
misalign the memory-mapped score arrays. Scores, labels, probabilities and the
decoded acceptances are memory-mapped NumPy arrays, so no pair table is ever
materialized in pandas.

The model (CatBoost or logistic regression) is fit on whole claim components of
the training partition (A), with a component-disjoint early-stopping split.
Thresholds and rules are chosen on the tune partition (B_REST); other partitions
are only reported. Every S1 of a population counts, including S1s without
candidates, and ``true_count`` includes unretrieved links. HOLDOUT-2 labels are
never opened: label directories must prove ``holdout2_rows_read == 0``.

Smoke run on a tiny synthetic fixture:
    PYTHONPATH=src python -m ber.resolver smoke --out /tmp/ber_resolver_smoke
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ber.arena_decision import component_bootstrap, evaluate, expected_f05_decode, iterated_best_response, one_owner

PARTITIONS = {"A": 0, "B_REST": 1, "SLICE": 2}
HOLDOUT2_MARKERS = ("holdout2", "holdout_2", "holdout-2")
DEFAULT_THRESHOLDS = tuple(round(float(t), 3) for t in np.arange(0.30, 0.9751, 0.025))
PAIR_COLUMNS = ("left_idx", "right_idx")
INT32_MAX = np.iinfo(np.int32).max


def log(message: str) -> None:
    print(time.strftime("%H:%M:%S"), message, flush=True)


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_json(path, value) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(_jsonable(value), stream, indent=2, allow_nan=False, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def atomic_memmap(path, dtype, length, fill=None):
    """Yield a writable .npy memmap that replaces ``path`` only if the block completes."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        array = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=(int(length),))
        if fill is not None:
            array[:] = fill
        yield array
        array.flush()
        del array
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_out(out) -> Path:
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing to overwrite nonempty resolver output: {out}")
    out.mkdir(parents=True, exist_ok=True)
    return out


def guard_labels(labels_dir, n_pairs: int, n_s1: int):
    """Open arena labels only after proving they contain no HOLDOUT-2 rows."""
    labels_dir = Path(labels_dir).resolve()
    if any(marker in part.lower() for part in labels_dir.parts for marker in HOLDOUT2_MARKERS):
        raise ValueError(f"Refusing to open HOLDOUT-2 labels: {labels_dir}")
    summary_path = labels_dir / "labels_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if summary.get("holdout2_rows_read") != 0:
        raise ValueError("Label artifact does not prove zero HOLDOUT-2 label rows (labels_summary.json)")
    if summary.get("pairs", n_pairs) != n_pairs:
        raise ValueError(f"labels_summary.json covers {summary['pairs']} pairs; pairs file has {n_pairs}")
    label = np.load(labels_dir / "label.npy", mmap_mode="r")
    true_count = np.load(labels_dir / "true_count.npy", mmap_mode="r")
    if label.shape != (n_pairs,) or label.dtype != np.bool_:
        raise ValueError(f"label.npy must be bool with {n_pairs} rows; got {label.dtype} {label.shape}")
    if true_count.shape != (n_s1,):
        raise ValueError(f"true_count.npy must have one row per S1 ({n_s1}); got {true_count.shape}")
    return label, np.asarray(true_count, dtype=np.int64), summary


def open_scores(score_paths: dict, n_pairs: int) -> dict:
    if not score_paths:
        raise ValueError("At least one aligned score array is required")
    scores = {}
    for name, path in score_paths.items():
        array = np.load(path, mmap_mode="r")
        if array.shape != (n_pairs,):
            raise ValueError(f"Score array {name} has shape {array.shape}; pairs file has {n_pairs} rows")
        scores[name] = array
    return scores


def stream_pairs(pairs_path, scores: dict, label, n_s1: int, out: Path, batch_rows: int) -> dict:
    """Validate pair order batch by batch, cache left/right as memmaps and collect per-S1 aggregates."""
    parquet = pq.ParquetFile(pairs_path)
    missing = set(PAIR_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"Pairs file lacks columns {sorted(missing)}")
    n = parquet.metadata.num_rows
    count = np.zeros(n_s1, dtype=np.int64)
    retrieved = np.zeros(n_s1, dtype=np.int64)
    s1_max = {name: np.full(n_s1, -np.inf, dtype=np.float32) for name in scores}
    last, pos = None, 0
    with atomic_memmap(out / "pairs_left_idx.npy", np.int32, n) as left_mm, \
            atomic_memmap(out / "pairs_right_idx.npy", np.int32, n) as right_mm:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=list(PAIR_COLUMNS)):
            left = batch.column("left_idx").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            right = batch.column("right_idx").to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            if left.size == 0:
                continue
            if left.min() < 0 or left.max() >= n_s1:
                raise ValueError("left_idx outside the S1 reference")
            if right.min() < 0 or right.max() > INT32_MAX:
                raise ValueError("right_idx outside the supported record range")
            dl, dr = np.diff(left), np.diff(right)
            bad = np.flatnonzero((dl < 0) | ((dl == 0) & (dr <= 0)))
            if last is not None and (left[0] < last[0] or (left[0] == last[0] and right[0] <= last[1])):
                bad = np.concatenate([[-1], bad])
            if bad.size:
                raise ValueError(f"Pairs are not strictly increasing by (left_idx, right_idx) at row "
                                 f"{pos + int(bad[0]) + 1}: duplicate or mismatched pair order")
            last = (int(left[-1]), int(right[-1]))
            stop = pos + len(left)
            left_mm[pos:stop] = left
            right_mm[pos:stop] = right
            groups, starts = np.unique(left, return_index=True)
            count[groups] += np.diff(np.append(starts, len(left)))
            retrieved[groups] += np.add.reduceat(np.asarray(label[pos:stop], dtype=np.int64), starts)
            for name, array in scores.items():
                x = np.asarray(array[pos:stop], dtype=np.float32)
                if not np.isfinite(x).all():
                    raise ValueError(f"Score array {name} has non-finite values in rows {pos}..{stop - 1}")
                s1_max[name][groups] = np.maximum(s1_max[name][groups], np.maximum.reduceat(x, starts))
            pos = stop
        if pos != n:
            raise ValueError(f"Streamed {pos} pairs; parquet metadata reports {n}")
    return {"n_pairs": n, "count": count, "retrieved": retrieved, "s1_max": s1_max}


def feature_names(score_names) -> list:
    names = []
    for name in score_names:
        names += [name, f"{name}_gap_to_s1_max", f"{name}_s1_max"]
    return names + ["log1p_s1_candidates"]


def build_features(rows, left_values, scores: dict, stats: dict) -> np.ndarray:
    columns = []
    for name, array in scores.items():
        x = np.asarray(array[rows], dtype=np.float32)
        top = stats["s1_max"][name][left_values]
        columns += [x, x - top, top]
    columns.append(np.log1p(stats["count"][left_values]).astype(np.float32))
    return np.column_stack(columns)


def rows_for_s1(s1_mask, left, batch_rows: int) -> np.ndarray:
    parts = []
    for start in range(0, len(left), batch_rows):
        hit = np.flatnonzero(s1_mask[left[start:start + batch_rows]])
        if hit.size:
            parts.append(hit + start)
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)


def select_training(partition, component, count, *, train_code, max_rows, early_stop_fraction, seed):
    """Whole training-partition components up to ``max_rows`` pairs, split component-disjointly for early stopping."""
    train_s1 = partition == train_code
    _, compact = np.unique(component, return_inverse=True)
    n_comp = int(compact.max()) + 1
    has_train = np.bincount(compact, weights=train_s1, minlength=n_comp) > 0
    has_other = np.bincount(compact, weights=~train_s1, minlength=n_comp) > 0
    if np.any(has_train & has_other):
        raise ValueError(f"{int((has_train & has_other).sum())} claim components span the training partition "
                         "and another partition")
    comp_rows = np.bincount(compact, weights=count, minlength=n_comp).astype(np.int64)
    order = np.random.default_rng(seed).permutation(np.flatnonzero(has_train & (comp_rows > 0)))
    if order.size == 0:
        raise ValueError("Training partition has no candidate pairs")
    take = order[np.cumsum(comp_rows[order]) <= max_rows]
    if take.size == 0:
        take = order[:1]
    n_es = max(1, int(round(len(take) * early_stop_fraction))) if early_stop_fraction > 0 and len(take) >= 2 else 0
    fit_comp = np.zeros(n_comp, dtype=bool)
    es_comp = np.zeros(n_comp, dtype=bool)
    es_comp[take[:n_es]] = True
    fit_comp[take[n_es:]] = True
    fit_s1, es_s1 = fit_comp[compact], es_comp[compact]
    info = {"train_partition_components": int(has_train.sum()), "fit_components": int(fit_comp.sum()),
            "early_stop_components": int(es_comp.sum()), "fit_s1": int(fit_s1.sum()), "early_stop_s1": int(es_s1.sum()),
            "max_rows": int(max_rows), "early_stop_fraction": early_stop_fraction, "seed": seed}
    return fit_s1, es_s1, info


class PairModel:
    def __init__(self, kind, *, seed=0, iterations=400, depth=6, learning_rate=0.08, threads=8):
        if kind not in ("catboost", "logistic"):
            raise ValueError(f"Unknown model kind {kind}")
        self.kind, self.seed, self.iterations = kind, seed, iterations
        self.depth, self.learning_rate, self.threads = depth, learning_rate, threads
        self.info = {}

    def fit(self, X, y, X_es=None, y_es=None):
        if y.size == 0 or y.min() == y.max():
            raise ValueError("Training rows need both classes")
        use_es = X_es is not None and len(y_es) > 0 and y_es.min() != y_es.max()
        if self.kind == "catboost":
            from catboost import CatBoostClassifier
            self.model = CatBoostClassifier(
                iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
                loss_function="Logloss", random_seed=self.seed, thread_count=self.threads,
                verbose=False, allow_writing_files=False)
            self.model.fit(X, y, eval_set=(X_es, y_es) if use_es else None,
                           early_stopping_rounds=50 if use_es else None, use_best_model=use_es)
            self.info = {"tree_count": int(self.model.tree_count_), "best_iteration": self.model.get_best_iteration()}
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            self.model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=self.iterations, C=1.0))
            self.model.fit(X, y)
            self.info = {"n_iter": int(self.model[-1].n_iter_[0])}
        self.info.update(early_stopping=bool(use_es), fit_rows=int(len(y)), fit_positive=int(y.sum()))
        if use_es:
            from sklearn.metrics import log_loss, roc_auc_score
            p = self.predict(X_es)
            self.info.update(early_stop_rows=int(len(y_es)), early_stop_logloss=float(log_loss(y_es, p, labels=[0, 1])),
                             early_stop_auc=float(roc_auc_score(y_es, p)))
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def save(self, out: Path, names) -> Path:
        if self.kind == "catboost":
            path = out / "model.cbm"
            temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            try:
                self.model.save_model(str(temporary))
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            path = out / "model_logistic.json"
            scaler, lr = self.model[0], self.model[-1]
            atomic_json(path, {"features": list(names), "mean": scaler.mean_, "scale": scaler.scale_,
                               "coef": lr.coef_[0], "intercept": float(lr.intercept_[0])})
        return path


def _strip_per_s1(result):
    result = dict(result)
    return result.pop("per_s1"), result


def decide(prob, left_mm, right_mm, label, true_count, partition, component, scores, *, tune, report, lam, max_k,
           thresholds, baseline, bootstrap, seed, country, batch_rows):
    """Evaluate threshold+one-owner, expected-F0.5 and iterated best response; choose on ``tune`` only."""
    n_s1 = len(true_count)
    pops = {name: partition == PARTITIONS[name] for name in (tune, *report)}
    scope = np.zeros(n_s1, dtype=bool)
    for mask in pops.values():
        scope |= mask
    rows = rows_for_s1(scope, left_mm, batch_rows)
    s1 = np.asarray(left_mm[rows], dtype=np.int64)
    rec = np.asarray(right_mm[rows], dtype=np.int64)
    p = np.asarray(prob[rows], dtype=np.float64)
    lab = np.asarray(label[rows], dtype=bool)

    def score_all(accept):
        per, res = {}, {}
        for name, mask in pops.items():
            per[name], res[name] = _strip_per_s1(evaluate(accept, s1, lab, true_count, population=mask,
                                                          country=country, return_per_s1=True))
            res[name]["s1_without_candidates"] = int((mask & (np.bincount(s1, minlength=n_s1) == 0)).sum())
        return per, res

    sweep = []
    for t in thresholds:
        r = evaluate(one_owner(p >= t, s1, rec, p), s1, lab, true_count, population=pops[tune])
        sweep.append({"threshold": t, "macro_f05": r["macro_f05"], "precision": r["precision"],
                      "recall": r["recall"], "false_links": r["false_links"]})
    best = max(sweep, key=lambda r: (round(r["macro_f05"], 12), r["precision"], r["threshold"]))
    accepts = {"threshold_one_owner": one_owner(p >= best["threshold"], s1, rec, p)}
    accepts["expected_f05"], _ = expected_f05_decode(p, s1, lam, max_k, n_s1=n_s1)
    accepts["iterated_best_response"], ibr_info = iterated_best_response(p, s1, rec, lam, max_k, n_s1=n_s1)
    if baseline is not None:
        name, threshold = baseline
        b = np.asarray(scores[name][rows], dtype=np.float64)
        accepts["baseline"] = one_owner(b >= threshold, s1, rec, b)

    per_s1, results = {}, {}
    for rule, accept in accepts.items():
        per_s1[rule], results[rule] = score_all(accept)
        accepted_recs = rec[accept]
        results[rule]["records_with_multiple_owners"] = int(accepted_recs.size - np.unique(accepted_recs).size)
    candidates = [r for r in accepts if r != "baseline"]
    selected = max(candidates, key=lambda r: (round(results[r][tune]["macro_f05"], 12), results[r][tune]["precision"]))
    comparator = "baseline" if baseline is not None else max(
        (r for r in candidates if r != selected), key=lambda r: results[r][tune]["macro_f05"])
    paired = {name: component_bootstrap(per_s1[comparator][name], per_s1[selected][name], component, n=bootstrap,
                                        seed=seed, population=mask)
              for name, mask in pops.items()}
    return {
        "rows": rows, "accept": accepts[selected], "selected_rule": selected, "comparator": comparator,
        "threshold_sweep": sweep, "chosen_threshold": best["threshold"], "results": results,
        "paired_bootstrap_selected_minus_comparator": paired,
        "ibr": {k: v for k, v in ibr_info.items() if k != "k"}, "decoded_pairs": int(len(rows)),
    }


def run(pairs, scores, labels, partition, components, out, *, reference=None, model="catboost",
        train_partition="A", tune_partition="B_REST", report_partitions=("SLICE",), max_train_rows=2_000_000,
        early_stop_fraction=0.1, iterations=400, depth=6, learning_rate=0.08, threads=8, seed=0,
        batch_rows=1_000_000, lam=None, max_k=15, thresholds=DEFAULT_THRESHOLDS, baseline=None, bootstrap=2000):
    started = time.time()
    timings = {}
    for name in (train_partition, tune_partition, *report_partitions):
        if name not in PARTITIONS:
            raise ValueError(f"Unknown partition {name}")
    if train_partition in (tune_partition, *report_partitions):
        raise ValueError("The training partition must differ from tune/report partitions")
    if baseline is not None and baseline[0] not in scores:
        raise ValueError(f"Baseline score {baseline[0]} is not among the supplied scores")
    out = _prepare_out(out)
    partition_arr = np.asarray(np.load(partition, mmap_mode="r"), dtype=np.uint8)
    component_arr = np.asarray(np.load(components, mmap_mode="r"), dtype=np.int64)
    n_s1 = len(partition_arr)
    if component_arr.shape != (n_s1,):
        raise ValueError("component_ids must have one row per S1 of the partition file")
    if not np.isin(partition_arr, list(PARTITIONS.values())).all():
        raise ValueError("Partition file contains unknown codes")
    country = None
    if reference is not None:
        country = pq.read_table(reference, columns=["country"]).column("country").to_numpy(zero_copy_only=False)
        if len(country) != n_s1:
            raise ValueError("Reference country column is not aligned to the S1 partition")
    n_pairs = pq.ParquetFile(pairs).metadata.num_rows
    label, true_count, label_summary = guard_labels(labels, n_pairs, n_s1)
    score_arrays = open_scores(scores, n_pairs)

    t0 = time.time()
    stats = stream_pairs(pairs, score_arrays, label, n_s1, out, batch_rows)
    if np.any(stats["retrieved"] > true_count):
        raise ValueError("true_count is smaller than the retrieved true links of some S1")
    left_mm = np.load(out / "pairs_left_idx.npy", mmap_mode="r")
    right_mm = np.load(out / "pairs_right_idx.npy", mmap_mode="r")
    timings["stream_validate"] = time.time() - t0
    log(f"streamed {n_pairs} pairs over {n_s1} S1s")

    t0 = time.time()
    fit_s1, es_s1, train_info = select_training(
        partition_arr, component_arr, stats["count"], train_code=PARTITIONS[train_partition],
        max_rows=max_train_rows, early_stop_fraction=early_stop_fraction, seed=seed)
    fit_rows = rows_for_s1(fit_s1, left_mm, batch_rows)
    es_rows = rows_for_s1(es_s1, left_mm, batch_rows)
    X = build_features(fit_rows, left_mm[fit_rows], score_arrays, stats)
    y = np.asarray(label[fit_rows], dtype=np.int8)
    X_es = build_features(es_rows, left_mm[es_rows], score_arrays, stats) if es_rows.size else None
    y_es = np.asarray(label[es_rows], dtype=np.int8) if es_rows.size else None
    names = feature_names(score_arrays)
    pair_model = PairModel(model, seed=seed, iterations=iterations, depth=depth, learning_rate=learning_rate,
                           threads=threads).fit(X, y, X_es, y_es)
    del X, y, X_es, y_es
    model_path = pair_model.save(out, names)
    unretrieved = (true_count - stats["retrieved"])[fit_s1]
    lam_used = float(unretrieved.mean()) if lam is None else float(lam)
    timings["fit"] = time.time() - t0
    log(f"fit {model} on {len(fit_rows)} rows; lam={lam_used:.5f}")

    t0 = time.time()
    with atomic_memmap(out / "prob.npy", np.float32, n_pairs) as prob:
        for start in range(0, n_pairs, batch_rows):
            window = slice(start, min(n_pairs, start + batch_rows))
            values = pair_model.predict(build_features(window, left_mm[window], score_arrays, stats))
            if not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
                raise ValueError(f"Model emitted invalid probabilities in rows {window.start}..{window.stop - 1}")
            prob[window] = values
    prob = np.load(out / "prob.npy", mmap_mode="r")
    timings["predict"] = time.time() - t0

    t0 = time.time()
    decision = decide(prob, left_mm, right_mm, label, true_count, partition_arr, component_arr, score_arrays,
                      tune=tune_partition, report=tuple(report_partitions), lam=lam_used, max_k=max_k,
                      thresholds=tuple(thresholds), baseline=baseline, bootstrap=bootstrap, seed=seed,
                      country=country, batch_rows=batch_rows)
    with atomic_memmap(out / "accept.npy", np.bool_, n_pairs, fill=False) as accept:
        accept[decision.pop("rows")] = decision.pop("accept")
    timings["decide"] = time.time() - t0
    log(f"selected {decision['selected_rule']}: tune macro F0.5 "
        f"{decision['results'][decision['selected_rule']][tune_partition]['macro_f05']:.6f}")

    outputs = {name: out / name for name in ("pairs_left_idx.npy", "pairs_right_idx.npy", "prob.npy", "accept.npy")}
    outputs[model_path.name] = model_path
    inputs = {"pairs": Path(pairs), "label": Path(labels) / "label.npy", "true_count": Path(labels) / "true_count.npy",
              "partition": Path(partition), "components": Path(components),
              **{f"score:{k}": Path(v) for k, v in scores.items()}}
    if reference is not None:
        inputs["reference"] = Path(reference)
    timings["total"] = time.time() - started
    metadata = {
        "kind": "ber.resolver", "holdout2_labels_opened": False, "pandas_pair_rows_loaded": 0,
        "config": {"model": model, "train_partition": train_partition, "tune_partition": tune_partition,
                   "report_partitions": list(report_partitions), "max_train_rows": max_train_rows,
                   "early_stop_fraction": early_stop_fraction, "iterations": iterations, "depth": depth,
                   "learning_rate": learning_rate, "threads": threads, "seed": seed, "batch_rows": batch_rows,
                   "lam": lam, "max_k": max_k, "thresholds": list(thresholds), "bootstrap": bootstrap,
                   "baseline": list(baseline) if baseline is not None else None},
        "inputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in inputs.items()},
        "label_summary": label_summary,
        "counts": {"pairs": n_pairs, "s1": n_s1, "s1_without_candidates": int((stats["count"] == 0).sum()),
                   "partition_s1": {k: int((partition_arr == c).sum()) for k, c in PARTITIONS.items()}},
        "features": names, "training": {**train_info, "fit_rows": int(len(fit_rows)), "early_stop_rows": int(len(es_rows)),
                                        "model": pair_model.info},
        "lam_used": lam_used, "lam_source": "fit S1 mean unretrieved links" if lam is None else "config",
        **decision,
        "outputs": {k: {"path": str(v), "sha256": sha256(v)} for k, v in outputs.items()},
        "timings_seconds": timings,
    }
    atomic_json(out / "metadata.json", metadata)
    return metadata


def make_fixture(root, *, n_s1=90, n_distractors=300, seed=7) -> dict:
    """Tiny arena-shaped fixture with contests, retrieval misses, singletons and S1s without candidates."""
    root = Path(root)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    true_count = rng.choice([0, 1, 2, 3], size=n_s1, p=[0.35, 0.4, 0.2, 0.05]).astype(np.int32)
    no_candidates = {4, 11, n_s1 - 5}
    starts = np.concatenate([[0], np.cumsum(true_count)])
    n_owned = int(starts[-1])
    owner = np.concatenate([np.repeat(np.arange(n_s1), true_count), np.full(n_distractors, -1)])
    component = np.arange(n_s1, dtype=np.int64) // 3
    pairs = []
    for s in range(n_s1):
        if s in no_candidates:
            continue
        own = range(starts[s], starts[s + 1])
        chosen = {r for r in own if rng.random() < 0.85}
        for sibling in np.flatnonzero(component == component[s]):
            if sibling != s:
                chosen |= {r for r in range(starts[sibling], starts[sibling + 1]) if rng.random() < 0.5}
        chosen |= set(rng.integers(n_owned, n_owned + n_distractors, size=int(rng.integers(1, 5))).tolist())
        pairs += [(s, r) for r in sorted(chosen)]
    left = np.array([p[0] for p in pairs], dtype=np.int32)
    right = np.array([p[1] for p in pairs], dtype=np.int32)
    label = owner[right] == left
    signal = np.where(label, 1.0, -1.0)
    blend = (1 / (1 + np.exp(-(2.5 * signal + rng.normal(0, 1.2, len(left)))))).astype(np.float32)
    tree = (1 / (1 + np.exp(-(2.0 * signal + rng.normal(0, 1.5, len(left)))))).astype(np.float32)
    partition = np.where(component % 5 < 3, PARTITIONS["A"],
                         np.where(component % 5 == 3, PARTITIONS["B_REST"], PARTITIONS["SLICE"])).astype(np.uint8)
    paths = {"pairs": root / "pairs.parquet", "labels": root / "labels", "partition": root / "s1_partition.npy",
             "components": root / "component_ids.npy", "reference": root / "reference.parquet",
             "scores": {"blend": root / "blend.npy", "tree": root / "tree.npy"}}
    pq.write_table(pa.table({"left_idx": left, "right_idx": right}), paths["pairs"], row_group_size=64)
    pq.write_table(pa.table({"country": np.where(np.arange(n_s1) % 2 == 0, "US", "IN")}), paths["reference"])
    np.save(paths["scores"]["blend"], blend)
    np.save(paths["scores"]["tree"], tree)
    np.save(root / "labels" / "label.npy", label)
    np.save(root / "labels" / "true_count.npy", true_count)
    atomic_json(root / "labels" / "labels_summary.json",
                {"pairs": int(len(label)), "positive_pairs": int(label.sum()), "true_links": int(true_count.sum()),
                 "holdout2_rows_read": 0, "synthetic": True})
    np.save(paths["partition"], partition)
    np.save(paths["components"], component)
    return paths


def smoke(out, *, model="catboost", seed=0, batch_rows=50) -> dict:
    out = _prepare_out(out)
    paths = make_fixture(out / "fixture", seed=7)
    return run(paths["pairs"], paths["scores"], paths["labels"], paths["partition"], paths["components"],
               out / "run", reference=paths["reference"], model=model, iterations=60, depth=3, threads=2,
               seed=seed, batch_rows=batch_rows, bootstrap=200, baseline=("blend", 0.5))


def _score_arg(text):
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError("Scores are given as NAME=PATH.npy")
    return name, path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    r = commands.add_parser("run")
    r.add_argument("--pairs", required=True)
    r.add_argument("--score", action="append", type=_score_arg, required=True, help="NAME=PATH.npy aligned to pairs")
    r.add_argument("--labels", required=True, help="Directory with label.npy, true_count.npy, labels_summary.json")
    r.add_argument("--partition", required=True)
    r.add_argument("--components", required=True)
    r.add_argument("--reference", help="Reference parquet with a country column aligned to S1s")
    r.add_argument("--out", required=True)
    r.add_argument("--model", choices=("catboost", "logistic"), default="catboost")
    r.add_argument("--report-partition", action="append", default=None)
    r.add_argument("--max-train-rows", type=int, default=2_000_000)
    r.add_argument("--early-stop-fraction", type=float, default=0.1)
    r.add_argument("--iterations", type=int, default=400)
    r.add_argument("--depth", type=int, default=6)
    r.add_argument("--learning-rate", type=float, default=0.08)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--batch-rows", type=int, default=1_000_000)
    r.add_argument("--lam", type=float, default=None, help="Default: mean unretrieved links of fit S1s")
    r.add_argument("--max-k", type=int, default=15)
    r.add_argument("--baseline-score")
    r.add_argument("--baseline-threshold", type=float, default=0.775)
    r.add_argument("--bootstrap", type=int, default=2000)
    s = commands.add_parser("smoke")
    s.add_argument("--out", required=True)
    s.add_argument("--model", choices=("catboost", "logistic"), default="catboost")
    s.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.command == "smoke":
        result = smoke(args.out, model=args.model, seed=args.seed)
    else:
        scores = dict(args.score)
        if len(scores) != len(args.score):
            parser.error("Duplicate score names")
        result = run(args.pairs, scores, args.labels, args.partition, args.components, args.out,
                     reference=args.reference, model=args.model,
                     report_partitions=tuple(args.report_partition or ("SLICE",)),
                     max_train_rows=args.max_train_rows, early_stop_fraction=args.early_stop_fraction,
                     iterations=args.iterations, depth=args.depth, learning_rate=args.learning_rate,
                     threads=args.threads, seed=args.seed, batch_rows=args.batch_rows, lam=args.lam,
                     max_k=args.max_k, bootstrap=args.bootstrap,
                     baseline=(args.baseline_score, args.baseline_threshold) if args.baseline_score else None)
    selected = result["selected_rule"]
    print(json.dumps(_jsonable({"selected_rule": selected, "tune": result["results"][selected][result["config"]
                                ["tune_partition"]]["macro_f05"], "outputs": result["outputs"]}), indent=2))


if __name__ == "__main__":
    main()
