"""CPU-only, competitor-complete U8t development evaluation.

    python -m ber.takeover_eval baseline --world brest_world.npz --output baseline
    python -m ber.takeover_eval screen --world brest_world.npz --scores bge.parquet \
        --scores byte.parquet --s1-rows screen_s1.npy --ids brest_screen.parquet --output screen

pair_row is a GLOBAL WORLD ROW, never a position in a score/band parquet.
Decoding is threshold >=, one owner (score, lower S1, lower row), then cap 11
(score, lower row). There is no reassignment after the cap. Only finite,
complete-case scored rows of selected S1s in .001 < baseline_p < .999 change;
all other probabilities AND thresholds stay at baseline. Even dominated fixed
claimants and pre-cap winners remain available when a changed row loses ownership.

Selection uses nested *small-model* component cross-fitting: outer fold f is
excluded from both fitting and the inner OOF threshold/blend search. The final
outer-OOF probabilities are decoded jointly against the whole saved world.
This is REUSED DEVELOPMENT selection, not a fresh or fully nested validation:
the cached baseline OOF features were themselves fitted on other B_REST folds.
Joint ownership and the inherited component graph can add dependence. Bootstrap
intervals condition on this selection and do not establish promotion readiness.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


BASE_THRESHOLD = 0.725
CAP = 11
BAND = (0.001, 0.999)
REFERENCE_THRESHOLDS = (0.70, 0.70, 0.70, 0.725, 0.70)
THRESHOLDS = (0.50, 0.60, 0.65, 0.70, 0.725, 0.75, 0.80, 0.85, 0.90)
BLENDS = (0.0, 0.10, 0.25, 0.50)
METHODS = ("blend_baseline_control", "convex_blend", "baseline_recalibration", "logistic_fusion")
CONTROLS = {"convex_blend": "blend_baseline_control", "logistic_fusion": "baseline_recalibration"}
CLASSIFIER = {"C": 0.1, "solver": "lbfgs", "max_iter": 400, "tol": 1e-8}
BYTE_COLUMNS = ("copy_nll", "nonmatch_nll", "token_count", "log_ratio", "log_ratio_normalized")
COUNTRIES = {0: "India", 1: "US"}
KNOWN_WORLD_SHA = "3308381202e5a78224c2ac7adb82f68115c4119435ee85a33c290fc7647297af"
LIMITATION = (
    "REUSED DEVELOPMENT selection on declared B_REST S1s, not fresh holdout or unseen-test performance. "
    "Cached baseline OOF features were fitted on other B_REST folds, so small-model cross-fitting is "
    "not a fully nested refit of the original stack. Joint ownership and inherited components may "
    "also induce dependence. Component-bootstrap intervals condition on model/parameter selection. "
    "Promotion requires a fresh held-out check or refitting the original feature stack."
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def array_digest(values):
    """Portable fingerprint for an ordered set of global integer row indices."""
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def _integer(values, name, size=None):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in "iu" or (size is not None and len(a) != size):
        raise ValueError(f"{name}: expected an aligned 1D integer array")
    if a.size and a.dtype.kind == "u" and a.max() > np.iinfo(np.int64).max:
        raise ValueError(f"{name}: integer overflow")
    return a


def _boolean(values, name, size):
    a = np.asarray(values)
    if a.ndim != 1 or len(a) != size or a.dtype.kind not in "biu" or not np.isin(a, (0, 1)).all():
        raise ValueError(f"{name}: expected an aligned boolean/0-1 array")
    return a.astype(bool, copy=False)


def _unique_rows(values, name, size):
    a = _integer(values, name).astype(np.int64, copy=False)
    if a.size and (a.min() < 0 or a.max() >= size):
        raise ValueError(f"{name}: row index outside [0, {size})")
    if len(np.unique(a)) != len(a):
        raise ValueError(f"{name}: duplicate row index")
    return a


def entity_f05(tp, count, truth):
    denominator = 4.0 * np.asarray(count) + truth
    result = np.divide(5.0 * np.asarray(tp), denominator, out=np.zeros_like(denominator, dtype=float),
                       where=denominator > 0)
    result[denominator == 0] = 1.0
    return result


def logit(prob):
    p = np.clip(np.asarray(prob, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p) - np.log1p(-p)


class OwnerIndex:
    """One static record/S1/row sort, reused for all score/threshold changes."""

    def __init__(self, left, right, rows=None):
        self.rows = np.arange(len(left), dtype=np.int64) if rows is None else np.asarray(rows, dtype=np.int64)
        self.left, self.right = np.asarray(left), np.asarray(right)
        self.order = np.lexsort((self.rows, self.left, self.right))
        r = self.right[self.order]
        self.starts = np.flatnonzero(np.r_[True, r[1:] != r[:-1]]) if len(r) else np.empty(0, np.int64)
        self.sizes = np.diff(np.r_[self.starts, len(r)])

    def winners(self, prob, thresholds=BASE_THRESHOLD):
        p = np.asarray(prob)
        if p.ndim != 1 or len(p) != len(self.rows) or not np.isfinite(p).all():
            raise ValueError("Owner scores must be finite and aligned")
        threshold = np.asarray(thresholds, dtype=p.dtype)
        if threshold.ndim > 1 or (threshold.ndim == 1 and threshold.shape != p.shape):
            raise ValueError("Thresholds must be scalar or pair-aligned")
        if not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold > 1)):
            raise ValueError("Invalid probability threshold")
        if not len(p):
            return self.rows.copy()
        score = np.where(p >= threshold, p, -np.inf)[self.order]
        best = np.maximum.reduceat(score, self.starts)
        candidate = np.isfinite(score) & (score == np.repeat(best, self.sizes))
        # First max in the static order implements lower-S1/lower-global-row ties.
        positions = np.where(candidate, np.arange(len(score)), len(score))
        chosen = np.minimum.reduceat(positions, self.starts)
        return self.rows[self.order[chosen[chosen < len(score)]]]


def _cap(rows, left, scores, cap=CAP):
    if not len(rows):
        return rows
    order = np.lexsort((rows, -np.asarray(scores, dtype=float), left[rows]))
    rows = rows[order]
    group = left[rows]
    rank = np.arange(len(rows))
    starts = np.maximum.accumulate(np.where(np.r_[True, group[1:] != group[:-1]], rank, 0))
    return np.sort(rows[rank - starts < cap])


@dataclass
class World:
    left: np.ndarray
    right: np.ndarray
    label: np.ndarray
    baseline_p: np.ndarray
    true_count: np.ndarray
    population: np.ndarray
    component: np.ndarray
    country_code: np.ndarray
    partition: np.ndarray
    folds: np.ndarray
    provenance: dict = field(default_factory=dict)
    index: OwnerIndex = field(init=False, repr=False)
    retrieved: np.ndarray = field(init=False, repr=False)
    candidate_count: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.true_count = _integer(self.true_count, "true_count")
        n = len(self.true_count)
        self.left = _integer(self.left, "left")
        m = len(self.left)
        self.right = _integer(self.right, "right", m)
        self.label = _boolean(self.label, "label", m)
        self.population = _boolean(self.population, "population", n)
        for name in ("component", "country_code", "partition", "folds"):
            setattr(self, name, _integer(getattr(self, name), name, n))
        self.baseline_p = np.asarray(self.baseline_p)
        if (self.baseline_p.shape != (m,) or self.baseline_p.dtype.kind != "f"
                or not np.isfinite(self.baseline_p).all()
                or np.any((self.baseline_p < 0) | (self.baseline_p > 1))):
            raise ValueError("baseline_p must be aligned finite probabilities in [0,1]")
        if np.any(self.true_count < 0) or not self.population.any():
            raise ValueError("Invalid true_count or empty world population")
        if m and (self.left.min() < 0 or self.left.max() >= n or self.right.min() < 0):
            raise ValueError("World left/right index out of bounds")
        if not np.array_equal(self.population, self.partition == 1) or not self.population[self.left].all():
            raise ValueError("World must contain the complete B_REST population and only its candidate rows")
        pop = np.flatnonzero(self.population)
        if (np.any(self.component[pop] < 0) or not np.isin(self.country_code[pop], (0, 1)).all()
                or not np.isin(self.folds[pop], np.arange(5)).all()
                or np.any(self.folds[~self.population] != -1)):
            raise ValueError("Invalid country, component or B_REST fold codes")
        ordered = pop[np.argsort(self.component[pop], kind="stable")]
        if np.any((self.component[ordered[1:]] == self.component[ordered[:-1]])
                  & (self.folds[ordered[1:]] != self.folds[ordered[:-1]])):
            raise ValueError("Component fold leak: one component occurs in multiple folds")
        self.index = OwnerIndex(self.left, self.right)
        order = self.index.order
        if m and np.any((self.right[order[1:]] == self.right[order[:-1]])
                        & (self.left[order[1:]] == self.left[order[:-1]])):
            raise ValueError("Duplicate (left,right) world pair")
        self.retrieved = np.bincount(self.left[self.label], minlength=n)
        self.candidate_count = np.bincount(self.left, minlength=n)
        if np.any(self.retrieved > self.true_count):
            raise ValueError("true_count is smaller than retrieved true links; missing truth accounting")
        if m and np.any(np.add.reduceat(self.label[order].astype(np.int8), self.index.starts) > 1):
            raise ValueError("A record has multiple true owners")

    def decode(self, prob=None, thresholds=BASE_THRESHOLD):
        p = self.baseline_p if prob is None else np.asarray(prob, dtype=self.baseline_p.dtype)
        rows = self.index.winners(p, thresholds)
        return _cap(rows, self.left, p[rows])

    def counts(self, accepted):
        n = len(self.true_count)
        return (np.bincount(self.left[accepted], minlength=n),
                np.bincount(self.left[accepted[self.label[accepted]]], minlength=n))


def load_world(path):
    path = Path(path)
    marker = path.with_name("brest_complete.json")
    meta = json.loads(marker.read_text())
    if meta.get("state") != "complete" or meta.get("holdout2_rows_read") != 0:
        raise ValueError("A complete B_REST marker with holdout2_rows_read == 0 is required")
    sha = digest(path)
    if sha != meta.get("world_sha256"):
        raise ValueError("World SHA256 differs from brest_complete.json")
    if (meta.get("band_low", BAND[0]), meta.get("band_high", BAND[1])) != BAND:
        raise ValueError("World marker does not describe the declared .001 < p < .999 score band")
    names = ("left", "right", "label", "baseline_p", "true_count", "population", "component",
             "country_code", "partition", "folds")
    with np.load(path, allow_pickle=False) as data:
        if any(name not in data for name in names):
            raise ValueError("Missing required world arrays")
        arrays = {name: data[name] for name in names}
    if meta.get("rows") != len(arrays["left"]):
        raise ValueError("World row count differs from its marker")
    return World(**arrays, provenance={"world": str(path), "world_sha256": sha,
                 "complete_sha256": digest(marker), "complete": meta})


def selected_rows(world, path=None):
    if path is None:
        return np.flatnonzero(world.population)
    rows = _unique_rows(np.load(path, allow_pickle=False), "s1_rows", len(world.population))
    if not len(rows) or not world.population[rows].all():
        raise ValueError("s1_rows must be nonempty and entirely within B_REST")
    return np.sort(rows)


class KeyRegistry:
    """Bijections detect conflicting pair IDs even without a canonical --ids file."""

    def __init__(self):
        self.by_index = ({}, {})
        self.by_id = ({}, {})

    def check(self, world, table, name):
        if any(c not in table.column_names for c in ("pair_row", "s1_id", "rec_id")):
            raise ValueError(f"{name}: pair_row, s1_id, rec_id are required")
        rows = _unique_rows(table["pair_row"].to_numpy(), f"{name}: pair_row", len(world.left))
        ids = []
        for j, (key, index) in enumerate((("s1_id", world.left[rows]), ("rec_id", world.right[rows]))):
            values = table[key].to_pylist()
            if any(not isinstance(v, str) or not v.strip() for v in values):
                raise ValueError(f"{name}: null/empty/non-string {key}")
            for ix, value in zip(index.tolist(), values):
                if self.by_index[j].setdefault(ix, value) != value or self.by_id[j].setdefault(value, ix) != ix:
                    raise ValueError(f"{name}: {key} is misaligned with world row indices")
            ids.append(values)
        if len(set(zip(*ids))) != len(rows):
            raise ValueError(f"{name}: duplicate ID pair")
        for name_, expected in (("left_idx", world.left[rows]), ("right_idx", world.right[rows])):
            if name_ in table.column_names:
                a = _integer(table[name_].to_numpy(), name_, len(rows))
                if not np.array_equal(a, expected):
                    raise ValueError(f"{name}: {name_} does not match GLOBAL pair_row")
        return rows


def _key_table(path):
    names = pq.ParquetFile(path).schema_arrow.names
    wanted = [c for c in ("pair_row", "s1_id", "rec_id", "left_idx", "right_idx") if c in names]
    return pq.read_table(path, columns=wanted)


def _numeric(table, key):
    a = table[key].to_numpy()
    if a.ndim != 1 or a.dtype.kind not in "fiu":
        raise ValueError(f"{key}: numeric scores required")
    return a.astype(float)


def _score_features(table):
    names = set(table.column_names)
    if set(BYTE_COLUMNS) & names:
        if not set(BYTE_COLUMNS) <= names or names & {"prob", "logit"}:
            raise ValueError("ByT5 requires all five score fields and an unambiguous score schema")
        x = np.column_stack([_numeric(table, c) for c in BYTE_COLUMNS])
        finite = np.isfinite(x).all(axis=1)
        if np.any(finite & ((x[:, 0] < 0) | (x[:, 1] < 0))):
            raise ValueError("Negative ByT5 NLL")
        valid = finite & (x[:, 2] > 0) & (x[:, 2] == np.floor(x[:, 2]))
        if (not np.allclose(x[valid, 3], x[valid, 1] - x[valid, 0], atol=1e-4, rtol=1e-4)
                or not np.allclose(x[valid, 4], x[valid, 3] / x[valid, 2], atol=1e-5, rtol=1e-4)):
            raise ValueError("Inconsistent ByT5 log-ratio fields")
        x[~valid] = np.nan
        return x, list(BYTE_COLUMNS), 4, {"kind": "ByT5", "invalid_rows": int((~valid).sum())}
    if not names & {"prob", "logit"}:
        raise ValueError("Scores require BGE prob/logit or ByT5 NLL fields")
    prob = _numeric(table, "prob") if "prob" in names else None
    raw = _numeric(table, "logit") if "logit" in names else np.full(len(table), np.nan)
    recovered = 0
    if prob is not None:
        finite = np.isfinite(prob)
        if np.any(finite & ((prob < 0) | (prob > 1))):
            raise ValueError("BGE probability outside [0,1]")
        both = finite & np.isfinite(raw)
        if not np.allclose(prob[both], expit(raw[both]), atol=2e-6, rtol=2e-5):
            raise ValueError("BGE prob/logit disagree")
        fill = finite & ~np.isfinite(raw)
        raw[fill] = logit(prob[fill])
        recovered = int(fill.sum())
    raw[~np.isfinite(raw)] = np.nan
    return raw[:, None], ["logit"], 0, {"kind": "BGE", "invalid_rows": int(np.isnan(raw).sum()),
                                       "logits_from_probability": recovered}


@dataclass
class Scores:
    rows: np.ndarray
    features: np.ndarray
    names: list
    primary_columns: list
    provenance: dict


def load_scores(paths, world, selection, ids_path=None):
    if not paths or len({str(Path(p).resolve()) for p in paths}) != len(paths):
        raise ValueError("Supply distinct score files")
    registry = KeyRegistry()
    canonical = None
    provenance = {"files": [], "ids": None, "key_verification": "internal bijections only; no canonical --ids"}
    if ids_path is not None:
        canonical = np.sort(registry.check(world, _key_table(ids_path), "--ids"))
        provenance.update(ids={"path": str(ids_path), "sha256": digest(ids_path), "rows": len(canonical)},
                          key_verification="GLOBAL pair_row and keys checked against --ids")
    pieces, names, primary = [], [], []
    for i, path in enumerate(paths):
        schema = pq.ParquetFile(path).schema_arrow.names
        columns = [c for c in ("pair_row", "s1_id", "rec_id", "left_idx", "right_idx", "prob", "logit",
                               *BYTE_COLUMNS) if c in schema]
        table = pq.read_table(path, columns=columns)
        rows = registry.check(world, table, str(path))
        if canonical is not None and not np.isin(rows, canonical).all():
            raise ValueError("Score pair_row absent from canonical --ids")
        x, columns, scalar, info = _score_features(table)
        primary.append(len(names) + scalar)
        names.extend(f"score{i}:{c}" for c in columns)
        pieces.append((rows, x))
        provenance["files"].append({"path": str(path), "sha256": digest(path), "rows": len(rows),
                                    "features": columns, **info})
    union = np.unique(np.concatenate([r for r, _ in pieces]))
    values = np.full((len(union), len(names)), np.nan)
    col = 0
    for rows, x in pieces:
        values[np.searchsorted(union, rows), col:col + x.shape[1]] = x
        col += x.shape[1]
    chosen = np.zeros(len(world.population), bool)
    chosen[selection] = True
    eligible = chosen[world.left[union]]
    band = (world.baseline_p[union] > BAND[0]) & (world.baseline_p[union] < BAND[1])
    complete = np.isfinite(values).all(axis=1)
    keep = eligible & band & complete
    expected = chosen[world.left] & (world.baseline_p > BAND[0]) & (world.baseline_p < BAND[1])
    provenance.update(union_rows=len(union), outside_selected_s1=int((~eligible).sum()),
                      outside_band=int((eligible & ~band).sum()),
                      incomplete_rows_in_selected_band=int((eligible & band & ~complete).sum()),
                      selected_band_rows=int(expected.sum()), usable_rows=int(keep.sum()),
                      unchanged_selected_band_rows=int(expected.sum() - keep.sum()),
                      missing_policy="Complete-case intersection across score files; otherwise baseline p AND threshold",
                      usable_pair_rows_sha256=array_digest(union[keep]), feature_order=names,
                      primary_columns=primary)
    return Scores(union[keep], values[keep], names, primary, provenance)


@dataclass
class LocalDecision:
    rows: np.ndarray
    count: np.ndarray
    tp: np.ndarray
    f05: np.ndarray


class ScreenDecoder:
    """Exact small mutable-world reduction, including all cap spillover.

    One-owner precedes the cap, so a fixed record's only possible new owner is
    its highest baseline-eligible fixed claimant. Untouched pre-cap winners of
    every potentially affected S1 must be retained, including cap-dropped rows.
    No iterative graph closure is necessary because the decoder never backfills.
    """

    def __init__(self, world, mutable_rows):
        self.world = world
        self.mutable = np.sort(_unique_rows(mutable_rows, "mutable_rows", len(world.left)))
        self.base_owners = world.index.winners(world.baseline_p)
        self.baseline = _cap(self.base_owners, world.left, world.baseline_p[self.base_owners])
        self.base_count, self.base_tp = world.counts(self.baseline)
        self.base_f05 = entity_f05(self.base_tp, self.base_count, world.true_count)
        # np.isin uses a dense lookup for the ~10M record range, allocated once.
        touched = np.isin(world.right, np.unique(world.right[self.mutable]))
        fixed = touched.copy()
        fixed[self.mutable] = False
        fixed_rows = np.flatnonzero(fixed & (world.baseline_p >= BASE_THRESHOLD))
        fi = OwnerIndex(world.left[fixed_rows], world.right[fixed_rows], fixed_rows)
        best_fixed = fi.winners(world.baseline_p[fixed_rows])
        self.contest_rows = np.r_[self.mutable, best_fixed]
        self.contests = OwnerIndex(world.left[self.contest_rows], world.right[self.contest_rows], self.contest_rows)
        self.affected = np.unique(world.left[self.contest_rows])
        self.affected_pos = np.full(len(world.population), -1, dtype=np.int64)
        self.affected_pos[self.affected] = np.arange(len(self.affected))
        untouched = self.base_owners[~touched[self.base_owners]]
        self.static_owners = untouched[self.affected_pos[world.left[untouched]] >= 0]
        self.baseline_local = self.baseline[self.affected_pos[world.left[self.baseline]] >= 0]
        self.mutable_pos = np.full(len(world.left), -1, dtype=np.int64)
        self.mutable_pos[self.mutable] = np.arange(len(self.mutable))

    def decode(self, prob, thresholds=BASE_THRESHOLD):
        w = self.world
        p = np.asarray(prob, dtype=w.baseline_p.dtype)
        if p.shape != self.mutable.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("Mutable probabilities must be finite, in [0,1], and aligned")
        scores = w.baseline_p[self.contest_rows].copy()
        scores[:len(p)] = p
        ts = np.full(len(scores), BASE_THRESHOLD, dtype=p.dtype)
        ts[:len(p)] = thresholds
        winners = np.r_[self.static_owners, self.contests.winners(scores, ts)]
        score = w.baseline_p[winners].copy()
        positions = self.mutable_pos[winners]
        changed = positions >= 0
        score[changed] = p[positions[changed]]
        accepted = _cap(winners, w.left, score)
        n = len(self.affected)
        count = np.bincount(self.affected_pos[w.left[accepted]], minlength=n)
        tp = np.bincount(self.affected_pos[w.left[accepted[w.label[accepted]]]], minlength=n)
        return LocalDecision(accepted, count, tp, entity_f05(tp, count, w.true_count[self.affected]))

    def counts_for(self, local, rows):
        count, tp = self.base_count[rows].copy(), self.base_tp[rows].copy()
        pos = self.affected_pos[rows]
        hit = pos >= 0
        count[hit], tp[hit] = local.count[pos[hit]], local.tp[pos[hit]]
        return count, tp

    def materialize(self, local):
        keep = self.baseline[self.affected_pos[self.world.left[self.baseline]] < 0]
        return np.sort(np.r_[keep, local.rows])


@dataclass
class SmallLogistic:
    mean: np.ndarray
    scale: np.ndarray
    coef: np.ndarray
    intercept: float
    status: str
    samples: int
    positives: int

    def predict(self, x, fallback):
        if self.status == "empty_training_baseline_fallback":
            return np.asarray(fallback, dtype=float).copy()
        z = np.clip((np.clip(x, -1e6, 1e6) - self.mean) / self.scale, -20, 20)
        return z @ self.coef + self.intercept

    def parameters(self):
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist(), "coef": self.coef.tolist(),
                "intercept": self.intercept, "status": self.status, "samples": self.samples,
                "positives": self.positives}


def fit_logistic(x, labels):
    x, labels = np.asarray(x, dtype=float), np.asarray(labels, dtype=bool)
    if x.ndim != 2 or labels.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError("Small logistic fit needs finite aligned features/labels")
    x = np.clip(x, -1e6, 1e6)
    width, n, positives = x.shape[1], len(x), int(labels.sum())
    if not n:
        return SmallLogistic(np.zeros(width), np.ones(width), np.zeros(width), 0.0,
                             "empty_training_baseline_fallback", 0, 0)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-12] = 1
    if positives in (0, n):
        prior = (positives + 0.5) / (n + 1)
        return SmallLogistic(mean, scale, np.zeros(width), float(logit(prior)), "single_class_Jeffreys_prior", n, positives)
    z = np.clip((x - mean) / scale, -20, 20)
    model = LogisticRegression(**CLASSIFIER)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(z, labels)
    return SmallLogistic(mean, scale, model.coef_[0], float(model.intercept_[0]), "fitted", n, positives)


class CrossfitModels:
    """Cache each excluded-fold set: 10 inner + 5 outer tiny packs for five folds."""

    def __init__(self, world, scores):
        self.rows = scores.rows
        self.folds = world.folds[world.left[self.rows]]
        self.labels = world.label[self.rows]
        self.base = logit(world.baseline_p[self.rows])
        self.x = np.column_stack([self.base, scores.features])
        self.primary = scores.primary_columns
        self.cache = {}
        self.audit = {}

    def predictions(self, excluded):
        excluded = tuple(sorted(excluded))
        key = ",".join(map(str, excluded))
        if key not in self.cache:
            train = ~np.isin(self.folds, excluded)
            control = fit_logistic(self.x[train, :1], self.labels[train])
            fusion = fit_logistic(self.x[train], self.labels[train])
            calibrators = [fit_logistic(self.x[train, j + 1:j + 2], self.labels[train]) for j in self.primary]
            ctrl = control.predict(self.x[:, :1], self.base)
            new = np.mean([model.predict(self.x[:, j + 1:j + 2], self.base)
                           for j, model in zip(self.primary, calibrators)], axis=0)
            self.cache[key] = {"control": ctrl, "new": new, "fusion": fusion.predict(self.x, self.base)}
            self.audit[key] = {"excluded_folds": list(excluded), "train_folds": np.unique(self.folds[train]).tolist(),
                               "train_pair_rows_sha256": array_digest(self.rows[train]), "train_rows": int(train.sum()),
                               "control": control.parameters(), "fusion": fusion.parameters(),
                               "new_score_calibrators": [m.parameters() for m in calibrators]}
        return self.cache[key]


def _method_prob(method, predictions, baseline, active, alpha=None):
    p = baseline.copy()
    if method.startswith("blend_") or method == "convex_blend":
        if alpha == 0:
            return p  # Preserve exact baseline ties; do not round-trip through sigmoid/logit.
        which = "control" if method == "blend_baseline_control" else "new"
        z = (1 - alpha) * logit(baseline[active]) + alpha * predictions[which][active]
    else:
        z = predictions["control" if method == "baseline_recalibration" else "fusion"][active]
    p[active] = expit(z)
    return p


def crossfit_screen(world, scores, selection, decoder=None):
    selection = _unique_rows(selection, "selection", len(world.population))
    if not len(selection) or not world.population[selection].all():
        raise ValueError("Selection must be nonempty and entirely B_REST")
    rows = _unique_rows(scores.rows, "score rows", len(world.left))
    if (scores.features.shape != (len(rows), len(scores.names)) or not np.isfinite(scores.features).all()
            or not scores.primary_columns or any(j < 0 or j >= len(scores.names) for j in scores.primary_columns)
            or not np.isin(world.left[rows], selection).all()
            or np.any((world.baseline_p[rows] <= BAND[0]) | (world.baseline_p[rows] >= BAND[1]))):
        raise ValueError("Fit rows must be finite selected-S1 score-band rows with aligned feature order")
    if not np.array_equal(rows, np.sort(rows)):
        raise ValueError("Usable score rows must be sorted by GLOBAL pair_row")
    decoder = decoder or ScreenDecoder(world, scores.rows)
    folds = np.unique(world.folds[selection])
    if len(folds) < 3:
        raise ValueError("Screen requires at least three selected component folds for inner/outer isolation")
    base = world.baseline_p[scores.rows]
    final = {m: {"prob": base.copy(), "threshold": np.full(len(base), BASE_THRESHOLD, dtype=base.dtype)} for m in METHODS}
    audit = {"per_outer_fold": {}, "model_packs": {}, "thresholds": list(THRESHOLDS), "blend_weights": list(BLENDS),
             "classifier": {**CLASSIFIER, "penalty": "L2", "class_weight": None,
                            "standardization": "training rows only", "raw_clip": [-1e6, 1e6], "z_clip": [-20, 20]},
             "blend": "(1-alpha)*baseline_logit + alpha*mean(per-file calibrated primary logit)",
             "primary_scores": [scores.names[j] for j in scores.primary_columns],
             "tie_break": "macro F05 (1e-12 tie), fewer false links, nearer .725, smaller alpha",
             "fit_scope": "Only selected S1 complete-case scored band; outer fold excluded from inner fits and selection"}
    if not len(base):
        audit["status"] = "No complete finite score rows; all methods use the unchanged baseline"
        return final, audit
    models = CrossfitModels(world, scores)
    for outer in folds:
        active = models.folds != outer
        selection_rows_ = selection[world.folds[selection] != outer]
        truth = world.true_count[selection_rows_]
        inner = {k: models.base.copy() for k in ("control", "new", "fusion")}
        for hold in folds[folds != outer]:
            prediction = models.predictions((int(outer), int(hold)))
            take = models.folds == hold
            for k in inner:
                inner[k][take] = prediction[k][take]
        chosen = {}
        for method in METHODS:
            best_key, best, grid = None, None, []
            alphas = BLENDS if method in ("blend_baseline_control", "convex_blend") else (None,)
            for alpha in alphas:
                prob = _method_prob(method, inner, base, active, alpha)
                for threshold in THRESHOLDS:
                    ts = np.full(len(base), BASE_THRESHOLD, dtype=base.dtype)
                    ts[active] = threshold
                    local = decoder.decode(prob, ts)
                    count, tp = decoder.counts_for(local, selection_rows_)
                    value = float(entity_f05(tp, count, truth).mean())
                    false = int((count - tp).sum())
                    key = (value, -false, -abs(threshold - BASE_THRESHOLD), -(alpha or 0))
                    item = {"alpha": alpha, "threshold": threshold, "macro_f05": value, "false_links": false}
                    grid.append(item)
                    if (best_key is None or value > best_key[0] + 1e-12
                            or (abs(value - best_key[0]) <= 1e-12 and key[1:] > best_key[1:])):
                        best_key, best = key, item
            chosen[method] = {**best, "search": grid}
        prediction = models.predictions((int(outer),))
        take = models.folds == outer
        for method, params in chosen.items():
            p = _method_prob(method, prediction, base, take, params["alpha"])
            final[method]["prob"][take] = p[take]
            final[method]["threshold"][take] = params["threshold"]
        audit["per_outer_fold"][str(outer)] = {
            "evaluation_s1": int((world.folds[selection] == outer).sum()),
            "selection_s1": len(selection_rows_), "evaluation_pair_rows_sha256": array_digest(scores.rows[take]),
            "selection_s1_rows_sha256": array_digest(selection_rows_),
            "inner_excluded_sets": [[int(outer), int(h)] for h in folds if h != outer], "methods": chosen}
    audit["model_packs"] = models.audit
    return final, audit


def summary(world, rows, count, tp):
    truth, retrieved = world.true_count[rows], world.retrieved[rows]
    total, positive, true = int(count.sum()), int(tp.sum()), int(truth.sum())
    singleton = truth == 0
    oracle = entity_f05(retrieved, retrieved, truth)
    return {"s1": len(rows), "macro_f05": float(entity_f05(tp, count, truth).mean()) if len(rows) else None,
            "micro_precision": positive / total if total else 0.0, "micro_recall": positive / true if true else 0.0,
            "predicted_links": total, "true_links": true, "true_positive_links": positive, "false_links": total - positive,
            "singletons": int(singleton.sum()),
            "singleton_accuracy": float((count[singleton] == 0).mean()) if singleton.any() else None,
            "singleton_false_links_s1": int((count[singleton] > 0).sum()),
            "empty_predictions": int((count == 0).sum()), "s1_without_candidates": int((world.candidate_count[rows] == 0).sum()),
            "candidate_pairs": int(world.candidate_count[rows].sum()), "retrieved_true_links": int(retrieved.sum()),
            "missed_true_links": int(true - retrieved.sum()),
            "retrieval_link_recall": float(retrieved.sum() / true) if true else 0.0,
            "retrieval_oracle_macro_f05": float(oracle.mean()) if len(rows) else None,
            "maximum_predicted_list": int(count.max()) if len(count) else 0}


def with_countries(world, rows, count, tp):
    result = summary(world, rows, count, tp)
    result["by_country"] = {}
    for code in np.unique(world.country_code[rows]):
        m = world.country_code[rows] == code
        result["by_country"][COUNTRIES[int(code)]] = summary(world, rows[m], count[m], tp[m])
    return result


def component_bootstrap(a, b, components, *, n=1000, seed=20260927):
    """Paired ordinary cluster bootstrap, S1-weighted within sampled components."""
    a, b, components = np.asarray(a, float), np.asarray(b, float), np.asarray(components)
    if n < 1000:
        raise ValueError("At least 1000 component-bootstrap resamples are required")
    if a.ndim != 1 or a.shape != b.shape or a.shape != components.shape or not len(a):
        raise ValueError("Bootstrap needs nonempty aligned per-S1 vectors")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Bootstrap scores must be finite")
    _, inv = np.unique(components, return_inverse=True)
    size = np.bincount(inv)
    sums = np.bincount(inv, weights=b - a)
    rng, k = np.random.default_rng(seed), len(size)
    samples = np.empty(n)
    if np.all(a == b):
        samples.fill(0)
    else:
        for i in range(n):
            draw = rng.integers(0, k, size=k)
            samples[i] = sums[draw].sum() / size[draw].sum()
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"delta": float((b - a).mean()), "ci_low": float(lo), "ci_high": float(hi),
            "bootstrap_mean": float(samples.mean()), "p_le_zero": float((samples <= 0).mean()),
            "components": k, "s1": len(a), "replicates": n, "seed": seed,
            "method": "paired percentile bootstrap, whole selected-S1 components with replacement, S1-weighted"}


def paired(world, rows, a, b, *, n, seed):
    sa, sb = with_countries(world, rows, *a), with_countries(world, rows, *b)
    fa, fb = (entity_f05(tp, count, world.true_count[rows]) for count, tp in (a, b))
    fields = ("macro_f05", "micro_precision", "micro_recall", "false_links", "true_positive_links",
              "predicted_links", "singleton_accuracy", "singleton_false_links_s1")

    def difference(x, y):
        return {k: y[k] - x[k] if y[k] is not None and x[k] is not None else None for k in fields}

    return {"delta": difference(sa, sb), "improved_s1": int((fb > fa).sum()),
            "worsened_s1": int((fb < fa).sum()), "tied_s1": int((fb == fa).sum()),
            "entity_points_delta": float((fb - fa).sum()),
            "by_country": {c: difference(sa["by_country"][c], sb["by_country"][c]) for c in sa["by_country"]},
            "macro_f05_component_bootstrap": component_bootstrap(fa, fb, world.component[rows], n=n, seed=seed)}


def _base_report(world, selection):
    order, starts = world.index.order, world.index.starts
    folds = world.folds[world.left[order]]
    cross_fold = int((np.maximum.reduceat(folds, starts) != np.minimum.reduceat(folds, starts)).sum()) if len(order) else 0
    return {"scope": "REUSED DEVELOPMENT / B_REST", "limitations": LIMITATION,
            "holdout2_rows_read": 0, "official_test_evaluated": False, "promotion_ready": False,
            "provenance": world.provenance, "source_sha256": digest(__file__),
            "population": {"world_s1_slots": len(world.population), "world_pairs": len(world.left),
                           "b_rest_s1": int(world.population.sum()), "evaluated_s1": len(selection),
                           "fixed_competitor_s1": int(world.population.sum() - len(selection)),
                           "s1_rows_sha256": array_digest(selection),
                           "fold_s1_counts": {str(f): int((world.folds[selection] == f).sum())
                                              for f in np.unique(world.folds[selection])},
                           "country_codes": COUNTRIES},
            "decode": {"threshold": BASE_THRESHOLD, "owner_tie": "lower S1 then lower global pair_row",
                       "cap": CAP, "cap_tie": "lower global pair_row", "order": "threshold, owner, cap; no backfill",
                       "probability_dtype": str(world.baseline_p.dtype),
                       "threshold_dtype": str(world.baseline_p.dtype), "world_records_crossing_inherited_folds": cross_fold},
            "versions": {"numpy": np.__version__, "pyarrow": pa.__version__,
                         "scikit_learn": __import__("sklearn").__version__}}


def _parity(world, fixed, reference):
    actual = {"fixed": fixed["macro_f05"], "reference_per_fold": reference["macro_f05"]}
    if world.provenance.get("world_sha256") != KNOWN_WORLD_SHA:
        return {"status": "Not the hash-identified U8t world; historical parity not asserted", "actual": actual}
    expected = {"fixed": 0.9889270323911976, "reference_per_fold": 0.988918311526446}
    if any(abs(actual[k] - expected[k]) > 1e-12 for k in actual):
        raise ValueError(f"U8t baseline parity failed: {actual}, expected {expected}")
    return {"status": "PASS", "actual": actual, "expected": expected, "atol": 1e-12}


def _output_dir(path):
    p = Path(path)
    if p.exists() and any(p.iterdir()):
        raise FileExistsError(f"Refusing to overwrite a nonempty output directory: {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def _json(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def _vectors(world, selection):
    rows = np.flatnonzero(world.population)
    return {"s1_rows": rows, "selected": np.isin(rows, selection), "true_count": world.true_count[rows],
            "retrieved_true_count": world.retrieved[rows], "candidate_count": world.candidate_count[rows],
            "component": world.component[rows], "country_code": world.country_code[rows], "folds": world.folds[rows]}


def _add_vectors(vectors, name, count, tp):
    vectors[name + "_predicted_count"] = count
    vectors[name + "_true_positive_count"] = tp
    vectors[name + "_f05"] = entity_f05(tp, count, vectors["true_count"])


def run_baseline(world, selection, output, *, n=1000, seed=20260927, ids_path=None):
    started = time.monotonic()
    report = _base_report(world, selection)
    if ids_path:
        rows = KeyRegistry().check(world, _key_table(ids_path), "--ids")
        report["ids"] = {"path": str(ids_path), "sha256": digest(ids_path), "rows": len(rows)}
    fixed = world.decode()
    threshold = np.asarray(REFERENCE_THRESHOLDS, dtype=world.baseline_p.dtype)[world.folds[world.left]]
    reference = world.decode(thresholds=threshold)
    counts = {"baseline_fixed": world.counts(fixed), "baseline_reference_oof": world.counts(reference)}
    all_rows = np.flatnonzero(world.population)
    full = {name: with_countries(world, all_rows, count[all_rows], tp[all_rows]) for name, (count, tp) in counts.items()}
    report["parity"] = _parity(world, full["baseline_fixed"], full["baseline_reference_oof"])
    report["metrics"] = {name: with_countries(world, selection, count[selection], tp[selection])
                         for name, (count, tp) in counts.items()}
    report["full_b_rest_baselines"] = full
    report["reference_per_fold_thresholds"] = list(REFERENCE_THRESHOLDS)
    subset = {name: (count[selection], tp[selection]) for name, (count, tp) in counts.items()}
    report["paired_reference_minus_fixed"] = paired(world, selection, subset["baseline_fixed"],
                                                    subset["baseline_reference_oof"], n=n, seed=seed)
    out, vectors = _output_dir(output), _vectors(world, selection)
    for name, (count, tp) in counts.items():
        _add_vectors(vectors, name, count[all_rows], tp[all_rows])
    np.savez_compressed(out / "per_s1.npz", **vectors)
    np.savez_compressed(out / "decisions.npz", baseline_fixed_rows=fixed, baseline_reference_oof_rows=reference)
    report["runtime_seconds"] = time.monotonic() - started
    report["artifacts"] = {name: digest(out / name) for name in ("per_s1.npz", "decisions.npz")}
    _json(out / "report.json", report)
    return report


def run_screen(world, scores, selection, output, *, n=1000, seed=20260927):
    started = time.monotonic()
    decoder = ScreenDecoder(world, scores.rows)
    predictions, parameters = crossfit_screen(world, scores, selection, decoder)
    report = _base_report(world, selection)
    report["scores"] = scores.provenance
    report["exact_reduction"] = {"mutable_rows": len(scores.rows), "contest_rows": len(decoder.contest_rows),
                                 "potentially_affected_s1": len(decoder.affected),
                                 "static_precap_owner_rows": len(decoder.static_owners)}
    report["metrics"] = {"baseline_fixed": with_countries(world, selection, decoder.base_count[selection],
                                                           decoder.base_tp[selection])}
    counts = {"baseline_fixed": (decoder.base_count[selection], decoder.base_tp[selection])}
    vectors = _vectors(world, selection)
    all_rows = vectors["s1_rows"]
    other = all_rows[~vectors["selected"]]
    _add_vectors(vectors, "baseline_fixed", decoder.base_count[all_rows], decoder.base_tp[all_rows])
    # These are exact partial-edit effects, not an extrapolated all-B_REST model score.
    report["world_with_only_screen_edits"] = {
        "scope": "Exact B_REST accounting after editing selected finite score-band rows only; no extrapolation",
        "baseline_fixed": with_countries(world, all_rows, decoder.base_count[all_rows], decoder.base_tp[all_rows])}
    report["fixed_competitors"] = {
        "scope": "Probabilities/thresholds fixed; accepted links may change through ownership or cap spillover",
        "baseline_fixed": with_countries(world, other, decoder.base_count[other], decoder.base_tp[other])}
    changes = {}
    replay = {"pair_row": scores.rows, "features": scores.features, "feature_names": np.asarray(scores.names),
              "baseline_p": world.baseline_p[scores.rows]}
    for method, pred in predictions.items():
        local = decoder.decode(pred["prob"], pred["threshold"])
        counts[method] = decoder.counts_for(local, selection)
        report["metrics"][method] = with_countries(world, selection, *counts[method])
        full_counts = decoder.counts_for(local, all_rows)
        _add_vectors(vectors, method, *full_counts)
        report["world_with_only_screen_edits"][method] = with_countries(world, all_rows, *full_counts)
        report["fixed_competitors"][method] = with_countries(world, other, *decoder.counts_for(local, other))
        changes[method + "_added_rows"] = np.setdiff1d(local.rows, decoder.baseline_local, assume_unique=True)
        changes[method + "_dropped_rows"] = np.setdiff1d(decoder.baseline_local, local.rows, assume_unique=True)
        replay[method + "_prob"] = pred["prob"]
        replay[method + "_threshold"] = pred["threshold"]
    report["paired"] = {}
    for method in METHODS:
        report["paired"][method + "_minus_baseline_fixed"] = paired(
            world, selection, counts["baseline_fixed"], counts[method], n=n, seed=seed)
    report["candidate_feature_usefulness"] = {}
    for method, control in CONTROLS.items():
        result = paired(world, selection, counts[control], counts[method], n=n, seed=seed)
        report["paired"][method + "_minus_" + control] = result
        ci = result["macro_f05_component_bootstrap"]
        finding = ("positive incremental reused-development signal" if ci["ci_low"] > 0 else
                   "inconclusive incremental reused-development signal" if ci["delta"] > 0 else
                   "no positive incremental effect on the declared screen")
        report["candidate_feature_usefulness"][method] = {"matched_control": control, "finding": finding,
                                                         "delta": ci["delta"], "ci95": [ci["ci_low"], ci["ci_high"]]}
    out = _output_dir(output)
    np.savez_compressed(out / "per_s1.npz", **vectors)
    np.savez_compressed(out / "screen_scores.npz", **replay)
    np.savez_compressed(out / "decision_changes.npz", **changes)
    _json(out / "selected_parameters.json", parameters)
    report["runtime_seconds"] = time.monotonic() - started
    report["artifacts"] = {name: digest(out / name) for name in ("per_s1.npz", "screen_scores.npz",
                                                                "decision_changes.npz", "selected_parameters.json")}
    _json(out / "report.json", report)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    for command in ("baseline", "screen"):
        sub = commands.add_parser(command)
        sub.add_argument("--world", required=True, type=Path)
        sub.add_argument("--output", required=True, type=Path)
        sub.add_argument("--s1-rows", required=command == "screen", type=Path)
        sub.add_argument("--ids", type=Path, help="Canonical pair-key parquet; pair_row must be a GLOBAL world row")
        sub.add_argument("--bootstrap", type=int, default=1000)
        sub.add_argument("--seed", type=int, default=20260927)
        sub.add_argument("--threads", type=int, default=1, help="CPU BLAS/Arrow threads; no GPU libraries used")
        if command == "screen":
            sub.add_argument("--scores", type=Path, action="append", required=True)
    args = p.parse_args(argv)
    if args.bootstrap < 1000 or args.threads < 1:
        p.error("--bootstrap must be >= 1000 and --threads must be >= 1")
    if args.output.exists() and any(args.output.iterdir()):
        p.error("--output must be new or empty")
    started = time.monotonic()
    pa.set_cpu_count(args.threads)
    pa.set_io_thread_count(args.threads)
    with threadpool_limits(limits=args.threads):
        world = load_world(args.world)
        selection = selected_rows(world, args.s1_rows)
        if args.s1_rows:
            world.provenance["s1_rows_file"] = {"path": str(args.s1_rows), "sha256": digest(args.s1_rows)}
        if args.command == "baseline":
            report = run_baseline(world, selection, args.output, n=args.bootstrap, seed=args.seed, ids_path=args.ids)
        else:
            scores = load_scores(args.scores, world, selection, args.ids)
            report = run_screen(world, scores, selection, args.output, n=args.bootstrap, seed=args.seed)
    report["total_runtime_seconds_including_load"] = time.monotonic() - started
    _json(args.output / "report.json", report)
    print(json.dumps({"report": str(args.output / "report.json"), "scope": report["scope"],
                      "metrics": {k: {f: v[f] for f in ("s1", "macro_f05", "false_links")}
                                  for k, v in report["metrics"].items()},
                      "seconds": report["total_runtime_seconds_including_load"]}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
