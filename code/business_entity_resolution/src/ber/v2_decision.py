"""CPU-only P1 decisions on the saved, label-free medium TRAIN/TUNE retrieval.

The outer entity folds evaluate country-wise calibration and the count-prior rule
out of fold. Source-balance weight is chosen by an inner entity split. The older
medium eval, sealed v2 holdout, and official test are never input paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from math import comb
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .metrics import pair_labels
from .normalization import normalize_text
from .uncertainty import paired_bootstrap


TUNE_ID_SHA256 = "5bb5759fa52ac19df8839520c7ccf147bae01a5499a2900af5d1b4ebeb1dbaa2"
FULL_POOL_ROWS = 10_320_219
BASELINE_THRESHOLD = 0.775
TREE_GATE = 0.01
DELTAS = (0.0, 0.05, 0.1, 0.2)
BALANCE_WEIGHTS = (0.0, 0.25, 0.5, 1.0)  # additive log odds, applied to the opposite source
ANCHOR_PROBABILITY = 0.95
BETA_STRENGTH = 2.0
ISOTONIC_BINS = 64
EPS = 1e-6
_SUFFIXES = {"llc", "llp", "ltd", "limited", "inc", "incorporated", "corp",
             "corporation", "company", "co", "pvt", "private", "plc"}
_DIGITS = re.compile(r"\d+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ids_digest(ids) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def frame_digest(frame: pd.DataFrame, columns: list[str]) -> str:
    values = pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def json_write(path: Path, document: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


@dataclass
class TuneData:
    references: pd.DataFrame
    pool: pd.DataFrame
    pairs: pd.DataFrame
    truth: dict[str, set[str]]
    left: np.ndarray  # positions in the 12,000 TUNE references, one for each TUNE pair
    right: np.ndarray  # positions in the saved compact pool (not a sampled search pool)
    raw: np.ndarray
    gate: np.ndarray
    base: np.ndarray
    labels: np.ndarray
    actual: np.ndarray
    country: np.ndarray
    source: np.ndarray
    offsets: np.ndarray
    original_left: np.ndarray
    original_pair_rows: np.ndarray
    train_reference: pd.DataFrame
    train_left: np.ndarray
    train_labels: np.ndarray
    train_base: np.ndarray
    provenance: dict
    saved_baseline: dict
    retrieval: dict


def load_verified(inputs: Path, prior_report: Path) -> TuneData:
    """Reject stale/misaligned or injected caches before accessing any pair labels."""
    report = json.loads(prior_report.read_text())
    expected = report["provenance"]
    paths = {
        "reference": inputs / "data/reference.parquet",
        "pool": inputs / "data/pool.parquet",
        "truth": inputs / "data/truth.json",
        "data_metadata": inputs / "data/metadata.json",
        "data_complete": inputs / "data/complete.json",
        "pairs": inputs / "scores/pairs.parquet",
        "base": inputs / "scores/base_probabilities.npy",
        "cross": inputs / "scores/cross_probabilities.npy",
        "gate": inputs / "scores/reranked_mask.npy",
        "selected_blend": inputs / "scores/probabilities.npy",
        "saved_config": inputs / "scores/experiment_config.json",
        "saved_report": inputs / "scores/results.json",
    }
    for name, path in paths.items():
        if path.stat().st_size != expected[name]["bytes"] or sha256_file(path) != expected[name]["sha256"]:
            raise ValueError(f"Saved input identity differs: {name}")
    metadata = json.loads(paths["data_metadata"].read_text())
    complete = json.loads(paths["data_complete"].read_text())
    config = json.loads(paths["saved_config"].read_text())
    saved = json.loads(paths["saved_report"].read_text())
    if (complete.get("state") != "complete" or complete.get("metadata") != metadata
            or complete["config"]["labels_used_for_mining"] is not False
            or metadata["positive_candidate_injection"] is not False
            or complete["config"]["pool_embedding_identity"]["scope"] != "full_pool"
            or metadata["mining_full_pool_records"] != FULL_POOL_ROWS
            or config["full_pool_records"] != FULL_POOL_ROWS
            or config["evaluation_used"] or saved["evaluation_used"]
            or not config["tune_only_inference"] or config["rerank_cutoff"] != TREE_GATE
            or config["reference_digest"] != expected["reference_digest"]
            or config["pool_digest"] != expected["pool_digest"]
            or config["pair_digest"] != expected["pair_digest"]
            or config["retrieval_checkpoint_sha256"] != complete["config"]["checkpoint_sha256"]):
        raise ValueError("Expected completed, tune-only, label-free full-pool inputs")
    reference = pd.read_parquet(paths["reference"]).fillna("")
    pool = pd.read_parquet(paths["pool"]).fillna("")
    if (reference.entity_id.duplicated().any() or pool.entity_id.duplicated().any()
            or set(reference.split) != {"train", "tune"}
            or frame_digest(reference, ["entity_id", "business_name", "business_address", "country", "split"])
            != config["reference_digest"]
            or frame_digest(pool, ["entity_id", "business_name", "business_address", "country"])
            != config["pool_digest"]):
        raise ValueError("Saved TRAIN/TUNE reference or compact pool identities differ")
    tune = reference.split.eq("tune").to_numpy()
    if tune.sum() != 12_000 or ids_digest(reference.entity_id[tune]) != TUNE_ID_SHA256:
        raise ValueError("The specified 12,000-S1 medium TUNE split is missing")
    truth = {key: set(value) for key, value in json.loads(paths["truth"].read_text()).items()}
    if set(truth) != set(reference.entity_id):
        raise ValueError("Truth keys must cover exactly the saved TRAIN/TUNE S1s")
    train_truth = {key: sorted(truth[key]) for key in reference.entity_id[~tune]}
    train_digest = hashlib.sha256(json.dumps(train_truth, sort_keys=True).encode()).hexdigest()
    if train_digest != config["training_truth_digest"]:
        raise ValueError("TRAIN-only truth digest differs from the saved CE experiment")
    pairs = pd.read_parquet(paths["pairs"])
    if (len(pairs) != metadata["candidate_pairs"] or pairs.duplicated(["left_idx", "right_idx"]).any()
            or frame_digest(pairs, ["left_idx", "right_idx"]) != config["pair_digest"]
            or pairs.left_idx.min() < 0 or pairs.left_idx.max() >= len(reference)
            or pairs.right_idx.min() < 0 or pairs.right_idx.max() >= len(pool)):
        raise ValueError("Candidate pair identity, uniqueness or positional bounds differ")
    left = pairs.left_idx.to_numpy(dtype=np.int64)
    right = pairs.right_idx.to_numpy(dtype=np.int64)
    if np.any(np.diff(left) < 0):
        raise ValueError("Saved candidate pairs must be ordered by S1 for complete-entity scoring")
    base = np.load(paths["base"], mmap_mode="r")
    cross = np.load(paths["cross"], mmap_mode="r")
    gate = np.load(paths["gate"], mmap_mode="r")
    blend = np.load(paths["selected_blend"], mmap_mode="r")
    if (any(a.shape != (len(pairs),) for a in (base, cross, gate, blend))
            or gate.dtype != np.bool_
            or any(not np.isfinite(a).all() or np.any((a < 0) | (a > 1)) for a in (base, cross, blend))
            or hashlib.sha256(base.tobytes()).hexdigest() != config["base_probability_digest"]
            or not np.array_equal(gate, (base >= TREE_GATE) & tune[left])
            or np.any(cross[~gate] != 0)):
        raise ValueError("CE gate, score range or aligned probability identity differs")
    reconstructed = np.zeros(len(pairs), dtype=np.float32)
    reconstructed[gate] = .25 * base[gate] + .75 * cross[gate]
    if not np.array_equal(blend, reconstructed):
        raise ValueError("Saved blend differs from the v1 gated 75% CE decision score")
    labels = pair_labels(reference, pool, pairs, truth).astype(bool)
    pair_rows = np.flatnonzero(tune[left])
    original_ref = np.flatnonzero(tune)
    ref_to_tune = np.full(len(reference), -1, dtype=np.int32)
    ref_to_tune[original_ref] = np.arange(len(original_ref))
    original_train = np.flatnonzero(~tune)
    ref_to_train = np.full(len(reference), -1, dtype=np.int32)
    ref_to_train[original_train] = np.arange(len(original_train))
    tune_left = ref_to_tune[left[pair_rows]]
    tune_right = right[pair_rows]
    source = pool.entity_id.str.slice(0, 3).to_numpy()[tune_right]
    if not set(source).issubset({"S2-", "S3-"}):
        raise ValueError("Candidate targets must belong to S2 or S3")
    tuned = reference.iloc[original_ref].reset_index(drop=True)
    offsets = np.r_[0, np.cumsum(np.bincount(tune_left, minlength=len(tuned)))]
    return TuneData(
        tuned, pool, pairs, truth, tune_left, tune_right,
        np.asarray(blend[pair_rows]), np.asarray(gate[pair_rows]), np.asarray(base[pair_rows]),
        labels[pair_rows], np.asarray([len(truth[key]) for key in tuned.entity_id]),
        tuned.country.to_numpy(), source, offsets, original_ref, pair_rows,
        reference.loc[~tune].reset_index(drop=True), ref_to_train[left[~tune[left]]], labels[~tune[left]],
        np.asarray(base[~tune[left]]),
        {"read_only_inputs": {name: {"path": str(path.resolve()), **expected[name]}
                              for name, path in paths.items()},
         "reference_digest": config["reference_digest"], "pool_digest": config["pool_digest"],
         "pair_digest": config["pair_digest"], "tune_id_sha256": TUNE_ID_SHA256,
         "encoder_checkpoint_sha256": expected["encoder_checkpoint_sha256"],
         "tree_checkpoint_sha256": expected["tree_checkpoint_sha256"],
         "reranker_checkpoint_sha256": expected["reranker_checkpoint_sha256"],
         "prior_cpu_report": {"path": str(prior_report.resolve()), "sha256": sha256_file(prior_report)}},
        saved["selected_by_tune"]["tune"], metadata["retrieval"]["tune"],
    )


def score_decisions(data: TuneData, keep: np.ndarray, selected: np.ndarray | None = None) -> tuple[dict, np.ndarray]:
    """The exact official singleton/macro convention, with complete S1 coverage."""
    if keep.shape != data.left.shape or keep.dtype != np.bool_:
        raise ValueError("One Boolean decision is required for every saved TUNE candidate")
    chosen = np.ones(len(data.references), dtype=bool) if selected is None else np.asarray(selected, dtype=bool)
    if chosen.shape != (len(data.references),) or not chosen.any() or np.any(keep & ~chosen[data.left]):
        raise ValueError("Entity selection must preserve all its candidate decisions")
    predicted = np.bincount(data.left[keep], minlength=len(chosen))
    tp = np.bincount(data.left[keep & data.labels], minlength=len(chosen))
    actual = data.actual
    denom = 4 * predicted + actual
    f05 = np.divide(5.0 * tp, denom, out=np.zeros(len(chosen)), where=denom > 0)
    f1 = np.divide(2.0 * tp, predicted + actual, out=np.zeros(len(chosen)), where=predicted + actual > 0)
    correct_singleton = (actual == 0) & (predicted == 0)
    f05[correct_singleton] = f1[correct_singleton] = 1.0

    def summary(mask: np.ndarray) -> dict:
        total_tp, total_pred, total_truth = int(tp[mask].sum()), int(predicted[mask].sum()), int(actual[mask].sum())
        single = mask & (actual == 0)
        false = total_pred - total_tp
        return {"reference_count": int(mask.sum()), "macro_f05": float(f05[mask].mean()),
                "macro_f1": float(f1[mask].mean()),
                "micro_precision": total_tp / total_pred if total_pred else 0.0,
                "micro_recall": total_tp / total_truth if total_truth else 0.0,
                "predicted_links": total_pred, "true_links": total_truth,
                "true_positive_links": total_tp, "false_positive_links": false,
                "missed_true_links": total_truth - total_tp,
                "true_singletons": int(single.sum()),
                "singleton_accuracy": float((predicted[single] == 0).mean()) if single.any() else None,
                "singleton_false_merges": int((predicted[single] > 0).sum()),
                "entities_with_false_positive": int(((predicted - tp > 0) & mask).sum())}

    result = summary(chosen)
    result["by_country"] = {key: summary(chosen & (data.country == key))
                            for key in sorted(set(data.country[chosen]))}
    result["retrieved_true_links_rejected"] = int((data.labels & chosen[data.left] & ~keep).sum())
    return result, f05


def assignment(keep: np.ndarray, targets: np.ndarray, scores: np.ndarray,
               left: np.ndarray, delta: float) -> tuple[np.ndarray, dict]:
    """Only an accepted target's weaker claims can be removed; ties are stable.

    For delta=0, even an exact tie is resolved to the lowest S1 position. For
    positive margins, near ties remain explicitly unresolved, as specified by P1.
    """
    if delta < 0 or any(len(a) != len(keep) for a in (targets, scores, left)):
        raise ValueError("Aligned claims and a nonnegative margin are required")
    output = keep.copy()
    ids = np.flatnonzero(keep)
    order = ids[np.lexsort((ids, left[ids], -scores[ids], targets[ids]))]
    removed = 0
    conflicting_targets = 0
    unresolved = 0
    for group in np.split(order, np.flatnonzero(np.diff(targets[order])) + 1) if len(order) else []:
        if len(group) < 2:
            continue
        conflicting_targets += 1
        winner = group[0]
        for loser in group[1:]:
            if delta == 0 or float(scores[winner]) - float(scores[loser]) >= delta:
                output[loser] = False
                removed += 1
            else:
                unresolved += 1
    return output, {"conflicting_targets": conflicting_targets, "removed_claims": removed,
                    "unresolved_extra_claims": unresolved}


def balanced_folds(reference: pd.DataFrame, seed: int) -> np.ndarray:
    """Country-balanced, reproducible entity-disjoint halves, independent of labels."""
    fold = np.empty(len(reference), dtype=np.uint8)
    for country, indices in reference.groupby("country", sort=True).indices.items():
        ordered = sorted(indices, key=lambda i: hashlib.sha256(
            f"{seed}:{reference.entity_id.iat[i]}".encode()).digest())
        fold[ordered[::2]] = 0
        fold[ordered[1::2]] = 1
    return fold


def train_count_prior(data: TuneData) -> dict:
    """The count prior and gated discovery rate use TRAIN truth, never TUNE."""
    refs = data.train_reference
    actual = np.asarray([len(data.truth[key]) for key in refs.entity_id])
    histogram = np.bincount(actual)
    prior = (histogram + .5) / (len(refs) + .5 * len(histogram))
    train_gate = data.train_base >= TREE_GATE
    gate_tp = int(data.train_labels[train_gate].sum())
    total = int(actual.sum())
    by_country = {}
    pair_country = refs.country.to_numpy()[data.train_left]
    for country in sorted(refs.country.unique()):
        local = pair_country == country
        gated = local & train_gate
        ungated = local & ~train_gate
        by_country[country] = {
            "gated_pair_prevalence": float(data.train_labels[gated].mean()),
            "ungated_pair_prevalence": float(data.train_labels[ungated].mean()),
        }
    return {"count_histogram": histogram.tolist(), "p_true_count": prior.tolist(),
            "gate_true_link_discovery": gate_tp / total,
            "train_references": len(refs), "train_true_links": total,
            "train_gated_true_links": gate_tp,
            "pair_prevalence": float(data.train_labels.mean()),
            "gated_pair_prevalence": float(data.train_labels[train_gate].mean()),
            "ungated_pair_prevalence": float(data.train_labels[~train_gate].mean()),
            "country_pair_prior": by_country,
            "train_both_source_fraction_of_non_singletons": float(np.mean([
                any(key.startswith("S2-") for key in data.truth[entity])
                and any(key.startswith("S3-") for key in data.truth[entity])
                for entity in refs.entity_id[actual > 0]]))}


def fit_calibration(raw: np.ndarray, gate: np.ndarray, labels: np.ndarray,
                    left: np.ndarray, country: np.ndarray, fit_refs: np.ndarray,
                    train_prior: dict) -> dict:
    """Beta-smoothed score bins followed by weighted, monotone isotonic regression."""
    if fit_refs.dtype != bool or fit_refs.shape != country.shape or not fit_refs.any():
        raise ValueError("Calibration needs a nonempty complete-entity fit set")
    selected = fit_refs[left]
    pair_country = country[left]
    models = {}
    for key in ("__pooled__", *sorted(set(country))):
        fit = selected & (np.ones(len(raw), dtype=bool) if key == "__pooled__" else pair_country == key)
        scored = fit & gate
        missed = fit & ~gate
        if scored.sum() < 100 or missed.sum() < 100:
            raise ValueError(f"Too few pairs to calibrate {key}")
        source = (train_prior["country_pair_prior"].get(key, {}) if key != "__pooled__" else {})
        gated_base = source.get("gated_pair_prevalence", train_prior["gated_pair_prevalence"])
        missed_base = source.get("ungated_pair_prevalence", train_prior["ungated_pair_prevalence"])
        x, y = raw[scored], labels[scored]
        boundaries = np.unique(np.quantile(x, np.linspace(0.0, 1.0, ISOTONIC_BINS + 1)))
        if len(boundaries) < 2:
            raise ValueError(f"No score variation for country {key}")
        bin_index = np.searchsorted(boundaries[1:-1], x, side="right")
        counts = np.bincount(bin_index, minlength=len(boundaries) - 1)
        positives = np.bincount(bin_index, weights=y, minlength=len(counts))
        raw_sum = np.bincount(bin_index, weights=x, minlength=len(counts))
        occupied = counts > 0
        bin_centers = raw_sum[occupied] / counts[occupied]
        # Beta(alpha=prior*2, beta=(1-prior)*2) posterior for each observed bin.
        beta_means = (positives[occupied] + BETA_STRENGTH * gated_base) / (counts[occupied] + BETA_STRENGTH)
        model = IsotonicRegression(y_min=EPS, y_max=1 - EPS, out_of_bounds="clip")
        model.fit(bin_centers, beta_means, sample_weight=counts[occupied])
        unscored_pos = int(labels[missed].sum())
        unscored_p = (unscored_pos + BETA_STRENGTH * missed_base) / (missed.sum() + BETA_STRENGTH)
        models[key] = {
            "x": model.X_thresholds_.tolist(), "y": model.y_thresholds_.tolist(),
            "bin_centers": bin_centers.tolist(), "bin_counts": counts[occupied].tolist(),
            "bin_positive_counts": positives[occupied].astype(int).tolist(),
            "bin_beta_posterior_means": beta_means.tolist(),
            "gated_pairs": int(scored.sum()), "gated_positives": int(y.sum()),
            "ungated_pairs": int(missed.sum()), "ungated_positives": unscored_pos,
            "p_ungated": float(np.clip(unscored_p, EPS, 1 - EPS)),
        }
    return models


def predict_calibration(models: dict, raw: np.ndarray, gate: np.ndarray,
                        left: np.ndarray, country: np.ndarray) -> np.ndarray:
    result = np.empty(len(raw), dtype=np.float64)
    for key in set(country):
        model = models.get(key, models["__pooled__"])
        mask = country[left] == key
        gated = mask & gate
        result[gated] = np.interp(raw[gated], model["x"], model["y"])
        result[mask & ~gate] = model["p_ungated"]
    return np.clip(result, EPS, 1 - EPS)


@dataclass
class CountPosterior:
    """Prior P(T), binomial gate misses, and symmetric-polynomial pair evidence."""
    prior: np.ndarray
    discovery: float
    base_pair_prevalence: float

    def __post_init__(self):
        self.max_true = len(self.prior) - 1
        if (not np.isclose(self.prior.sum(), 1.0) or not 0 < self.discovery <= 1
                or not 0 < self.base_pair_prevalence < 1):
            raise ValueError("Count posterior needs a normalized TRAIN count prior")
        self._cache = {}

    def coefficients(self, n: int):
        if n not in self._cache:
            a = np.zeros((len(self.prior), len(self.prior)), dtype=np.float64)
            for t, mass in enumerate(self.prior):
                for j in range(min(t, n) + 1):
                    a[t, j] = (mass * comb(t, j) * self.discovery**j
                               * (1 - self.discovery)**(t - j) / comb(n, j))
            coef = np.zeros((n + 1, len(self.prior)), dtype=np.float64)
            for k in range(1, n + 1):
                for t in range(1, len(self.prior)):
                    coef[k] += 5 * a[t] / (4 * k + t)
            self._cache[n] = a, coef
        return self._cache[n]

    def best_prefix(self, gated_p: np.ndarray, ungated_p: float, n_total: int) -> tuple[int, np.ndarray]:
        """Exact expected official F0.5 under this stated count/evidence model.

        E_j is the sum of odds products over j retrieved positives; G_j(k) is
        the same sum weighted by TP in the first k. T also includes gate misses.
        The singleton action has utility P(T=0|scores), not a made-up threshold.
        """
        gated_p = np.asarray(gated_p, dtype=np.float64)
        if n_total < len(gated_p) or np.any((gated_p < 0) | (gated_p > 1)):
            raise ValueError("Invalid saved candidate count or calibrated scores")
        a, coef = self.coefficients(n_total)
        size = len(self.prior)
        odds_base = self.base_pair_prevalence / (1 - self.base_pair_prevalence)
        ungated_odds = (ungated_p / (1 - ungated_p)) / odds_base
        unscored = n_total - len(gated_p)
        suffix = np.zeros((len(gated_p) + 1, size), dtype=np.float64)
        for j in range(min(size - 1, unscored) + 1):
            suffix[-1, j] = comb(unscored, j) * ungated_odds**j
        weights = (gated_p / (1 - gated_p)) / odds_base
        for i in range(len(weights) - 1, -1, -1):
            suffix[i] = suffix[i + 1]
            suffix[i, 1:] += weights[i] * suffix[i + 1, :-1]
        normalizer = float(np.einsum("tj,j->", a, suffix[0]))
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("Count-posterior evidence is not finite and positive")
        best_k, best = 0, float(a[0, 0] / normalizer)
        utility = np.empty(len(weights) + 1, dtype=np.float64)
        utility[0] = best
        prefix = np.zeros(size, dtype=np.float64)
        prefix[0] = 1.0
        hit_mass = np.zeros(size, dtype=np.float64)
        for i, weight in enumerate(weights):
            without_i = np.convolve(prefix, suffix[i + 1])[:size]
            hit_mass[1:] += weight * without_i[:-1]
            prefix[1:] += weight * prefix[:-1].copy()
            utility[i + 1] = float(np.dot(hit_mass, coef[i + 1]) / normalizer)
            if utility[i + 1] > best + 1e-12:
                best_k, best = i + 1, utility[i + 1]
        return best_k, utility


def adjust_source(p: np.ndarray, gate: np.ndarray, source: np.ndarray, weight: float) -> np.ndarray:
    """One confident source can boost only the opposite source; no feedback loop."""
    adjusted = p.copy()
    if weight == 0 or not gate.any():
        return adjusted
    first = gate & (source == "S2-")
    second = gate & (source == "S3-")
    high2 = np.any(first & (p >= ANCHOR_PROBABILITY))
    high3 = np.any(second & (p >= ANCHOR_PROBABILITY))
    chosen = second if high2 and not high3 else first if high3 and not high2 else np.zeros(len(p), dtype=bool)
    if chosen.any():
        odds = p[chosen] / (1 - p[chosen])
        adjusted[chosen] = np.clip(odds * np.exp(weight) / (1 + odds * np.exp(weight)), EPS, 1 - EPS)
    return adjusted


def expected_decisions(data: TuneData, calibrated: np.ndarray, selected: np.ndarray,
                       model: CountPosterior, weight: float) -> tuple[np.ndarray, np.ndarray, dict]:
    decisions = np.zeros(len(data.left), dtype=bool)
    adjusted = calibrated.copy()
    chosen_counts = Counter()
    for ref in np.flatnonzero(selected):
        start, end = data.offsets[ref:ref + 2]
        rows = np.arange(start, end)
        g = data.gate[start:end]
        if end == start:
            chosen_counts[0] += 1
            continue
        p = adjust_source(calibrated[start:end], g, data.source[start:end], weight)
        adjusted[start:end] = p
        eligible = rows[g]
        order = eligible[np.lexsort((data.right[eligible], -p[g]))]
        best, _ = model.best_prefix(p[order - start], float(p[~g][0]) if (~g).any() else EPS, end - start)
        decisions[order[:best]] = True
        chosen_counts[best] += 1
    return decisions, adjusted, {str(k): chosen_counts[k] for k in sorted(chosen_counts)}


def reliability(raw: np.ndarray, calibrated: np.ndarray, labels: np.ndarray,
                gate: np.ndarray, left: np.ndarray, country: np.ndarray) -> dict:
    """Validation-fold reliability for actually reranked pairs only."""
    boundaries = (0, .01, .05, .1, .2, .4, .6, .8, .9, .95, .99, 1.000001)
    output = {}
    for key in ("all", *sorted(set(country))):
        mask = gate & (np.ones(len(raw), dtype=bool) if key == "all" else country[left] == key)
        curves = {}
        for label, values in (("raw_blend", raw), ("calibrated", calibrated)):
            bins = []
            for low, high in zip(boundaries[:-1], boundaries[1:]):
                chosen = mask & (values >= low) & (values < high)
                if chosen.any():
                    bins.append({"low": low, "high": high, "pairs": int(chosen.sum()),
                                 "mean_probability": float(values[chosen].mean()),
                                 "empirical_positive_rate": float(labels[chosen].mean())})
            curves[label] = {"brier": float(np.mean((values[mask] - labels[mask]) ** 2)),
                             "ece": float(sum(row["pairs"] * abs(row["mean_probability"] - row["empirical_positive_rate"])
                                              for row in bins) / mask.sum()), "bins": bins}
        output[key] = {"gated_pairs": int(mask.sum()), "raw_and_calibrated": curves}
    return output


def composite_key(name: str, address: str, country: str) -> tuple[str, str, tuple[str, ...]] | None:
    words = normalize_text(name).split()
    while words and words[-1] in _SUFFIXES:
        words.pop()
    digits = tuple(_DIGITS.findall(normalize_text(address)))
    if not words or not digits:
        return None
    return country, " ".join(words), digits


def address_sanity(left: str, right: str, minimum: float = .7) -> bool:
    a = {word for word in normalize_text(left).split() if not word.isdigit()}
    b = {word for word in normalize_text(right).split() if not word.isdigit()}
    return bool(a and b and len(a & b) / len(a | b) >= minimum)


def composite_candidates(data: TuneData) -> tuple[np.ndarray, dict]:
    """Only override *existing retrieved edges*, using text without truth labels."""
    keys = [composite_key(n, a, c) for n, a, c in zip(
        data.train_reference.business_name, data.train_reference.business_address, data.train_reference.country)]
    keys += [composite_key(n, a, c) for n, a, c in zip(
        data.references.business_name, data.references.business_address, data.references.country)]
    key_counts = Counter(key for key in keys if key is not None)
    reference_keys = [composite_key(n, a, c) for n, a, c in zip(
        data.references.business_name, data.references.business_address, data.references.country)]
    eligible_refs = {i: key for i, key in enumerate(reference_keys) if key is not None and key_counts[key] == 1}
    target_ids = np.unique(data.right)
    # Only target records that were actually retrieved for TUNE are inspected.
    pool_keys = {int(i): composite_key(data.pool.business_name.iat[i], data.pool.business_address.iat[i],
                                       data.pool.country.iat[i]) for i in target_ids}
    output = np.zeros(len(data.left), dtype=bool)
    matching_key = 0
    for i, (ref, target) in enumerate(zip(data.left, data.right)):
        key = eligible_refs.get(int(ref))
        if key is not None and key == pool_keys[int(target)]:
            matching_key += 1
            if address_sanity(data.references.business_address.iat[ref], data.pool.business_address.iat[target]):
                output[i] = True
    return output, {"actual_retrieved_tune_pairs": len(output),
                    "reference_key_uniqueness_scope": "47,999 saved TRAIN+TUNE references only",
                    "retrieved_target_records_inspected": len(target_ids),
                    "eligible_tune_references": len(eligible_refs),
                    "exact_key_candidate_pairs": matching_key,
                    "passing_address_sanity": int(output.sum()),
                    "sanity_minimum_alpha_token_jaccard": .7,
                    "retrieval_branch_added": False}


def compare(data: TuneData, name: str, keep: np.ndarray, baseline: np.ndarray,
            baseline_scores: np.ndarray, replicates: int, seed: int,
            *, detail: dict | None = None) -> tuple[dict, np.ndarray]:
    score, per_ref = score_decisions(data, keep)
    changed = per_ref - baseline_scores
    added, removed = keep & ~baseline, baseline & ~keep
    result = {"variant": name, "score": score,
              "delta_macro_f05": float(changed.mean()),
              "improved_entities": int((changed > 0).sum()),
              "worsened_entities": int((changed < 0).sum()),
              "unchanged_entities": int((changed == 0).sum()),
              "added_true_links": int((added & data.labels).sum()),
              "added_false_links": int((added & ~data.labels).sum()),
              "removed_true_links": int((removed & data.labels).sum()),
              "removed_false_links": int((removed & ~data.labels).sum()),
              "paired_tune_entity_bootstrap": paired_bootstrap(
                  per_ref, baseline_scores, n_bootstrap=replicates, seed=seed)}
    if detail is not None:
        result["detail"] = detail
    return result, per_ref


def inner_balance_choice(data: TuneData, outer_fit: np.ndarray, posterior: CountPosterior,
                         train_prior: dict, seed: int) -> tuple[float, dict]:
    """Select source weight using only entities in the opposite outer fold."""
    indices = np.flatnonzero(outer_fit)
    inner = balanced_folds(data.references.iloc[indices].reset_index(drop=True), seed + 1)
    sums = {weight: 0.0 for weight in BALANCE_WEIGHTS}
    validation_folds = []
    for held in (0, 1):
        validation = np.zeros(len(data.references), dtype=bool)
        validation[indices[inner == held]] = True
        fit = outer_fit & ~validation
        models = fit_calibration(data.raw, data.gate, data.labels, data.left, data.country, fit, train_prior)
        p = predict_calibration(models, data.raw, data.gate, data.left, data.country)
        measures = {}
        for weight in BALANCE_WEIGHTS:
            keep, _, _ = expected_decisions(data, p, validation, posterior, weight)
            score, _ = score_decisions(data, keep, validation)
            sums[weight] += score["macro_f05"] * validation.sum()
            measures[str(weight)] = score["macro_f05"]
        validation_folds.append({"validation_references": int(validation.sum()),
                                 "validation_ids_sha256": ids_digest(data.references.entity_id[validation]),
                                 "fit_references": int(fit.sum()),
                                 "fit_ids_sha256": ids_digest(data.references.entity_id[fit]),
                                 "macro_f05_by_source_weight": measures})
    scores = {str(weight): sums[weight] / len(indices) for weight in BALANCE_WEIGHTS}
    choice = max(BALANCE_WEIGHTS, key=lambda weight: (sums[weight], -weight))
    return choice, {"selected_weight": choice, "score_by_weight": scores,
                    "inner_entity_disjoint_validation": validation_folds}


def cross_fit(data: TuneData, train_prior: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """An outer entity's label cannot fit its calibration or its source weight."""
    fold = balanced_folds(data.references, seed)
    posterior = CountPosterior(np.asarray(train_prior["p_true_count"]),
                               train_prior["gate_true_link_discovery"], train_prior["pair_prevalence"])
    calibrated = np.empty(len(data.left), dtype=np.float64)
    adjusted = np.empty(len(data.left), dtype=np.float64)
    decisions = np.zeros(len(data.left), dtype=bool)
    records = []
    for held in (0, 1):
        validation = fold == held
        fit = ~validation
        weight, inner = inner_balance_choice(data, fit, posterior, train_prior, seed + 100 * held)
        models = fit_calibration(data.raw, data.gate, data.labels, data.left, data.country, fit, train_prior)
        probabilities = predict_calibration(models, data.raw, data.gate, data.left, data.country)
        predicted, adjusted_probabilities, distribution = expected_decisions(
            data, probabilities, validation, posterior, weight)
        rows = validation[data.left]
        calibrated[rows] = probabilities[rows]
        adjusted[rows] = adjusted_probabilities[rows]
        decisions[rows] = predicted[rows]
        measure, _ = score_decisions(data, predicted, validation)
        records.append({"outer_validation_fold": held,
                        "calibration_fit_references": int(fit.sum()),
                        "calibration_fit_ids_sha256": ids_digest(data.references.entity_id[fit]),
                        "validation_references": int(validation.sum()),
                        "validation_ids_sha256": ids_digest(data.references.entity_id[validation]),
                        "calibration_models": models,
                        "source_weight_selection": inner,
                        "validation_score": measure,
                        "chosen_prefix_distribution": distribution})
    final_choice = max(BALANCE_WEIGHTS, key=lambda weight: (
        sum(record["source_weight_selection"]["score_by_weight"][str(weight)]
            * record["calibration_fit_references"] for record in records), -weight))
    final_models = fit_calibration(data.raw, data.gate, data.labels, data.left, data.country,
                                   np.ones(len(data.references), dtype=bool), train_prior)
    return calibrated, adjusted, decisions, {
        "outer_fold": fold, "outer_records": records,
        "final_weight_for_future_inference": final_choice,
        "final_calibrator_fitted_on_all_tune": final_models,
        "inner_selection_scope": "Entity-disjoint within each outer calibration half; no outer validation labels fit a calibration or source weight",
    }, reliability(data.raw, calibrated, data.labels, data.gate, data.left, data.country)


def run(args) -> dict:
    started = time.monotonic()
    inputs, report_file, output = Path(args.inputs), Path(args.cpu_report), Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace existing decision artifacts: {output}")
    data = load_verified(inputs, report_file)
    if len(data.left) != 600_000 or len(data.references) != 12_000:
        raise ValueError("P1 requires all 600,000 saved TUNE pairs / 12,000 entities")
    baseline = np.asarray(data.raw >= BASELINE_THRESHOLD)
    baseline_report, baseline_scores = score_decisions(data, baseline)
    for key, expected in data.saved_baseline.items():
        if key == "threshold":
            observed = BASELINE_THRESHOLD
        elif key.startswith("f05_"):
            observed = baseline_report["by_country"][key[4:]]["macro_f05"]
        else:
            observed = baseline_report.get(key)
        if observed is not None and not np.isclose(observed, expected, atol=1e-12, rtol=0):
            raise ValueError(f"Frozen v1 TUNE baseline changed: {key}: {observed} != {expected}")
    train_prior = train_count_prior(data)
    calibrated, adjusted, rule, folds, curves = cross_fit(data, train_prior, args.seed)
    if np.any(rule & ~data.gate) or not np.isfinite(calibrated).all():
        raise ValueError("Decision rule accepted a CE-unscored candidate or has nonfinite calibration")
    rule_report, _ = compare(data, "calibrated_expected_f05_oof", rule, baseline,
                             baseline_scores, args.replicates, args.seed)
    a_variants, combination_variants = [], []
    a_masks, combination_masks = {}, {}
    for delta in DELTAS:
        key = f"{delta:.2f}"
        a, counts = assignment(baseline, data.right, data.raw, data.left, delta)
        a_masks[key] = a
        a_report, _ = compare(data, f"assignment_delta_{key}", a, baseline, baseline_scores,
                              args.replicates, args.seed, detail=counts)
        a_variants.append(a_report)
        combined, counts = assignment(rule, data.right, adjusted, data.left, delta)
        combination_masks[key] = combined
        combined_report, _ = compare(data, f"oof_rule_plus_assignment_delta_{key}", combined,
                                      baseline, baseline_scores, args.replicates, args.seed, detail=counts)
        combination_variants.append(combined_report)
    selected_delta = max(DELTAS, key=lambda d: (
        a_variants[DELTAS.index(d)]["score"]["macro_f05"], -d))
    key_mask = None
    composite_report = None
    if not args.skip_composite:
        key_mask, key_details = composite_candidates(data)
        extras = key_mask & ~baseline
        key_details.update({"newly_accepted_true_links_vs_baseline": int((extras & data.labels).sum()),
                            "newly_accepted_false_links_vs_baseline": int((extras & ~data.labels).sum()),
                            "truth_used_to_create_key_mask": False})
        key_baseline, _ = compare(data, "retrieved_composite_key_plus_baseline", baseline | key_mask,
                                  baseline, baseline_scores, args.replicates, args.seed)
        key_rule, counts = assignment(rule | key_mask, data.right, adjusted, data.left, 0.0)
        key_combined, _ = compare(data, "retrieved_composite_key_plus_rule_plus_assignment_0",
                                  key_rule, baseline, baseline_scores, args.replicates, args.seed, detail=counts)
        composite_report = {"scope_and_coverage": key_details,
                            "baseline_plus_key": key_baseline, "rule_plus_key_and_pure_assignment": key_combined}
    output.mkdir(parents=True, exist_ok=True)
    fold = folds.pop("outer_fold")
    pd.DataFrame({"source1_entity_id": data.references.entity_id, "country": data.country,
                  "outer_validation_fold": fold, "original_left_idx": data.original_left,
                  "true_count": data.actual, "baseline_f05": baseline_scores,
                  "rule_f05": score_decisions(data, rule)[1]}).to_parquet(
                      output / "tune_references.parquet", index=False)
    pd.DataFrame({"original_pair_row": data.original_pair_rows,
                  "left_idx": data.original_left[data.left], "right_idx": data.right,
                  "source1_entity_id": data.references.entity_id.to_numpy()[data.left],
                  "candidate_entity_id": data.pool.entity_id.to_numpy()[data.right],
                  "candidate_source": data.source,
                  "original_tree_probability": data.base,
                  "original_blend_probability": data.raw,
                  "was_ce_scored": data.gate, "calibrated_oof_probability": calibrated.astype(np.float32),
                  "source_adjusted_oof_probability": adjusted.astype(np.float32),
                  "is_truth_for_scoring_only": data.labels,
                  "baseline_accept": baseline, "rule_accept_oof": rule,
                  "assignment_accept_selected": a_masks[f"{selected_delta:.2f}"],
                  "rule_plus_assignment_accept_selected": combination_masks[f"{selected_delta:.2f}"],
                  "composite_key_pass": key_mask if key_mask is not None else np.zeros(len(rule), dtype=bool),
                  }).to_parquet(output / "tune_pairs.parquet", index=False)
    selection = {"assignment_delta": selected_delta,
                 "source_weight_for_future_inference": folds["final_weight_for_future_inference"],
                 "rule_requires_new_holdout": True,
                 "winner_claim": "None: all reported selection/validation is medium TUNE development data"}
    config = {"schema": 1, "hypothesis": "Entity-level count posterior and label-free one-to-one/key constraints can improve F0.5 at fixed retrieved pairs and matcher scores",
              "changed_variable": "decision and calibration only, no mining/model training",
              "seed": args.seed, "bootstrap_replicates": args.replicates,
              "full_competing_target_pool": FULL_POOL_ROWS, "tune_references": len(data.references),
              "tune_candidate_pairs": len(data.left), "tune_id_sha256": TUNE_ID_SHA256,
              "reference_id_sha256_by_fold": [ids_digest(data.references.entity_id[fold == i]) for i in (0, 1)],
              "provenance": data.provenance, "tree_gate": TREE_GATE, "blend_ce_weight": .75,
              "baseline_threshold": BASELINE_THRESHOLD, "assignment_deltas": list(DELTAS),
              "source_balance_weights": list(BALANCE_WEIGHTS), "anchor_probability": ANCHOR_PROBABILITY,
              "calibration": {"type": "country-wise Beta-bin isotonic", "bins": ISOTONIC_BINS,
                              "beta_prior_strength": BETA_STRENGTH,
                              "fit_scope": "only the opposite outer TUNE half, inner weight tuning entity-disjoint",
                              "unscored_candidates": "separate country-wise ungated prevalence for count evidence; never eligible for prediction"},
              "count_model": {"training_only_prior": train_prior,
                              "definition": "P(T=t) Binomial(t,j; TRAIN gate discovery) times symmetric-polynomial likelihood ratios of calibrated candidate odds relative to TRAIN pair prevalence; maximize exact expected official singleton/macro F0.5 over gated score prefixes",
                              "assumptions": "independent pair evidence conditional on count; uniform subset at fixed retrieved count; binomial gate misses"},
              "outer_cross_fit": folds, "selection_on_tune": selection,
              "evaluation_or_old_holdout_used": False, "france_labels_used": False}
    json_write(output / "experiment_config.json", config)
    results = {"schema": 1, "population": "12,000 medium TUNE S1s; 600,000 saved dense top-50 pairs from a full 10,320,219-target competition",
               "baseline": baseline_report, "assignment": a_variants,
               "calibrated_expected_rule_oof": rule_report,
               "rule_plus_assignment": combination_variants,
               "composite_key": composite_report,
               "retrieval_ceiling": data.retrieval, "reliability_oof": curves,
               "selection": selection,
               "keep_discard": {},
               "limitations": ["Calibration/source weights are out-of-fold on reused development TUNE, not a fresh holdout; bootstrap CIs condition on model and choices and do not correct repeated selection.",
                               "The prior comes from 35,999 TRAIN S1s, the score calibrators and source weight from entity-disjoint TUNE folds.",
                               "One-to-one conflicts are visible among these 12,000 TUNE S1s only; a full submission has many more competing references.",
                               "Composite-key uniqueness is measured only among 47,999 saved TRAIN+TUNE references, and only existing dense-retrieved candidate edges are tested; no new retrieval ceiling or France accuracy is claimed."],
               "output_files": {name: {"sha256": sha256_file(output / name),
                                        "bytes": (output / name).stat().st_size}
                                for name in ("tune_references.parquet", "tune_pairs.parquet", "experiment_config.json")},
               "elapsed_seconds": time.monotonic() - started}
    for variant in [*a_variants, rule_report, *combination_variants]:
        lower = variant["paired_tune_entity_bootstrap"]["confidence_intervals"]["left_minus_right"][0]
        delta_f = variant["delta_macro_f05"]
        redundant_margin = (variant["variant"].startswith("oof_rule_plus_assignment_delta_")
                            and variant["variant"] != "oof_rule_plus_assignment_delta_0.00")
        results["keep_discard"][variant["variant"]] = (
            "Discard as redundant margin: it leaves both TUNE conflicts unresolved"
            if redundant_margin else
            "Keep as TUNE candidate for fresh holdout" if delta_f > 0 and lower >= -1e-12
            else "Discard as F0.5 leader on this reused TUNE; retain measured tradeoff")
    if composite_report is not None:
        for name in ("baseline_plus_key", "rule_plus_key_and_pure_assignment"):
            variant = composite_report[name]
            lower = variant["paired_tune_entity_bootstrap"]["confidence_intervals"]["left_minus_right"][0]
            results["keep_discard"][variant["variant"]] = (
                "Discard composite-key override: it adds zero retrieved links versus its parent rule; "
                "a new retrieval branch remains untested")
    json_write(output / "results.json", results)
    print(json.dumps({"output": str(output.resolve()), "baseline": baseline_report["macro_f05"],
                      "assignment": [(v["variant"], v["score"]["macro_f05"]) for v in a_variants],
                      "rule": rule_report["score"]["macro_f05"],
                      "rule_plus_assignment": [(v["variant"], v["score"]["macro_f05"])
                                               for v in combination_variants],
                      "composite": composite_report["scope_and_coverage"] if composite_report else None,
                      "seconds": results["elapsed_seconds"]}, allow_nan=False), flush=True)
    return results


def analyze_saved(args) -> dict:
    """Independent saved-output replay and OOF source-balance ablation, no refit."""
    output = Path(args.output)
    supplement = output / "source_balance_ablation.json"
    if supplement.exists():
        raise FileExistsError(f"Refusing to replace prior analysis: {supplement}")
    data = load_verified(Path(args.inputs), Path(args.cpu_report))
    config = json.loads((output / "experiment_config.json").read_text())
    results = json.loads((output / "results.json").read_text())
    refs = pd.read_parquet(output / "tune_references.parquet")
    pairs = pd.read_parquet(output / "tune_pairs.parquet")
    if (len(refs) != len(data.references) or len(pairs) != len(data.left)
            or ids_digest(refs.source1_entity_id) != TUNE_ID_SHA256
            or not np.array_equal(refs.source1_entity_id.to_numpy(), data.references.entity_id.to_numpy())
            or not np.array_equal(pairs.left_idx, data.original_left[data.left])
            or not np.array_equal(pairs.right_idx, data.right)
            or not np.array_equal(pairs.is_truth_for_scoring_only, data.labels)
            or not np.array_equal(pairs.was_ce_scored, data.gate)
            or not np.array_equal(pairs.original_blend_probability, data.raw)
            or config["provenance"]["pair_digest"] != data.provenance["pair_digest"]
            or config["tune_id_sha256"] != TUNE_ID_SHA256):
        raise ValueError("Emitted complete TUNE pairs or S1 IDs do not replay the verified inputs")
    for name, column in (("baseline", "baseline_accept"),
                         ("calibrated_expected_rule_oof", "rule_accept_oof")):
        score, per_ref = score_decisions(data, pairs[column].to_numpy(dtype=bool))
        saved = results[name] if name == "baseline" else results[name]["score"]
        if (not np.isclose(score["macro_f05"], saved["macro_f05"], atol=1e-12, rtol=0)
                or not np.array_equal(per_ref, refs[f"{'baseline' if name == 'baseline' else 'rule'}_f05"])):
            raise ValueError(f"Emitted decisions do not reproduce the {name} score")
    baseline = pairs.baseline_accept.to_numpy(dtype=bool)
    chosen = pairs.rule_accept_oof.to_numpy(dtype=bool)
    calibrated = pairs.calibrated_oof_probability.to_numpy(dtype=np.float64)
    train_prior = config["count_model"]["training_only_prior"]
    posterior = CountPosterior(np.asarray(train_prior["p_true_count"]),
                               train_prior["gate_true_link_discovery"], train_prior["pair_prevalence"])
    unweighted, _, counts = expected_decisions(data, calibrated,
                                                np.ones(len(data.references), dtype=bool), posterior, 0.0)
    base_score, base_entities = score_decisions(data, baseline)
    no_balance_score, no_balance_entities = score_decisions(data, unweighted)
    balanced_score, balanced_entities = score_decisions(data, chosen)
    by_fold = {}
    for fold in (0, 1):
        selected = refs.outer_validation_fold.to_numpy() == fold
        by_fold[str(fold)] = {key: score_decisions(data, decision & selected[data.left], selected)[0]
                              for key, decision in (("baseline", baseline), ("no_source_balance", unweighted),
                                                    ("nested_source_balance", chosen))}
    by_country = {}
    for country in sorted(set(data.country)):
        selected = data.country == country
        by_country[country] = paired_bootstrap(balanced_entities[selected], base_entities[selected],
                                               n_bootstrap=args.replicates, seed=args.seed)
    key = pairs.composite_key_pass.to_numpy(dtype=bool)
    output_report = {
        "schema": 1, "purpose": "Ablate the source bonus while holding the exact OOF calibrated probabilities fixed",
        "saved_results_sha256": sha256_file(output / "results.json"),
        "saved_config_sha256": sha256_file(output / "experiment_config.json"),
        "saved_pairs_sha256": sha256_file(output / "tune_pairs.parquet"),
        "bootstrap_seed": args.seed, "bootstrap_replicates": args.replicates,
        "baseline": base_score, "calibrated_rule_without_source_bonus": no_balance_score,
        "calibrated_rule_with_nested_source_bonus": balanced_score,
        "nested_bonus_minus_no_bonus": paired_bootstrap(
            balanced_entities, no_balance_entities, n_bootstrap=args.replicates, seed=args.seed),
        "no_bonus_minus_baseline": paired_bootstrap(
            no_balance_entities, base_entities, n_bootstrap=args.replicates, seed=args.seed),
        "no_bonus_prefix_distribution": counts,
        "country_nested_bonus_minus_baseline": by_country, "scores_by_outer_fold": by_fold,
        "source_probabilities_modified": int(np.count_nonzero(np.abs(
            pairs.source_adjusted_oof_probability.to_numpy() - calibrated) > 1e-7)),
        "composite_key_passed": int(key.sum()),
        "composite_key_passed_true": int((key & data.labels).sum()),
        "composite_key_passed_false": int((key & ~data.labels).sum()),
        "new_composite_links_vs_baseline": int((key & ~baseline).sum()),
        "limitations": "All scores are still on reused development TUNE; source weight was selected via inner TUNE entity splits, not a fresh holdout.",
    }
    json_write(supplement, output_report)
    print(json.dumps({"output": str(supplement.resolve()), "no_bonus_f05": no_balance_score["macro_f05"],
                      "nested_bonus_f05": balanced_score["macro_f05"],
                      "source_bonus_delta": output_report["nested_bonus_minus_no_bonus"]["estimates"]["left_minus_right"],
                      "composite_passed_true": output_report["composite_key_passed_true"]},
                     allow_nan=False), flush=True)
    return output_report


def review_saved(args) -> dict:
    """Prefer the higher-precision near-tie using the saved entity OOF scores."""
    output = Path(args.output)
    destination = output / "decision_review.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to replace prior review: {destination}")
    ablation = json.loads((output / "source_balance_ablation.json").read_text())
    report = json.loads((output / "results.json").read_text())
    config = json.loads((output / "experiment_config.json").read_text())
    data = load_verified(Path(args.inputs), Path(args.cpu_report))
    pairs = pd.read_parquet(output / "tune_pairs.parquet")
    if (sha256_file(output / "tune_pairs.parquet") != ablation["saved_pairs_sha256"]
            or sha256_file(output / "results.json") != ablation["saved_results_sha256"]
            or sha256_file(output / "experiment_config.json") != ablation["saved_config_sha256"]
            or not np.array_equal(pairs.right_idx, data.right)
            or not np.array_equal(pairs.is_truth_for_scoring_only, data.labels)):
        raise ValueError("The saved ablation and scored pairs are not the same verified run")
    prior = config["count_model"]["training_only_prior"]
    posterior = CountPosterior(np.asarray(prior["p_true_count"]),
                               prior["gate_true_link_discovery"], prior["pair_prevalence"])
    no_balance, _, _ = expected_decisions(
        data, pairs.calibrated_oof_probability.to_numpy(dtype=np.float64),
        np.ones(len(data.references), dtype=bool), posterior, 0)
    no_balance_a, no_balance_claims = assignment(
        no_balance, data.right, pairs.calibrated_oof_probability.to_numpy(dtype=np.float64), data.left, 0)
    balanced_a = pairs.rule_plus_assignment_accept_selected.to_numpy(dtype=bool)
    baseline = pairs.baseline_accept.to_numpy(dtype=bool)
    baseline_score, baseline_entities = score_decisions(data, baseline)
    no_balance_score, no_balance_entities = score_decisions(data, no_balance_a)
    balanced_score, balanced_entities = score_decisions(data, balanced_a)
    if (not np.isclose(balanced_score["macro_f05"], report["rule_plus_assignment"][0]["score"]["macro_f05"])
            or not np.isclose(baseline_score["macro_f05"], report["baseline"]["macro_f05"])):
        raise ValueError("Saved baseline or selected assignment does not replay")
    np.save(output / "no_source_rule_accept.npy", no_balance)
    np.save(output / "no_source_rule_plus_assignment_accept.npy", no_balance_a)
    lower_bonus = ablation["nested_bonus_minus_no_bonus"]["confidence_intervals"]["left_minus_right"][0]
    if not lower_bonus < 0 or not no_balance_score["micro_precision"] > balanced_score["micro_precision"]:
        raise ValueError("The measured source-balance near-tie no longer supports a precision preference")
    by_country = {country: {"no_source_plus_assignment": no_balance_score["by_country"][country],
                            "nested_bonus_plus_assignment": balanced_score["by_country"][country]}
                  for country in sorted(set(data.country))}
    review = {
        "schema": 1, "population": "reused 12,000 medium TUNE S1s; OOF country calibration, no fresh holdout",
        "baseline": baseline_score,
        "no_source_rule_plus_pure_assignment": no_balance_score,
        "nested_source_rule_plus_pure_assignment": balanced_score,
        "no_source_minus_baseline": paired_bootstrap(
            no_balance_entities, baseline_entities, n_bootstrap=args.replicates, seed=args.seed),
        "nested_source_minus_no_source": paired_bootstrap(
            balanced_entities, no_balance_entities, n_bootstrap=args.replicates, seed=args.seed),
        "by_country": by_country, "no_source_assignment_conflicts": no_balance_claims,
        "composite_key": {"passing_retrieved_pairs": ablation["composite_key_passed"],
                          "true": ablation["composite_key_passed_true"],
                          "false": ablation["composite_key_passed_false"],
                          "incremental_links": ablation["new_composite_links_vs_baseline"]},
        "keep_discard": {
            "one_to_one_delta_0": "Keep as constraint candidate: drops two false claims and zero true on this TUNE; 95% paired CI [0, 0.000111], but only two conflicts were visible.",
            "one_to_one_positive_margins": "Discard: all leave those two conflicts unresolved and tie the baseline.",
            "beta_isotonic_plus_expected_f05_count_prior": "Keep as TUNE candidate: OOF no-source rule improves F0.5 and singleton accuracy with fewer false links; require fresh v2 holdout before promotion.",
            "source_balance_log_odds_0_25": "Discard for current finalist: inner folds selected it, but the isolated paired gain CI crosses zero and it lowers precision versus no bonus.",
            "composite_key_override_on_dense_pairs": "Discard: 3,967 passing retrieved pairs (one false) add zero accepted links. A new retrieval branch and recomputed features would be a separate experiment.",
        },
        "recommendation_for_new_holdout": {"decision": "country-wise isotonic, TRAIN count-prior expected-F0.5 prefix, source weight 0, then one-to-one delta 0",
                                            "final_calibrator": "experiment_config.json/outer_cross_fit/final_calibrator_fitted_on_all_tune",
                                            "source_balance_weight": 0.0, "assignment_margin": 0.0,
                                            "status": "TUNE-only candidate, not a release freeze"},
        "artifact_sha256": {name: sha256_file(output / name) for name in (
            "results.json", "experiment_config.json", "tune_references.parquet",
            "tune_pairs.parquet", "source_balance_ablation.json",
            "no_source_rule_accept.npy", "no_source_rule_plus_assignment_accept.npy")},
        "source_sha256": sha256_file(Path(__file__)),
        "bootstrap_seed": args.seed, "bootstrap_replicates": args.replicates,
        "remaining_dependency": "A single pre-frozen v2 candidate must be tested against the full pool on a new entity-disjoint holdout. France has no labels; full-reference conflicts and key retrieval gain remain unmeasured.",
    }
    json_write(destination, review)
    print(json.dumps({"output": str(destination.resolve()), "preferred_f05": no_balance_score["macro_f05"],
                      "preferred_false_links": no_balance_score["false_positive_links"],
                      "bonus_f05": balanced_score["macro_f05"],
                      "preferred_ci": review["no_source_minus_baseline"]["confidence_intervals"]["left_minus_right"]},
                     allow_nan=False), flush=True)
    return review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, help="Existing cpu-decision/inputs directory (read only)")
    parser.add_argument("--cpu-report", required=True, help="Existing cpu-decision/results.json (read only)")
    parser.add_argument("--output", default="artifacts/v2_p1_decision")
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--skip-composite", action="store_true")
    parser.add_argument("--analyze-saved", action="store_true",
                        help="Replay the finished P1 output and ablate the source bonus without fitting again")
    parser.add_argument("--review-saved", action="store_true",
                        help="Record the higher-precision near-tie using the saved OOF probabilities")
    args = parser.parse_args()
    if args.replicates != 10_000:
        raise ValueError("P1 requires 10,000 paired entity bootstrap resamples")
    if args.analyze_saved and args.review_saved:
        raise ValueError("Choose only one saved-output analysis")
    if args.analyze_saved:
        analyze_saved(args)
    elif args.review_saved:
        review_saved(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
