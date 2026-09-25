"""Small, reproducible matchers with a common probability interface.

Threshold selection belongs to the entity-level evaluation runner. Classifier
probabilities alone do not optimize the challenge's macro F0.5 metric.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class HeuristicModel:
    """Untrained reference score; useful for checking the value of supervision."""
    feature_names_in_: list[str]

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        name = np.maximum(features["core_token_sort"], features["name_token_sort"])
        address = features["address_token_sort"].to_numpy()
        number_conflict = features["number_disjoint"].to_numpy()
        long_number_conflict = features["long_number_disjoint"].to_numpy()
        # Address support is essential when business names are common or reused.
        score = 0.55 * name + 0.45 * address
        score -= 0.12 * number_conflict + 0.10 * long_number_conflict
        score *= features["country_equal"].to_numpy()
        score = np.asarray(np.clip(score, 0, 1), dtype=np.float32)
        return np.column_stack((1 - score, score))


def fit_model(X: pd.DataFrame, y: np.ndarray, kind: str = "catboost",
              device: str = "CPU", **kwargs: Any) -> Any:
    """Fit a joblib-serializable matcher.

    Common options: ``seed`` (42), ``sample_weight`` and ``eval_set=(Xv, yv)``.
    CatBoost additionally accepts its standard constructor arguments, including
    iterations, depth, learning_rate, thread_count, and early_stopping_rounds.
    Logistic regression accepts C, max_iter, class_weight, and solver.
    GPU CatBoost is requested explicitly via ``device='GPU'``; failures are not
    hidden by an automatic CPU fallback.
    """
    y = np.asarray(y, dtype=np.int8)
    if len(X) != len(y):
        raise ValueError("X and y lengths differ")
    if not len(y):
        raise ValueError("Cannot train a model on zero pairs")
    kind = kind.lower().replace("-", "_")
    if kind in {"heuristic", "rules"}:
        return HeuristicModel(list(X.columns))
    if np.unique(y).size != 2:
        raise ValueError("Training labels must contain both matching and nonmatching pairs")
    seed = kwargs.pop("seed", kwargs.pop("random_state", 42))
    sample_weight = kwargs.pop("sample_weight", None)
    eval_set = kwargs.pop("eval_set", None)
    if kind in {"catboost", "cat"}:
        from catboost import CatBoostClassifier

        early_stopping_rounds = kwargs.pop("early_stopping_rounds", 60)
        verbose = kwargs.pop("verbose", False)
        params = {
            "iterations": 550, "depth": 7, "learning_rate": 0.08,
            "loss_function": "Logloss", "eval_metric": "Logloss",
            "random_seed": seed, "task_type": device.upper(),
            "thread_count": 6, "allow_writing_files": False,
            "l2_leaf_reg": 5.0,
        }
        params.update(kwargs)
        model = CatBoostClassifier(**params)
        fit_options: dict[str, Any] = {"sample_weight": sample_weight, "verbose": verbose}
        if eval_set is not None:
            fit_options.update(eval_set=eval_set, early_stopping_rounds=early_stopping_rounds,
                               use_best_model=True)
        model.fit(X, y, **fit_options)
        return model
    if kind in {"logistic", "logistic_regression", "lr"}:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        # Standardization is fitted on training data only. Do not balance classes
        # by default: threshold calibration needs the natural candidate mix.
        params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "random_state": seed}
        params.update(kwargs)
        model = Pipeline([("scale", StandardScaler()), ("classifier", LogisticRegression(**params))])
        model.fit(X, y, **({"classifier__sample_weight": sample_weight} if sample_weight is not None else {}))
        return model
    if kind in {"histgb", "hist_gradient_boosting"}:
        from sklearn.ensemble import HistGradientBoostingClassifier

        params = {"learning_rate": 0.08, "max_iter": 300, "max_leaf_nodes": 31,
                  "l2_regularization": 2.0, "random_state": seed}
        params.update(kwargs)
        model = HistGradientBoostingClassifier(**params)
        model.fit(X, y, sample_weight=sample_weight)
        return model
    raise ValueError(f"Unknown model kind: {kind!r}")


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Positive-class probability or baseline score in candidate row order."""
    if len(X) == 0:
        return np.empty(0, dtype=np.float32)
    prediction = np.asarray(model.predict_proba(X))
    if prediction.ndim != 2 or prediction.shape != (len(X), 2):
        raise ValueError(f"Expected two-class predictions, got shape {prediction.shape}")
    result = np.asarray(prediction[:, 1], dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Model produced non-finite predictions")
    return result
