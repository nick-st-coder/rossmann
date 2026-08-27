"""Model training pipelines for the Rossmann forecasting task.

Deterministic, seeded training helpers using scikit-learn / XGBoost /
LightGBM. Splits are chronological via :class:`sklearn.model_selection.TimeSeriesSplit`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

from src.evaluation.metrics_report import evaluate_model

LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": 46,
    "verbose": -1,
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "random_state": 44,
}


def make_baseline() -> DummyRegressor:
    """Return a mean-predictor baseline regressor."""
    return DummyRegressor(strategy="mean")


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Return a Ridge pipeline with robust scaling (the linear baseline)."""
    return Pipeline(
        [
            ("scaler", RobustScaler()),
            ("model", Ridge(alpha=alpha)),
        ]
    )


def make_model(model_name: str, **params: Any) -> BaseEstimator:
    """Build a regressor by name, overriding defaults with ``params``.

    Args:
        model_name: One of ``"lightgbm"``, ``"xgboost"``, ``"ridge"``.
        **params: Hyperparameters to override the library defaults.

    Raises:
        ValueError: If ``model_name`` is not supported.

    Returns:
        An unfitted scikit-learn-compatible regressor.
    """
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(**{**LGBM_PARAMS, **params})
    if model_name == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(**{**XGB_PARAMS, **params})
    if model_name == "ridge":
        return make_ridge_pipeline(**params)
    raise ValueError(f"Unsupported model_name: {model_name!r}")


def clip_predictions(y_pred: np.ndarray | pd.Series) -> np.ndarray:
    """Clip predictions at 0 since sales cannot be negative."""
    return np.clip(y_pred, 0, None)


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "lightgbm",
    **params: Any,
) -> BaseEstimator:
    """Fit a model on the full training set.

    Args:
        X_train: Feature matrix.
        y_train: Target vector.
        model_name: Model family (see :func:`make_model`).
        **params: Hyperparameters passed to :func:`make_model`.

    Returns:
        A fitted model.
    """
    model = make_model(model_name, **params)
    model.fit(X_train, y_train)
    return model


def cross_validate(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 7,
) -> pd.DataFrame:
    """Run chronological time-series cross-validation.

    Args:
        model: An unfitted regressor.
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of forward splits.
        gap: Days to leave between consecutive folds.

    Returns:
        DataFrame of per-fold metric rows (MAE, RMSE, RMSLE).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows: list[dict[str, float]] = []
    for train_idx, valid_idx in tscv.split(X):
        X_fold_train, X_fold_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_fold_train, y_fold_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred = clip_predictions(model.predict(X_fold_valid))
        rows.append(evaluate_model(y_fold_valid, y_pred))

    return pd.DataFrame(rows)
