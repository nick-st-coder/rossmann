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


def predict_two_stage(
    model: BaseEstimator,
    X: pd.DataFrame,
    open_col: str = "Open",
) -> np.ndarray:
    """Predict sales with the two-stage rule.

    Closed rows (``Open == 0``) are predicted as exactly 0; open rows are
    predicted by ``model``. This mirrors how the model was trained (regressor
    fit on open rows only) and is the inference path the serving app uses.

    Args:
        model: A fitted regressor trained on open-store rows.
        X: Feature matrix that still contains the ``Open`` column.
        open_col: Name of the ``Open`` indicator column.

    Returns:
        Array of sales predictions, 0 for closed rows.
    """
    open_mask = X[open_col].to_numpy() == 1
    preds = np.zeros(len(X), dtype=float)
    if open_mask.any():
        X_open = X.loc[open_mask].drop(columns=[open_col])
        preds[open_mask] = clip_predictions(model.predict(X_open))
    return preds


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str = "xgboost",
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


def cross_validate_by_date(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    dates: pd.Series,
    n_splits: int = 5,
    gap_days: int = 7,
) -> pd.DataFrame:
    """Run chronological cross-validation split on unique dates.

    The dataset has one row per store per date, so row-index splits (as in
    :func:`cross_validate`) mix dates across folds and ``gap`` counts rows,
    not days. This variant splits on the unique sorted dates so every fold
    is a clean time window with a real day gap.

    Args:
        model: An unfitted regressor.
        X: Feature matrix (open-store rows only).
        y: Target vector aligned with ``X``.
        dates: Date for each row, aligned with ``X``/``y``.
        n_splits: Number of forward folds.
        gap_days: Calendar days to leave between consecutive folds.

    Returns:
        DataFrame of per-fold metric rows (MAE, RMSE, RMSLE).
    """
    unique_dates = np.sort(dates.unique())
    n_dates = len(unique_dates)
    # Each fold trains on an expanding window and validates on the next block.
    fold_size = n_dates // (n_splits + 1)
    rows: list[dict[str, float]] = []

    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        valid_start = train_end + gap_days
        valid_end = min(valid_start + fold_size, n_dates)

        if valid_start >= n_dates:
            break

        train_dates = unique_dates[:train_end]
        valid_dates = unique_dates[valid_start:valid_end]

        train_mask = dates.isin(train_dates).to_numpy()
        valid_mask = dates.isin(valid_dates).to_numpy()

        model.fit(X[train_mask], y[train_mask])
        y_pred = clip_predictions(model.predict(X[valid_mask]))
        rows.append(evaluate_model(y[valid_mask], y_pred))

    return pd.DataFrame(rows)
