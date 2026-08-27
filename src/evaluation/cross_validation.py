"""Cross-validation and evaluation helpers for the forecasting task."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import TimeSeriesSplit

from src.evaluation.metrics_report import evaluate_model
from src.models.train import clip_predictions


def cross_validate_scores(
    model: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    gap: int = 7,
) -> pd.DataFrame:
    """Run time-series CV, returning per-fold metric rows.

    Args:
        model: An unfitted regressor.
        X: Feature matrix.
        y: Target vector.
        n_splits: Number of forward splits.
        gap: Days to leave between consecutive folds.

    Returns:
        DataFrame with one row per fold and columns ``MAE``, ``RMSE``,
        ``RMSLE``.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    rows: list[dict[str, float]] = []
    for train_idx, valid_idx in tscv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        y_pred = clip_predictions(model.predict(X.iloc[valid_idx]))
        rows.append(evaluate_model(y.iloc[valid_idx], y_pred))
    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Summarize per-fold scores with mean ± std for each metric.

    Args:
        scores: Per-fold metric DataFrame (see :func:`cross_validate_scores`).

    Returns:
        Mapping of metric name to ``{"mean", "std"}``.
    """
    return {
        metric: {
            "mean": float(scores[metric].mean()),
            "std": float(scores[metric].std()),
        }
        for metric in scores.columns
    }


def mean_metric(scores: pd.DataFrame, metric: str = "RMSLE") -> float:
    """Return the mean value of a single metric across folds.

    Args:
        scores: Per-fold metric DataFrame.
        metric: Metric column name.

    Returns:
        The mean score; ``np.nan`` if the column is absent.
    """
    if metric not in scores.columns:
        return np.nan
    return float(scores[metric].mean())
