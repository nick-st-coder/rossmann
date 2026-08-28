"""Regression metric helpers for the Rossmann forecasting task."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)

METRICS = ("MAE", "RMSE", "RMSLE")


def evaluate_model(y_true, y_pred):
    """Compute MAE, RMSE, and RMSLE for a single prediction set.

    Args:
        y_true: Ground-truth target values.
        y_pred: Model predictions (clipped at 0 for RMSLE).

    Returns:
        Dict mapping metric name to value.
    """
    y_pred_clipped = np.clip(y_pred, 0, None)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "RMSLE": root_mean_squared_log_error(y_true, y_pred_clipped),
    }


def evaluate_train_test(
    y_train: Any,
    y_pred_train: Any,
    y_test: Any,
    y_pred_test: Any,
) -> dict[str, float]:
    """Compute train/test metrics and the gap between them.

    The train/test gap is a quick overfitting signal: a large gap means the
    model fits the training data far better than it generalizes.

    Args:
        y_train: Ground-truth training targets.
        y_pred_train: Predictions on the training set.
        y_test: Ground-truth test targets.
        y_pred_test: Predictions on the test set.

    Returns:
        Dict with ``train_<metric>``, ``test_<metric>``, and
        ``gap_<metric>`` keys for each metric in :data:`METRICS`.
    """
    train_metrics = evaluate_model(y_train, y_pred_train)
    test_metrics = evaluate_model(y_test, y_pred_test)

    report: dict[str, float] = {}
    for metric in METRICS:
        train_value = train_metrics[metric]
        test_value = test_metrics[metric]
        report[f"train_{metric}"] = train_value
        report[f"test_{metric}"] = test_value
        report[f"gap_{metric}"] = test_value - train_value

    return report
