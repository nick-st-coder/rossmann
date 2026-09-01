"""Tests for regression metrics in ``src/evaluation/metrics_report.py``."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.metrics_report import METRICS, evaluate_model, evaluate_train_test


def test_evaluate_model_perfect_predictions() -> None:
    """Perfect predictions yield zero error on all metrics."""
    y_true = np.array([100.0, 200.0, 300.0])
    report = evaluate_model(y_true, y_true)
    assert report["MAE"] == pytest.approx(0.0)
    assert report["RMSE"] == pytest.approx(0.0)
    assert report["RMSLE"] == pytest.approx(0.0)


def test_evaluate_model_known_error() -> None:
    """MAE/RMSE match hand-computed values for a small example."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 30.0])
    report = evaluate_model(y_true, y_pred)
    assert report["MAE"] == pytest.approx((2 + 2 + 0) / 3)
    assert report["RMSE"] == pytest.approx(np.sqrt((4 + 4 + 0) / 3))
    # RMSLE uses log1p: log(13/11), log(19/21), log(31/31).
    expected_rmsle = np.sqrt(
        (np.log1p(12) - np.log1p(10)) ** 2 + (np.log1p(18) - np.log1p(20)) ** 2 + 0.0
    ) / np.sqrt(3)
    assert report["RMSLE"] == pytest.approx(expected_rmsle)


def test_evaluate_model_clips_negative_predictions() -> None:
    """Negative predictions are clipped at 0 for RMSLE (log domain)."""
    y_true = np.array([5.0, 5.0])
    y_pred = np.array([-3.0, 5.0])
    report = evaluate_model(y_true, y_pred)
    # Clipped prediction is 0, so RMSLE = sqrt((log1p(5)-log1p(0))^2 + 0)/sqrt(2).
    expected = np.sqrt((np.log1p(5) - np.log1p(0)) ** 2) / np.sqrt(2)
    assert report["RMSLE"] == pytest.approx(expected)


def test_evaluate_model_returns_all_metrics() -> None:
    """The report contains exactly the documented metric keys."""
    report = evaluate_model([1.0, 2.0], [1.0, 2.0])
    assert set(report) == set(METRICS)


def test_evaluate_train_test_reports_gap() -> None:
    """Train/test metrics and the gap between them are all present."""
    y_train = np.array([10.0, 20.0])
    y_pred_train = np.array([10.0, 20.0])
    y_test = np.array([10.0, 20.0])
    y_pred_test = np.array([12.0, 18.0])
    report = evaluate_train_test(y_train, y_pred_train, y_test, y_pred_test)
    for metric in METRICS:
        assert f"train_{metric}" in report
        assert f"test_{metric}" in report
        assert f"gap_{metric}" in report
    # Perfect train predictions -> zero train error -> gap equals test error.
    assert report["gap_MAE"] == pytest.approx(report["test_MAE"])
