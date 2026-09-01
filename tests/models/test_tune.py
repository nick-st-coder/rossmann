"""Tests for Optuna tuning helpers in ``src/models/tune.py``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.tune import time_slice_sample, tune


@pytest.fixture
def df() -> pd.DataFrame:
    """Small dated frame for sampling tests."""
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"]
            ),
            "Sales": [100.0, 110.0, 120.0, 130.0, 140.0],
            "Store": [1, 1, 1, 1, 1],
        }
    )


def test_time_slice_sample_by_end_date(df: pd.DataFrame) -> None:
    """``end_date`` keeps only rows before the cutoff."""
    sampled = time_slice_sample(df, end_date="2020-01-03")
    assert len(sampled) == 2
    assert (sampled["Date"] < pd.Timestamp("2020-01-03")).all()


def test_time_slice_sample_by_frac(df: pd.DataFrame) -> None:
    """``frac`` keeps a reproducible random fraction of rows."""
    sampled = time_slice_sample(df, frac=0.6)
    assert 0 < len(sampled) < len(df)


def test_time_slice_sample_requires_one_strategy(df: pd.DataFrame) -> None:
    """Calling with neither end_date nor frac raises ValueError."""
    with pytest.raises(ValueError, match="end_date or frac"):
        time_slice_sample(df)


def test_tune_returns_study_with_best_params() -> None:
    """A tiny tuning run completes and exposes best params."""
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [2.0, 4.0, 6.0, 8.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    study = tune(X, y, model_name="xgboost", n_trials=2, n_splits=2, gap=0)
    assert study.best_params
    assert study.best_value >= 0.0


def test_tune_unsupported_model_raises() -> None:
    """An unknown model family raises ValueError."""
    X = pd.DataFrame({"a": [1.0, 2.0]})
    y = pd.Series([1.0, 2.0])
    with pytest.raises(ValueError, match="No search space"):
        tune(X, y, model_name="not-a-model", n_trials=1)
