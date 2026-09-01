"""Tests for model training helpers in ``src/models/train.py``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from src.models.train import (
    clip_predictions,
    make_baseline,
    make_model,
    make_ridge_pipeline,
    predict_two_stage,
    train_model,
)


@pytest.fixture
def X() -> pd.DataFrame:
    """Small numeric feature frame."""
    return pd.DataFrame(
        {
            "Open": [1, 1, 0, 1, 0, 1],
            "Promo": [0, 1, 0, 1, 0, 1],
            "Store": [1, 1, 1, 2, 2, 2],
        }
    )


@pytest.fixture
def y() -> pd.Series:
    """Target aligned with ``X`` (closed rows have 0 sales)."""
    return pd.Series([100.0, 120.0, 0.0, 200.0, 0.0, 210.0])


def test_make_baseline_returns_dummy_regressor() -> None:
    """The baseline is a mean-predicting DummyRegressor."""
    model = make_baseline()
    assert isinstance(model, DummyRegressor)
    assert model.strategy == "mean"


def test_make_ridge_pipeline_returns_pipeline() -> None:
    """The ridge baseline is a scaler + ridge pipeline."""
    model = make_ridge_pipeline(alpha=0.5)
    assert isinstance(model, Pipeline)
    assert model.steps[0][0] == "scaler"
    assert model.steps[1][0] == "model"


def test_make_model_ridge_returns_pipeline() -> None:
    """``make_model("ridge")`` returns the ridge pipeline."""
    assert isinstance(make_model("ridge"), Pipeline)


def test_make_model_unsupported_raises() -> None:
    """An unknown model name raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_name"):
        make_model("not-a-model")


def test_clip_predictions_clamps_negatives() -> None:
    """Negative predictions are clamped to 0."""
    preds = np.array([-5.0, 0.0, 3.0])
    assert np.array_equal(clip_predictions(preds), np.array([0.0, 0.0, 3.0]))


def test_predict_two_stage_zeroes_closed_rows(X: pd.DataFrame, y: pd.Series) -> None:
    """Closed rows (Open == 0) are predicted as exactly 0."""
    model = DummyRegressor(strategy="mean")
    model.fit(X[X["Open"] == 1].drop(columns=["Open"]), y[X["Open"] == 1])
    preds = predict_two_stage(model, X)
    assert preds[X["Open"] == 0].tolist() == [0.0, 0.0]
    assert (preds[X["Open"] == 1] > 0).all()


def test_train_model_fits_and_predicts(X: pd.DataFrame, y: pd.Series) -> None:
    """Training a ridge model produces finite predictions."""
    model = train_model(X, y, model_name="ridge")
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert len(preds) == len(X)
