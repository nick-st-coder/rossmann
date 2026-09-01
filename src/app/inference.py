from __future__ import annotations

import logging
from datetime import date
from typing import Any

import pandas as pd
from mlflow.types import DataType

from models.loader import load_model
from src.app import features
from src.utils.validation import validate_features

logger = logging.getLogger(__name__)


def _model() -> Any:
    """Return the loaded model, raising if it is not yet available."""
    if _MODEL is None:
        raise RuntimeError("Model not loaded. Call load() at startup first.")
    return _MODEL


_MODEL: Any | None = None
FEATURE_COLUMNS: list[str] = []
_INT_COLUMNS: set[str] = set()
_FLOAT_COLUMNS: set[str] = set()


def load() -> None:
    """Load the model and history once at startup. Fails fast on error."""
    global _MODEL, FEATURE_COLUMNS, _INT_COLUMNS, _FLOAT_COLUMNS
    _MODEL = load_model()
    names = _MODEL.metadata.signature.inputs.input_names()
    types = _MODEL.metadata.signature.inputs.input_types()
    FEATURE_COLUMNS = list(names)
    features.FEATURE_COLUMNS = list(names)
    _INT_COLUMNS = {n for n, t in zip(names, types, strict=True) if t == DataType.long}
    _FLOAT_COLUMNS = {
        n for n, t in zip(names, types, strict=True) if t == DataType.double
    }
    features.load_history()
    logger.info(
        "Model loaded: %s (%d features)", _MODEL.metadata.run_id, len(FEATURE_COLUMNS)
    )


def _coerce_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to the dtypes the model signature expects.

    The model was logged with ``long`` (int) and ``double`` (float) inputs.
    Gradio may send whole numbers as ints (e.g. ``CompetitionDistance`` as
    int64), which MLflow's schema enforcement rejects. Casting to the
    signature dtypes avoids that mismatch.
    """
    df = df.copy()
    for col in _INT_COLUMNS & set(df.columns):
        df[col] = df[col].astype("int64")
    for col in _FLOAT_COLUMNS & set(df.columns):
        df[col] = df[col].astype("float64")
    return df


def predict(features: dict[str, Any]) -> dict[str, float]:
    """Predict sales for a single feature dict.

    Args:
        features: Raw feature values keyed by feature name.

    Returns:
        ``{"sales": <prediction>}``.

    Raises:
        ValueError: If the feature dict is missing/extra columns or has
            missing values.
    """
    df = pd.DataFrame([features])
    df = _coerce_dtypes(df)
    validate_features(df, FEATURE_COLUMNS)
    pred = _model().predict(df)
    return {"sales": float(pred[0])}


def predict_raw(
    store: int,
    pred_date: date,
    store_type: str,
    assortment: str,
    state_holiday: str,
    school_holiday: bool,
    promo: bool,
    competition_distance: float | None,
    competition_open_since_month: int | None,
    competition_open_since_year: int | None,
    promo2: bool,
    promo2_since_week: int | None,
    promo2_since_year: int | None,
    promo_interval: str,
) -> dict[str, float]:
    """Predict sales from raw, human-friendly inputs.

    Builds the full feature vector from the raw inputs (computing store
    aggregates, lags, and one-hot encodings from history), then predicts.

    Returns:
        ``{"sales": <prediction>}``.
    """
    feats = features.build_features(
        store=store,
        pred_date=pred_date,
        store_type=store_type,
        assortment=assortment,
        state_holiday=state_holiday,
        school_holiday=school_holiday,
        promo=promo,
        competition_distance=competition_distance,
        competition_open_since_month=competition_open_since_month,
        competition_open_since_year=competition_open_since_year,
        promo2=promo2,
        promo2_since_week=promo2_since_week,
        promo2_since_year=promo2_since_year,
        promo_interval=promo_interval,
    )
    return predict(feats)
