from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from mlflow.types import DataType

from models.loader import load_model
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
    """Load the model once at startup. Fails fast on error."""
    global _MODEL, FEATURE_COLUMNS, _INT_COLUMNS, _FLOAT_COLUMNS
    _MODEL = load_model()
    names = _MODEL.metadata.signature.inputs.input_names()
    types = _MODEL.metadata.signature.inputs.input_types()
    FEATURE_COLUMNS = list(names)
    _INT_COLUMNS = {n for n, t in zip(names, types, strict=True) if t == DataType.long}
    _FLOAT_COLUMNS = {n for n, t in zip(names, types, strict=True) if t == DataType.double}
    logger.info("Model loaded: %s (%d features)", _MODEL.metadata.run_id, len(FEATURE_COLUMNS))


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
