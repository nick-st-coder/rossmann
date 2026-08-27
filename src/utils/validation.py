"""Data validation helpers shared by training and serving code.

These checks keep the model's input contract explicit so the FastAPI/Gradio
app can validate incoming requests and training can fail fast on bad data.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "Store",
    "DayOfWeek",
    "Open",
    "Promo",
    "StateHoliday",
    "SchoolHoliday",
    "StoreType",
    "Assortment",
    "CompetitionDistance",
    "Promo2",
    "Promo2SinceWeek",
    "Promo2SinceYear",
)


def require_no_missing(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Raise if the DataFrame has missing values (optionally in given columns)."""
    cols = list(columns) if columns is not None else list(df.columns)
    missing = df[cols].isna().any()
    bad = missing[missing].index.tolist()
    if bad:
        raise ValueError(f"Missing values found in columns: {bad}")


def require_columns(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Raise if any required column is absent from the DataFrame."""
    cols = list(columns) if columns is not None else list(REQUIRED_COLUMNS)
    absent = [c for c in cols if c not in df.columns]
    if absent:
        raise ValueError(f"Missing required columns: {absent}")


def validate_features(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Validate a feature frame against an explicit expected feature list.

    Args:
        df: The input DataFrame.
        feature_columns: The exact ordered feature names the model expects.

    Raises:
        ValueError: If the DataFrame isn't non-empty, lacks expected
        columns, has extra unknown columns, or contains missing values.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    absent = [c for c in feature_columns if c not in df.columns]
    if absent:
        raise ValueError(f"Missing expected features: {absent}")

    extra = [c for c in df.columns if c not in feature_columns]
    if extra:
        raise ValueError(f"Unexpected columns present: {extra}")

    require_no_missing(df, columns=feature_columns)


def ensure_numeric(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    """Raise if any (given) column is not numeric."""
    cols = list(columns) if columns is not None else list(df.columns)
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise TypeError(f"Expected numeric column, got {col!r}: {df[col].dtype}")
