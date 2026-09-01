"""Tests for input validation helpers in ``src/utils/validation.py``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.utils.validation import (
    ensure_numeric,
    require_columns,
    require_no_missing,
    validate_features,
)


@pytest.fixture
def df() -> pd.DataFrame:
    """A small clean feature frame."""
    return pd.DataFrame(
        {
            "Store": [1, 2],
            "DayOfWeek": [1, 2],
            "Open": [1, 0],
            "Promo": [0, 1],
            "StateHoliday": ["0", "a"],
            "SchoolHoliday": [0, 1],
            "StoreType": ["a", "b"],
            "Assortment": ["a", "b"],
            "CompetitionDistance": [100.0, 200.0],
            "Promo2": [0, 1],
            "Promo2SinceWeek": [0, 10],
            "Promo2SinceYear": [0, 2015],
        }
    )


def test_require_columns_passes_when_all_present(df: pd.DataFrame) -> None:
    """No exception when every required column exists."""
    require_columns(df)  # should not raise


def test_require_columns_raises_on_missing(df: pd.DataFrame) -> None:
    """A missing required column raises ValueError naming it."""
    with pytest.raises(ValueError, match="Store"):
        require_columns(df.drop(columns=["Store"]))


def test_require_no_missing_passes_on_clean_frame(df: pd.DataFrame) -> None:
    """No exception when there are no missing values."""
    require_no_missing(df)


def test_require_no_missing_raises_on_nan(df: pd.DataFrame) -> None:
    """A NaN in a checked column raises ValueError naming it."""
    df.loc[0, "Promo"] = pd.NA
    with pytest.raises(ValueError, match="Promo"):
        require_no_missing(df)


def test_validate_features_passes_on_exact_columns(df: pd.DataFrame) -> None:
    """Exact feature list with no extras passes."""
    validate_features(df, list(df.columns))


def test_validate_features_raises_on_empty() -> None:
    """An empty DataFrame is rejected."""
    with pytest.raises(ValueError, match="empty"):
        validate_features(pd.DataFrame(), ["Store"])


def test_validate_features_raises_on_missing_column(df: pd.DataFrame) -> None:
    """A missing expected feature raises ValueError."""
    with pytest.raises(ValueError, match="Missing expected features"):
        validate_features(df, ["Store", "NotAFeature"])


def test_validate_features_raises_on_extra_column(df: pd.DataFrame) -> None:
    """An unexpected extra column raises ValueError."""
    with pytest.raises(ValueError, match="Unexpected columns"):
        validate_features(df, ["Store"])


def test_ensure_numeric_raises_on_non_numeric(df: pd.DataFrame) -> None:
    """A string column raises TypeError."""
    with pytest.raises(TypeError, match="StoreType"):
        ensure_numeric(df, columns=["StoreType"])


def test_ensure_numeric_passes_on_numeric(df: pd.DataFrame) -> None:
    """Numeric columns pass without exception."""
    ensure_numeric(df, columns=["Store", "Promo"])  # should not raise
