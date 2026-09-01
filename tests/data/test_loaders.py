"""Tests for data loading and preprocessing in ``src/data/loaders.py``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.loaders import (
    drop_leakage,
    feature_columns,
    load_raw_data,
    preprocess,
)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """A minimal raw merged train+store frame with all required columns."""
    return pd.DataFrame(
        {
            "Store": [1, 1, 1, 1, 1, 1, 1, 1],
            "DayOfWeek": [1, 2, 3, 4, 5, 6, 7, 1],
            "Date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-05",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-08",
                ]
            ),
            "Open": [1, 1, 1, 1, 1, 1, 1, 1],
            "Promo": [0, 1, 0, 1, 0, 1, 0, 1],
            "StateHoliday": ["0", "0", "0", "0", "0", "0", "0", "0"],
            "SchoolHoliday": [0, 0, 0, 0, 0, 0, 0, 0],
            "StoreType": ["a", "a", "a", "a", "a", "a", "a", "a"],
            "Assortment": ["a", "a", "a", "a", "a", "a", "a", "a"],
            "CompetitionDistance": [
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
                100.0,
            ],
            "CompetitionOpenSinceMonth": [1, 1, 1, 1, 1, 1, 1, 1],
            "CompetitionOpenSinceYear": [
                2015,
                2015,
                2015,
                2015,
                2015,
                2015,
                2015,
                2015,
            ],
            "Promo2": [0, 0, 0, 0, 0, 0, 0, 0],
            "Promo2SinceWeek": [0, 0, 0, 0, 0, 0, 0, 0],
            "Promo2SinceYear": [0, 0, 0, 0, 0, 0, 0, 0],
            "PromoInterval": [
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
            ],
            "Customers": [50, 60, 55, 70, 65, 80, 75, 60],
            "Sales": [100, 120, 110, 140, 130, 160, 150, 120],
        }
    )


def test_load_raw_data_merges_train_and_store(tmp_path) -> None:
    """``load_raw_data`` merges train and store CSVs on Store."""
    train = pd.DataFrame(
        {"Store": [1, 2], "Date": ["2020-01-01", "2020-01-01"], "Sales": [100, 200]}
    )
    store = pd.DataFrame({"Store": [1, 2], "StoreType": ["a", "b"]})
    train.to_csv(tmp_path / "train.csv", index=False)
    store.to_csv(tmp_path / "store.csv", index=False)

    merged = load_raw_data(tmp_path)
    assert list(merged.columns) == ["Store", "Date", "Sales", "StoreType"]
    assert len(merged) == 2


def test_preprocess_produces_numeric_no_missing(raw_df: pd.DataFrame) -> None:
    """The full pipeline yields numeric features with no missing values."""
    processed = preprocess(raw_df)
    assert not processed.isna().any().any()
    # Every column except the datetime ``Date`` is numeric.
    non_date = processed.drop(columns=["Date"])
    numeric = non_date.select_dtypes(include="number")
    assert len(numeric.columns) == len(non_date.columns)


def test_preprocess_drops_competition_date_columns(raw_df: pd.DataFrame) -> None:
    """Raw competition date columns are replaced by derived features."""
    processed = preprocess(raw_df)
    assert "CompetitionOpenSinceMonth" not in processed.columns
    assert "CompetitionOpenSinceYear" not in processed.columns
    assert "competition_age_months" in processed.columns
    assert "has_competition" in processed.columns


def test_preprocess_adds_time_features(raw_df: pd.DataFrame) -> None:
    """Calendar-derived features are present after preprocessing."""
    processed = preprocess(raw_df)
    for col in ["year", "month", "day", "week_of_year"]:
        assert col in processed.columns


def test_preprocess_adds_payday_features(raw_df: pd.DataFrame) -> None:
    """Day-of-month and payday flags are derived."""
    processed = preprocess(raw_df)
    assert "day_of_month" in processed.columns
    assert "is_payday" in processed.columns
    # Jan 1 and Jan 15 are paydays; Jan 1 is in the fixture.
    assert processed.loc[processed["day_of_month"] == 1, "is_payday"].eq(1).all()


def test_preprocess_adds_store_aggregates(raw_df: pd.DataFrame) -> None:
    """Per-store mean sales features are derived."""
    processed = preprocess(raw_df)
    assert "store_mean_sales" in processed.columns
    assert "store_mean_sales_dow" in processed.columns
    assert processed["store_mean_sales"].nunique() == 1  # single store


def test_preprocess_adds_lag_features(raw_df: pd.DataFrame) -> None:
    """Lag and rolling features exist after preprocessing."""
    processed = preprocess(raw_df)
    for col in ["sales_lag_1", "sales_lag_7", "sales_rolling_7"]:
        assert col in processed.columns


def test_preprocess_one_hot_encodes_categoricals(raw_df: pd.DataFrame) -> None:
    """Categorical columns are one-hot encoded with sanitized names."""
    processed = preprocess(raw_df)
    assert "StoreType" not in processed.columns
    assert "StoreType_a" in processed.columns
    assert "Assortment_a" in processed.columns
    assert "StateHoliday_0" in processed.columns


def test_feature_columns_excludes_target_date_leakage_split(
    raw_df: pd.DataFrame,
) -> None:
    """Feature columns exclude Sales, Date, Customers, and Open."""
    processed = preprocess(raw_df)
    cols = feature_columns(processed)
    assert "Sales" not in cols
    assert "Date" not in cols
    assert "Customers" not in cols
    assert "Open" not in cols


def test_drop_leakage_removes_customers(raw_df: pd.DataFrame) -> None:
    """``drop_leakage`` removes the Customers column."""
    processed = preprocess(raw_df)
    assert "Customers" in processed.columns
    dropped = drop_leakage(processed)
    assert "Customers" not in dropped.columns
