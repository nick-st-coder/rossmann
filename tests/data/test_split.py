"""Tests for chronological splitting helpers in ``src/data/split.py``."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.split import split_by_date, split_features_target, split_open_closed


@pytest.fixture
def df() -> pd.DataFrame:
    """Small time-ordered DataFrame with open and closed rows."""
    return pd.DataFrame(
        {
            "Store": [1, 1, 1, 2, 2, 2],
            "Date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                ]
            ),
            "Open": [1, 0, 1, 1, 1, 0],
            "Sales": [100, 0, 120, 200, 210, 0],
            "Promo": [0, 0, 1, 1, 0, 0],
        }
    )


def test_split_by_date_partitions_on_cutoff(df: pd.DataFrame) -> None:
    """Rows before the cutoff go to train, at/after go to test."""
    train, test = split_by_date(df, cutoff="2020-01-02")
    assert set(train["Date"]) == {pd.Timestamp("2020-01-01")}
    assert set(test["Date"]) == {pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-03")}


def test_split_by_date_sorts_output(df: pd.DataFrame) -> None:
    """Both splits are sorted by date regardless of input order."""
    shuffled = df.sample(frac=1, random_state=0)
    train, test = split_by_date(shuffled, cutoff="2020-01-02")
    assert train["Date"].is_monotonic_increasing
    assert test["Date"].is_monotonic_increasing


def test_split_by_date_empty_test_when_cutoff_after_all(df: pd.DataFrame) -> None:
    """A cutoff past the last date yields an empty test set."""
    train, test = split_by_date(df, cutoff="2021-01-01")
    assert len(test) == 0
    assert len(train) == len(df)


def test_split_features_target_drops_date_and_target(df: pd.DataFrame) -> None:
    """X excludes the target and date columns; y is the target."""
    X, y = split_features_target(df)
    assert "Sales" not in X.columns
    assert "Date" not in X.columns
    assert list(y) == list(df["Sales"])


def test_split_open_closed_partitions_on_open_flag(df: pd.DataFrame) -> None:
    """Open rows and closed rows are separated by the Open flag."""
    open_df, closed_df = split_open_closed(df)
    assert (open_df["Open"] == 1).all()
    assert (closed_df["Open"] == 0).all()
    assert len(open_df) + len(closed_df) == len(df)
