"""Chronological train/test splitting for the Rossmann forecasting task.

Data is time-ordered, so splits are always chronological — never shuffled.
"""

from __future__ import annotations

import pandas as pd

TARGET_COL = "Sales"
DATE_COL = "Date"


def split_by_date(
    df: pd.DataFrame,
    cutoff: str,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a processed DataFrame into train/test on a date cutoff.

    Args:
        df: Processed DataFrame containing ``date_col`` and ``target_col``.
        cutoff: ISO date string; rows before it form the train set.
        target_col: Name of the target column.
        date_col: Name of the date column.

    Returns:
        ``(train, test)`` DataFrames, each sorted by date.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    train = df[df[date_col] < cutoff].copy()
    test = df[df[date_col] >= cutoff].copy()
    return train, test


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    date_col: str = DATE_COL,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features from the target, dropping the date column.

    Args:
        df: Processed DataFrame.
        target_col: Name of the target column.
        date_col: Name of the date column to drop (not a model feature).

    Returns:
        ``(X, y)`` where ``X`` excludes ``target_col`` and ``date_col``.
    """
    X = df.drop(columns=[target_col, date_col])
    y = df[target_col]
    return X, y
