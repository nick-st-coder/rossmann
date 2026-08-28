"""Data loading and preprocessing for the Rossmann forecasting pipeline.

Loads the raw `train.csv` / `store.csv` files, merges them, and applies the
cleaning + feature-engineering steps from `notebooks/Cleaning.ipynb` so the
same transforms are reusable in training and serving.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns that are one-hot encoded from categorical features.
CATEGORICAL_COLS = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]

# Target and date columns are never model features.
TARGET_COL = "Sales"
DATE_COL = "Date"

# Columns excluded from the feature set:
# - ``Customers`` is a same-day count that is nearly identical to ``Sales``
#   (target leakage) and is not known at prediction time.
# - ``Open`` is handled by the two-stage split: ``Open == 0`` implies
#   ``Sales == 0`` deterministically, so the regressor only sees open rows.
LEAKAGE_COLS = ["Customers", "Open"]

# Columns dropped after deriving competition-age features.
_COMPETITION_DATE_COLS = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]

# Lag/rolling features that require a full history per store before they exist.
_LAG_COLS = ["sales_lag_1", "sales_lag_7", "sales_rolling_7"]


def load_raw_data(data_dir: str | Path) -> pd.DataFrame:
    """Load and merge the raw train and store CSVs.

    Args:
        data_dir: Directory containing ``train.csv`` and ``store.csv``.

    Returns:
        Merged DataFrame (left join on ``Store``).
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv", dtype={"StateHoliday": str})
    store = pd.read_csv(data_dir / "store.csv")
    return pd.merge(train, store, how="left", on="Store")


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values per the cleaning notebook's decisions."""
    df = df.copy()
    # Promo2 features: 0 means the store never ran the second promo.
    df["Promo2SinceWeek"] = df["Promo2SinceWeek"].fillna(0)
    df["Promo2SinceYear"] = df["Promo2SinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    # Competition distance is right-skewed with outliers -> median fill plus a
    # missingness indicator so the model can learn the "no competitor" case.
    df["CompetitionDistance_missing"] = df["CompetitionDistance"].isna().astype(int)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    return df


def _add_competition_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive competition age and presence, then drop the raw date columns."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    competition_date = pd.to_datetime(
        df["CompetitionOpenSinceYear"].astype("Int64").astype(str)
        + "-"
        + df["CompetitionOpenSinceMonth"].astype("Int64").astype(str)
        + "-01",
        errors="coerce",
    )

    df["competition_age_months"] = (
        (df["Date"].dt.year - competition_date.dt.year) * 12
        + (df["Date"].dt.month - competition_date.dt.month)
    ).fillna(0)
    df["has_competition"] = competition_date.notna().astype(int)

    return df.drop(columns=_COMPETITION_DATE_COLS)


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived features from the ``Date`` column."""
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    df["is_Saturday"] = (df["DayOfWeek"] == 6).astype(int)
    df["is_Sunday"] = (df["DayOfWeek"] == 7).astype(int)
    return df


def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-store lag and rolling sales features, dropping rows without them."""
    df = df.copy().sort_values(["Store", "Date"])
    df["sales_lag_1"] = df.groupby("Store")["Sales"].shift(1)
    df["sales_lag_7"] = df.groupby("Store")["Sales"].shift(7)
    df["sales_rolling_7"] = df.groupby("Store")["Sales"].transform(
        lambda x: x.shift(1).rolling(7).mean()
    )
    return df.dropna(subset=_LAG_COLS)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns and sanitize column names."""
    df = df.copy()
    # StateHoliday mixes int 0 and string "0" -> normalize to string first.
    df["StateHoliday"] = df["StateHoliday"].astype(str)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, dtype=int)
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[^A-Za-z0-9_]+", "_", regex=True)
        .str.strip("_")
    )
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning + feature-engineering pipeline.

    Args:
        df: Raw merged DataFrame (see :func:`load_raw_data`).

    Returns:
        Model-ready DataFrame with numeric features only, no missing values.
    """
    df = _fill_missing(df)
    df = _add_competition_features(df)
    df = _add_time_features(df)
    df = _add_lag_features(df)
    return _encode_categoricals(df)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature columns for a processed DataFrame.

    Excludes the target, date, and leakage columns (``Customers``, ``Open``)
    so the same feature set is used for training and serving.

    Args:
        df: Processed DataFrame.

    Returns:
        Ordered list of feature column names.
    """
    return [
        col
        for col in df.columns
        if col not in {TARGET_COL, DATE_COL, *LEAKAGE_COLS}
    ]


def load_processed(data_dir: str | Path) -> pd.DataFrame:
    """Load a preprocessed dataset from ``data/processed``.

    Args:
        data_dir: Directory containing the processed CSV.

    Returns:
        Processed DataFrame with ``Date`` parsed as datetime.
    """
    df = pd.read_csv(Path(data_dir) / "rossmannV2.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df
