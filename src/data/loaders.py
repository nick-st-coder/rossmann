"""Data loading and preprocessing for the Rossmann forecasting pipeline.

Loads the raw `train.csv` / `store.csv` files, merges them, and applies the
cleaning + feature-engineering steps from `notebooks/Cleaning.ipynb` so the
same transforms are reusable in training and serving.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Columns that are one-hot encoded from categorical features.
CATEGORICAL_COLS = ["StateHoliday", "StoreType", "Assortment", "PromoInterval"]

# Target and date columns are never model features.
TARGET_COL = "Sales"
DATE_COL = "Date"

# Columns excluded from the feature set:
# - ``Customers`` is a same-day count that is nearly identical to ``Sales``
#   (target leakage) and is not known at prediction time. It is dropped in
#   cleaning so the dataset is final before modeling.
# - ``Open`` is not a regressor feature: ``Open == 0`` implies ``Sales == 0``
#   deterministically, so the two-stage split uses it to predict closed rows
#   as zero. It stays in the dataset as a split column only.
LEAKAGE_COLS = ["Customers"]
SPLIT_COLS = ["Open"]

# Columns dropped after deriving competition-age features.
_COMPETITION_DATE_COLS = ["CompetitionOpenSinceMonth", "CompetitionOpenSinceYear"]

# Lag/rolling features that require a full history per store before they exist.
_LAG_COLS = ["sales_lag_1", "sales_lag_7", "sales_rolling_7"]

# Store-level aggregate features derived from historical sales.
_STORE_AGG_COLS = ["store_mean_sales", "store_mean_sales_dow"]

# Payday features derived from the day of month.
_PAYDAY_COLS = ["day_of_month", "is_payday"]

# Holiday-proximity features derived from the state-holiday calendar.
_HOLIDAY_PROX_COLS = ["days_to_holiday", "days_since_holiday"]


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
    # Add calendar-derived features from the ``Date`` column.
    df = df.copy()
    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
    return df


def _add_promo2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive promo2-active and promo2-age features.

    ``Promo2`` is a continuous promo that runs in specific months each year
    (``PromoInterval``). The raw ``Promo2SinceWeek/Year`` columns only say
    when the promo started, not whether it is running on a given date, so
    the model cannot tell active promo2 periods apart. This adds:

    - ``promo2_active``: 1 if the store runs promo2 and the date falls in an
      active interval month, else 0.
    - ``promo2_age_months``: months since the promo2 start (0 if never ran).
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # The three intervals are exactly the months where month % 3 is constant:
    # Jan,Apr,Jul,Oct -> 1, Feb,May,Aug,Nov -> 2, Mar,Jun,Sept,Dec -> 0.
    interval_mod = {
        "Jan,Apr,Jul,Oct": 1,
        "Feb,May,Aug,Nov": 2,
        "Mar,Jun,Sept,Dec": 0,
    }
    interval = df["PromoInterval"].map(interval_mod)

    # Promo2 is active when the store participates and the current month is
    # in its interval. Stores without promo2 (interval "None") are inactive.
    df["promo2_active"] = (
        (df["Promo2"] == 1) & (df["Date"].dt.month % 3 == interval)
    ).astype(int)

    # Months since the promo2 start; 0 for stores that never ran it.
    promo2_start = pd.to_datetime(
        df["Promo2SinceYear"].astype("Int64").astype(str)
        + "-"
        + df["Promo2SinceWeek"].astype("Int64").astype(str)
        + "-1",
        format="%Y-%W-%w",
        errors="coerce",
    )
    df["promo2_age_months"] = (
        (
            (df["Date"].dt.year - promo2_start.dt.year) * 12
            + (df["Date"].dt.month - promo2_start.dt.month)
        )
        .fillna(0)
        .astype(int)
    )

    return df


def _add_store_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-store historical sales aggregates.

    Store size is the dominant driver of sales, but the model only sees the
    store id and static store attributes. These aggregates give it a direct
    measure of each store's typical sales level:

    - ``store_mean_sales``: mean sales per store over the full history.
    - ``store_mean_sales_dow``: mean sales per store per day-of-week.
    """
    df = df.copy()
    df["store_mean_sales"] = df.groupby("Store")["Sales"].transform("mean")
    df["store_mean_sales_dow"] = df.groupby(["Store", "DayOfWeek"])["Sales"].transform(
        "mean"
    )
    return df


def _add_payday_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-month and payday features.

    Rossmann sales spike around the 1st and 15th of the month (payday
    effect). ``day_of_month`` lets the model learn the shape; ``is_payday``
    is a direct flag for the two known spike days.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["day_of_month"] = df["Date"].dt.day
    df["is_payday"] = df["day_of_month"].isin([1, 15]).astype(int)
    return df


def _add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Add days-to-next / days-since-last state-holiday features.

    Sales dip in the days before a holiday and spike after it. These
    features give the model a continuous measure of holiday proximity
    instead of only the binary ``StateHoliday`` flags.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # StateHoliday is a string at this point ("0", "a", "b", "c").
    holiday_dates = pd.to_datetime(df.loc[df["StateHoliday"] != "0", "Date"].unique())
    holiday_dates = np.sort(holiday_dates)

    # Vectorized nearest-holiday search via searchsorted.
    # ``side="left"`` gives the first holiday at/after each date; dates past
    # the last holiday clamp to it, so cap the "days to" at 0 there.
    date_idx = np.searchsorted(holiday_dates, df["Date"].to_numpy(), side="left")
    date_idx = np.clip(date_idx, 0, len(holiday_dates) - 1)

    next_holiday = holiday_dates[date_idx]
    prev_holiday = holiday_dates[np.maximum(date_idx - 1, 0)]

    df["days_to_holiday"] = (next_holiday - df["Date"]).dt.days.clip(lower=0)
    df["days_since_holiday"] = (df["Date"] - prev_holiday).dt.days
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
    df = _add_promo2_features(df)
    df = _add_store_aggregates(df)
    df = _add_payday_feature(df)
    df = _add_holiday_proximity(df)
    df = _add_lag_features(df)
    return _encode_categoricals(df)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the model feature columns for a processed DataFrame.

    Excludes the target, date, leakage (``Customers``), and split (``Open``)
    columns so the same feature set is used for training and serving.

    Args:
        df: Processed DataFrame.

    Returns:
        Ordered list of feature column names.
    """
    excluded = {TARGET_COL, DATE_COL, *LEAKAGE_COLS, *SPLIT_COLS}
    return [col for col in df.columns if col not in excluded]


def drop_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Drop target-leakage columns from a processed DataFrame.

    ``Customers`` is a same-day count that is nearly identical to ``Sales``
    (log-space correlation ~0.996) and is not known at prediction time, so it
    must not reach the model. This is applied in cleaning so the dataset is
    final before modeling.

    Args:
        df: Processed DataFrame.

    Returns:
        DataFrame without the leakage columns.
    """
    return df.drop(columns=[col for col in LEAKAGE_COLS if col in df.columns])


def load_processed(data_dir: str | Path) -> pd.DataFrame:
    """Load a preprocessed dataset from ``data/processed``.

    Args:
        data_dir: Directory containing the processed CSV.

    Returns:
        Processed DataFrame with ``Date`` parsed as datetime.
    """
    df = pd.read_csv(Path(data_dir) / "rossmann.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df
