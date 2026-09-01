"""Build compact serving artifacts from the raw Rossmann data.

The FastAPI/Gradio app needs store aggregates, per-store sales history, and
holiday dates to compute lag/rolling features for the raw-input prediction
path. Instead of shipping the ~36MB ``train.csv`` / ``store.csv`` in the
Docker image, this script precomputes the small derived artifacts the app
actually needs and writes them to ``artifacts/serving/``.

Usage:
    uv run python scripts/build_serving_artifacts.py

Outputs (all small, committed to the repo so the Docker build needs no data):
    artifacts/serving/history.parquet   per-store Date/Sales series
    artifacts/serving/store_means.parquet  store_mean_sales + store_mean_sales_dow
    artifacts/serving/holiday_dates.parquet  state-holiday dates
    artifacts/serving/meta.json         comp_dist_median, global_mean
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW_TRAIN_PATH = Path("data/raw/train.csv")
RAW_STORE_PATH = Path("data/raw/store.csv")
OUT_DIR = Path("artifacts/serving")


def main() -> None:
    """Generate the serving artifacts from the raw CSVs."""
    train = pd.read_csv(
        RAW_TRAIN_PATH,
        usecols=["Store", "Date", "Sales", "DayOfWeek", "StateHoliday"],
        dtype={"StateHoliday": str},
    )
    train["Date"] = pd.to_datetime(train["Date"])
    train = train.sort_values(["Store", "Date"]).reset_index(drop=True)

    store = pd.read_csv(RAW_STORE_PATH, usecols=["CompetitionDistance"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-store Date/Sales series for lag and rolling features.
    train[["Store", "Date", "Sales"]].to_parquet(
        OUT_DIR / "history.parquet", index=False
    )

    # Store-level aggregates.
    store_means = train.groupby("Store")["Sales"].mean().rename("store_mean_sales")
    store_means_dow = (
        train.groupby(["Store", "DayOfWeek"])["Sales"]
        .mean()
        .rename("store_mean_sales_dow")
    )
    store_means.to_frame().to_parquet(OUT_DIR / "store_means.parquet")
    store_means_dow.to_frame().to_parquet(OUT_DIR / "store_means_dow.parquet")

    # State-holiday dates for proximity features.
    holiday_dates = pd.DataFrame(
        {
            "Date": np.sort(
                pd.to_datetime(train.loc[train["StateHoliday"] != "0", "Date"].unique())
            )
        }
    )
    holiday_dates.to_parquet(OUT_DIR / "holiday_dates.parquet", index=False)

    # Scalars.
    meta = {
        "comp_dist_median": float(store["CompetitionDistance"].median()),
        "global_mean": float(train["Sales"].mean()),
    }
    (OUT_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote serving artifacts to {OUT_DIR}")


if __name__ == "__main__":
    main()
