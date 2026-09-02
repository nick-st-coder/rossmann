"""Smoke-test the served MLflow model in CI.

Loads the model the same way the serving app does — registry first, local
artifact fallback — and runs a small, realistic prediction to confirm the
model loads and the prediction shape is correct. Fails fast with a clear
error otherwise.

Usage (from CI):
    uv run python scripts/validate_model.py

Environment:
    MLFLOW_TRACKING_URI  optional; if set and reachable, the registered
                         model is validated. If unset or unreachable, the
                         local artifact copy (what ships in the Docker
                         image) is validated instead.
    MLFLOW_MODEL_URI     optional; defaults to ``models:/Rossmann/2``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.loader import get_model_uri, get_tracking_uri, load_model

# A realistic single-row payload matching the model's 41-feature signature.
SAMPLE_PAYLOAD = {
    "Store": 1097,
    "DayOfWeek": 4,
    "Promo": 0,
    "SchoolHoliday": 1,
    "CompetitionDistance": 720.0,
    "Promo2": 0,
    "Promo2SinceWeek": 0.0,
    "Promo2SinceYear": 0.0,
    "CompetitionDistance_missing": 0,
    "competition_age_months": 154.0,
    "has_competition": 1,
    "year": 2015,
    "month": 1,
    "day": 1,
    "week_of_year": 1,
    "promo2_active": 0,
    "promo2_age_months": 0,
    "store_mean_sales": 5000.0,
    "store_mean_sales_dow": 5200.0,
    "day_of_month": 1,
    "is_payday": 1,
    "days_to_holiday": 3,
    "days_since_holiday": 20,
    "sales_lag_1": 4800.0,
    "sales_lag_7": 5100.0,
    "sales_rolling_7": 5050.0,
    "StateHoliday_0": 0,
    "StateHoliday_a": 1,
    "StateHoliday_b": 0,
    "StateHoliday_c": 0,
    "StoreType_a": 1,
    "StoreType_b": 0,
    "StoreType_c": 0,
    "StoreType_d": 0,
    "Assortment_a": 1,
    "Assortment_b": 0,
    "Assortment_c": 0,
    "PromoInterval_Feb_May_Aug_Nov": 0,
    "PromoInterval_Jan_Apr_Jul_Oct": 0,
    "PromoInterval_Mar_Jun_Sept_Dec": 0,
    "PromoInterval_None": 1,
}


def main() -> None:
    """Load the model and run a smoke prediction, failing fast on error."""
    tracking_uri = get_tracking_uri()
    model_uri = get_model_uri()
    print(f"Loading model from {model_uri} (tracking URI: {tracking_uri})")
    # Use the same loader as the serving app: registry first, local artifact
    # fallback when the tracking server is unreachable (e.g. in CI).
    model = load_model()

    df = pd.DataFrame([SAMPLE_PAYLOAD])
    pred = model.predict(df)

    # XGBoost returns a 1-D array for a single-row prediction; accept both
    # (1,) and (1, 1) so the check is robust to the flavor's output shape.
    if pred.shape not in {(1,), (1, 1)}:
        raise RuntimeError(
            f"Unexpected prediction shape {pred.shape}; expected (1,) or (1, 1)."
        )
    value = float(pred.ravel()[0])
    if value < 0:
        raise RuntimeError(f"Negative prediction {value}; expected non-negative sales.")
    print(f"Smoke prediction OK: sales = {value:.2f}")


if __name__ == "__main__":
    main()
