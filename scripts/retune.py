"""Re-tune XGBoost on a time-slice sample of all stores, then re-validate on full data.

Usage:
    uv run python scripts/retune.py [--n-trials 40] [--end-date 2014-06-01]

The tuning sample keeps all 1115 stores (store heterogeneity) but only rows
before ``end_date`` (fewer rows -> faster trials). The winner is then
re-scored on the full training set with the same 5-fold date CV used for the
baseline comparison, so the tuning score and the validation score are
comparable.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import mlflow
import optuna
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import feature_columns, load_processed, split_open_closed
from src.models import train
from src.models.tune import time_slice_sample, tune
from src.tracking import start_run

DATA_DIR = Path("data/processed")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--end-date", default="2014-06-01", help="Tuning slice cutoff.")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--gap-days", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")

    df = load_processed(DATA_DIR)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    cutoff = "2015-01-01"
    train_df = df[df["Date"] < cutoff].copy()

    train_open, _ = split_open_closed(train_df)

    features = feature_columns(df)
    X_full = train_open[features]
    y_full = train_open["Sales"]

    # 1) Tuning slice: all stores, rows before end_date.
    sample = time_slice_sample(train_open, end_date=args.end_date)
    X_sample = sample[features]
    y_sample = sample["Sales"]
    print(
        f"Tuning slice: {len(sample):,} rows ({len(sample) / len(train_open):.0%} of train)"
    )

    # 2) Optuna study with pruning, logged to MLflow as nested runs.
    with start_run(
        model_type="xgboost",
        stage="dev",
        dataset_version="rossmann-v2",
        n_trials=args.n_trials,
        end_date=args.end_date,
        seed=args.seed,
    ):
        study = tune(
            X_sample,
            y_sample,
            model_name="xgboost",
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            gap=args.gap_days,
            random_state=args.seed,
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            log_to_mlflow=True,
        )

        best_params = study.best_params
        best_params.update({"tree_method": "hist", "random_state": 45, "n_jobs": -1})
        mlflow.log_params(best_params)
        mlflow.log_metric("tuning_best_RMSLE", study.best_value)

        # 3) Re-validate the winner on FULL data with the same CV scheme.
        model = train.make_model("xgboost", **best_params)
        full_scores = train.cross_validate_by_date(
            model,
            X_full,
            y_full,
            train_open["Date"],
            n_splits=args.n_splits,
            gap_days=args.gap_days,
        )
        full_rmsle = float(full_scores["RMSLE"].mean())
        mlflow.log_metric("full_cv_RMSLE", full_rmsle)

        # 4) Compare against the current default params on the same CV.
        defaults = {
            "n_estimators": 300,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 44,
        }
        default_model = train.make_model("xgboost", **defaults)
        default_scores = train.cross_validate_by_date(
            default_model,
            X_full,
            y_full,
            train_open["Date"],
            n_splits=args.n_splits,
            gap_days=args.gap_days,
        )
        default_rmsle = float(default_scores["RMSLE"].mean())
        mlflow.log_metric("default_cv_RMSLE", default_rmsle)
        mlflow.log_metric("delta_vs_default", default_rmsle - full_rmsle)

        print(f"\nTuning best (sample): {study.best_value:.5f} RMSLE")
        print(f"Full-data CV, tuned:   {full_rmsle:.5f} RMSLE")
        print(f"Full-data CV, defaults: {default_rmsle:.5f} RMSLE")
        print(f"Delta (default - tuned): {default_rmsle - full_rmsle:+.5f}")
        print(f"\nBest params:\n{best_params}")


if __name__ == "__main__":
    main()
