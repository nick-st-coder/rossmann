"""Hyperparameter tuning with Optuna for LightGBM / XGBoost regressors.

Tuning uses chronological time-series cross-validation so validation folds
never leak future information into training.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import optuna
import pandas as pd

from src.models import train


def _lgbm_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample LightGBM hyperparameters from the search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


def _xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    """Sample XGBoost hyperparameters from the search space."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    }


_PARAM_FN: dict[str, Callable[[optuna.Trial], dict[str, Any]]] = {
    "lightgbm": _lgbm_params,
    "xgboost": _xgb_params,
}


def tune(
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "lightgbm",
    n_trials: int = 30,
    n_splits: int = 5,
    gap: int = 7,
    random_state: int = 42,
) -> optuna.Study:
    """Run an Optuna study for the given model family.

    Args:
        X: Feature matrix.
        y: Target vector.
        model_name: ``"lightgbm"`` or ``"xgboost"``.
        n_trials: Number of Optuna trials.
        n_splits / gap: Time-series CV configuration.
        random_state: Seed for reproducible sampling.

    Raises:
        ValueError: If ``model_name`` has no search-space function.

    Returns:
        A completed Optuna study; ``study.best_params`` holds the winners.
    """
    if model_name not in _PARAM_FN:
        raise ValueError(f"No search space for model_name: {model_name!r}")

    param_fn = _PARAM_FN[model_name]

    def _objective(trial: optuna.Trial) -> float:
        model = train.make_model(model_name, **param_fn(trial))
        scores = train.cross_validate(model, X, y, n_splits=n_splits, gap=gap)
        return float(scores["RMSLE"].mean())

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(_objective, n_trials=n_trials)
    return study
