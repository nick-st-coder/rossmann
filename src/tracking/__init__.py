"""MLflow tracking helpers.

Naming, tagging, and logging conventions for reproducible training runs.
Follow `mlflow.instructions.md` — never hardcode a tracking URI, and always
log params, metrics, artifacts, and the model itself.
"""

from __future__ import annotations

import logging
import os
from datetime import UTC, datetime
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI from the environment.

    Raises:
        RuntimeError: If ``MLFLOW_TRACKING_URI`` is not set.
    """
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError(
            "MLFLOW_TRACKING_URI is not set. Set it before logging a run."
        )
    return uri


def log_train_test_metrics(
    report: dict[str, float],
    run_id: str | None = None,
) -> None:
    """Log train/test metrics and their gaps to the active MLflow run.

    Args:
        report: Output of
            :func:`src.evaluation.metrics_report.evaluate_train_test`.
        run_id: Optional explicit run id; defaults to the active run.
    """
    mlflow.log_metrics(report, run_id=run_id)
    logger.info("Logged %d train/test metrics to MLflow.", len(report))


def start_run(
    model_type: str,
    stage: str = "dev",
    git_sha: str | None = None,
    dataset_version: str | None = None,
    **params: Any,
) -> mlflow.ActiveRun:
    """Start a named, tagged MLflow run following repo conventions.

    Args:
        model_type: Model family, e.g. ``"lightgbm"``.
        stage: Promotion stage: ``dev`` | ``staging`` | ``prod``.
        git_sha: Short commit SHA the run executes from.
        dataset_version: Identifier/hash of the dataset or feature set.
        **params: Hyperparameters to log on the run.

    Returns:
        The active MLflow run.
    """
    mlflow.set_tracking_uri(get_tracking_uri())

    run_name = f"{model_type}_{datetime.now(UTC):%Y%m%d_%H%M%S}"
    run = mlflow.start_run(run_name=run_name)

    mlflow.set_tags(
        {
            "stage": stage,
            "git_sha": git_sha or "dirty",
            "dataset_version": dataset_version or "unknown",
        }
    )
    if params:
        mlflow.log_params(params)

    return run
