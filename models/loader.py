"""Model loading helpers for the serving app.

Loads the Rossmann model so the FastAPI / Gradio app can serve predictions.
The model URI is read from the environment so the Dockerfile can override it
at deploy time, falling back to the local dev server and ``Rossmann`` v2.

Loading order:
1. ``MLFLOW_MODEL_URI`` if set (e.g. a local path baked into the image).
2. The MLflow registry: ``models:/<name>/<version>`` (default ``Rossmann`` v2).
"""

from __future__ import annotations

import logging
import os

import mlflow
from mlflow.pyfunc import PyFuncModel

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_NAME = "Rossmann"
DEFAULT_MODEL_VERSION = "2"


def get_tracking_uri() -> str:
    """Return the MLflow tracking URI, defaulting to the local dev server."""
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def get_model_uri() -> str:
    """Build the model URI from environment variables.

    Returns:
        ``MLFLOW_MODEL_URI`` if set, else ``models:/<name>/<version>``.
    """
    explicit = os.environ.get("MLFLOW_MODEL_URI")
    if explicit:
        return explicit
    name = os.environ.get("MLFLOW_MODEL_NAME", DEFAULT_MODEL_NAME)
    version = os.environ.get("MLFLOW_MODEL_VERSION", DEFAULT_MODEL_VERSION)
    return f"models:/{name}/{version}"


def load_model() -> PyFuncModel:
    """Load the Rossmann model.

    Returns:
        The loaded pyfunc model wrapper, callable via ``predict``.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    model_uri = get_model_uri()
    logger.info("Loading model from %s (tracking URI: %s)", model_uri, get_tracking_uri())
    return mlflow.pyfunc.load_model(model_uri)
