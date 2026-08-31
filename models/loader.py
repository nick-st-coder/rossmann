"""Model loading helpers for the serving app.

Loads the Rossmann model so the FastAPI / Gradio app can serve predictions.
The model URI is read from the environment so the Dockerfile can override it
at deploy time, falling back to the local dev server and ``Rossmann`` v2.

Loading order:
1. ``MLFLOW_MODEL_URI`` if set (e.g. a local path baked into the image).
2. The MLflow registry: ``models:/<name>/<version>`` (default ``Rossmann`` v2).
3. If the registry is unreachable, fall back to the local artifact folder
   (``mlartifacts/1/models/<model_id>/artifacts``), patching the
   ``artifact_path`` in ``MLmodel`` so MLflow can load it without the server.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path

import mlflow
from mlflow.pyfunc import PyFuncModel

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_URI = "http://127.0.0.1:5000"
DEFAULT_MODEL_NAME = "Rossmann"
DEFAULT_MODEL_VERSION = "2"
DEFAULT_MODEL_ID = "m-82c25dbfddd54b05a64f407fe6f54a82"
LOCAL_ARTIFACTS_DIR = Path("mlartifacts/1/models")
REACHABILITY_TIMEOUT_SECONDS = 2.0


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


def _local_artifact_dir() -> Path | None:
    """Return the local artifact folder for the model, if it exists."""
    model_id = os.environ.get("MLFLOW_MODEL_ID", DEFAULT_MODEL_ID)
    candidate = LOCAL_ARTIFACTS_DIR / model_id / "artifacts"
    return candidate if candidate.is_dir() else None


def _load_from_local(artifact_dir: Path) -> PyFuncModel:
    """Load a model from a local artifact folder, patching ``artifact_path``.

    The ``MLmodel`` file inside the artifact folder records the original
    ``artifact_path`` as an MLflow server URI (``mlflow-artifacts:/...``),
    which MLflow cannot resolve without the server. Copy the folder to a
    temp dir and rewrite that field to a relative value so the model loads
    purely from disk.
    """
    tmp = Path(tempfile.mkdtemp(prefix="rossmann-model-"))
    shutil.copytree(artifact_dir, tmp, dirs_exist_ok=True)
    mlmodel = tmp / "MLmodel"
    text = mlmodel.read_text()
    text = text.replace("artifact_path: mlflow-artifacts:/", "artifact_path: model")
    mlmodel.write_text(text)
    logger.info("Loading model from local artifacts: %s", artifact_dir)
    return mlflow.pyfunc.load_model(str(tmp))


def _tracking_server_reachable() -> bool:
    """Return True if the MLflow tracking server responds quickly.

    A short timeout avoids hanging on urllib3's retry/backoff when the
    server is down, which would otherwise delay the local fallback.
    """
    uri = get_tracking_uri()
    if not uri.startswith("http"):
        return True  # file/sqlite tracking URIs don't need a server
    try:
        with urllib.request.urlopen(uri, timeout=REACHABILITY_TIMEOUT_SECONDS):
            return True
    except (OSError, urllib.error.URLError):
        return False


def load_model() -> PyFuncModel:
    """Load the Rossmann model.

    Tries the registry first, then falls back to the local artifact folder
    if the MLflow server is unreachable.

    Returns:
        The loaded pyfunc model wrapper, callable via ``predict``.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    model_uri = get_model_uri()
    if not _tracking_server_reachable():
        logger.warning(
            "MLflow tracking server %s unreachable; using local artifacts.",
            get_tracking_uri(),
        )
        return _load_from_local_or_raise(model_uri)
    try:
        logger.info("Loading model from %s (tracking URI: %s)", model_uri, get_tracking_uri())
        return mlflow.pyfunc.load_model(model_uri)
    except (OSError, mlflow.MlflowException) as exc:
        logger.warning("Registry load failed (%s); trying local artifacts.", exc)
        return _load_from_local_or_raise(model_uri)


def _load_from_local_or_raise(model_uri: str) -> PyFuncModel:
    """Load from local artifacts, raising a clear error if unavailable."""
    artifact_dir = _local_artifact_dir()
    if artifact_dir is None:
        raise RuntimeError(
            f"Could not load model from registry ({model_uri}) and no local "
            f"artifacts found under {LOCAL_ARTIFACTS_DIR}."
        )
    return _load_from_local(artifact_dir)
