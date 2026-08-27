"""Model training pipelines and evaluation logic.

Deterministic, seeded training scripts and model definitions. Every
training run must be reproducible from a committed script/config and logged
to MLflow (see `src/tracking/`).
"""

from . import train

# ``tune`` (Optuna) is imported lazily because it lives in the ``training``
# dependency group, which the serving app doesn't install. Import it via
# ``from src.models.tune import tune`` when tuning.
__all__ = ["train"]
