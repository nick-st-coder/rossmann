"""Data loading, preprocessing, and splitting.

Anything data-related that runs more than once lives here instead of in a
notebook cell: raw loaders, the cleaning/feature pipeline, and
chronological train/test splits.
"""

from .loaders import feature_columns, load_processed, load_raw_data, preprocess
from .split import split_by_date, split_features_target, split_open_closed

__all__ = [
    "feature_columns",
    "load_processed",
    "load_raw_data",
    "preprocess",
    "split_by_date",
    "split_features_target",
    "split_open_closed",
]
