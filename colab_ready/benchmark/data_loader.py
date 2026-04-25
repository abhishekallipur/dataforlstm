"""
Data loading module — delegates to the existing NSRDB loader and applies
the full feature engineering pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure the project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Ensure models package is importable (create __init__.py if missing)
_models_dir = _PROJECT_ROOT / "models"
for _pkg in [_models_dir, _models_dir / "baseline_lstm"]:
    _init = _pkg / "__init__.py"
    if _pkg.is_dir() and not _init.exists():
        _init.write_text("")

import models.baseline_lstm.model as _baseline_loader

logger = logging.getLogger("benchmark.data_loader")


def load_raw_nsrdb(data_path: str) -> pd.DataFrame:
    """
    Load raw NSRDB data and build the baseline feature table.

    Delegates to the existing ``models.baseline_lstm.model.build_feature_table``
    which handles:
      - Multi-CSV directory loading
      - Timestamp construction from Year/Month/Day/Hour/Minute
      - Column alias resolution (GHI, Temperature, etc.)
      - Solar zenith computation (NOAA approximation fallback)
      - Cyclical time features (sin/cos hour & day-of-year)
      - Lag features (ghi_lag_1/2/3/24)
      - Rolling features (mean/std over 3/6/24 windows)
      - Day/night flag (is_daylight, cos_zenith)

    Returns a chronologically sorted DataFrame with no NaN values.
    """
    logger.info("Loading NSRDB data from: %s", data_path)
    df = _baseline_loader.build_feature_table(data_path)
    logger.info(
        "Loaded %d rows × %d columns  |  time range: %s → %s",
        len(df),
        len(df.columns),
        df["timestamp"].iloc[0],
        df["timestamp"].iloc[-1],
    )
    return df


def load_benchmark_data(data_path: str, extra_features: bool = True) -> pd.DataFrame:
    """
    Full pipeline: load raw NSRDB → enrich with additional benchmark features.

    Parameters
    ----------
    data_path : str
        Path to the dataset directory or CSV file.
    extra_features : bool
        If True, add volatility, solar geometry, regime, and residual-ready
        features on top of the baseline feature table.

    Returns
    -------
    pd.DataFrame
        Fully-featured, chronologically sorted, NaN-free DataFrame.
    """
    from benchmark.features import enrich_features

    df = load_raw_nsrdb(data_path)
    if extra_features:
        df = enrich_features(df)
    return df
