"""
Walk-forward and rolling-horizon validation.

All validation modes use strictly causal splitting — the training window
always precedes the test window.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from benchmark.evaluation import compute_metrics

logger = logging.getLogger("benchmark.validation")


# ---------------------------------------------------------------------------
# Walk-forward validation
# ---------------------------------------------------------------------------


def walk_forward_validate(
    X: np.ndarray,
    y_raw: np.ndarray,
    train_fn: Callable,
    predict_fn: Callable,
    peak_threshold: float,
    n_splits: int = 5,
    timestamps: Optional[np.ndarray] = None,
    is_daylight: Optional[np.ndarray] = None,
    regime_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Walk-forward (expanding window) cross-validation.

    Parameters
    ----------
    X : array (n_samples, …) — features.
    y_raw : 1-D array — raw (un-scaled) target values.
    train_fn : callable(X_train, y_train, X_val, y_val) → model
        Must return an object with a ``.predict(X)`` method returning raw
        predictions (un-scaled).
    predict_fn : callable(model, X) → y_pred_raw
    peak_threshold : float — P90 threshold for peak MAE.
    n_splits : int — number of CV folds.

    Returns
    -------
    DataFrame with per-fold metrics.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_results: List[Dict[str, Any]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
        logger.info("Walk-forward fold %d / %d — train: %d  val: %d", fold_idx, n_splits, len(train_idx), len(val_idx))

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        model = train_fn(X_train, y_train, X_val, y_val)
        y_pred = predict_fn(model, X_val)

        fold_ts = timestamps[val_idx] if timestamps is not None else None
        fold_day = is_daylight[val_idx] if is_daylight is not None else None
        fold_regime = regime_ids[val_idx] if regime_ids is not None else None

        metrics = compute_metrics(
            y_val, y_pred, peak_threshold,
            timestamps=fold_ts,
            is_daylight=fold_day,
            regime_ids=fold_regime,
        )
        metrics["fold"] = fold_idx
        metrics["train_size"] = len(train_idx)
        metrics["val_size"] = len(val_idx)
        fold_results.append(metrics)

    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# Rolling-horizon validation
# ---------------------------------------------------------------------------


def rolling_horizon_validate(
    X: np.ndarray,
    y_raw: np.ndarray,
    train_fn: Callable,
    predict_fn: Callable,
    peak_threshold: float,
    train_window: int,
    test_window: int,
    step: int,
    timestamps: Optional[np.ndarray] = None,
    is_daylight: Optional[np.ndarray] = None,
    regime_ids: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Rolling (fixed-size) horizon validation.

    The training window slides forward by ``step`` samples each iteration.
    """
    n = len(X)
    fold_results: List[Dict[str, Any]] = []
    fold_idx = 0

    start = 0
    while start + train_window + test_window <= n:
        fold_idx += 1
        train_end = start + train_window
        test_end = train_end + test_window

        train_idx = np.arange(start, train_end)
        val_idx = np.arange(train_end, test_end)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_raw[train_idx], y_raw[val_idx]

        model = train_fn(X_train, y_train, X_val, y_val)
        y_pred = predict_fn(model, X_val)

        fold_ts = timestamps[val_idx] if timestamps is not None else None
        fold_day = is_daylight[val_idx] if is_daylight is not None else None
        fold_regime = regime_ids[val_idx] if regime_ids is not None else None

        metrics = compute_metrics(
            y_val, y_pred, peak_threshold,
            timestamps=fold_ts,
            is_daylight=fold_day,
            regime_ids=fold_regime,
        )
        metrics["fold"] = fold_idx
        metrics["train_start"] = int(start)
        metrics["train_end"] = int(train_end)
        metrics["test_end"] = int(test_end)
        fold_results.append(metrics)

        start += step

    logger.info("Rolling-horizon validation complete — %d folds", fold_idx)
    return pd.DataFrame(fold_results)


# ---------------------------------------------------------------------------
# Regime-specific day-level evaluation
# ---------------------------------------------------------------------------


def evaluate_by_day_type(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    timestamps: np.ndarray,
    regime_ids: np.ndarray,
    peak_threshold: float,
) -> pd.DataFrame:
    """
    Evaluate separately for clear-sky days, cloudy days, and mixed-weather days.

    A day is classified by the dominant (most frequent) regime across its
    daylight hours.
    """
    ts = pd.to_datetime(timestamps)
    dates = ts.date
    unique_dates = np.unique(dates)

    rows: List[Dict[str, Any]] = []
    for d in unique_dates:
        mask = dates == d
        if mask.sum() < 6:  # skip very short days
            continue

        day_regimes = regime_ids[mask]
        dominant = int(np.bincount(day_regimes.astype(int), minlength=3).argmax())

        yt = y_true[mask]
        yp = y_pred[mask]

        if yt.max() < 10:  # skip all-night days
            continue

        from sklearn.metrics import mean_absolute_error, mean_squared_error

        rows.append({
            "date": str(d),
            "dominant_regime": REGIME_NAMES[dominant] if dominant < len(REGIME_NAMES) else "unknown",
            "n_hours": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "peak_mae": float(mean_absolute_error(yt[yt >= peak_threshold], yp[yt >= peak_threshold])) if (yt >= peak_threshold).any() else float("nan"),
        })

    return pd.DataFrame(rows)


from benchmark.features import REGIME_NAMES
