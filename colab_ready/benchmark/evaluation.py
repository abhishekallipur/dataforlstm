"""
Unified evaluation system for the benchmarking framework.

Computes all metrics for every model and produces comparison tables
with composite-score rankings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from benchmark.config import EvalConfig, GHI_DAYLIGHT_THRESHOLD
from benchmark.features import REGIME_NAMES

logger = logging.getLogger("benchmark.evaluation")


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    peak_threshold: float,
    timestamps: Optional[np.ndarray] = None,
    is_daylight: Optional[np.ndarray] = None,
    regime_ids: Optional[np.ndarray] = None,
    transition_window: int = 2,
) -> Dict[str, float]:
    """
    Compute the full metric suite.

    Parameters
    ----------
    y_true, y_pred : 1-D float arrays of equal length.
    peak_threshold : GHI value (W/m²) above which a sample is "peak".
    timestamps : optional, for transition-hour detection.
    is_daylight : optional, explicit daylight mask.
    regime_ids : optional, integer regime codes (0=clear, 1=partly, 2=cloudy).
    transition_window : hours around sunrise/sunset to count as transition.

    Returns
    -------
    dict with keys: rmse, mae, r2, day_mae, peak_mae, cloud_mae, transition_mae
    """
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    assert len(y_true) == len(y_pred), "Length mismatch in y_true / y_pred."

    metrics: Dict[str, float] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if np.std(y_true) > 1e-8 else float("nan"),
    }

    # Daytime MAE
    if is_daylight is not None:
        day_mask = is_daylight.ravel() > 0.5
    else:
        day_mask = y_true > GHI_DAYLIGHT_THRESHOLD
    if day_mask.any():
        metrics["day_mae"] = float(mean_absolute_error(y_true[day_mask], y_pred[day_mask]))
    else:
        metrics["day_mae"] = float("nan")

    # Peak MAE
    peak_mask = y_true >= peak_threshold
    if peak_mask.any():
        metrics["peak_mae"] = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask]))
    else:
        metrics["peak_mae"] = float("nan")

    # Cloud-event MAE (regime_id == 2 = cloudy)
    if regime_ids is not None:
        cloud_mask = (regime_ids.ravel() == 2) & day_mask
        if cloud_mask.any():
            metrics["cloud_mae"] = float(mean_absolute_error(y_true[cloud_mask], y_pred[cloud_mask]))
        else:
            metrics["cloud_mae"] = float("nan")
    else:
        metrics["cloud_mae"] = float("nan")

    # Sunrise/sunset transition MAE
    if timestamps is not None:
        ts = pd.to_datetime(timestamps)
        hours = ts.hour.values
        # Transition = within ±transition_window hours of sunrise (6-8) or sunset (17-19)
        sunrise_mask = (hours >= (6 - transition_window)) & (hours <= (8 + transition_window))
        sunset_mask = (hours >= (17 - transition_window)) & (hours <= (19 + transition_window))
        transition_mask = sunrise_mask | sunset_mask
        if transition_mask.any():
            metrics["transition_mae"] = float(mean_absolute_error(y_true[transition_mask], y_pred[transition_mask]))
        else:
            metrics["transition_mae"] = float("nan")
    else:
        metrics["transition_mae"] = float("nan")

    return metrics


# ---------------------------------------------------------------------------
# Per-regime metrics
# ---------------------------------------------------------------------------


def compute_regime_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    regime_ids: np.ndarray,
    peak_threshold: float,
    is_daylight: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute metrics broken down by weather regime."""
    rows: List[Dict[str, object]] = []
    for rid, rname in enumerate(REGIME_NAMES):
        mask = regime_ids.ravel() == rid
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        row: Dict[str, object] = {
            "regime": rname,
            "count": int(mask.sum()),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "mae": float(mean_absolute_error(yt, yp)),
            "bias": float(np.mean(yp - yt)),
        }
        peak_m = yt >= peak_threshold
        row["peak_mae"] = float(mean_absolute_error(yt[peak_m], yp[peak_m])) if peak_m.any() else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Composite ranking
# ---------------------------------------------------------------------------


def compute_composite_score(
    metrics: Dict[str, float],
    config: Optional[EvalConfig] = None,
) -> float:
    """
    Compute a single composite score for ranking.

    Lower is better.  NaN metrics contribute nothing (treated as 0).
    """
    cfg = config or EvalConfig()

    def _safe(val: float) -> float:
        return val if np.isfinite(val) else 0.0

    score = (
        cfg.weight_rmse * _safe(metrics.get("rmse", 0.0))
        + cfg.weight_mae * _safe(metrics.get("mae", 0.0))
        + cfg.weight_peak_mae * _safe(metrics.get("peak_mae", 0.0))
        + cfg.weight_cloud_mae * _safe(metrics.get("cloud_mae", 0.0))
        + cfg.weight_transition_mae * _safe(metrics.get("transition_mae", 0.0))
    )
    return float(score)


def build_comparison_table(
    results: Dict[str, Dict[str, float]],
    config: Optional[EvalConfig] = None,
) -> pd.DataFrame:
    """
    Build a comparison DataFrame for all models, ranked by composite score.

    Parameters
    ----------
    results : dict
        ``{model_name: metrics_dict}`` where each ``metrics_dict`` has keys
        ``rmse, mae, r2, day_mae, peak_mae, cloud_mae, transition_mae``.

    Returns
    -------
    DataFrame with one row per model, sorted by ``composite_score`` (ascending).
    """
    rows: List[Dict[str, object]] = []
    for model_name, metrics in results.items():
        row = {"model": model_name, **metrics}
        row["composite_score"] = compute_composite_score(metrics, config)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("composite_score").reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df


# ---------------------------------------------------------------------------
# Robustness ranking
# ---------------------------------------------------------------------------


def build_robustness_ranking(
    sanity_reports: Dict[str, dict],
) -> pd.DataFrame:
    """
    Rank models by prediction robustness (fewer corrections = better).
    """
    rows = []
    for model_name, report in sanity_reports.items():
        rows.append({
            "model": model_name,
            "n_nan": report.get("n_nan", 0),
            "n_negative": report.get("n_negative", 0),
            "n_exploding": report.get("n_exploding", 0),
            "n_total_corrected": report.get("n_total_corrected", 0),
        })
    df = pd.DataFrame(rows).sort_values("n_total_corrected").reset_index(drop=True)
    df["robustness_rank"] = range(1, len(df) + 1)
    return df
