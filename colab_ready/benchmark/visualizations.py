"""
Research-quality visualization system for the benchmarking framework.

All plots are saved at ≥200 DPI with consistent styling suitable
for academic publication.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark.visualizations")

# ---------------------------------------------------------------------------
# Style configuration
# ---------------------------------------------------------------------------

MODEL_PALETTE = {
    "GBT (LightGBM)": "#2196F3",
    "SVM (SVR-RBF)": "#FF9800",
    "ANN": "#4CAF50",
    "DNN": "#9C27B0",
    "LSTM": "#F44336",
    "CNN-DNN": "#00BCD4",
    "CNN-LSTM": "#795548",
    "CNN-A-LSTM": "#E91E63",
    "Hybrid Residual": "#607D8B",
}

DEFAULT_COLOR = "#333333"

DPI = 200
FIG_SCALE = (14, 6)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
})


def _color(model_name: str) -> str:
    return MODEL_PALETTE.get(model_name, DEFAULT_COLOR)


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved plot: %s", path)


# ---------------------------------------------------------------------------
# 1. Actual vs Predicted overlay
# ---------------------------------------------------------------------------


def plot_actual_vs_predicted(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str,
    max_points: int = 720,
    title: str = "Actual vs Predicted GHI — All Models",
) -> None:
    """30-day overlay of actual GHI and all model predictions."""
    n = min(len(y_true), max_points)
    ts = pd.to_datetime(timestamps[:n])

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(ts, y_true[:n], color="black", linewidth=2.0, label="Actual GHI", zorder=10)
    for model_name, preds in predictions.items():
        ax.plot(ts, preds[:n], linewidth=1.3, alpha=0.85, color=_color(model_name), label=model_name)

    ax.set_title(title)
    ax.set_ylabel("GHI (W/m²)")
    ax.set_xlabel("Timestamp")
    ax.legend(ncol=3, fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, output_path)

def plot_actual_vs_predicted_subplots(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str,
    max_points: int = 720,
    title: str = "30-Day Actual vs Predicted GHI (Separated)",
) -> None:
    """30-day overlay with a separate facet for each model."""
    n = min(len(y_true), max_points)
    ts = pd.to_datetime(timestamps[:n])

    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (model_name, preds) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        ax.plot(ts, y_true[:n], color="black", linewidth=2.0, label="Actual GHI", zorder=10)
        ax.plot(ts, preds[:n], linewidth=1.5, alpha=0.9, color=_color(model_name), label=model_name)
        ax.set_title(model_name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        
        if idx % cols == 0:
            ax.set_ylabel("GHI (W/m²)")
            
    for ax in axes_flat[-cols:]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        
    # Hide unused axes
    for idx in range(len(predictions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.autofmt_xdate()
    fig.tight_layout()
    _save(fig, output_path)

def plot_1day_prediction_subplots(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str,
    start_hour: int = 48,
    title: str = "1-Day Actual vs Predicted GHI Profile",
) -> None:
    """1-day overlay with a separate facet for each model."""
    n = min(len(y_true), start_hour + 24)
    ts = pd.to_datetime(timestamps[start_hour:n])
    yt = y_true[start_hour:n]

    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False, sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, (model_name, preds) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        ax.plot(ts, yt, color="black", linewidth=2.5, label="Actual GHI", zorder=10)
        ax.plot(ts, preds[start_hour:n], linewidth=2.0, alpha=0.9, color=_color(model_name), label=model_name, linestyle="--")
        ax.set_title(model_name)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)
        
        if idx % cols == 0:
            ax.set_ylabel("GHI (W/m²)")
            
    for ax in axes_flat[-cols:]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax.set_xlabel("Hour of Day")
        
    # Hide unused axes
    for idx in range(len(predictions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(title, fontsize=15, y=1.02)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Training curves
# ---------------------------------------------------------------------------


def plot_training_curves(
    histories: Dict[str, Dict[str, list]],
    output_path: str,
) -> None:
    """Loss and MAE curves for all neural-network models."""
    n_models = len(histories)
    if n_models == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for model_name, history in histories.items():
        color = _color(model_name)
        if "loss" in history:
            axes[0].plot(history["loss"], label=f"{model_name} train", color=color, linewidth=1.5)
        if "val_loss" in history:
            axes[0].plot(history["val_loss"], label=f"{model_name} val", color=color, linewidth=1.5, linestyle="--")

    axes[0].set_title("Training Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(True, alpha=0.25)

    for model_name, history in histories.items():
        color = _color(model_name)
        if "mae" in history:
            axes[1].plot(history["mae"], label=f"{model_name} train", color=color, linewidth=1.5)
        if "val_mae" in history:
            axes[1].plot(history["val_mae"], label=f"{model_name} val", color=color, linewidth=1.5, linestyle="--")

    axes[1].set_title("MAE Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].legend(fontsize=7, ncol=2)
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Residual distributions
# ---------------------------------------------------------------------------


def plot_residual_distributions(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Histogram of (predicted − actual) for each model."""
    n_models = len(predictions)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (model_name, preds) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        residuals = preds - y_true
        ax.hist(residuals, bins=60, alpha=0.75, color=_color(model_name), density=True)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.set_title(f"{model_name}\nMean={np.mean(residuals):.1f}, Std={np.std(residuals):.1f}")
        ax.set_xlabel("Prediction Error (W/m²)")
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for idx in range(len(predictions), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Residual Distributions", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------


def plot_feature_importance(
    importances: Dict[str, pd.DataFrame],
    output_path: str,
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top features for models that support it."""
    n = len(importances)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8), squeeze=False)

    for idx, (model_name, imp_df) in enumerate(importances.items()):
        ax = axes[0][idx]
        top = imp_df.sort_values("importance", ascending=False).head(top_n).iloc[::-1]
        ax.barh(top["feature"], top["importance"], color=_color(model_name))
        ax.set_title(f"{model_name} — Top {top_n} Features")
        ax.set_xlabel("Importance (Gain)")
        ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Attention weight heatmap
# ---------------------------------------------------------------------------


def plot_attention_weights(
    weights: np.ndarray,
    output_path: str,
    title: str = "CNN-A-LSTM Attention Weights",
) -> None:
    """Heatmap of attention weights across timesteps (averaged over samples)."""
    if weights.ndim == 3:
        avg_weights = weights.mean(axis=0).squeeze()
    elif weights.ndim == 2:
        avg_weights = weights.mean(axis=0)
    else:
        avg_weights = weights

    fig, ax = plt.subplots(figsize=(12, 4))
    steps = np.arange(1, len(avg_weights) + 1)
    ax.bar(steps, avg_weights, color="#E91E63", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel("Timestep (1 = oldest)")
    ax.set_ylabel("Attention Weight")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Model ranking chart
# ---------------------------------------------------------------------------


def plot_model_ranking(
    comparison_df: pd.DataFrame,
    output_path: str,
) -> None:
    """Horizontal bar chart of composite scores (lower = better)."""
    fig, ax = plt.subplots(figsize=(12, 6))
    sorted_df = comparison_df.sort_values("composite_score", ascending=True)
    colors = [_color(name) for name in sorted_df["model"]]
    ax.barh(sorted_df["model"], sorted_df["composite_score"], color=colors)
    ax.set_title("Model Ranking by Composite Score (lower = better)")
    ax.set_xlabel("Composite Score")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 7. Regime-wise box plots
# ---------------------------------------------------------------------------


def plot_regime_comparison(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    regime_ids: np.ndarray,
    output_path: str,
) -> None:
    """Bar chart of MAE per regime per model."""
    from benchmark.features import REGIME_NAMES
    from sklearn.metrics import mean_absolute_error

    models = list(predictions.keys())
    regime_names = REGIME_NAMES

    fig, ax = plt.subplots(figsize=(14, 6))
    n_regimes = len(regime_names)
    n_models = len(models)
    width = 0.8 / n_models
    x = np.arange(n_regimes)

    for i, model_name in enumerate(models):
        preds = predictions[model_name]
        maes = []
        for rid in range(n_regimes):
            mask = regime_ids == rid
            if mask.any():
                maes.append(mean_absolute_error(y_true[mask], preds[mask]))
            else:
                maes.append(0.0)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(x + offset, maes, width=width, label=model_name, color=_color(model_name))

    ax.set_xticks(x)
    ax.set_xticklabels(regime_names)
    ax.set_ylabel("MAE (W/m²)")
    ax.set_title("MAE by Weather Regime — All Models")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 8. Scatter: actual vs predicted
# ---------------------------------------------------------------------------


def plot_scatter_all(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Scatter plot of actual vs predicted for all models."""
    n = len(predictions)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (model_name, preds) in enumerate(predictions.items()):
        ax = axes_flat[idx]
        ax.scatter(y_true, preds, s=2, alpha=0.15, color=_color(model_name))
        max_val = max(y_true.max(), preds.max())
        ax.plot([0, max_val], [0, max_val], "k--", linewidth=1)
        ax.set_title(model_name)
        ax.set_xlabel("Actual GHI (W/m²)")
        ax.set_ylabel("Predicted GHI (W/m²)")
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="box")

    for idx in range(n, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Actual vs Predicted", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 9. Hourly error profile
# ---------------------------------------------------------------------------


def plot_hourly_error_profile(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    timestamps: np.ndarray,
    output_path: str,
) -> None:
    """MAE by hour-of-day for all models."""
    from sklearn.metrics import mean_absolute_error

    hours = pd.to_datetime(timestamps).hour

    fig, ax = plt.subplots(figsize=(14, 6))
    for model_name, preds in predictions.items():
        hourly_mae = []
        for h in range(24):
            mask = hours == h
            if mask.any():
                hourly_mae.append(mean_absolute_error(y_true[mask], preds[mask]))
            else:
                hourly_mae.append(0.0)
        ax.plot(range(24), hourly_mae, marker="o", linewidth=1.8, label=model_name, color=_color(model_name))

    ax.set_title("Hourly MAE Profile — All Models")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE (W/m²)")
    ax.set_xticks(range(24))
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 10. Peak prediction comparison
# ---------------------------------------------------------------------------


def plot_peak_comparison(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    peak_threshold: float,
    output_path: str,
    max_points: int = 200,
) -> None:
    """Zoomed view of peak-hour predictions only."""
    peak_mask = y_true >= peak_threshold
    if not peak_mask.any():
        logger.warning("No peak samples found — skipping peak comparison plot.")
        return

    # Take a contiguous window of peak samples
    peak_idx = np.where(peak_mask)[0][:max_points]
    ts = pd.to_datetime(timestamps[peak_idx])

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(ts, y_true[peak_idx], color="black", linewidth=2.0, label="Actual GHI", zorder=10)
    for model_name, preds in predictions.items():
        ax.plot(ts, preds[peak_idx], linewidth=1.3, alpha=0.85, color=_color(model_name), label=model_name)

    ax.set_title(f"Peak GHI Prediction Comparison (GHI ≥ {peak_threshold:.0f} W/m²)")
    ax.set_ylabel("GHI (W/m²)")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 11. Cloud-event failure analysis
# ---------------------------------------------------------------------------


def plot_cloud_event_errors(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    regime_ids: np.ndarray,
    output_path: str,
) -> None:
    """Scatter of absolute error vs actual GHI for cloudy events."""
    cloud_mask = regime_ids == 2  # cloudy
    if not cloud_mask.any():
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, preds in predictions.items():
        abs_error = np.abs(preds[cloud_mask] - y_true[cloud_mask])
        ax.scatter(y_true[cloud_mask], abs_error, s=4, alpha=0.3, color=_color(model_name), label=model_name)

    ax.set_title("Cloud-Event Prediction Error Analysis")
    ax.set_xlabel("Actual GHI (W/m²)")
    ax.set_ylabel("Absolute Error (W/m²)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Master plot generator
# ---------------------------------------------------------------------------


def generate_all_plots(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    regime_ids: np.ndarray,
    peak_threshold: float,
    training_histories: Dict[str, Dict[str, list]],
    feature_importances: Dict[str, pd.DataFrame],
    comparison_df: pd.DataFrame,
    attention_weights: Optional[np.ndarray],
    plots_dir: str,
) -> List[str]:
    """Generate all research plots and return the list of saved paths."""
    saved: List[str] = []

    # 1. Actual vs predicted
    p = os.path.join(plots_dir, "actual_vs_predicted_all.png")
    plot_actual_vs_predicted(timestamps, y_true, predictions, p)
    saved.append(p)
    
    # 1a. Actual vs predicted (Subplots - 30 days)
    p = os.path.join(plots_dir, "actual_vs_predicted_subplots_30days.png")
    plot_actual_vs_predicted_subplots(timestamps, y_true, predictions, p)
    saved.append(p)

    # 1b. Actual vs predicted (Subplots - 1 day)
    # Start at hour 72 to skip past any initial artifacts and hit a normal day
    p = os.path.join(plots_dir, "actual_vs_predicted_subplots_1day.png")
    plot_1day_prediction_subplots(timestamps, y_true, predictions, p, start_hour=72)
    saved.append(p)

    # 2. Training curves
    if training_histories:
        p = os.path.join(plots_dir, "training_curves.png")
        plot_training_curves(training_histories, p)
        saved.append(p)

    # 3. Residual distributions
    p = os.path.join(plots_dir, "residual_distributions.png")
    plot_residual_distributions(y_true, predictions, p)
    saved.append(p)

    # 4. Feature importance
    if feature_importances:
        p = os.path.join(plots_dir, "feature_importance.png")
        plot_feature_importance(feature_importances, p)
        saved.append(p)

    # 5. Attention weights
    if attention_weights is not None:
        p = os.path.join(plots_dir, "attention_weights.png")
        plot_attention_weights(attention_weights, p)
        saved.append(p)

    # 6. Model ranking
    p = os.path.join(plots_dir, "model_ranking.png")
    plot_model_ranking(comparison_df, p)
    saved.append(p)

    # 7. Regime comparison
    p = os.path.join(plots_dir, "regime_comparison.png")
    plot_regime_comparison(y_true, predictions, regime_ids, p)
    saved.append(p)

    # 8. Scatter
    p = os.path.join(plots_dir, "scatter_actual_vs_predicted.png")
    plot_scatter_all(y_true, predictions, p)
    saved.append(p)

    # 9. Hourly error
    p = os.path.join(plots_dir, "hourly_error_profile.png")
    plot_hourly_error_profile(y_true, predictions, timestamps, p)
    saved.append(p)

    # 10. Peak comparison
    p = os.path.join(plots_dir, "peak_prediction_comparison.png")
    plot_peak_comparison(timestamps, y_true, predictions, peak_threshold, p)
    saved.append(p)

    # 11. Cloud-event errors
    p = os.path.join(plots_dir, "cloud_event_errors.png")
    plot_cloud_event_errors(y_true, predictions, regime_ids, p)
    saved.append(p)

    logger.info("Generated %d plots in %s", len(saved), plots_dir)
    return saved
