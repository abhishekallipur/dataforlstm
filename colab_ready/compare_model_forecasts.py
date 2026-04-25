"""Generate aligned forecast comparison plots for the available GHI models.

This utility loads the saved baseline, attention, and residual-hybrid artifacts,
aligns them on the common test timestamps, and writes:

- a 30-day overlay plot for all models
- one-day overlay plots for representative sunny/cloudy/mixed cases
- accuracy tables overall and by regime
- a merged prediction CSV for downstream analysis

Rain is not directly observed in the hourly NSRDB rows used by this project,
so cloudy/overcast is the closest available proxy for low-irradiance weather.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import models.baseline_lstm.model as baseline
import models.agent.ghi_sprint_parallel as sprint_parallel


plt.switch_backend("Agg")

MODEL_COLUMNS = ["baseline_pred", "attention_pred", "hybrid_pred"]
MODEL_LABELS = {
    "baseline_pred": "Baseline LSTM",
    "attention_pred": "Attention LSTM",
    "hybrid_pred": "Residual Hybrid",
}
MODEL_COLORS = {
    "baseline_pred": "#1f77b4",
    "attention_pred": "#ff7f0e",
    "hybrid_pred": "#2ca02c",
}


@dataclass
class ForecastComparisonArtifacts:
    predictions_csv: str = "outputs/reports/model_forecast_comparison_predictions.csv"
    metrics_csv: str = "outputs/reports/model_forecast_comparison_metrics.csv"
    regime_metrics_csv: str = "outputs/reports/model_forecast_comparison_regime_metrics.csv"
    selected_days_csv: str = "outputs/reports/model_forecast_comparison_selected_days.csv"
    summary_json: str = "outputs/reports/model_forecast_comparison_summary.json"
    comparison_30d_plot: str = "outputs/plots/comparison/model_forecast_comparison_30_days.png"
    month_day_plot: str = "outputs/plots/comparison/model_forecast_month_day_comparison.png"
    regime_plot: str = "outputs/plots/comparison/model_forecast_regime_mae.png"
    forecast_day_prefix: str = "outputs/plots/comparison/model_forecast_day"


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _resolve_existing_path(candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of these artifacts exist: {', '.join(candidates)}")


def _load_model(model_path: Path) -> keras.Model:
    try:
        return keras.models.load_model(model_path, compile=False, safe_mode=False)
    except TypeError:
        return keras.models.load_model(model_path, compile=False)


def _compute_clear_sky_ghi(solar_zenith_angle: pd.Series) -> pd.Series:
    zenith = solar_zenith_angle.clip(0.0, 180.0)
    cos_zenith = np.cos(np.deg2rad(zenith)).clip(0.0, 1.0)
    denom = np.clip(cos_zenith, 0.05, None)
    clear_sky = 1098.0 * cos_zenith * np.exp(-0.059 / denom)
    clear_sky = np.where(cos_zenith > 0.0, clear_sky, 0.0)
    return pd.Series(clear_sky, index=solar_zenith_angle.index)


def _classify_regime(clear_sky_index: pd.Series, is_daylight: pd.Series) -> pd.Series:
    regime = np.full(len(clear_sky_index), "cloudy", dtype=object)
    daylight = is_daylight.to_numpy(dtype=np.float32) > 0.5
    idx = clear_sky_index.to_numpy(dtype=np.float32)

    clear_mask = daylight & (idx >= 0.80)
    partly_mask = daylight & (idx >= 0.45) & (idx < 0.80)
    night_mask = ~daylight

    regime[clear_mask] = "sunny"
    regime[partly_mask] = "partly_cloudy"
    regime[night_mask] = "night"
    return pd.Series(regime, index=clear_sky_index.index)


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    daylight_mask = y_true > 10.0
    peak_mask = y_true >= peak_threshold

    metrics: Dict[str, float] = {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "day_mae": float(mean_absolute_error(y_true[daylight_mask], y_pred[daylight_mask])) if np.any(daylight_mask) else float("nan"),
        "peak_mae": float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else float("nan"),
    }
    return metrics


def _load_baseline_predictions(
    data_path: str,
    model_path: Path,
    sequence_length: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
) -> Dict[str, object]:
    raw_df = baseline.build_feature_table(data_path)
    bundle = baseline.build_sequences(
        df=raw_df,
        sequence_length=sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    model = _load_model(model_path)
    pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1, 1)

    actual_raw = bundle.target_scaler.inverse_transform(bundle.y_test_scaled.reshape(-1, 1)).reshape(-1)
    pred_raw = bundle.target_scaler.inverse_transform(pred_scaled).reshape(-1)
    pred_raw = np.clip(pred_raw, 0.0, None)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(bundle.test_timestamps),
            "actual_ghi": actual_raw,
            "baseline_pred": pred_raw,
        }
    )
    frame = frame.merge(
        raw_df[
            [
                "timestamp",
                "solar_zenith_angle",
                "temperature",
                "relative_humidity",
                "wind_speed",
                "pressure",
                "hour",
                "cos_zenith",
                "is_daylight",
            ]
        ],
        on="timestamp",
        how="left",
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    return {
        "frame": frame,
        "peak_threshold": float(bundle.peak_threshold_raw),
    }


def _load_attention_predictions(
    data_path: str,
    model_path: Path,
    sequence_length: int,
    train_ratio: float,
    val_ratio: float,
    batch_size: int,
) -> pd.DataFrame:
    raw_df = baseline.build_feature_table(data_path)
    bundle = baseline.build_sequences(
        df=raw_df,
        sequence_length=sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    peak_threshold_scaled = float(np.percentile(bundle.y_train.reshape(-1), 90))
    model = sprint_parallel.build_attention_for_scaled_target(
        sequence_length=sequence_length,
        num_features=bundle.X_test.shape[-1],
        learning_rate=1e-3,
        peak_threshold_scaled=peak_threshold_scaled,
        peak_weight=2.0,
    )
    model.load_weights(str(model_path))
    pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1)

    pred_raw = bundle.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    pred_raw = np.clip(pred_raw, 0.0, None)
    actual_raw = bundle.target_scaler.inverse_transform(bundle.y_test_scaled.reshape(-1, 1)).reshape(-1)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(bundle.test_timestamps),
            "actual_ghi": actual_raw,
            "attention_pred": pred_raw,
        }
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _load_hybrid_predictions(hybrid_csv: str) -> pd.DataFrame:
    path = Path(hybrid_csv)
    if not path.exists():
        raise FileNotFoundError(f"Residual hybrid predictions CSV not found: {hybrid_csv}")

    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "hybrid_pred"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Residual hybrid CSV is missing required columns: {sorted(missing)}")

    if "actual_ghi" not in frame.columns and "ghi" in frame.columns:
        frame = frame.rename(columns={"ghi": "actual_ghi"})

    keep_cols = [col for col in ["timestamp", "actual_ghi", "baseline_pred", "hybrid_pred"] if col in frame.columns]
    frame = frame[keep_cols].copy()
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def _merge_predictions(
    baseline_frame: pd.DataFrame,
    attention_frame: pd.DataFrame,
    hybrid_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = baseline_frame.merge(
        attention_frame[["timestamp", "attention_pred"]],
        on="timestamp",
        how="inner",
    )
    merged = merged.merge(
        hybrid_frame[["timestamp", "hybrid_pred"]],
        on="timestamp",
        how="inner",
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["clear_sky_ghi_est"] = _compute_clear_sky_ghi(merged["solar_zenith_angle"])
    merged["clear_sky_index"] = merged["actual_ghi"] / np.clip(merged["clear_sky_ghi_est"], 1.0, None)
    merged["regime_label"] = _classify_regime(merged["clear_sky_index"], merged["is_daylight"])
    merged["month"] = merged["timestamp"].dt.month
    merged["date"] = merged["timestamp"].dt.date
    merged["hour"] = merged["timestamp"].dt.hour

    for column in MODEL_COLUMNS:
        merged[f"{column}_error"] = merged[column] - merged["actual_ghi"]
        merged[f"{column}_abs_error"] = merged[f"{column}_error"].abs()

    return merged


def _build_metrics_table(frame: pd.DataFrame, peak_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for column in MODEL_COLUMNS:
        metrics = _score_predictions(frame["actual_ghi"].to_numpy(dtype=np.float32), frame[column].to_numpy(dtype=np.float32), peak_threshold)
        rows.append({"model": MODEL_LABELS[column], **metrics})
    return pd.DataFrame(rows)


def _build_regime_metrics(frame: pd.DataFrame, peak_threshold: float) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    regime_order = ["sunny", "partly_cloudy", "cloudy", "night"]
    for regime in regime_order:
        subset = frame[frame["regime_label"] == regime]
        if subset.empty:
            continue
        for column in MODEL_COLUMNS:
            metrics = _score_predictions(subset["actual_ghi"].to_numpy(dtype=np.float32), subset[column].to_numpy(dtype=np.float32), peak_threshold)
            rows.append(
                {
                    "regime": regime,
                    "model": MODEL_LABELS[column],
                    "count": int(len(subset)),
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "day_mae": metrics["day_mae"],
                    "peak_mae": metrics["peak_mae"],
                }
            )
    return pd.DataFrame(rows)


def _build_day_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for date, subset in frame.groupby("date"):
        daylight = subset[subset["is_daylight"] > 0.5]
        if daylight.empty:
            continue
        clear_mean = float(daylight["clear_sky_index"].mean())
        dominant_regime = daylight["regime_label"].mode().iloc[0] if not daylight["regime_label"].mode().empty else "mixed"
        row: Dict[str, object] = {
            "date": str(date),
            "month": int(pd.Timestamp(date).month),
            "daylight_count": int(len(daylight)),
            "clear_sky_index_mean": clear_mean,
            "dominant_regime": dominant_regime,
        }
        for column in MODEL_COLUMNS:
            row[f"{column}_mae"] = float(mean_absolute_error(daylight["actual_ghi"], daylight[column]))
        rows.append(row)
    return pd.DataFrame(rows)


def _pick_representative_days(day_summary: pd.DataFrame, month: Optional[int] = None) -> Dict[str, str]:
    if day_summary.empty:
        return {}

    filtered = day_summary.copy()
    if month is not None:
        month_filtered = filtered[filtered["month"] == month]
        if not month_filtered.empty:
            filtered = month_filtered

    filtered = filtered[filtered["daylight_count"] >= 6]
    if filtered.empty:
        filtered = day_summary[day_summary["daylight_count"] >= 6]
    if filtered.empty:
        filtered = day_summary.copy()

    sunny_date = filtered.sort_values(["clear_sky_index_mean", "daylight_count"], ascending=[False, False]).iloc[0]["date"]
    cloudy_date = filtered.sort_values(["clear_sky_index_mean", "daylight_count"], ascending=[True, False]).iloc[0]["date"]

    median_index = float(filtered["clear_sky_index_mean"].median())
    mixed_date = filtered.iloc[(filtered["clear_sky_index_mean"] - median_index).abs().argsort()].iloc[0]["date"]

    return {
        "sunny": str(sunny_date),
        "cloudy": str(cloudy_date),
        "mixed": str(mixed_date),
    }


def _window_for_day(frame: pd.DataFrame, date_text: str) -> pd.DataFrame:
    date = pd.to_datetime(date_text).date()
    window = frame[frame["date"] == date].copy()
    if window.empty:
        raise ValueError(f"No data available for date {date_text}")
    return window.sort_values("timestamp").reset_index(drop=True)


def _window_for_horizon(frame: pd.DataFrame, forecast_days: int, start_date: Optional[str] = None) -> pd.DataFrame:
    if start_date:
        start_ts = pd.to_datetime(start_date)
        window = frame[frame["timestamp"] >= start_ts].copy()
    else:
        window = frame.copy()

    points = min(len(window), forecast_days * 24)
    if points <= 0:
        raise ValueError("No samples available for the requested forecast horizon.")
    return window.iloc[:points].reset_index(drop=True)


def _plot_comparison_window(frame: pd.DataFrame, output_path: str, title: str) -> None:
    if frame.empty:
        raise ValueError("Cannot plot an empty comparison window.")

    _ensure_parent_dir(output_path)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    timestamps = pd.to_datetime(frame["timestamp"])
    ax_top.plot(timestamps, frame["actual_ghi"], color="black", linewidth=2.2, label="Actual GHI")
    for column in MODEL_COLUMNS:
        ax_top.plot(
            timestamps,
            frame[column],
            linewidth=1.7,
            alpha=0.95,
            color=MODEL_COLORS[column],
            label=MODEL_LABELS[column],
        )
    ax_top.set_title(title)
    ax_top.set_ylabel("GHI (W/m^2)")
    ax_top.grid(True, alpha=0.3)
    ax_top.legend(ncol=2, fontsize=9)

    for column in MODEL_COLUMNS:
        ax_bottom.plot(
            timestamps,
            frame[f"{column}_abs_error"],
            linewidth=1.7,
            alpha=0.95,
            color=MODEL_COLORS[column],
            label=f"{MODEL_LABELS[column]} abs error",
        )
    ax_bottom.set_ylabel("Absolute Error (W/m^2)")
    ax_bottom.set_xlabel("Timestamp")
    ax_bottom.grid(True, alpha=0.3)
    ax_bottom.legend(ncol=2, fontsize=9)

    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_metrics(regime_metrics: pd.DataFrame, output_path: str) -> None:
    if regime_metrics.empty:
        return

    _ensure_parent_dir(output_path)
    order = ["sunny", "partly_cloudy", "cloudy", "night"]
    filtered = regime_metrics[regime_metrics["regime"].isin(order)].copy()
    if filtered.empty:
        return

    pivot = filtered.pivot(index="regime", columns="model", values="mae").reindex(order).dropna(how="all")
    regimes = list(pivot.index)
    x_pos = np.arange(len(regimes))
    width = 0.24

    fig, ax = plt.subplots(figsize=(14, 6))
    offsets = {
        "Baseline LSTM": -width,
        "Attention LSTM": 0.0,
        "Residual Hybrid": width,
    }

    for model_label, offset in offsets.items():
        if model_label not in pivot.columns:
            continue
        ax.bar(x_pos + offset, pivot[model_label].values, width=width, label=model_label)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("MAE (W/m^2)")
    ax.set_title("Model MAE by Weather Regime")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_month_and_day_comparison(month_frame: pd.DataFrame, day_frame: pd.DataFrame, output_path: str) -> None:
    if month_frame.empty:
        raise ValueError("Month comparison frame is empty.")
    if day_frame.empty:
        raise ValueError("Day comparison frame is empty.")

    _ensure_parent_dir(output_path)
    fig, axes = plt.subplots(2, 1, figsize=(18, 12), sharex=False)

    def _plot_panel(ax: plt.Axes, frame: pd.DataFrame, title: str) -> None:
        timestamps = pd.to_datetime(frame["timestamp"])
        ax.plot(timestamps, frame["actual_ghi"], color="black", linewidth=2.4, label="Actual GHI")
        for column in MODEL_COLUMNS:
            ax.plot(
                timestamps,
                frame[column],
                linewidth=1.8,
                alpha=0.95,
                color=MODEL_COLORS[column],
                label=MODEL_LABELS[column],
            )
        ax.set_title(title)
        ax.set_ylabel("GHI (W/m^2)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=9)

    _plot_panel(axes[0], month_frame, "Month Window: Actual vs Predicted for All Models")
    _plot_panel(axes[1], day_frame, "Single-Day Window: Actual vs Predicted for All Models")
    axes[1].set_xlabel("Timestamp")

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _format_focus_name(label: str) -> str:
    return label.replace("_", "-")


def run_pipeline(
    data_path: str,
    baseline_model_path: str,
    attention_model_path: str,
    hybrid_predictions_csv: str,
    baseline_sequence_length: int = 48,
    attention_sequence_length: int = 48,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 128,
    forecast_days: int = 30,
    focus_month: Optional[int] = None,
    focus_date: Optional[str] = None,
    artifacts: ForecastComparisonArtifacts = ForecastComparisonArtifacts(),
) -> Dict[str, object]:
    baseline_result = _load_baseline_predictions(
        data_path=data_path,
        model_path=Path(baseline_model_path),
        sequence_length=baseline_sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
    )
    baseline_frame = baseline_result["frame"]
    peak_threshold = float(baseline_result["peak_threshold"])

    attention_frame = _load_attention_predictions(
        data_path=data_path,
        model_path=Path(attention_model_path),
        sequence_length=attention_sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        batch_size=batch_size,
    )
    hybrid_frame = _load_hybrid_predictions(hybrid_predictions_csv)

    common = _merge_predictions(baseline_frame, attention_frame, hybrid_frame)

    metrics_table = _build_metrics_table(common, peak_threshold)
    regime_metrics = _build_regime_metrics(common, peak_threshold)
    day_summary = _build_day_summary(common)

    _ensure_parent_dir(artifacts.predictions_csv)
    common.to_csv(artifacts.predictions_csv, index=False)

    _ensure_parent_dir(artifacts.metrics_csv)
    metrics_table.to_csv(artifacts.metrics_csv, index=False)

    _ensure_parent_dir(artifacts.regime_metrics_csv)
    regime_metrics.to_csv(artifacts.regime_metrics_csv, index=False)

    _ensure_parent_dir(artifacts.selected_days_csv)
    day_summary.to_csv(artifacts.selected_days_csv, index=False)

    horizon_window = _window_for_horizon(common, forecast_days=forecast_days, start_date=focus_date)
    _plot_comparison_window(
        horizon_window,
        artifacts.comparison_30d_plot,
        title=f"{forecast_days}-Day Forecast Horizon: Actual vs All Models",
    )

    if focus_month is not None:
        month_window_source = common[common["month"] == focus_month].copy()
        if month_window_source.empty:
            month_window_source = common.copy()
    else:
        month_window_source = common.copy()
    month_window = _window_for_horizon(month_window_source, forecast_days=forecast_days, start_date=focus_date)

    if focus_date:
        selected_days = {"selected": pd.to_datetime(focus_date).date().isoformat()}
    else:
        selected_days = _pick_representative_days(day_summary, month=focus_month)

    if selected_days:
        day_key = "selected" if "selected" in selected_days else ("mixed" if "mixed" in selected_days else next(iter(selected_days)))
        day_window = _window_for_day(common, selected_days[day_key])
    else:
        day_window = _window_for_day(common, common.loc[0, "date"])

    _plot_month_and_day_comparison(month_window, day_window, artifacts.month_day_plot)

    one_day_outputs: Dict[str, str] = {}
    for label, date_text in selected_days.items():
        day_window = _window_for_day(common, date_text)
        suffix = _format_focus_name(label)
        output_path = f"{artifacts.forecast_day_prefix}_{suffix}.png"
        title = f"One-Day Comparison ({label.title()}): {date_text}"
        _plot_comparison_window(day_window, output_path, title=title)
        one_day_outputs[label] = output_path

    _plot_regime_metrics(regime_metrics, artifacts.regime_plot)

    summary = {
        "peak_threshold_raw": peak_threshold,
        "metrics": metrics_table.to_dict(orient="records"),
        "regime_metrics": regime_metrics.to_dict(orient="records"),
        "selected_days": selected_days,
        "artifacts": asdict(artifacts),
        "one_day_outputs": one_day_outputs,
    }

    _ensure_parent_dir(artifacts.summary_json)
    with open(artifacts.summary_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nModel comparison summary")
    for row in metrics_table.to_dict(orient="records"):
        print(
            f"{row['model']}: RMSE {row['rmse']:.2f}, MAE {row['mae']:.2f}, "
            f"Day MAE {row['day_mae']:.2f}, Peak MAE {row['peak_mae']:.2f}"
        )
    print(f"Saved 30-day comparison to: {artifacts.comparison_30d_plot}")
    print(f"Saved month/day comparison to: {artifacts.month_day_plot}")
    print(f"Saved regime comparison to: {artifacts.regime_plot}")
    if one_day_outputs:
        for label, path in one_day_outputs.items():
            print(f"Saved {label} one-day comparison to: {path}")
    print(f"Saved merged predictions to: {artifacts.predictions_csv}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GHI forecasts across all trained models.")
    parser.add_argument("--data-path", type=str, default="dataset", help="Folder or CSV path with the NSRDB data.")
    parser.add_argument(
        "--baseline-model-path",
        type=str,
        default="outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5",
        help="Path to the baseline LSTM model artifact.",
    )
    parser.add_argument(
        "--attention-model-path",
        type=str,
        default="outputs/artifacts/task_b_attention_lstm_trained.h5",
        help="Path to the attention LSTM model artifact.",
    )
    parser.add_argument(
        "--hybrid-predictions-csv",
        type=str,
        default="outputs/reports/residual_hybrid_predictions.csv",
        help="Path to the saved residual-hybrid predictions CSV.",
    )
    parser.add_argument("--baseline-sequence-length", type=int, default=48, help="Sequence length for the baseline and hybrid alignment.")
    parser.add_argument("--attention-sequence-length", type=int, default=48, help="Sequence length used by the sprint attention artifact.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Chronological validation split ratio.")
    parser.add_argument("--batch-size", type=int, default=128, help="Inference batch size.")
    parser.add_argument("--forecast-days", type=int, default=30, help="Number of days to show in the main comparison plot.")
    parser.add_argument(
        "--focus-month",
        type=int,
        default=None,
        help="Optional month number (1-12) to pick representative one-day examples from.",
    )
    parser.add_argument(
        "--focus-date",
        type=str,
        default=None,
        help="Optional ISO date (YYYY-MM-DD) for the one-day comparison and 30-day window start.",
    )
    parser.add_argument("--predictions-csv", type=str, default="outputs/reports/model_forecast_comparison_predictions.csv", help="Output CSV for the merged predictions.")
    parser.add_argument("--metrics-csv", type=str, default="outputs/reports/model_forecast_comparison_metrics.csv", help="Output CSV for the overall metrics.")
    parser.add_argument("--regime-metrics-csv", type=str, default="outputs/reports/model_forecast_comparison_regime_metrics.csv", help="Output CSV for regime-based metrics.")
    parser.add_argument("--selected-days-csv", type=str, default="outputs/reports/model_forecast_comparison_selected_days.csv", help="Output CSV for the chosen day summary.")
    parser.add_argument("--summary-json", type=str, default="outputs/reports/model_forecast_comparison_summary.json", help="Summary JSON for the comparison run.")
    parser.add_argument("--comparison-30d-plot", type=str, default="outputs/plots/comparison/model_forecast_comparison_30_days.png", help="Output plot for the 30-day comparison.")
    parser.add_argument("--month-day-plot", type=str, default="outputs/plots/comparison/model_forecast_month_day_comparison.png", help="Output plot for the combined month/day comparison.")
    parser.add_argument("--regime-plot", type=str, default="outputs/plots/comparison/model_forecast_regime_mae.png", help="Output plot for the regime metrics.")
    parser.add_argument("--forecast-day-prefix", type=str, default="outputs/plots/comparison/model_forecast_day", help="Prefix for one-day comparison plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = ForecastComparisonArtifacts(
        predictions_csv=args.predictions_csv,
        metrics_csv=args.metrics_csv,
        regime_metrics_csv=args.regime_metrics_csv,
        selected_days_csv=args.selected_days_csv,
        summary_json=args.summary_json,
        comparison_30d_plot=args.comparison_30d_plot,
        month_day_plot=args.month_day_plot,
        regime_plot=args.regime_plot,
        forecast_day_prefix=args.forecast_day_prefix,
    )

    baseline_model_path = _resolve_existing_path(
        [
            args.baseline_model_path,
            "outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5",
            "outputs/artifacts/task_d_baseline_original_huber.h5",
        ]
    )
    attention_model_path = _resolve_existing_path(
        [
            args.attention_model_path,
            "outputs/artifacts/attention_lstm_50epochs.h5",
        ]
    )

    if args.focus_date is not None:
        pd.to_datetime(args.focus_date)

    run_pipeline(
        data_path=args.data_path,
        baseline_model_path=str(baseline_model_path),
        attention_model_path=str(attention_model_path),
        hybrid_predictions_csv=args.hybrid_predictions_csv,
        baseline_sequence_length=args.baseline_sequence_length,
        attention_sequence_length=args.attention_sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        forecast_days=args.forecast_days,
        focus_month=args.focus_month,
        focus_date=args.focus_date,
        artifacts=artifacts,
    )


if __name__ == "__main__":
    main()