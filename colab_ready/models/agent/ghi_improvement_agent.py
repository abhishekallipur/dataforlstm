import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras

import attention_lstm_model as attn
import lstm_model as base


def setup_logging(log_path: str) -> logging.Logger:
    log_parent = Path(log_path).parent
    if str(log_parent) not in ("", "."):
        log_parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("ghi_improvement_agent")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    day_mask = y_true > 10.0
    day_rmse = float(np.sqrt(mean_squared_error(y_true[day_mask], y_pred[day_mask]))) if np.any(day_mask) else np.nan
    day_mae = float(mean_absolute_error(y_true[day_mask], y_pred[day_mask])) if np.any(day_mask) else np.nan

    peak_mask = y_true >= peak_threshold
    peak_rmse = float(np.sqrt(mean_squared_error(y_true[peak_mask], y_pred[peak_mask]))) if np.any(peak_mask) else np.nan
    peak_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "day_rmse": day_rmse,
        "day_mae": day_mae,
        "peak_rmse": peak_rmse,
        "peak_mae": peak_mae,
        "pred_min": float(np.min(y_pred)),
        "pred_max": float(np.max(y_pred)),
        "true_min": float(np.min(y_true)),
        "true_max": float(np.max(y_true)),
    }


@dataclass
class Tier1Artifacts:
    blend_csv: str = "outputs/reports/blend_search_results.csv"
    blend_plot: str = "outputs/plots/sprint/blend_ratio_vs_metrics.png"
    attention_model: str = "outputs/artifacts/attention_lstm_50epochs.h5"
    attention_history: str = "outputs/reports/attention_training_history.json"
    attention_curve_plot: str = "outputs/plots/sprint/attention_training_curve.png"
    attention_pred_plot: str = "outputs/plots/sprint/attention_actual_vs_pred_50epochs.png"
    attention_weights_plot: str = "outputs/plots/sprint/attention_weights_50epochs.png"
    comparison_csv: str = "outputs/reports/tier1_comparison.csv"
    comparison_rmse_plot: str = "outputs/plots/sprint/comparison_rmse_mae_r2.png"
    comparison_day_plot: str = "outputs/plots/sprint/comparison_daytime_metrics.png"
    comparison_peak_plot: str = "outputs/plots/sprint/comparison_peak_metrics.png"
    summary_txt: str = "outputs/reports/tier1_summary.txt"
    log_file: str = "outputs/reports/ghi_agent_log.txt"


def _inverse_baseline_predictions(bundle: base.DataBundle, pred_scaled: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pred_raw = bundle.target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)
    y_true_raw = bundle.target_scaler.inverse_transform(bundle.y_test_scaled.reshape(-1, 1)).reshape(-1)
    return pred_raw, y_true_raw


def run_blend_search(
    bundle: base.DataBundle,
    baseline_pred_raw: np.ndarray,
    artifacts: Tier1Artifacts,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, float]:
    logger.info("Tier 1, Task 1.1: Blend ratio search started")
    blends = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

    rows: List[Dict[str, float]] = []
    y_true = bundle.y_test_raw.reshape(-1)

    for blend in blends:
        blended = blend * bundle.last_ghi_test + (1.0 - blend) * baseline_pred_raw
        blended = np.clip(blended, 0.0, None)
        metrics = calculate_metrics(y_true, blended, bundle.peak_threshold_raw)
        metrics["blend"] = blend
        rows.append(metrics)
        logger.info(
            "Blend=%.2f -> RMSE=%.2f MAE=%.2f Day_MAE=%.2f Peak_MAE=%.2f",
            blend,
            metrics["rmse"],
            metrics["mae"],
            metrics["day_mae"],
            metrics["peak_mae"],
        )

    df = pd.DataFrame(rows).sort_values("blend").reset_index(drop=True)

    # Winner: prioritize lower day+peak error equally, then MAE
    score = df["day_mae"] + df["peak_mae"] + 0.1 * df["mae"]
    winner_idx = int(np.nanargmin(score.to_numpy()))
    winner_blend = float(df.loc[winner_idx, "blend"])
    df["winner"] = ""
    df.loc[winner_idx, "winner"] = "yes"

    df.to_csv(artifacts.blend_csv, index=False)

    plt.figure(figsize=(9, 5))
    plt.plot(df["blend"], df["rmse"], marker="o", label="RMSE")
    plt.plot(df["blend"], df["day_mae"], marker="o", label="Day MAE")
    plt.plot(df["blend"], df["peak_mae"], marker="o", label="Peak MAE")
    plt.xlabel("Blend ratio")
    plt.ylabel("Error (W/m^2)")
    plt.title("Blend Ratio vs Metrics")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts.blend_plot, dpi=150)
    plt.close()

    logger.info("Tier 1, Task 1.1: Completed. Winner blend=%.2f", winner_blend)
    return df, winner_blend


class EpochMetricsLogger(keras.callbacks.Callback):
    def __init__(self, bundle: attn.DataBundle, peak_threshold_raw: float):
        super().__init__()
        self.bundle = bundle
        self.peak_threshold_raw = peak_threshold_raw
        self.rows: List[Dict[str, float]] = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred_scaled = self.model.predict(self.bundle.X_val, verbose=0).reshape(-1)
        y_true_scaled = self.bundle.y_val.reshape(-1)

        y_pred_raw = attn.inverse_transform_y(y_pred_scaled, self.bundle.y_scaler)
        y_true_raw = attn.inverse_transform_y(y_true_scaled, self.bundle.y_scaler)
        y_pred_raw = np.clip(y_pred_raw, 0.0, None)

        metrics = calculate_metrics(y_true_raw, y_pred_raw, self.peak_threshold_raw)
        row = {
            "epoch": int(epoch + 1),
            "train_loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "day_mae": metrics["day_mae"],
            "peak_mae": metrics["peak_mae"],
        }
        self.rows.append(row)


def train_attention_model(
    data_path: str,
    artifacts: Tier1Artifacts,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    logger: logging.Logger,
) -> Tuple[keras.Model, keras.Model, attn.DataBundle, Dict[str, float], List[Dict[str, float]]]:
    logger.info("Tier 1, Task 1.2: Attention retraining started")

    attn.set_seed(seed)
    tf.keras.backend.clear_session()

    df = attn.load_time_series(data_path)
    feat_df = attn.create_features(df)
    bundle = attn.prepare_sequences(feat_df, sequence_length=24, train_ratio=0.7, val_ratio=0.15)

    peak_threshold_scaled = float(bundle.y_scaler.transform([[bundle.train_peak_threshold_raw]])[0, 0])

    model, attention_model = attn.build_attention_lstm(
        sequence_length=24,
        num_features=bundle.X_train.shape[-1],
        learning_rate=learning_rate,
        peak_threshold_scaled=peak_threshold_scaled,
        peak_weight=2.0,
        use_peak_weighted_loss=True,
    )

    epoch_logger = EpochMetricsLogger(bundle=bundle, peak_threshold_raw=bundle.train_peak_threshold_raw)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
            start_from_epoch=50,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
        epoch_logger,
        keras.callbacks.ModelCheckpoint(
            filepath=artifacts.attention_model,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]

    history = model.fit(
        bundle.X_train,
        bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    # Ensure final best model is saved if checkpoint skipped for any reason.
    model.save(artifacts.attention_model)

    y_pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1)
    y_true_scaled = bundle.y_test.reshape(-1)
    y_pred_raw = attn.inverse_transform_y(y_pred_scaled, bundle.y_scaler)
    y_true_raw = attn.inverse_transform_y(y_true_scaled, bundle.y_scaler)
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    metrics = calculate_metrics(y_true_raw, y_pred_raw, bundle.train_peak_threshold_raw)

    # Save training history.
    history_payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "fit_history": {k: [float(vv) for vv in vals] for k, vals in history.history.items()},
        "epoch_metrics": epoch_logger.rows,
        "final_test_metrics": metrics,
    }
    with open(artifacts.attention_history, "w", encoding="utf-8") as f:
        json.dump(history_payload, f, indent=2)

    # Training curve plot.
    plt.figure(figsize=(9, 5))
    plt.plot(history.history.get("loss", []), label="Train loss")
    plt.plot(history.history.get("val_loss", []), label="Val loss")
    plt.title("Attention LSTM Training Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(artifacts.attention_curve_plot, dpi=150)
    plt.close()

    # Prediction and attention plots.
    attn.plot_actual_vs_predicted(
        timestamps=bundle.test_timestamps,
        y_true=y_true_raw,
        y_pred=y_pred_raw,
        output_path=artifacts.attention_pred_plot,
        max_points=720,
        show_plot=False,
    )
    sample_input = bundle.X_test[0:1]
    attn_weights = attention_model.predict(sample_input, verbose=0)
    attn.plot_attention_weights(attn_weights, output_path=artifacts.attention_weights_plot, show_plot=False)

    logger.info(
        "Tier 1, Task 1.2: Completed. RMSE=%.2f MAE=%.2f Day_MAE=%.2f Peak_MAE=%.2f",
        metrics["rmse"],
        metrics["mae"],
        metrics["day_mae"],
        metrics["peak_mae"],
    )

    return model, attention_model, bundle, metrics, epoch_logger.rows


def _plot_comparison_bars(df_comp: pd.DataFrame, metric_names: List[str], title: str, output_path: str) -> None:
    subset = df_comp[df_comp["metric"].isin(metric_names)].copy()
    x = np.arange(len(subset))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, subset["baseline"], width=width, label="Baseline")
    plt.bar(x + width / 2, subset["attention"], width=width, label="Attention")
    plt.xticks(x, subset["metric"], rotation=20)
    plt.ylabel("Value")
    plt.title(title)
    plt.grid(True, axis="y", linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_comparison(
    baseline_metrics: Dict[str, float],
    attention_metrics: Dict[str, float],
    best_blend: float,
    artifacts: Tier1Artifacts,
    logger: logging.Logger,
) -> pd.DataFrame:
    logger.info("Tier 1, Task 1.3: Side-by-side comparison started")

    metric_order = [
        "rmse",
        "mae",
        "r2",
        "day_rmse",
        "day_mae",
        "peak_rmse",
        "peak_mae",
        "pred_min",
        "pred_max",
        "true_min",
        "true_max",
    ]

    rows = []
    for metric in metric_order:
        b = float(baseline_metrics[metric])
        a = float(attention_metrics[metric])
        diff = a - b

        if metric == "r2":
            winner = "baseline" if b >= a else "attention"
        elif metric.startswith("true_"):
            winner = "n/a"
        else:
            winner = "baseline" if b <= a else "attention"

        rows.append({
            "metric": metric,
            "baseline": b,
            "attention": a,
            "difference_attention_minus_baseline": diff,
            "winner": winner,
        })

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(artifacts.comparison_csv, index=False)

    _plot_comparison_bars(df_comp, ["rmse", "mae", "r2"], "RMSE / MAE / R2 Comparison", artifacts.comparison_rmse_plot)
    _plot_comparison_bars(df_comp, ["day_rmse", "day_mae"], "Daytime Metrics Comparison", artifacts.comparison_day_plot)
    _plot_comparison_bars(df_comp, ["peak_rmse", "peak_mae"], "Peak Metrics Comparison", artifacts.comparison_peak_plot)

    baseline_wins = int((df_comp["winner"] == "baseline").sum())
    attention_wins = int((df_comp["winner"] == "attention").sum())

    with open(artifacts.summary_txt, "w", encoding="utf-8") as f:
        f.write("Tier 1 Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Best baseline blend: {best_blend:.2f}\n")
        f.write(f"Baseline wins: {baseline_wins}\n")
        f.write(f"Attention wins: {attention_wins}\n\n")

        if baseline_wins >= attention_wins:
            f.write("Decision: Baseline remains production model.\n")
            f.write("Next step: Tier 2 tuning focus on baseline and ensemble testing.\n")
        else:
            f.write("Decision: Attention is competitive; continue Tier 2 and Tier 3 attention exploration.\n")

        f.write("\nBaseline metrics:\n")
        for k, v in baseline_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")

        f.write("\nAttention metrics:\n")
        for k, v in attention_metrics.items():
            f.write(f"- {k}: {v:.4f}\n")

    logger.info("Tier 1, Task 1.3: Comparison completed")
    return df_comp


def train_baseline_for_tier1(
    data_path: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    logger: logging.Logger,
) -> Tuple[keras.Model, base.DataBundle, np.ndarray, Dict[str, float]]:
    logger.info("Preparing baseline model and data")

    base.set_seed(seed)
    tf.keras.backend.clear_session()

    df = base.build_feature_table(data_path)
    bundle = base.build_sequences(df=df, sequence_length=48, train_ratio=0.7, val_ratio=0.15)

    model = base.build_model(sequence_length=48, num_features=bundle.X_train.shape[-1], learning_rate=learning_rate)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=12,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    model.fit(
        bundle.X_train,
        bundle.y_train,
        validation_data=(bundle.X_val, bundle.y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1)
    pred_raw, y_true_raw = _inverse_baseline_predictions(bundle, pred_scaled)
    pred_raw = np.clip(pred_raw, 0.0, None)

    default_blend = 0.10
    blended_default = np.clip(default_blend * bundle.last_ghi_test + (1.0 - default_blend) * pred_raw, 0.0, None)
    baseline_metrics = calculate_metrics(y_true_raw, blended_default, bundle.peak_threshold_raw)
    logger.info(
        "Baseline default blend metrics RMSE=%.2f MAE=%.2f Day_MAE=%.2f Peak_MAE=%.2f",
        baseline_metrics["rmse"],
        baseline_metrics["mae"],
        baseline_metrics["day_mae"],
        baseline_metrics["peak_mae"],
    )

    return model, bundle, pred_raw, baseline_metrics


def run_tier1(args: argparse.Namespace) -> None:
    artifacts = Tier1Artifacts()

    # Ensure output directories exist.
    for output_path in asdict(artifacts).values():
        parent = Path(output_path).parent
        if str(parent) not in ("", "."):
            parent.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(artifacts.log_file)

    logger.info("Tier 1 workflow started")

    baseline_model, base_bundle, baseline_pred_raw, _ = train_baseline_for_tier1(
        data_path=args.data_path,
        epochs=args.baseline_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logger=logger,
    )

    blend_df, best_blend = run_blend_search(base_bundle, baseline_pred_raw, artifacts, logger)

    y_true_base = base_bundle.y_test_raw.reshape(-1)
    baseline_best_pred = np.clip(best_blend * base_bundle.last_ghi_test + (1.0 - best_blend) * baseline_pred_raw, 0.0, None)
    baseline_best_metrics = calculate_metrics(y_true_base, baseline_best_pred, base_bundle.peak_threshold_raw)

    _, _, _, attention_metrics, epoch_rows = train_attention_model(
        data_path=args.data_path,
        artifacts=artifacts,
        epochs=args.attention_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        logger=logger,
    )

    run_comparison(
        baseline_metrics=baseline_best_metrics,
        attention_metrics=attention_metrics,
        best_blend=best_blend,
        artifacts=artifacts,
        logger=logger,
    )

    logger.info("Tier 1 workflow finished")
    logger.info("Artifacts generated: %s", json.dumps(asdict(artifacts), indent=2))
    logger.info("Blend search rows: %d, Attention epoch rows: %d", len(blend_df), len(epoch_rows))



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GHI model improvement agent - Tier 1 automation")
    parser.add_argument("--data-path", type=str, default="dataset", help="Path to dataset folder or CSV")
    parser.add_argument("--baseline-epochs", type=int, default=5, help="Epochs for baseline model refresh run")
    parser.add_argument("--attention-epochs", type=int, default=100, help="Max epochs for attention retraining")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


if __name__ == "__main__":
    run_tier1(parse_args())
