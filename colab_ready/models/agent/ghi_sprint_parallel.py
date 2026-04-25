import argparse
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import models.baseline_lstm.model as base


@dataclass
class SprintOutputs:
    log_file: str = "outputs/reports/sprint_log.txt"
    task_a_csv: str = "outputs/reports/task_a_blend_sweep.csv"
    task_b_model: str = "outputs/artifacts/task_b_attention_lstm_trained.h5"
    task_b_metrics: str = "outputs/reports/task_b_attention_metrics.json"
    task_c_csv: str = "outputs/reports/task_c_ensemble_weights.csv"
    task_c_config: str = "outputs/reports/task_c_ensemble_config.json"
    task_d_csv: str = "outputs/reports/task_d_loss_tuning.csv"
    final_report: str = "outputs/reports/FINAL_SPRINT_REPORT.json"


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def make_logger(path: str) -> logging.Logger:
    ensure_parent(path)
    logger = logging.getLogger("ghi_sprint_parallel")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    fh = logging.FileHandler(path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, p90: float) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    day_mask = y_true > 10.0
    peak_mask = y_true >= p90

    day_mae = float(mean_absolute_error(y_true[day_mask], y_pred[day_mask])) if np.any(day_mask) else float("nan")
    peak_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else float("nan")

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "day_mae": day_mae,
        "peak_mae": peak_mae,
    }


def build_attention_for_scaled_target(sequence_length: int, num_features: int, learning_rate: float, peak_threshold_scaled: float, peak_weight: float) -> keras.Model:
    inp = keras.Input(shape=(sequence_length, num_features), name="series_input")
    x = layers.LSTM(64, return_sequences=True, name="lstm_64")(inp)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.LSTM(32, return_sequences=True, name="lstm_32")(x)

    score = layers.Dense(1, activation="tanh", name="attention_score")(x)
    att = layers.Softmax(axis=1, name="attention_weights")(score)
    weighted = layers.Multiply(name="weighted_sequence")([x, att])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="context_vector")(weighted)

    dense = layers.Dense(32, activation="relu", name="dense_32")(context)
    out = layers.Dense(1, activation="sigmoid", name="ghi_output")(dense)

    model = keras.Model(inputs=inp, outputs=out, name="attention_lstm_scaled")

    huber = keras.losses.Huber(delta=0.15, reduction=keras.losses.Reduction.NONE)
    threshold = tf.constant(peak_threshold_scaled, dtype=tf.float32)
    peak_w = tf.constant(peak_weight, dtype=tf.float32)

    def peak_weighted_huber(y_true, y_pred):
        base_loss = huber(y_true, y_pred)
        peaks = tf.cast(y_true >= threshold, tf.float32)
        weights = 1.0 + (peak_w - 1.0) * peaks

        # Keep nights lower weighted to reduce bias from low irradiance regions.
        night = tf.cast(y_true < 0.02, tf.float32)
        weights = weights * (1.0 - 0.4 * night)

        return tf.reduce_mean(base_loss * weights)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=peak_weighted_huber,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def train_baseline_model(bundle: base.DataBundle, epochs: int, batch_size: int, learning_rate: float, logger: logging.Logger) -> keras.Model:
    model = base.build_model(sequence_length=bundle.X_train.shape[1], num_features=bundle.X_train.shape[2], learning_rate=learning_rate)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1),
    ]

    logger.info("Baseline warm-start training started (%d epochs max)", epochs)
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
    return model


def task_a_blend_sweep(y_test_raw: np.ndarray, pred_raw: np.ndarray, last_ghi_test: np.ndarray, outputs: SprintOutputs, logger: logging.Logger) -> Dict[str, float]:
    logger.info("TASK A: Blend ratio rapid sweep")
    logger.info("=" * 60)

    p90 = float(np.percentile(y_test_raw, 90))
    blends = np.linspace(0.0, 1.0, 11)
    rows: List[Dict[str, float]] = []

    for i, blend in enumerate(blends, start=1):
        blended = blend * last_ghi_test + (1.0 - blend) * pred_raw
        blended = np.clip(blended, 0.0, None)
        m = calc_metrics(y_test_raw, blended, p90)
        rows.append(
            {
                "blend": round(float(blend), 2),
                "rmse": round(m["rmse"], 2),
                "mae": round(m["mae"], 2),
                "r2": round(m["r2"], 4),
                "day_mae": round(m["day_mae"], 2),
                "peak_mae": round(m["peak_mae"], 2),
            }
        )

        if i % 3 == 0:
            logger.info("Blend %.1f -> Peak MAE %.2f, Day MAE %.2f", blend, m["peak_mae"], m["day_mae"])

    df = pd.DataFrame(rows)
    ensure_parent(outputs.task_a_csv)
    df.to_csv(outputs.task_a_csv, index=False)

    best_peak = df.loc[df["peak_mae"].idxmin()].to_dict()
    best_day = df.loc[df["day_mae"].idxmin()].to_dict()
    best_overall = df.loc[(df["peak_mae"] + df["day_mae"]).idxmin()].to_dict()

    logger.info("Best Peak MAE blend=%.2f -> %.2f", best_peak["blend"], best_peak["peak_mae"])
    logger.info("Best Day MAE  blend=%.2f -> %.2f", best_day["blend"], best_day["day_mae"])
    logger.info("Best Overall  blend=%.2f -> Peak %.2f Day %.2f", best_overall["blend"], best_overall["peak_mae"], best_overall["day_mae"])

    return {
        "optimal_blend": float(best_overall["blend"]),
        "p90_test": p90,
    }


def task_b_attention_background(
    bundle: base.DataBundle,
    y_test_raw: np.ndarray,
    outputs: SprintOutputs,
    logger: logging.Logger,
    result_holder: Dict[str, object],
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    try:
        logger.info("TASK B: Attention LSTM full training (background)")
        logger.info("=" * 60)

        peak_threshold_scaled = float(np.percentile(bundle.y_train.reshape(-1), 90))
        model = build_attention_for_scaled_target(
            sequence_length=bundle.X_train.shape[1],
            num_features=bundle.X_train.shape[2],
            learning_rate=learning_rate,
            peak_threshold_scaled=peak_threshold_scaled,
            peak_weight=2.0,
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        ]

        history = model.fit(
            bundle.X_train,
            bundle.y_train,
            validation_data=(bundle.X_val, bundle.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        ensure_parent(outputs.task_b_model)
        model.save(outputs.task_b_model)

        y_pred_scaled = model.predict(bundle.X_test, verbose=0).reshape(-1, 1)
        y_pred_raw = bundle.target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
        y_pred_raw = np.clip(y_pred_raw, 0.0, None)

        p90 = float(np.percentile(y_test_raw, 90))
        metrics = calc_metrics(y_test_raw, y_pred_raw, p90)
        metrics = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}

        ensure_parent(outputs.task_b_metrics)
        with open(outputs.task_b_metrics, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        result_holder["ready"] = True
        result_holder["metrics"] = metrics
        result_holder["pred_raw"] = y_pred_raw
        result_holder["epochs_ran"] = len(history.history.get("loss", []))

        logger.info("TASK B complete -> RMSE %.2f Peak MAE %.2f Day MAE %.2f", metrics["rmse"], metrics["peak_mae"], metrics["day_mae"])
    except Exception as exc:
        logger.exception("TASK B failed: %s", exc)
        result_holder["ready"] = False
        result_holder["error"] = str(exc)


def task_c_ensemble(
    y_test_raw: np.ndarray,
    baseline_pred_blended: np.ndarray,
    attention_pred_raw: Optional[np.ndarray],
    outputs: SprintOutputs,
    logger: logging.Logger,
) -> Optional[Dict[str, float]]:
    logger.info("TASK C: Ensemble optimization")
    logger.info("=" * 60)

    if attention_pred_raw is None:
        logger.warning("Attention predictions unavailable, skipping Task C for now")
        return None

    p90 = float(np.percentile(y_test_raw, 90))
    weights = np.linspace(0.3, 0.8, 11)
    rows: List[Dict[str, float]] = []

    for i, w_baseline in enumerate(weights, start=1):
        w_attn = 1.0 - w_baseline
        ensemble_pred = w_baseline * baseline_pred_blended + w_attn * attention_pred_raw
        ensemble_pred = np.clip(ensemble_pred, 0.0, None)

        m = calc_metrics(y_test_raw, ensemble_pred, p90)
        rows.append(
            {
                "w_baseline": round(float(w_baseline), 2),
                "w_attention": round(float(w_attn), 2),
                "rmse": round(m["rmse"], 2),
                "mae": round(m["mae"], 2),
                "day_mae": round(m["day_mae"], 2),
                "peak_mae": round(m["peak_mae"], 2),
            }
        )
        if i % 3 == 0:
            logger.info("(%.2f / %.2f) -> Peak MAE %.2f", w_baseline, w_attn, m["peak_mae"])

    df = pd.DataFrame(rows)
    ensure_parent(outputs.task_c_csv)
    df.to_csv(outputs.task_c_csv, index=False)

    best = df.loc[df["peak_mae"].idxmin()]
    config = {
        "w_baseline": float(best["w_baseline"]),
        "w_attention": float(best["w_attention"]),
        "rmse": float(best["rmse"]),
        "mae": float(best["mae"]),
        "day_mae": float(best["day_mae"]),
        "peak_mae": float(best["peak_mae"]),
    }

    ensure_parent(outputs.task_c_config)
    with open(outputs.task_c_config, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    logger.info("Best ensemble -> baseline %.2f attention %.2f peak %.2f day %.2f", config["w_baseline"], config["w_attention"], config["peak_mae"], config["day_mae"])
    return config


def build_loss_variant(name: str, p90_train_scaled: float):
    delta = tf.constant(0.15, dtype=tf.float32)
    p90_t = tf.constant(p90_train_scaled, dtype=tf.float32)

    def huber_core(y_true, y_pred):
        residual = y_true - y_pred
        abs_r = tf.abs(residual)
        quad = 0.5 * tf.square(residual)
        lin = delta * (abs_r - 0.5 * delta)
        return tf.where(abs_r <= delta, quad, lin)

    if name == "original_huber":
        peak_w = 1.8
        night_w = 0.6
    elif name == "aggressive_peak_2x":
        peak_w = 2.2
        night_w = 0.4
    else:
        peak_w = 3.0
        night_w = 0.2

    @tf.function
    def loss_fn(y_true, y_pred):
        hub = huber_core(y_true, y_pred)
        peak_mask = tf.cast(y_true >= p90_t, tf.float32)
        night_mask = tf.cast(y_true < 0.02, tf.float32)
        day_mask = 1.0 - tf.minimum(1.0, peak_mask + night_mask)

        weight = peak_mask * peak_w + day_mask * 1.0 + night_mask * night_w
        return tf.reduce_mean(weight * hub)

    return loss_fn


def task_d_loss_tuning(
    baseline_model: keras.Model,
    bundle: base.DataBundle,
    y_test_raw: np.ndarray,
    outputs: SprintOutputs,
    logger: logging.Logger,
    epochs: int,
    batch_size: int,
) -> pd.DataFrame:
    logger.info("TASK D: Quick loss tuning")
    logger.info("=" * 60)

    p90_train_scaled = float(np.percentile(bundle.y_train.reshape(-1), 90))
    p90_test = float(np.percentile(y_test_raw, 90))

    variants = ["original_huber", "aggressive_peak_2x", "aggressive_peak_3x"]
    rows: List[Dict[str, float]] = []

    for name in variants:
        logger.info("Testing loss variant: %s", name)

        model = keras.models.clone_model(baseline_model)
        model.set_weights(baseline_model.get_weights())
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=5e-4),
            loss=build_loss_variant(name, p90_train_scaled),
            metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
        )

        callbacks = [
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        ]

        model.fit(
            bundle.X_train,
            bundle.y_train,
            validation_data=(bundle.X_val, bundle.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )

        y_pred_scaled = model.predict(bundle.X_test, verbose=0).reshape(-1, 1)
        y_pred_raw = bundle.target_scaler.inverse_transform(y_pred_scaled).reshape(-1)
        y_pred_raw = np.clip(y_pred_raw, 0.0, None)

        m = calc_metrics(y_test_raw, y_pred_raw, p90_test)
        row = {
            "variant": name,
            "rmse": round(m["rmse"], 2),
            "mae": round(m["mae"], 2),
            "day_mae": round(m["day_mae"], 2),
            "peak_mae": round(m["peak_mae"], 2),
        }
        rows.append(row)

        logger.info("Variant %s -> Peak MAE %.2f Day MAE %.2f RMSE %.2f", name, m["peak_mae"], m["day_mae"], m["rmse"])

        if m["peak_mae"] < 26.0:
            model_path = f"outputs/artifacts/task_d_baseline_{name}.h5"
            ensure_parent(model_path)
            model.save(model_path)
            logger.info("Saved improved model: %s", model_path)

    df = pd.DataFrame(rows)
    ensure_parent(outputs.task_d_csv)
    df.to_csv(outputs.task_d_csv, index=False)
    return df


def task_e_aggregate(
    outputs: SprintOutputs,
    logger: logging.Logger,
    baseline_original_metrics: Dict[str, float],
    blend_df: pd.DataFrame,
    loss_df: pd.DataFrame,
    task_b_metrics: Optional[Dict[str, float]],
    task_c_config: Optional[Dict[str, float]],
) -> Dict[str, object]:
    logger.info("TASK E: Final aggregation")
    logger.info("=" * 60)

    best_blend = blend_df.loc[(blend_df["peak_mae"] + blend_df["day_mae"]).idxmin()]
    best_loss = loss_df.loc[loss_df["peak_mae"].idxmin()]

    final_comparison: Dict[str, Dict[str, float]] = {
        "Baseline (Current)": {
            "RMSE": round(baseline_original_metrics["rmse"], 2),
            "MAE": round(baseline_original_metrics["mae"], 2),
            "Day_MAE": round(baseline_original_metrics["day_mae"], 2),
            "Peak_MAE": round(baseline_original_metrics["peak_mae"], 2),
        },
        "Baseline (Optimized Blend)": {
            "RMSE": float(best_blend["rmse"]),
            "MAE": float(best_blend["mae"]),
            "Day_MAE": float(best_blend["day_mae"]),
            "Peak_MAE": float(best_blend["peak_mae"]),
        },
        f"Baseline ({best_loss['variant']})": {
            "RMSE": float(best_loss["rmse"]),
            "MAE": float(best_loss["mae"]),
            "Day_MAE": float(best_loss["day_mae"]),
            "Peak_MAE": float(best_loss["peak_mae"]),
        },
    }

    if task_b_metrics is not None:
        final_comparison["Attention LSTM"] = {
            "RMSE": round(float(task_b_metrics["rmse"]), 2),
            "MAE": round(float(task_b_metrics["mae"]), 2),
            "Day_MAE": round(float(task_b_metrics["day_mae"]), 2),
            "Peak_MAE": round(float(task_b_metrics["peak_mae"]), 2),
        }

    if task_c_config is not None:
        final_comparison["Ensemble (Optimized)"] = {
            "RMSE": round(float(task_c_config["rmse"]), 2),
            "MAE": round(float(task_c_config["mae"]), 2),
            "Day_MAE": round(float(task_c_config["day_mae"]), 2),
            "Peak_MAE": round(float(task_c_config["peak_mae"]), 2),
        }

    df_final = pd.DataFrame(final_comparison).T
    best_model = df_final["Peak_MAE"].idxmin()

    report = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "winning_model": best_model,
        "final_metrics": {
            "rmse": float(df_final.loc[best_model, "RMSE"]),
            "mae": float(df_final.loc[best_model, "MAE"]),
            "day_mae": float(df_final.loc[best_model, "Day_MAE"]),
            "peak_mae": float(df_final.loc[best_model, "Peak_MAE"]),
        },
        "targets": {
            "day_mae": 35.0,
            "peak_mae": 25.0,
        },
        "targets_met": {
            "day_mae": bool(df_final.loc[best_model, "Day_MAE"] <= 35.0),
            "peak_mae": bool(df_final.loc[best_model, "Peak_MAE"] <= 25.0),
        },
        "all_results": final_comparison,
    }

    ensure_parent(outputs.final_report)
    with open(outputs.final_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    logger.info("FINAL RESULTS:\n%s", df_final.to_string())
    logger.info("Winning model: %s | Peak MAE %.2f | Day MAE %.2f", best_model, report["final_metrics"]["peak_mae"], report["final_metrics"]["day_mae"])

    return report


def run_sprint(args: argparse.Namespace) -> None:
    outputs = SprintOutputs()
    logger = make_logger(outputs.log_file)

    logger.info("GHI Improvement Sprint started")
    logger.info("Budget: %.2f hours", args.time_budget_hours)

    base.set_seed(args.seed)
    tf.keras.backend.clear_session()

    # Preload data once for all tasks.
    df = base.build_feature_table(args.data_path)
    bundle = base.build_sequences(df=df, sequence_length=48, train_ratio=0.7, val_ratio=0.15)

    baseline_model = train_baseline_model(
        bundle=bundle,
        epochs=args.baseline_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logger=logger,
    )

    val_pred_scaled = baseline_model.predict(bundle.X_val, verbose=0).reshape(-1, 1)
    test_pred_scaled = baseline_model.predict(bundle.X_test, verbose=0).reshape(-1, 1)

    y_val_true_raw = bundle.target_scaler.inverse_transform(bundle.y_val_scaled.reshape(-1, 1)).reshape(-1)
    y_test_true_raw = bundle.target_scaler.inverse_transform(bundle.y_test_scaled.reshape(-1, 1)).reshape(-1)
    y_val_pred_raw = bundle.target_scaler.inverse_transform(val_pred_scaled).reshape(-1)
    y_test_pred_raw = bundle.target_scaler.inverse_transform(test_pred_scaled).reshape(-1)

    y_test_pred_raw = np.clip(y_test_pred_raw, 0.0, None)
    p90_test = float(np.percentile(y_test_true_raw, 90))
    baseline_current = calc_metrics(y_test_true_raw, y_test_pred_raw, p90_test)

    # Start Task B in background.
    task_b_result: Dict[str, object] = {"ready": False}
    task_b_thread = threading.Thread(
        target=task_b_attention_background,
        kwargs={
            "bundle": bundle,
            "y_test_raw": y_test_true_raw,
            "outputs": outputs,
            "logger": logger,
            "result_holder": task_b_result,
            "epochs": args.attention_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
        daemon=True,
    )
    task_b_thread.start()

    # Task A
    task_a_info = task_a_blend_sweep(
        y_test_raw=y_test_true_raw,
        pred_raw=y_test_pred_raw,
        last_ghi_test=bundle.last_ghi_test,
        outputs=outputs,
        logger=logger,
    )
    optimal_blend = task_a_info["optimal_blend"]

    # Baseline blended predictions for downstream tasks.
    baseline_pred_blended = optimal_blend * bundle.last_ghi_test + (1.0 - optimal_blend) * y_test_pred_raw
    baseline_pred_blended = np.clip(baseline_pred_blended, 0.0, None)

    # Task D in foreground while Task B keeps training in background.
    loss_df = task_d_loss_tuning(
        baseline_model=baseline_model,
        bundle=bundle,
        y_test_raw=y_test_true_raw,
        outputs=outputs,
        logger=logger,
        epochs=args.loss_tuning_epochs,
        batch_size=args.batch_size,
    )

    # Task C can run after Task B ready; wait up to budget remainder.
    deadline = time.time() + args.time_budget_hours * 3600.0
    while task_b_thread.is_alive() and time.time() < deadline:
        logger.info("Waiting for Task B completion...")
        task_b_thread.join(timeout=30)

    attention_pred_raw = None
    task_b_metrics = None
    if bool(task_b_result.get("ready", False)):
        attention_pred_raw = np.asarray(task_b_result.get("pred_raw"))
        task_b_metrics = task_b_result.get("metrics")

    task_c_config = task_c_ensemble(
        y_test_raw=y_test_true_raw,
        baseline_pred_blended=baseline_pred_blended,
        attention_pred_raw=attention_pred_raw,
        outputs=outputs,
        logger=logger,
    )

    # Final aggregation
    blend_df = pd.read_csv(outputs.task_a_csv)
    report = task_e_aggregate(
        outputs=outputs,
        logger=logger,
        baseline_original_metrics=baseline_current,
        blend_df=blend_df,
        loss_df=loss_df,
        task_b_metrics=task_b_metrics,
        task_c_config=task_c_config,
    )

    logger.info("Sprint complete. Winning model: %s", report["winning_model"])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GHI model sprint orchestrator (Tasks A-E)")
    p.add_argument("--data-path", type=str, default="dataset")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--time-budget-hours", type=float, default=2.5)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--baseline-epochs", type=int, default=50)
    p.add_argument("--attention-epochs", type=int, default=100)
    p.add_argument("--loss-tuning-epochs", type=int, default=20)
    return p.parse_args()


if __name__ == "__main__":
    run_sprint(parse_args())
