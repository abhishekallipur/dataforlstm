import argparse
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow import keras
from tensorflow.keras import layers, regularizers

import models.attention_lstm.model as attn
import models.baseline_lstm.model as base


@dataclass
class Tier2Outputs:
    sequence_length_csv: str = "outputs/reports/sequence_length_sweep.csv"
    sequence_length_plot: str = "outputs/plots/sprint/sequence_length_vs_metrics.png"
    best_regularization_json: str = "outputs/reports/best_regularization_config.json"
    regularization_plot: str = "outputs/plots/sprint/regularization_heatmap.png"
    loss_variant_csv: str = "outputs/reports/loss_variant_results.csv"
    loss_variant_plot: str = "outputs/plots/sprint/loss_variant_comparison.png"
    best_loss_config_py: str = "outputs/reports/best_loss_config.py"
    augmentation_csv: str = "outputs/reports/augmentation_results.csv"
    augmentation_plot: str = "outputs/plots/sprint/augmentation_impact.png"
    augmentation_pipeline_py: str = "outputs/reports/data_augmentation.py"
    log_file: str = "outputs/reports/tier2_log.txt"


def ensure_parent(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def setup_logger(path: str) -> logging.Logger:
    ensure_parent(path)
    logger = logging.getLogger("tier2")
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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float) -> Dict[str, float]:
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    day_mask = y_true > 10.0
    peak_mask = y_true >= peak_threshold

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "day_rmse": float(np.sqrt(mean_squared_error(y_true[day_mask], y_pred[day_mask]))) if np.any(day_mask) else np.nan,
        "day_mae": float(mean_absolute_error(y_true[day_mask], y_pred[day_mask])) if np.any(day_mask) else np.nan,
        "peak_rmse": float(np.sqrt(mean_squared_error(y_true[peak_mask], y_pred[peak_mask]))) if np.any(peak_mask) else np.nan,
        "peak_mae": float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else np.nan,
    }


def build_baseline_model_custom(sequence_length: int, num_features: int, learning_rate: float, dropout: float, l2_reg: float) -> keras.Model:
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None
    inp = keras.Input(shape=(sequence_length, num_features))
    x = layers.LSTM(64, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(32, return_sequences=False, kernel_regularizer=reg, recurrent_regularizer=reg)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(16, activation="relu", kernel_regularizer=reg)(x)
    out = layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)

    model = keras.Model(inp, out, name="baseline_custom")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=base.peak_weighted_huber_loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def build_attention_model_custom(
    sequence_length: int,
    num_features: int,
    learning_rate: float,
    peak_threshold_scaled: float,
    peak_weight: float,
    dropout: float,
    l2_reg: float,
    loss_variant: str,
) -> keras.Model:
    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    inp = keras.Input(shape=(sequence_length, num_features))
    x = layers.LSTM(64, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(inp)
    x = layers.Dropout(dropout)(x)
    x = layers.LSTM(32, return_sequences=True, kernel_regularizer=reg, recurrent_regularizer=reg)(x)
    score = layers.Dense(1, activation="tanh", kernel_regularizer=reg)(x)
    att = layers.Softmax(axis=1, name="attention_weights")(score)
    weighted = layers.Multiply()([x, att])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1))(weighted)
    d = layers.Dense(32, activation="relu", kernel_regularizer=reg)(context)
    out = layers.Dense(1, kernel_regularizer=reg)(d)

    model = keras.Model(inp, out, name=f"attn_custom_{loss_variant}")

    if loss_variant == "peak_weighted":
        huber = keras.losses.Huber(delta=1.0, reduction=keras.losses.Reduction.NONE)
        threshold = tf.constant(peak_threshold_scaled, dtype=tf.float32)
        p_w = tf.constant(peak_weight, dtype=tf.float32)

        def loss_fn(y_true, y_pred):
            base_loss = huber(y_true, y_pred)
            peak = tf.cast(y_true >= threshold, tf.float32)
            w = 1.0 + (p_w - 1.0) * peak
            return tf.reduce_mean(base_loss * w)

    elif loss_variant == "focal_peak":
        huber = keras.losses.Huber(delta=1.0, reduction=keras.losses.Reduction.NONE)
        threshold = tf.constant(peak_threshold_scaled, dtype=tf.float32)

        def loss_fn(y_true, y_pred):
            err = tf.abs(y_true - y_pred)
            err2 = tf.square(err)
            base_loss = huber(y_true, y_pred)
            high = tf.cast(y_true >= threshold, tf.float32)
            mid = tf.cast((y_true < threshold) & (y_true >= 0.0), tf.float32)
            low = 1.0 - tf.cast(y_true >= 0.0, tf.float32) + tf.cast(y_true < 0.0, tf.float32)
            w = high * (1.0 + err2) + mid * 0.5 + low * 0.1
            return tf.reduce_mean(base_loss * w)

    else:
        loss_fn = keras.losses.Huber(delta=1.0)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def augment_sequences(X: np.ndarray, noise_sigma: float, max_shift: int) -> np.ndarray:
    X_aug = X.copy()

    if noise_sigma > 0:
        # Add noise only on first feature channel (GHI-like feature in current setup).
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=X_aug[:, :, 0].shape).astype(np.float32)
        X_aug[:, :, 0] = X_aug[:, :, 0] + noise

    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=X_aug.shape[0])
        for i, s in enumerate(shifts):
            if s != 0:
                X_aug[i] = np.roll(X_aug[i], shift=s, axis=0)

    return X_aug


def make_callbacks() -> List[keras.callbacks.Callback]:
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=0,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=0,
        ),
    ]


def run_attention_experiment(sequence_length: int, epochs: int, batch_size: int, learning_rate: float, dropout: float, l2_reg: float, loss_variant: str) -> Dict[str, float]:
    df = attn.load_time_series("dataset")
    feat = attn.create_features(df)
    bundle = attn.prepare_sequences(feat, sequence_length=sequence_length, train_ratio=0.7, val_ratio=0.15)

    peak_threshold_scaled = float(bundle.y_scaler.transform([[bundle.train_peak_threshold_raw]])[0, 0])
    model = build_attention_model_custom(
        sequence_length=sequence_length,
        num_features=bundle.X_train.shape[-1],
        learning_rate=learning_rate,
        peak_threshold_scaled=peak_threshold_scaled,
        peak_weight=2.0,
        dropout=dropout,
        l2_reg=l2_reg,
        loss_variant=loss_variant,
    )

    fit_ok = False
    for bs in [batch_size, max(64, batch_size // 2)]:
        try:
            model.fit(
                bundle.X_train,
                bundle.y_train,
                validation_data=(bundle.X_val, bundle.y_val),
                epochs=epochs,
                batch_size=bs,
                shuffle=True,
                callbacks=make_callbacks(),
                verbose=0,
            )
            fit_ok = True
            batch_size = bs
            break
        except tf.errors.ResourceExhaustedError:
            tf.keras.backend.clear_session()
            model = build_attention_model_custom(
                sequence_length=sequence_length,
                num_features=bundle.X_train.shape[-1],
                learning_rate=learning_rate,
                peak_threshold_scaled=peak_threshold_scaled,
                peak_weight=2.0,
                dropout=dropout,
                l2_reg=l2_reg,
                loss_variant=loss_variant,
            )

    if not fit_ok:
        raise RuntimeError("Attention experiment failed after batch-size retries")

    pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1)
    y_true_raw = attn.inverse_transform_y(bundle.y_test.reshape(-1), bundle.y_scaler)
    y_pred_raw = attn.inverse_transform_y(pred_scaled, bundle.y_scaler)
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    metrics = calculate_metrics(y_true_raw, y_pred_raw, bundle.train_peak_threshold_raw)
    metrics["sequence_length"] = sequence_length
    metrics["dropout"] = dropout
    metrics["l2_reg"] = l2_reg
    metrics["loss_variant"] = loss_variant
    return metrics


def run_baseline_experiment(epochs: int, batch_size: int, learning_rate: float, dropout: float, l2_reg: float, aug_noise: float = 0.0, aug_shift: int = 0) -> Dict[str, float]:
    df = base.build_feature_table("dataset")
    bundle = base.build_sequences(df=df, sequence_length=48, train_ratio=0.7, val_ratio=0.15)

    X_train = bundle.X_train
    if aug_noise > 0 or aug_shift > 0:
        X_train = augment_sequences(X_train, noise_sigma=aug_noise, max_shift=aug_shift)

    model = build_baseline_model_custom(
        sequence_length=48,
        num_features=bundle.X_train.shape[-1],
        learning_rate=learning_rate,
        dropout=dropout,
        l2_reg=l2_reg,
    )

    fit_ok = False
    for bs in [batch_size, max(64, batch_size // 2)]:
        try:
            model.fit(
                X_train,
                bundle.y_train,
                validation_data=(bundle.X_val, bundle.y_val),
                epochs=epochs,
                batch_size=bs,
                shuffle=True,
                callbacks=make_callbacks(),
                verbose=0,
            )
            fit_ok = True
            batch_size = bs
            break
        except tf.errors.ResourceExhaustedError:
            tf.keras.backend.clear_session()
            model = build_baseline_model_custom(
                sequence_length=48,
                num_features=bundle.X_train.shape[-1],
                learning_rate=learning_rate,
                dropout=dropout,
                l2_reg=l2_reg,
            )

    if not fit_ok:
        raise RuntimeError("Baseline experiment failed after batch-size retries")

    val_pred_scaled = model.predict(bundle.X_val, batch_size=batch_size, verbose=0).reshape(-1, 1)
    test_pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1, 1)

    y_val_true_raw = bundle.target_scaler.inverse_transform(bundle.y_val_scaled.reshape(-1, 1)).reshape(-1)
    y_test_true_raw = bundle.target_scaler.inverse_transform(bundle.y_test_scaled.reshape(-1, 1)).reshape(-1)
    y_val_pred_raw = bundle.target_scaler.inverse_transform(val_pred_scaled).reshape(-1)
    y_test_pred_raw = bundle.target_scaler.inverse_transform(test_pred_scaled).reshape(-1)

    blend = base._best_blend_from_validation(
        y_val_true_raw=y_val_true_raw,
        y_val_pred_raw=y_val_pred_raw,
        last_ghi_val=bundle.last_ghi_val,
        is_daylight_val=bundle.is_daylight_val,
        blend_min=0.0,
        blend_max=0.5,
        step=0.05,
    )

    y_test_pred_raw = np.clip((1.0 - blend) * y_test_pred_raw + blend * bundle.last_ghi_test, 0.0, None)
    metrics = calculate_metrics(y_test_true_raw, y_test_pred_raw, bundle.peak_threshold_raw)
    metrics["dropout"] = dropout
    metrics["l2_reg"] = l2_reg
    metrics["blend"] = blend
    metrics["aug_noise"] = aug_noise
    metrics["aug_shift"] = aug_shift
    return metrics


def save_plot_lines(df: pd.DataFrame, x_col: str, y_cols: List[str], title: str, out_path: str) -> None:
    ensure_parent(out_path)
    plt.figure(figsize=(9, 5))
    for c in y_cols:
        plt.plot(df[x_col], df[c], marker="o", label=c)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel("Metric")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_heatmap(df: pd.DataFrame, row: str, col: str, val: str, title: str, out_path: str) -> None:
    ensure_parent(out_path)
    pivot = df.pivot_table(index=row, columns=col, values=val, aggfunc="mean")
    plt.figure(figsize=(7, 5))
    plt.imshow(pivot.values, cmap="viridis", aspect="auto")
    plt.xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    plt.yticks(range(len(pivot.index)), [str(r) for r in pivot.index])
    plt.colorbar(label=val)
    plt.xlabel(col)
    plt.ylabel(row)
    plt.title(title)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            plt.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def write_best_loss_config(path: str, best_variant: str) -> None:
    ensure_parent(path)
    content = f'''"""Best Tier 2 loss config selected by automation."""

BEST_LOSS_VARIANT = "{best_variant}"
'''
    Path(path).write_text(content, encoding="utf-8")


def write_augmentation_pipeline(path: str) -> None:
    ensure_parent(path)
    content = '''"""Data augmentation utilities selected by Tier 2 automation."""

import numpy as np


def augment_sequences(X, noise_sigma=0.0, max_shift=0):
    X_aug = X.copy()
    if noise_sigma > 0:
        noise = np.random.normal(0.0, noise_sigma, size=X_aug[:, :, 0].shape)
        X_aug[:, :, 0] = X_aug[:, :, 0] + noise
    if max_shift > 0:
        shifts = np.random.randint(-max_shift, max_shift + 1, size=X_aug.shape[0])
        for i, s in enumerate(shifts):
            if s != 0:
                X_aug[i] = np.roll(X_aug[i], shift=s, axis=0)
    return X_aug
'''
    Path(path).write_text(content, encoding="utf-8")


def within_budget(deadline: float) -> bool:
    return time.time() < deadline


def run_tier2(args: argparse.Namespace) -> None:
    outputs = Tier2Outputs()
    for p in asdict(outputs).values():
        ensure_parent(p)

    logger = setup_logger(outputs.log_file)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    deadline = time.time() + args.time_budget_hours * 3600.0
    logger.info("Tier 2 started with budget %.2f hours", args.time_budget_hours)

    # Task 2.1: sequence length sweep (attention)
    seq_rows = []
    for seq in [24, 48, 72]:
        if not within_budget(deadline):
            logger.warning("Budget reached before sequence sweep finished")
            break
        logger.info("Task 2.1 -> sequence_length=%d", seq)
        try:
            row = run_attention_experiment(
                sequence_length=seq,
                epochs=args.epochs_attention,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dropout=0.2,
                l2_reg=0.0,
                loss_variant="peak_weighted",
            )
            seq_rows.append(row)
        except Exception as exc:
            logger.exception("Sequence experiment failed for %d: %s", seq, exc)

    if seq_rows:
        df_seq = pd.DataFrame(seq_rows)
        df_seq.to_csv(outputs.sequence_length_csv, index=False)
        save_plot_lines(df_seq.sort_values("sequence_length"), "sequence_length", ["rmse", "day_mae", "peak_mae"], "Sequence Length vs Metrics", outputs.sequence_length_plot)
    else:
        df_seq = pd.DataFrame()

    # Task 2.2: regularization tuning (fast budget-aware)
    reg_rows = []
    reg_grid = [(0.2, 0.0), (0.3, 0.001), (0.4, 0.01), (0.3, 0.0)]
    for model_type in ["baseline", "attention"]:
        for dropout, l2v in reg_grid:
            if not within_budget(deadline):
                logger.warning("Budget reached during regularization tuning")
                break
            logger.info("Task 2.2 -> %s dropout=%.1f l2=%s", model_type, dropout, l2v)
            try:
                if model_type == "baseline":
                    row = run_baseline_experiment(
                        epochs=args.epochs_baseline,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        dropout=dropout,
                        l2_reg=l2v,
                    )
                else:
                    row = run_attention_experiment(
                        sequence_length=24,
                        epochs=args.epochs_attention,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        dropout=dropout,
                        l2_reg=l2v,
                        loss_variant="peak_weighted",
                    )
                row["model"] = model_type
                reg_rows.append(row)
            except Exception as exc:
                logger.exception("Regularization experiment failed for %s d=%.2f l2=%s: %s", model_type, dropout, l2v, exc)

    if reg_rows:
        df_reg = pd.DataFrame(reg_rows)
        best_idx = df_reg["day_mae"].idxmin()
        best_cfg = df_reg.loc[best_idx].to_dict()
        Path(outputs.best_regularization_json).write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

        # Heatmap for baseline subset (stable matrix)
        baseline_reg = df_reg[df_reg["model"] == "baseline"]
        if not baseline_reg.empty:
            save_heatmap(baseline_reg, "dropout", "l2_reg", "day_mae", "Baseline Regularization Heatmap (Day MAE)", outputs.regularization_plot)
    else:
        df_reg = pd.DataFrame()

    # Task 2.3: loss refinement (attention)
    loss_rows = []
    for variant in ["huber", "peak_weighted", "focal_peak"]:
        if not within_budget(deadline):
            logger.warning("Budget reached during loss variant sweep")
            break
        logger.info("Task 2.3 -> loss variant %s", variant)
        try:
            row = run_attention_experiment(
                sequence_length=24,
                epochs=args.epochs_attention,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                dropout=0.2,
                l2_reg=0.0,
                loss_variant=variant,
            )
            loss_rows.append(row)
        except Exception as exc:
            logger.exception("Loss experiment failed for variant=%s: %s", variant, exc)

    if loss_rows:
        df_loss = pd.DataFrame(loss_rows)
        df_loss.to_csv(outputs.loss_variant_csv, index=False)
        save_plot_lines(df_loss, "loss_variant", ["rmse", "peak_mae"], "Loss Variant Comparison", outputs.loss_variant_plot)
        best_loss_variant = df_loss.sort_values("peak_mae").iloc[0]["loss_variant"]
        write_best_loss_config(outputs.best_loss_config_py, str(best_loss_variant))

    # Task 2.4: augmentation impact (both models, reduced variants)
    aug_rows = []
    aug_variants = [
        ("no_aug", 0.0, 0),
        ("noise", 0.03, 0),
        ("shift", 0.0, 2),
        ("both", 0.03, 2),
    ]

    for model_type in ["baseline", "attention"]:
        for name, noise, shift in aug_variants:
            if not within_budget(deadline):
                logger.warning("Budget reached during augmentation tests")
                break
            logger.info("Task 2.4 -> %s %s", model_type, name)
            try:
                if model_type == "baseline":
                    row = run_baseline_experiment(
                        epochs=args.epochs_baseline,
                        batch_size=args.batch_size,
                        learning_rate=args.learning_rate,
                        dropout=0.2,
                        l2_reg=0.0,
                        aug_noise=noise,
                        aug_shift=shift,
                    )
                else:
                    # Attention augmentation: apply on train split via temporary replacement
                    df = attn.load_time_series("dataset")
                    feat = attn.create_features(df)
                    bundle = attn.prepare_sequences(feat, sequence_length=24, train_ratio=0.7, val_ratio=0.15)
                    X_train_aug = augment_sequences(bundle.X_train, noise_sigma=noise, max_shift=shift)
                    peak_threshold_scaled = float(bundle.y_scaler.transform([[bundle.train_peak_threshold_raw]])[0, 0])
                    model = build_attention_model_custom(
                        sequence_length=24,
                        num_features=bundle.X_train.shape[-1],
                        learning_rate=args.learning_rate,
                        peak_threshold_scaled=peak_threshold_scaled,
                        peak_weight=2.0,
                        dropout=0.2,
                        l2_reg=0.0,
                        loss_variant="peak_weighted",
                    )

                    fit_ok = False
                    used_bs = args.batch_size
                    for bs in [args.batch_size, max(64, args.batch_size // 2)]:
                        try:
                            model.fit(
                                X_train_aug,
                                bundle.y_train,
                                validation_data=(bundle.X_val, bundle.y_val),
                                epochs=args.epochs_attention,
                                batch_size=bs,
                                shuffle=True,
                                callbacks=make_callbacks(),
                                verbose=0,
                            )
                            fit_ok = True
                            used_bs = bs
                            break
                        except tf.errors.ResourceExhaustedError:
                            tf.keras.backend.clear_session()
                            model = build_attention_model_custom(
                                sequence_length=24,
                                num_features=bundle.X_train.shape[-1],
                                learning_rate=args.learning_rate,
                                peak_threshold_scaled=peak_threshold_scaled,
                                peak_weight=2.0,
                                dropout=0.2,
                                l2_reg=0.0,
                                loss_variant="peak_weighted",
                            )

                    if not fit_ok:
                        raise RuntimeError("Attention augmentation experiment failed after batch-size retries")

                    pred_scaled = model.predict(bundle.X_test, batch_size=used_bs, verbose=0).reshape(-1)
                    y_true_raw = attn.inverse_transform_y(bundle.y_test.reshape(-1), bundle.y_scaler)
                    y_pred_raw = np.clip(attn.inverse_transform_y(pred_scaled, bundle.y_scaler), 0.0, None)
                    row = calculate_metrics(y_true_raw, y_pred_raw, bundle.train_peak_threshold_raw)

                row["model"] = model_type
                row["augmentation"] = name
                row["noise_sigma"] = noise
                row["max_shift"] = shift
                aug_rows.append(row)
            except Exception as exc:
                logger.exception("Augmentation experiment failed for %s %s: %s", model_type, name, exc)

    if aug_rows:
        df_aug = pd.DataFrame(aug_rows)
        df_aug.to_csv(outputs.augmentation_csv, index=False)
        plot_df = df_aug.copy()
        plot_df["model_aug"] = plot_df["model"] + ":" + plot_df["augmentation"]
        save_plot_lines(plot_df, "model_aug", ["rmse", "day_mae", "peak_mae"], "Augmentation Impact", outputs.augmentation_plot)
        write_augmentation_pipeline(outputs.augmentation_pipeline_py)

    logger.info("Tier 2 complete (or budget reached). Outputs written under outputs/reports and outputs/plots")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier 2 automation (budget-aware)")
    p.add_argument("--time-budget-hours", type=float, default=2.5)
    p.add_argument("--epochs-baseline", type=int, default=22)
    p.add_argument("--epochs-attention", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    run_tier2(parse_args())
