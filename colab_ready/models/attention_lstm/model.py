import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


def _ensure_parent_dir(file_path: str) -> None:
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _find_column(df: pd.DataFrame, aliases: List[str]) -> str:
    norm_map = {_normalize_col(c): c for c in df.columns}
    for alias in aliases:
        key = _normalize_col(alias)
        if key in norm_map:
            return norm_map[key]
    return ""


def _load_single_csv(path: str) -> pd.DataFrame:
    # Supports both NSRDB-style (metadata rows) and regular CSVs.
    try:
        raw = pd.read_csv(path)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV {path}: {exc}") from exc

    # NSRDB files often need skiprows=2 to reach data headers.
    if "Year" not in raw.columns and "Month" not in raw.columns and "Day" not in raw.columns:
        try:
            raw2 = pd.read_csv(path, skiprows=2)
            if "Year" in raw2.columns and "Month" in raw2.columns and "Day" in raw2.columns:
                raw = raw2
        except Exception:
            pass

    return raw


def load_time_series(path: str) -> pd.DataFrame:
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not files:
            raise FileNotFoundError(f"No CSV files found in: {path}")
    else:
        files = [path]

    frames: List[pd.DataFrame] = []
    for file_path in files:
        frames.append(_load_single_csv(file_path))

    raw = pd.concat(frames, ignore_index=True)

    ghi_col = _find_column(raw, ["GHI", "Global Horizontal Irradiance", "ghi"])
    if not ghi_col:
        raise ValueError("Missing GHI column. Expected one of: GHI, Global Horizontal Irradiance, ghi")

    if all(c in raw.columns for c in ["Year", "Month", "Day", "Hour", "Minute"]):
        timestamp = pd.to_datetime(
            {
                "year": raw["Year"],
                "month": raw["Month"],
                "day": raw["Day"],
                "hour": raw["Hour"],
                "minute": raw["Minute"],
            },
            errors="coerce",
        )
    else:
        dt_col = _find_column(raw, ["datetime", "timestamp", "date time", "date"])
        if not dt_col:
            raise ValueError(
                "Missing datetime columns. Provide either Year/Month/Day/Hour/Minute or a datetime/timestamp column."
            )
        timestamp = pd.to_datetime(raw[dt_col], errors="coerce")

    df = pd.DataFrame(
        {
            "timestamp": timestamp,
            "ghi": pd.to_numeric(raw[ghi_col], errors="coerce"),
        }
    )

    df = df.dropna(subset=["timestamp", "ghi"]).sort_values("timestamp").reset_index(drop=True)

    # Keep one row per timestamp if duplicates exist.
    df = df.groupby("timestamp", as_index=False)["ghi"].mean()

    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = df.copy()
    ts = feat["timestamp"]

    hour = ts.dt.hour.astype(float)
    doy = ts.dt.dayofyear.astype(float)

    feat["sin_hour"] = np.sin(2.0 * np.pi * hour / 24.0)
    feat["cos_hour"] = np.cos(2.0 * np.pi * hour / 24.0)
    feat["sin_doy"] = np.sin(2.0 * np.pi * doy / 365.25)
    feat["cos_doy"] = np.cos(2.0 * np.pi * doy / 365.25)

    for lag in range(1, 25):
        feat[f"ghi_lag_{lag}"] = feat["ghi"].shift(lag)

    for window in [3, 6, 12]:
        feat[f"ghi_roll_mean_{window}"] = feat["ghi"].rolling(window=window).mean()

    feat = feat.dropna().reset_index(drop=True)
    return feat


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    y_test_raw: np.ndarray
    test_timestamps: np.ndarray
    x_scaler: StandardScaler
    y_scaler: StandardScaler
    train_peak_threshold_raw: float


def prepare_sequences(
    feat_df: pd.DataFrame,
    sequence_length: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> DataBundle:
    feature_cols = [
        "ghi",
        "sin_hour",
        "cos_hour",
        "sin_doy",
        "cos_doy",
        *[f"ghi_lag_{i}" for i in range(1, 25)],
        "ghi_roll_mean_3",
        "ghi_roll_mean_6",
        "ghi_roll_mean_12",
    ]

    values_x = feat_df[feature_cols].to_numpy(dtype=np.float32)
    values_y_raw = feat_df["ghi"].to_numpy(dtype=np.float32)
    timestamps = feat_df["timestamp"].to_numpy()

    n = len(feat_df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= sequence_length + 10 or val_end <= train_end + 10 or n <= val_end + 10:
        raise ValueError("Not enough samples for requested sequence length and split ratios.")

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_scaler.fit(values_x[:train_end])
    y_scaler.fit(values_y_raw[:train_end].reshape(-1, 1))

    x_scaled = x_scaler.transform(values_x).astype(np.float32)
    y_scaled = y_scaler.transform(values_y_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    x_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    y_raw_seq: List[float] = []
    ts_seq: List[np.datetime64] = []
    target_indices: List[int] = []

    for idx in range(sequence_length, n):
        x_seq.append(x_scaled[idx - sequence_length:idx, :])
        y_seq.append(float(y_scaled[idx]))
        y_raw_seq.append(float(values_y_raw[idx]))
        ts_seq.append(timestamps[idx])
        target_indices.append(idx)

    X = np.asarray(x_seq, dtype=np.float32)
    y = np.asarray(y_seq, dtype=np.float32).reshape(-1, 1)
    y_raw = np.asarray(y_raw_seq, dtype=np.float32)
    ts = np.asarray(ts_seq)
    target_indices_np = np.asarray(target_indices)

    train_mask = target_indices_np < train_end
    val_mask = (target_indices_np >= train_end) & (target_indices_np < val_end)
    test_mask = target_indices_np >= val_end

    y_train_raw = y_raw[train_mask]
    train_peak_threshold_raw = float(np.percentile(y_train_raw, 90))

    return DataBundle(
        X_train=X[train_mask],
        y_train=y[train_mask],
        X_val=X[val_mask],
        y_val=y[val_mask],
        X_test=X[test_mask],
        y_test=y[test_mask],
        y_test_raw=y_raw[test_mask],
        test_timestamps=ts[test_mask],
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        train_peak_threshold_raw=train_peak_threshold_raw,
    )


@tf.keras.utils.register_keras_serializable()
def _identity(x):
    return x


def build_attention_lstm(
    sequence_length: int,
    num_features: int,
    learning_rate: float,
    peak_threshold_scaled: float,
    peak_weight: float,
    use_peak_weighted_loss: bool,
) -> Tuple[keras.Model, keras.Model]:
    inputs = keras.Input(shape=(sequence_length, num_features), name="series_input")

    x = layers.LSTM(64, return_sequences=True, name="lstm_64")(inputs)
    x = layers.Dropout(0.2, name="dropout_1")(x)
    x = layers.LSTM(32, return_sequences=True, name="lstm_32")(x)

    score = layers.Dense(1, activation="tanh", name="attention_score")(x)
    attention_weights = layers.Softmax(axis=1, name="attention_weights")(score)

    weighted_sequence = layers.Multiply(name="weighted_sequence")([x, attention_weights])
    context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="context_vector")(weighted_sequence)

    dense = layers.Dense(32, activation="relu", name="dense_32")(context)
    output = layers.Dense(1, name="ghi_output")(dense)

    model = keras.Model(inputs=inputs, outputs=output, name="attention_lstm_ghi")
    attention_model = keras.Model(inputs=inputs, outputs=attention_weights, name="attention_extractor")

    if use_peak_weighted_loss:
        huber = keras.losses.Huber(delta=1.0, reduction=keras.losses.Reduction.NONE)
        threshold = tf.constant(peak_threshold_scaled, dtype=tf.float32)
        peak_w = tf.constant(peak_weight, dtype=tf.float32)

        def weighted_huber(y_true, y_pred):
            base = huber(y_true, y_pred)
            peaks = tf.cast(y_true >= threshold, tf.float32)
            weights = 1.0 + (peak_w - 1.0) * peaks
            return tf.reduce_mean(base * weights)

        loss_fn = weighted_huber
    else:
        loss_fn = keras.losses.Huber(delta=1.0)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss_fn,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )

    return model, attention_model


def inverse_transform_y(y_scaled: np.ndarray, y_scaler: StandardScaler) -> np.ndarray:
    return y_scaler.inverse_transform(y_scaled.reshape(-1, 1)).reshape(-1)


def evaluate_predictions(
    y_true_raw: np.ndarray,
    y_pred_raw: np.ndarray,
    peak_threshold_raw: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true_raw, y_pred_raw)))
    metrics["mae"] = float(mean_absolute_error(y_true_raw, y_pred_raw))
    metrics["r2"] = float(r2_score(y_true_raw, y_pred_raw))

    day_mask = y_true_raw > 10.0
    if np.any(day_mask):
        metrics["day_rmse"] = float(np.sqrt(mean_squared_error(y_true_raw[day_mask], y_pred_raw[day_mask])))
        metrics["day_mae"] = float(mean_absolute_error(y_true_raw[day_mask], y_pred_raw[day_mask]))
    else:
        metrics["day_rmse"] = float("nan")
        metrics["day_mae"] = float("nan")

    peak_mask = y_true_raw >= peak_threshold_raw
    if np.any(peak_mask):
        metrics["peak_rmse"] = float(np.sqrt(mean_squared_error(y_true_raw[peak_mask], y_pred_raw[peak_mask])))
        metrics["peak_mae"] = float(mean_absolute_error(y_true_raw[peak_mask], y_pred_raw[peak_mask]))
    else:
        metrics["peak_rmse"] = float("nan")
        metrics["peak_mae"] = float("nan")

    return metrics


def plot_actual_vs_predicted(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    max_points: int = 720,
    show_plot: bool = True,
) -> None:
    points = min(len(y_true), max_points)
    ts = pd.to_datetime(timestamps[:points])

    _ensure_parent_dir(output_path)
    plt.figure(figsize=(14, 5))
    plt.plot(ts, y_true[:points], label="Actual GHI", linewidth=1.8)
    plt.plot(ts, y_pred[:points], label="Predicted GHI", linewidth=1.8)
    plt.title("Actual vs Predicted GHI")
    plt.xlabel("Timestamp")
    plt.ylabel("GHI (W/m^2)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_attention_weights(
    attention_weights: np.ndarray,
    output_path: str,
    show_plot: bool = True,
) -> None:
    # attention_weights expected shape: (1, timesteps, 1)
    weights = attention_weights[0, :, 0]
    steps = np.arange(1, len(weights) + 1)

    _ensure_parent_dir(output_path)
    plt.figure(figsize=(10, 4))
    plt.plot(steps, weights, marker="o", linewidth=1.5)
    plt.title("Attention Weights Across Input Timesteps")
    plt.xlabel("Timestep (1 = oldest, 24 = latest)")
    plt.ylabel("Attention Weight")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()


def run_training(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    tf.keras.backend.clear_session()

    df = load_time_series(args.data_path)
    feat_df = create_features(df)
    bundle = prepare_sequences(
        feat_df,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    peak_threshold_scaled = float(bundle.y_scaler.transform([[bundle.train_peak_threshold_raw]])[0, 0])

    model, attention_model = build_attention_lstm(
        sequence_length=args.sequence_length,
        num_features=bundle.X_train.shape[-1],
        learning_rate=args.learning_rate,
        peak_threshold_scaled=peak_threshold_scaled,
        peak_weight=args.peak_weight,
        use_peak_weighted_loss=(not args.disable_peak_weight),
    )

    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.early_stopping_patience,
            restore_best_weights=True,
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
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred_scaled = model.predict(bundle.X_test, batch_size=args.batch_size, verbose=0).reshape(-1)
    y_true_scaled = bundle.y_test.reshape(-1)

    y_pred_raw = inverse_transform_y(y_pred_scaled, bundle.y_scaler)
    y_true_raw = inverse_transform_y(y_true_scaled, bundle.y_scaler)

    # Physically valid GHI floor.
    y_pred_raw = np.clip(y_pred_raw, 0.0, None)

    metrics = evaluate_predictions(
        y_true_raw=y_true_raw,
        y_pred_raw=y_pred_raw,
        peak_threshold_raw=bundle.train_peak_threshold_raw,
    )

    print("\n===== OUTPUT CHECKS =====")
    print(f"Pred min/max: {y_pred_raw.min():.3f} / {y_pred_raw.max():.3f} W/m^2")
    print(f"True min/max: {y_true_raw.min():.3f} / {y_true_raw.max():.3f} W/m^2")

    print("\n===== EVALUATION METRICS =====")
    print(f"RMSE: {metrics['rmse']:.3f} W/m^2")
    print(f"MAE : {metrics['mae']:.3f} W/m^2")
    print(f"R^2 : {metrics['r2']:.4f}")
    print(f"Day RMSE (GHI > 10): {metrics['day_rmse']:.3f} W/m^2")
    print(f"Day MAE  (GHI > 10): {metrics['day_mae']:.3f} W/m^2")
    print(f"Peak RMSE (Top 10% train): {metrics['peak_rmse']:.3f} W/m^2")
    print(f"Peak MAE  (Top 10% train): {metrics['peak_mae']:.3f} W/m^2")

    show_plot = not args.no_plot

    pred_plot_path = args.prediction_plot
    attn_plot_path = args.attention_plot

    plot_actual_vs_predicted(
        timestamps=bundle.test_timestamps,
        y_true=y_true_raw,
        y_pred=y_pred_raw,
        output_path=pred_plot_path,
        max_points=args.plot_points,
        show_plot=show_plot,
    )
    print(f"Saved prediction plot to: {pred_plot_path}")

    sample_idx = int(np.clip(args.attention_sample_index, 0, len(bundle.X_test) - 1))
    sample_input = bundle.X_test[sample_idx:sample_idx + 1]
    attn_weights = attention_model.predict(sample_input, verbose=0)

    plot_attention_weights(
        attention_weights=attn_weights,
        output_path=attn_plot_path,
        show_plot=show_plot,
    )
    print(f"Saved attention plot to: {attn_plot_path}")

    print("\n===== BRIEF NOTE =====")
    print(
        "Attention helps peak prediction because it learns which timesteps in the last 24 hours are most informative "
        "for the current target, instead of compressing all history uniformly as a standard LSTM does."
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attention-based LSTM for GHI forecasting")
    parser.add_argument("--data-path", type=str, default="dataset", help="CSV file or folder containing CSV files")
    parser.add_argument("--sequence-length", type=int, default=24, help="Input timesteps")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Train split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio")
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs (recommended 50-100)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate")
    parser.add_argument("--peak-weight", type=float, default=2.0, help="Loss weight for peak targets")
    parser.add_argument("--disable-peak-weight", action="store_true", help="Disable peak weighting in Huber loss")
    parser.add_argument("--early-stopping-patience", type=int, default=10, help="EarlyStopping patience")
    parser.add_argument("--prediction-plot", type=str, default="outputs/plots/attention/attention_actual_vs_pred.png", help="Path to save Actual vs Predicted plot")
    parser.add_argument("--attention-plot", type=str, default="outputs/plots/attention/attention_weights.png", help="Path to save attention weights plot")
    parser.add_argument("--plot-points", type=int, default=720, help="Max test points in prediction plot")
    parser.add_argument("--attention-sample-index", type=int, default=0, help="Test sample index for attention visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-plot", action="store_true", help="Save plots only, do not open interactive windows")
    return parser.parse_args()


if __name__ == "__main__":
    run_training(parse_args())
