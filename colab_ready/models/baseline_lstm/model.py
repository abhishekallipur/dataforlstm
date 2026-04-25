"""
Debugged and refactored GHI forecasting pipeline (TensorFlow/Keras).

This version is designed to prevent collapse-to-zero behavior by:
1) Leakage-safe scaling (fit scalers on train only).
2) Day/night-aware weighting (without breaking contiguous sequences).
3) Peak-aware weighted Huber training.
4) Simple but stronger LSTM architecture (64 -> 32) with non-negative output.
"""

import argparse
import glob
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import tensorflow as tf
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


def _normalize(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _find_column(df: pd.DataFrame, aliases: List[str]) -> str:
    normalized = {_normalize(c): c for c in df.columns}
    for alias in aliases:
        key = _normalize(alias)
        if key in normalized:
            return normalized[key]
    return ""


def _load_single_nsrdb_csv(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    # NSRDB format:
    # row1 metadata headers, row2 metadata values, row3 actual data headers, row4+ data rows.
    metadata_df = pd.read_csv(path, nrows=1)
    metadata = {str(k): str(v) for k, v in metadata_df.iloc[0].to_dict().items()}
    data_df = pd.read_csv(path, skiprows=2)
    return data_df, metadata


def _load_nsrdb(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
    if os.path.isdir(path):
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {path}")
    else:
        csv_files = [path]

    frames = []
    metadata: Dict[str, str] = {}
    for i, csv_file in enumerate(csv_files):
        df, meta = _load_single_nsrdb_csv(csv_file)
        frames.append(df)
        if i == 0:
            metadata = meta

    combined = pd.concat(frames, ignore_index=True)
    return combined, metadata


def _compute_zenith_from_timestamp(
    timestamps: pd.Series, latitude: float, longitude: float, timezone_offset_hours: float
) -> np.ndarray:
    # NOAA solar position approximation (good enough for model features).
    day_of_year = timestamps.dt.dayofyear.to_numpy(dtype=np.float64)
    hour = timestamps.dt.hour.to_numpy(dtype=np.float64)
    minute = timestamps.dt.minute.to_numpy(dtype=np.float64)

    gamma = 2.0 * np.pi / 365.0 * (day_of_year - 1.0 + (hour - 12.0) / 24.0)
    decl = (
        0.006918
        - 0.399912 * np.cos(gamma)
        + 0.070257 * np.sin(gamma)
        - 0.006758 * np.cos(2 * gamma)
        + 0.000907 * np.sin(2 * gamma)
        - 0.002697 * np.cos(3 * gamma)
        + 0.00148 * np.sin(3 * gamma)
    )
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * np.cos(gamma)
        - 0.032077 * np.sin(gamma)
        - 0.014615 * np.cos(2 * gamma)
        - 0.040849 * np.sin(2 * gamma)
    )

    time_offset = eqtime + 4.0 * longitude - 60.0 * timezone_offset_hours
    true_solar_time = (hour * 60.0 + minute + time_offset) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0
    hour_angle[hour_angle < -180.0] += 360.0

    lat_rad = np.deg2rad(latitude)
    ha_rad = np.deg2rad(hour_angle)
    cos_zenith = (
        np.sin(lat_rad) * np.sin(decl)
        + np.cos(lat_rad) * np.cos(decl) * np.cos(ha_rad)
    )
    cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_zenith))


def build_feature_table(path: str) -> pd.DataFrame:
    raw, metadata = _load_nsrdb(path)

    aliases = {
        "ghi": ["GHI", "Global Horizontal Irradiance"],
        "temp": ["Temperature", "Air Temperature", "Temp"],
        "rh": ["Relative Humidity", "RH", "Humidity"],
        "wind": ["Wind Speed", "WindSpeed"],
        "pressure": ["Pressure", "Surface Pressure", "Atmospheric Pressure"],
        "zenith": ["Solar Zenith Angle", "Zenith Angle", "Solar Zenith"],
    }

    selected = {}
    for name, options in aliases.items():
        selected[name] = _find_column(raw, options)
        if not selected[name] and name != "zenith":
            raise ValueError(f"Missing required column '{name}'. Available columns: {list(raw.columns)}")

    time_cols = ["Year", "Month", "Day", "Hour", "Minute"]
    if not all(col in raw.columns for col in time_cols):
        raise ValueError("NSRDB data must include Year, Month, Day, Hour, Minute.")

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
    if timestamp.isna().any():
        raise ValueError("Failed to parse timestamps from Year/Month/Day/Hour/Minute.")

    df = pd.DataFrame({"timestamp": timestamp})
    df["ghi"] = pd.to_numeric(raw[selected["ghi"]], errors="coerce")
    df["temperature"] = pd.to_numeric(raw[selected["temp"]], errors="coerce")
    df["relative_humidity"] = pd.to_numeric(raw[selected["rh"]], errors="coerce")
    df["wind_speed"] = pd.to_numeric(raw[selected["wind"]], errors="coerce")
    df["pressure"] = pd.to_numeric(raw[selected["pressure"]], errors="coerce")

    if selected["zenith"]:
        df["solar_zenith_angle"] = pd.to_numeric(raw[selected["zenith"]], errors="coerce")
    else:
        lat = float(metadata["Latitude"])
        lon = float(metadata["Longitude"])
        tz = float(metadata["Local Time Zone"])
        df["solar_zenith_angle"] = _compute_zenith_from_timestamp(df["timestamp"], lat, lon, tz)

    # sort to guarantee chronology after concatenation
    df = df.sort_values("timestamp").reset_index(drop=True)

    # cyclical time features
    hour = df["timestamp"].dt.hour.astype(float)
    day_of_year = df["timestamp"].dt.dayofyear.astype(float)
    df["hour"] = hour
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24.0)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24.0)
    df["sin_doy"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_doy"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # day/night features
    cos_zenith = np.cos(np.deg2rad(df["solar_zenith_angle"].clip(0, 180)))
    df["cos_zenith"] = np.clip(cos_zenith, 0.0, 1.0)
    df["is_daylight"] = (df["cos_zenith"] > 0.0).astype(float)

    # lag and rolling features
    df["ghi_lag_1"] = df["ghi"].shift(1)
    df["ghi_lag_2"] = df["ghi"].shift(2)
    df["ghi_lag_3"] = df["ghi"].shift(3)
    df["ghi_lag_24"] = df["ghi"].shift(24)
    df["ghi_diff_1"] = df["ghi"] - df["ghi"].shift(1)
    df["ghi_diff_3"] = df["ghi"] - df["ghi"].shift(3)
    df["ghi_roll_mean_3"] = df["ghi"].rolling(window=3).mean()
    df["ghi_roll_mean_6"] = df["ghi"].rolling(window=6).mean()
    df["ghi_roll_mean_24"] = df["ghi"].rolling(window=24).mean()
    df["ghi_roll_std_6"] = df["ghi"].rolling(window=6).std()
    df["ghi_roll_std_24"] = df["ghi"].rolling(window=24).std()

    # fill missing values from lag/rolling and any source gaps
    df = df.ffill().bfill()
    return df


@dataclass
class DataBundle:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    y_val_scaled: np.ndarray
    y_val_raw: np.ndarray
    last_ghi_val: np.ndarray
    is_daylight_val: np.ndarray
    X_test: np.ndarray
    y_test_scaled: np.ndarray
    y_test_raw: np.ndarray
    test_timestamps: np.ndarray
    last_ghi_test: np.ndarray
    is_daylight_test: np.ndarray
    target_scaler: MinMaxScaler
    peak_threshold_raw: float


@tf.keras.utils.register_keras_serializable()
def peak_weighted_huber_loss(y_true, y_pred):
    """
    Shape-safe dynamic weighting computed inside the graph.
    No external sample_weight array required.
    Assumes target is MinMax-scaled to [0, 1].
    """
    huber = tf.keras.losses.Huber(
        delta=0.15, reduction=tf.keras.losses.Reduction.NONE
    )
    base = huber(y_true, y_pred)  # (batch,)

    y_clip = tf.clip_by_value(y_true, 0.0, 1.0)
    y_flat = tf.squeeze(y_clip, axis=-1)  # (batch,)

    # Moderate peak emphasis to avoid over-sharp midday-only spikes.
    weights = 1.0 + 1.8 * tf.pow(y_flat, 1.25)

    # Downweight night/near-zero targets to avoid collapse-to-zero bias.
    night_mask = y_flat < 0.02
    weights = tf.where(night_mask, 0.60, weights)

    return tf.reduce_mean(base * weights)


def build_sequences(
    df: pd.DataFrame,
    sequence_length: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> DataBundle:
    feature_cols = [
        "ghi",
        "temperature",
        "relative_humidity",
        "wind_speed",
        "pressure",
        "solar_zenith_angle",
        "hour",
        "sin_hour",
        "cos_hour",
        "sin_doy",
        "cos_doy",
        "cos_zenith",
        "is_daylight",
        "ghi_lag_1",
        "ghi_lag_2",
        "ghi_lag_3",
        "ghi_lag_24",
        "ghi_diff_1",
        "ghi_diff_3",
        "ghi_roll_mean_3",
        "ghi_roll_mean_6",
        "ghi_roll_mean_24",
        "ghi_roll_std_6",
        "ghi_roll_std_24",
    ]

    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["ghi"].to_numpy(dtype=np.float32)
    is_daylight = df["is_daylight"].to_numpy(dtype=np.float32)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= sequence_length + 10 or val_end <= train_end + 10 or n <= val_end + 10:
        raise ValueError("Not enough rows for requested sequence length and split.")

    # leakage-safe scaling: fit only on train rows
    x_scaler = StandardScaler()
    X_train_rows = X_raw[:train_end]
    x_scaler.fit(X_train_rows)
    X_scaled = x_scaler.transform(X_raw).astype(np.float32)

    target_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler.fit(y_raw[:train_end].reshape(-1, 1))
    y_scaled = target_scaler.transform(y_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    X_seq, y_seq = [], []
    y_raw_seq = []
    day_seq = []
    last_ghi_seq = []
    target_indices = []
    timestamp_values = df["timestamp"].to_numpy()

    for i in range(sequence_length, n):
        X_seq.append(X_scaled[i - sequence_length:i, :])
        y_seq.append(y_scaled[i])
        y_raw_seq.append(y_raw[i])
        day_seq.append(is_daylight[i])
        last_ghi_seq.append(y_raw[i - 1])
        target_indices.append(i)

    X_seq = np.asarray(X_seq, dtype=np.float32)
    y_seq = np.asarray(y_seq, dtype=np.float32).reshape(-1, 1)
    y_raw_seq = np.asarray(y_raw_seq, dtype=np.float32).reshape(-1)
    day_seq = np.asarray(day_seq, dtype=np.float32).reshape(-1)
    last_ghi_seq = np.asarray(last_ghi_seq, dtype=np.float32).reshape(-1)
    target_indices = np.asarray(target_indices)

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    X_train, y_train = X_seq[train_mask], y_seq[train_mask]
    X_val, y_val = X_seq[val_mask], y_seq[val_mask]
    y_val_scaled = y_seq[val_mask]
    y_val_raw = y_raw_seq[val_mask]
    last_ghi_val = last_ghi_seq[val_mask]
    is_daylight_val = day_seq[val_mask]
    X_test, y_test_scaled = X_seq[test_mask], y_seq[test_mask]
    y_test_raw = y_raw_seq[test_mask]
    test_timestamps = timestamp_values[target_indices[test_mask]]
    last_ghi_test = last_ghi_seq[test_mask]
    is_daylight_test = day_seq[test_mask]

    y_train_raw = y_raw_seq[train_mask]
    day_train = day_seq[train_mask]

    day_train_values = y_train_raw[day_train > 0.5]
    if day_train_values.size > 0:
        peak_threshold_raw = float(np.percentile(day_train_values, 90))
    else:
        peak_threshold_raw = float(np.percentile(y_train_raw, 90))

    return DataBundle(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        y_val_scaled=y_val_scaled,
        y_val_raw=y_val_raw,
        last_ghi_val=last_ghi_val,
        is_daylight_val=is_daylight_val,
        X_test=X_test,
        y_test_scaled=y_test_scaled,
        y_test_raw=y_test_raw,
        test_timestamps=test_timestamps,
        last_ghi_test=last_ghi_test,
        is_daylight_test=is_daylight_test,
        target_scaler=target_scaler,
        peak_threshold_raw=peak_threshold_raw
    )


def build_model(sequence_length: int, num_features: int, learning_rate: float) -> keras.Model:
    inputs = keras.Input(shape=(sequence_length, num_features))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=False)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(16, activation="relu")(x)
    # Target is MinMax-scaled to [0, 1], so sigmoid is a stable bounded head.
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="ghi_lstm_refactored")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=peak_weighted_huber_loss,
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold_raw: float) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    daylight_mask = y_true > 10.0
    day_rmse = float(np.sqrt(mean_squared_error(y_true[daylight_mask], y_pred[daylight_mask]))) if np.any(daylight_mask) else np.nan
    day_mae = float(mean_absolute_error(y_true[daylight_mask], y_pred[daylight_mask])) if np.any(daylight_mask) else np.nan

    peak_mask = y_true >= peak_threshold_raw
    peak_mae = float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else np.nan
    peak_rmse = float(np.sqrt(mean_squared_error(y_true[peak_mask], y_pred[peak_mask]))) if np.any(peak_mask) else np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "day_rmse": day_rmse,
        "day_mae": day_mae,
        "peak_mae": peak_mae,
        "peak_rmse": peak_rmse,
    }


def _best_blend_from_validation(
    y_val_true_raw: np.ndarray,
    y_val_pred_raw: np.ndarray,
    last_ghi_val: np.ndarray,
    is_daylight_val: np.ndarray,
    blend_min: float = 0.0,
    blend_max: float = 0.5,
    step: float = 0.05,
) -> float:
    daylight_mask = is_daylight_val > 0.5
    if not np.any(daylight_mask):
        return 0.25

    candidates = np.arange(blend_min, blend_max + 0.5 * step, step)
    best_blend = float(candidates[0])
    best_score = np.inf

    for blend in candidates:
        blended = (1.0 - blend) * y_val_pred_raw + blend * last_ghi_val
        blended = np.clip(blended, 0.0, None)
        score = mean_absolute_error(y_val_true_raw[daylight_mask], blended[daylight_mask])
        if score < best_score:
            best_score = score
            best_blend = float(blend)

    return best_blend


def plot_three_day_window(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # pick a 72-hour window centered around the highest true GHI point in test set.
    peak_idx = int(np.argmax(y_true))
    start = max(0, peak_idx - 24)
    end = min(len(y_true), start + 72)
    start = max(0, end - 72)

    x = np.arange(start, end)
    plt.figure(figsize=(14, 5))
    plt.plot(x, y_true[start:end], label="Actual GHI", linewidth=2.0, color="#1f77b4")
    plt.plot(x, y_pred[start:end], label="Predicted GHI", linewidth=2.0, color="#ff7f0e")
    plt.fill_between(x, y_true[start:end], y_pred[start:end], alpha=0.12, color="#ff7f0e")
    plt.title("3-Day Window Around Peak (72 Hours): Actual vs Predicted")
    plt.xlabel("Test Sample Index (Hourly)")
    plt.ylabel("GHI (W/m^2)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def save_thirty_day_plot(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    prediction_days: int = 30,
) -> None:
    points = min(len(y_true), prediction_days * 24)
    if points <= 0:
        raise ValueError("No samples available to render 30-day plot.")

    ts = pd.to_datetime(timestamps[:points])
    _ensure_parent_dir(output_path)
    plt.figure(figsize=(15, 5))
    plt.plot(ts, y_true[:points], label="Actual GHI", linewidth=1.8, color="#1f77b4")
    plt.plot(ts, y_pred[:points], label="Predicted GHI", linewidth=1.8, color="#ff7f0e")
    plt.title(f"{prediction_days}-Day Forecast Horizon: Actual vs Predicted")
    plt.xlabel("Timestamp")
    plt.ylabel("GHI (W/m^2)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_one_day_plot(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str,
    day_index: int = 0,
) -> None:
    points_per_day = 24
    total_points = len(y_true)
    if total_points <= 0:
        raise ValueError("No samples available to render 1-day plot.")

    start = max(0, day_index * points_per_day)
    if start >= total_points:
        start = max(0, total_points - points_per_day)
    end = min(total_points, start + points_per_day)

    ts = pd.to_datetime(timestamps[start:end])

    _ensure_parent_dir(output_path)
    plt.figure(figsize=(12, 4))
    plt.plot(ts, y_true[start:end], label="Actual GHI", linewidth=2.0, color="#1f77b4")
    plt.plot(ts, y_pred[start:end], label="Predicted GHI", linewidth=2.0, color="#ff7f0e")
    plt.title("1-Day Forecast Horizon: Actual vs Predicted")
    plt.xlabel("Timestamp")
    plt.ylabel("GHI (W/m^2)")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_training_improvement_plot(history: keras.callbacks.History, output_path: str) -> None:
    _ensure_parent_dir(output_path)
    epochs = np.arange(1, len(history.history.get("loss", [])) + 1)
    if len(epochs) == 0:
        return

    train_loss = np.asarray(history.history.get("loss", []), dtype=float)
    val_loss = np.asarray(history.history.get("val_loss", []), dtype=float)
    train_mae = np.asarray(history.history.get("mae", []), dtype=float)
    val_mae = np.asarray(history.history.get("val_mae", []), dtype=float)

    best_val_loss = np.minimum.accumulate(val_loss) if val_loss.size else np.asarray([])

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label="Train Loss", linewidth=2.0)
    if val_loss.size:
        plt.plot(epochs, val_loss, label="Val Loss", linewidth=2.0)
        plt.plot(epochs, best_val_loss, label="Best Val Loss So Far", linewidth=1.8, linestyle="--")
    plt.title("Loss Improvement During Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()

    plt.subplot(1, 2, 2)
    if train_mae.size:
        plt.plot(epochs, train_mae, label="Train MAE", linewidth=2.0)
    if val_mae.size:
        plt.plot(epochs, val_mae, label="Val MAE", linewidth=2.0)
    plt.title("MAE Trend During Training")
    plt.xlabel("Epoch")
    plt.ylabel("MAE")
    plt.grid(True, linestyle="--", alpha=0.3)
    if train_mae.size or val_mae.size:
        plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_pipeline(
    data_path: str,
    sequence_length: int = 48,
    epochs: int = 80,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    persistence_blend: float = -1.0,
    prediction_days: int = 30,
    plot_output_path: str = "outputs/plots/baseline/prediction_30_days.png",
    one_day_plot_output_path: str = "outputs/plots/baseline/prediction_1_day.png",
    training_progress_output_path: str = "outputs/plots/baseline/training_improvement.png",
    one_day_start_index: int = 0,
    seed: int = 42,
    no_plot: bool = False,
) -> None:
    set_seed(seed)
    tf.keras.backend.clear_session()

    df = build_feature_table(data_path)
    bundle = build_sequences(
        df=df,
        sequence_length=sequence_length,
        train_ratio=0.7,
        val_ratio=0.15,
    )

    model = build_model(
        sequence_length=sequence_length,
        num_features=bundle.X_train.shape[-1],
        learning_rate=learning_rate
    )
    model.summary()

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

    val_pred_scaled = model.predict(bundle.X_val, batch_size=batch_size, verbose=0).reshape(-1, 1)
    pred_scaled = model.predict(bundle.X_test, batch_size=batch_size, verbose=0).reshape(-1, 1)
    y_test_scaled = bundle.y_test_scaled.reshape(-1, 1)
    y_val_scaled = bundle.y_val_scaled.reshape(-1, 1)

    # correct inverse-transform with the same train-fitted target scaler
    val_pred_raw = bundle.target_scaler.inverse_transform(val_pred_scaled).reshape(-1)
    y_val_true_raw = bundle.target_scaler.inverse_transform(y_val_scaled).reshape(-1)
    pred_raw = bundle.target_scaler.inverse_transform(pred_scaled).reshape(-1)
    y_true_raw = bundle.target_scaler.inverse_transform(y_test_scaled).reshape(-1)

    # Auto-tune blend on validation when negative value is passed.
    # 0.0 = pure model, 1.0 = pure persistence.
    if persistence_blend < 0.0:
        persistence_blend = _best_blend_from_validation(
            y_val_true_raw=y_val_true_raw,
            y_val_pred_raw=val_pred_raw,
            last_ghi_val=bundle.last_ghi_val,
            is_daylight_val=bundle.is_daylight_val,
            blend_min=0.0,
            blend_max=0.5,
            step=0.05,
        )
    persistence_blend = float(np.clip(persistence_blend, 0.0, 1.0))
    pred_raw = (1.0 - persistence_blend) * pred_raw + persistence_blend * bundle.last_ghi_test

    # enforce physics: no negative irradiance
    pred_raw = np.clip(pred_raw, 0.0, None)

    metrics = evaluate_metrics(y_true_raw, pred_raw, bundle.peak_threshold_raw)

    print("\n===== DEBUG CHECKS ====x=")
    print(f"Pred min/max: {pred_raw.min():.3f} / {pred_raw.max():.3f} W/m^2")
    print(f"True min/max: {y_true_raw.min():.3f} / {y_true_raw.max():.3f} W/m^2")

    print("\n===== TEST METRICS =====")
    print(f"RMSE: {metrics['rmse']:.2f} W/m^2")
    print(f"MAE : {metrics['mae']:.2f} W/m^2")
    print(f"R^2 : {metrics['r2']:.4f}")
    print(f"Day RMSE (GHI>10): {metrics['day_rmse']:.2f} W/m^2")
    print(f"Day MAE  (GHI>10): {metrics['day_mae']:.2f} W/m^2")
    print(f"Peak MAE (>=train daytime P90): {metrics['peak_mae']:.2f} W/m^2")
    print(f"Peak RMSE(>=train daytime P90): {metrics['peak_rmse']:.2f} W/m^2")
    print(f"Persistence blend used: {persistence_blend:.2f}")

    save_thirty_day_plot(
        timestamps=bundle.test_timestamps,
        y_true=y_true_raw,
        y_pred=pred_raw,
        output_path=plot_output_path,
        prediction_days=prediction_days,
    )
    print(f"Saved {prediction_days}-day prediction graph to: {plot_output_path}")

    save_one_day_plot(
        timestamps=bundle.test_timestamps,
        y_true=y_true_raw,
        y_pred=pred_raw,
        output_path=one_day_plot_output_path,
        day_index=one_day_start_index,
    )
    print(f"Saved 1-day prediction graph to: {one_day_plot_output_path}")

    save_training_improvement_plot(history, training_progress_output_path)
    print(f"Saved training improvement graph to: {training_progress_output_path}")

    if not no_plot:
        plot_three_day_window(y_true_raw, pred_raw)

        # optional full-test diagnostic plot
        n = min(len(y_true_raw), 500)
        plt.figure(figsize=(14, 5))
        plt.plot(y_true_raw[:n], label="Actual GHI", linewidth=1.8)
        plt.plot(pred_raw[:n], label="Predicted GHI", linewidth=1.8)
        plt.title("First 500 Test Hours: Actual vs Predicted")
        plt.xlabel("Test Sample Index")
        plt.ylabel("GHI (W/m^2)")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.plot(history.history["loss"], label="Train loss")
        plt.plot(history.history["val_loss"], label="Val loss")
        plt.title("Training Curves (Weighted Huber)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refactored LSTM pipeline for NSRDB GHI forecasting.")
    parser.add_argument("--data-path", type=str, default="dataset", help="Folder or single NSRDB CSV path.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Past hours used to predict next hour.")
    parser.add_argument("--epochs", type=int, default=80, help="Max training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument(
        "--persistence-blend",
        type=float,
        default=-1.0,
        help="Blend weight for persistence baseline. Use negative value for auto-tune on validation.",
    )
    parser.add_argument("--prediction-days", type=int, default=30, help="Number of days to include in saved forecast plot.")
    parser.add_argument("--plot-output", type=str, default="outputs/plots/baseline/prediction_30_days.png", help="File path for saved prediction plot.")
    parser.add_argument("--one-day-plot-output", type=str, default="outputs/plots/baseline/prediction_1_day.png", help="File path for saved 1-day prediction plot.")
    parser.add_argument("--training-progress-plot-output", type=str, default="outputs/plots/baseline/training_improvement.png", help="File path for saved training improvement plot.")
    parser.add_argument("--one-day-start-index", type=int, default=0, help="0-based day index within the test split for 1-day plot.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-plot", action="store_true", help="Disable plotting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        persistence_blend=args.persistence_blend,
        prediction_days=args.prediction_days,
        plot_output_path=args.plot_output,
        one_day_plot_output_path=args.one_day_plot_output,
        training_progress_output_path=args.training_progress_plot_output,
        one_day_start_index=args.one_day_start_index,
        seed=args.seed,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()

