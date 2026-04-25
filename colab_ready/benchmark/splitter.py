"""
Chronological time-series splitter with leakage-safe scaling.

Rules enforced:
  1. Train / val / test are contiguous, non-overlapping time slices.
  2. No random shuffling — ever.
  3. Scalers are fit ONLY on the training partition.
  4. Sequence construction guarantees no future leakage (each window
     ends before the target timestamp).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger("benchmark.splitter")


# ---------------------------------------------------------------------------
# Data bundles
# ---------------------------------------------------------------------------


@dataclass
class TabularBundle:
    """Holds train / val / test arrays for tabular (flat-feature) models."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    train_timestamps: np.ndarray
    val_timestamps: np.ndarray
    test_timestamps: np.ndarray

    feature_scaler: StandardScaler
    target_scaler: MinMaxScaler
    peak_threshold_raw: float

    # Raw (un-scaled) targets for evaluation
    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray

    # Masks for regime / daylight analysis
    is_daylight_test: np.ndarray
    regime_id_test: np.ndarray


@dataclass
class SequenceBundle:
    """Holds train / val / test arrays for sequence models (LSTM, CNN, etc.)."""

    X_train: np.ndarray   # (N_train, seq_len, n_features)
    y_train: np.ndarray   # (N_train, 1) — scaled
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray     # scaled

    train_timestamps: np.ndarray
    val_timestamps: np.ndarray
    test_timestamps: np.ndarray

    feature_scaler: StandardScaler
    target_scaler: MinMaxScaler
    peak_threshold_raw: float

    y_train_raw: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray

    is_daylight_test: np.ndarray
    regime_id_test: np.ndarray

    sequence_length: int
    n_features: int


# ---------------------------------------------------------------------------
# Tabular split
# ---------------------------------------------------------------------------


def build_tabular_bundle(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "ghi",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> TabularBundle:
    """
    Split a chronologically sorted DataFrame into train/val/test for
    tabular models.  Scalers are fit **only on training rows**.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end < 100 or val_end <= train_end + 10 or n <= val_end + 10:
        raise ValueError("Dataset too small for the requested split ratios.")

    # Verify chronological order
    ts = df["timestamp"].values
    assert np.all(ts[1:] >= ts[:-1]), "Timestamps are NOT monotonically increasing."

    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()
    is_daylight = df["is_daylight"].to_numpy(dtype=np.float32)
    regime_id = df["regime_id"].to_numpy(dtype=np.int64) if "regime_id" in df.columns else np.zeros(n, dtype=np.int64)

    # --- Fit scalers on train only ------------------------------------
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_raw[:train_end])
    X_scaled = feature_scaler.transform(X_raw).astype(np.float32)

    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler.fit(y_raw[:train_end].reshape(-1, 1))
    y_scaled = target_scaler.transform(y_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    # --- Peak threshold from daytime training data --------------------
    train_daylight = is_daylight[:train_end] > 0.5
    train_day_ghi = y_raw[:train_end][train_daylight]
    peak_threshold = float(np.percentile(train_day_ghi, 90)) if train_day_ghi.size > 0 else float(np.percentile(y_raw[:train_end], 90))

    logger.info(
        "Tabular split — train: %d  val: %d  test: %d  features: %d  peak_threshold: %.1f",
        train_end, val_end - train_end, n - val_end, len(feature_cols), peak_threshold,
    )

    return TabularBundle(
        X_train=X_scaled[:train_end],
        y_train=y_scaled[:train_end],
        X_val=X_scaled[train_end:val_end],
        y_val=y_scaled[train_end:val_end],
        X_test=X_scaled[val_end:],
        y_test=y_scaled[val_end:],
        train_timestamps=timestamps[:train_end],
        val_timestamps=timestamps[train_end:val_end],
        test_timestamps=timestamps[val_end:],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        peak_threshold_raw=peak_threshold,
        y_train_raw=y_raw[:train_end],
        y_val_raw=y_raw[train_end:val_end],
        y_test_raw=y_raw[val_end:],
        is_daylight_test=is_daylight[val_end:],
        regime_id_test=regime_id[val_end:],
    )


# ---------------------------------------------------------------------------
# Sequence split
# ---------------------------------------------------------------------------


def build_sequence_bundle(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "ghi",
    sequence_length: int = 48,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> SequenceBundle:
    """
    Build overlapping look-back sequences for LSTM / CNN models.
    The target at index *i* uses features from indices [i-seq_len, i).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    if train_end <= sequence_length + 10:
        raise ValueError("Not enough rows for the requested sequence length and split.")

    ts_all = df["timestamp"].to_numpy()
    assert np.all(ts_all[1:] >= ts_all[:-1]), "Timestamps must be monotonically increasing."

    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df[target_col].to_numpy(dtype=np.float32)
    is_daylight = df["is_daylight"].to_numpy(dtype=np.float32)
    regime_id = df["regime_id"].to_numpy(dtype=np.int64) if "regime_id" in df.columns else np.zeros(n, dtype=np.int64)

    # --- Fit scalers on train only ------------------------------------
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_raw[:train_end])
    X_scaled = feature_scaler.transform(X_raw).astype(np.float32)

    target_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    target_scaler.fit(y_raw[:train_end].reshape(-1, 1))
    y_scaled = target_scaler.transform(y_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    # --- Build sequences ----------------------------------------------
    X_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    y_raw_seq: List[float] = []
    ts_seq: List = []
    day_seq: List[float] = []
    regime_seq: List[int] = []
    target_indices: List[int] = []

    for i in range(sequence_length, n):
        X_seq.append(X_scaled[i - sequence_length:i, :])
        y_seq.append(y_scaled[i])
        y_raw_seq.append(y_raw[i])
        ts_seq.append(ts_all[i])
        day_seq.append(is_daylight[i])
        regime_seq.append(int(regime_id[i]))
        target_indices.append(i)

    X_arr = np.asarray(X_seq, dtype=np.float32)
    y_arr = np.asarray(y_seq, dtype=np.float32).reshape(-1, 1)
    y_raw_arr = np.asarray(y_raw_seq, dtype=np.float32)
    ts_arr = np.asarray(ts_seq)
    day_arr = np.asarray(day_seq, dtype=np.float32)
    regime_arr = np.asarray(regime_seq, dtype=np.int64)
    idx_arr = np.asarray(target_indices)

    train_mask = idx_arr < train_end
    val_mask = (idx_arr >= train_end) & (idx_arr < val_end)
    test_mask = idx_arr >= val_end

    # Peak threshold from daytime train targets
    train_day_ghi = y_raw_arr[train_mask & (day_arr > 0.5)]
    peak_threshold = float(np.percentile(train_day_ghi, 90)) if train_day_ghi.size > 0 else float(np.percentile(y_raw_arr[train_mask], 90))

    logger.info(
        "Sequence split — train: %d  val: %d  test: %d  seq_len: %d  features: %d  peak: %.1f",
        train_mask.sum(), val_mask.sum(), test_mask.sum(),
        sequence_length, X_arr.shape[-1], peak_threshold,
    )

    return SequenceBundle(
        X_train=X_arr[train_mask],
        y_train=y_arr[train_mask],
        X_val=X_arr[val_mask],
        y_val=y_arr[val_mask],
        X_test=X_arr[test_mask],
        y_test=y_arr[test_mask],
        train_timestamps=ts_arr[train_mask],
        val_timestamps=ts_arr[val_mask],
        test_timestamps=ts_arr[test_mask],
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        peak_threshold_raw=peak_threshold,
        y_train_raw=y_raw_arr[train_mask],
        y_val_raw=y_raw_arr[val_mask],
        y_test_raw=y_raw_arr[test_mask],
        is_daylight_test=day_arr[test_mask],
        regime_id_test=regime_arr[test_mask],
        sequence_length=sequence_length,
        n_features=X_arr.shape[-1],
    )
