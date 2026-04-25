"""
Residual hybrid solar irradiance forecasting pipeline.

Final prediction = baseline prediction + residual correction.
The residual model is a LightGBM regressor trained on leakage-safe
lagged features, regime probabilities, and baseline forecast context.
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow import keras

import models.baseline_lstm.model as base


plt.switch_backend("Agg")

REGIME_NAMES = ["clear", "partly_cloudy", "cloudy"]
REGIME_TO_ID = {name: idx for idx, name in enumerate(REGIME_NAMES)}


def _ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def _compute_dew_point(temp_c: pd.Series, relative_humidity: pd.Series) -> pd.Series:
    rh = relative_humidity.clip(1.0, 100.0)
    a = 17.27
    b = 237.7
    alpha = ((a * temp_c) / (b + temp_c)) + np.log(rh / 100.0)
    return (b * alpha) / (a - alpha)


def _compute_air_mass(solar_zenith_angle: pd.Series) -> pd.Series:
    zenith = solar_zenith_angle.clip(0.0, 180.0)
    cos_zenith = np.cos(np.deg2rad(zenith)).clip(0.0, 1.0)
    denom = cos_zenith + 0.15 * np.power(np.clip(93.885 - zenith, 1e-3, None), -1.253)
    air_mass = np.where(cos_zenith > 0.0, 1.0 / np.clip(denom, 1e-3, None), 0.0)
    return pd.Series(air_mass, index=solar_zenith_angle.index)


def _compute_clear_sky_ghi(solar_zenith_angle: pd.Series) -> pd.Series:
    zenith = solar_zenith_angle.clip(0.0, 180.0)
    cos_zenith = np.cos(np.deg2rad(zenith)).clip(0.0, 1.0)
    denom = np.clip(cos_zenith, 0.05, None)
    clear_sky = 1098.0 * cos_zenith * np.exp(-0.059 / denom)
    clear_sky = np.where(cos_zenith > 0.0, clear_sky, 0.0)
    return pd.Series(clear_sky, index=solar_zenith_angle.index)


def _regime_from_clear_sky_index(clear_sky_index: pd.Series, is_daylight: pd.Series) -> pd.Series:
    regime = np.full(len(clear_sky_index), REGIME_TO_ID["cloudy"], dtype=np.int64)
    daylight = is_daylight.to_numpy(dtype=np.float32) > 0.5
    idx = clear_sky_index.to_numpy(dtype=np.float32)

    clear_mask = daylight & (idx >= 0.80)
    partly_mask = daylight & (idx >= 0.45) & (idx < 0.80)

    regime[clear_mask] = REGIME_TO_ID["clear"]
    regime[partly_mask] = REGIME_TO_ID["partly_cloudy"]
    return pd.Series(regime, index=clear_sky_index.index)


def _safe_quantile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def _load_reference_report(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _select_columns(frame: pd.DataFrame, extra_exclude: Sequence[str] = ()) -> List[str]:
    exclude = {
        "timestamp",
        "actual_ghi",
        "residual_target",
        "regime_label_true",
        "regime_label_pred",
        "split",
        "target_index",
        *extra_exclude,
    }
    columns = []
    for name in frame.columns:
        if name in exclude:
            continue
        if pd.api.types.is_numeric_dtype(frame[name]):
            columns.append(name)
    return columns


@dataclass
class BaselineLedger:
    X_all: np.ndarray
    y_all_raw: np.ndarray
    timestamps: np.ndarray
    target_indices: np.ndarray
    train_mask: np.ndarray
    val_mask: np.ndarray
    test_mask: np.ndarray
    x_scaler: StandardScaler
    y_scaler: MinMaxScaler
    peak_threshold_raw: float
    last_ghi: np.ndarray
    is_daylight: np.ndarray


@dataclass
class HybridArtifacts:
    report_json: str = "outputs/reports/residual_hybrid_report.json"
    predictions_csv: str = "outputs/reports/residual_hybrid_predictions.csv"
    cv_results_csv: str = "outputs/reports/residual_hybrid_walk_forward.csv"
    feature_importance_csv: str = "outputs/reports/residual_hybrid_feature_importance.csv"
    optional_models_csv: str = "outputs/reports/residual_hybrid_optional_models.csv"
    baseline_model_path: str = "outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5"
    classifier_model_path: str = "outputs/artifacts/residual_regime_classifier.txt"
    regressor_dir: str = "outputs/artifacts/residual_regressor_ensemble"
    calibrator_path: str = "outputs/artifacts/residual_calibrator.joblib"
    metrics_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_metrics.png"
    feature_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_feature_importance.png"
    hour_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_hour_errors.png"
    regime_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_regime_errors.png"
    residual_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_residual_distribution.png"
    cv_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_walk_forward.png"
    scatter_plot: str = "outputs/plots/residual_hybrid/residual_hybrid_scatter.png"


def build_baseline_ledger(
    df: pd.DataFrame,
    sequence_length: int = 48,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> BaselineLedger:
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

    x_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_raw = df["ghi"].to_numpy(dtype=np.float32)
    timestamps = df["timestamp"].to_numpy()
    is_daylight = df["is_daylight"].to_numpy(dtype=np.float32)

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    if train_end <= sequence_length + 10 or val_end <= train_end + 10 or n <= val_end + 10:
        raise ValueError("Not enough rows for baseline ledger construction.")

    x_scaler = StandardScaler()
    x_scaler.fit(x_raw[:train_end])
    x_scaled = x_scaler.transform(x_raw).astype(np.float32)

    y_scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    y_scaler.fit(y_raw[:train_end].reshape(-1, 1))
    y_scaled = y_scaler.transform(y_raw.reshape(-1, 1)).astype(np.float32).reshape(-1)

    x_seq: List[np.ndarray] = []
    y_seq: List[float] = []
    indices: List[int] = []
    last_ghi: List[float] = []
    day_flags: List[float] = []

    for idx in range(sequence_length, n):
        x_seq.append(x_scaled[idx - sequence_length : idx, :])
        y_seq.append(float(y_scaled[idx]))
        indices.append(idx)
        last_ghi.append(float(y_raw[idx - 1]))
        day_flags.append(float(is_daylight[idx]))

    x_all = np.asarray(x_seq, dtype=np.float32)
    y_all_raw = np.asarray([y_raw[idx] for idx in indices], dtype=np.float32)
    target_indices = np.asarray(indices)
    last_ghi_arr = np.asarray(last_ghi, dtype=np.float32)
    day_flags_arr = np.asarray(day_flags, dtype=np.float32)

    train_mask = target_indices < train_end
    val_mask = (target_indices >= train_end) & (target_indices < val_end)
    test_mask = target_indices >= val_end

    train_day_values = y_all_raw[train_mask & (day_flags_arr > 0.5)]
    if train_day_values.size > 0:
        peak_threshold_raw = _safe_quantile(train_day_values, 0.90)
    else:
        peak_threshold_raw = _safe_quantile(y_all_raw[train_mask], 0.90)

    return BaselineLedger(
        X_all=x_all,
        y_all_raw=y_all_raw,
        timestamps=timestamps[target_indices],
        target_indices=target_indices,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        x_scaler=x_scaler,
        y_scaler=y_scaler,
        peak_threshold_raw=peak_threshold_raw,
        last_ghi=last_ghi_arr,
        is_daylight=day_flags_arr,
    )


def _load_or_train_baseline_model(
    ledger: BaselineLedger,
    sequence_length: int,
    artifacts: HybridArtifacts,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    retrain: bool = False,
) -> keras.Model:
    set_seed(seed)
    model_path = Path(artifacts.baseline_model_path)
    if model_path.exists() and not retrain:
        return keras.models.load_model(model_path, compile=False)

    model = base.build_model(
        sequence_length=sequence_length,
        num_features=ledger.X_all.shape[-1],
        learning_rate=learning_rate,
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=8,
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    model.fit(
        ledger.X_all[ledger.train_mask],
        ledger.y_scaler.transform(ledger.y_all_raw[ledger.train_mask].reshape(-1, 1)).astype(np.float32),
        validation_data=(
            ledger.X_all[ledger.val_mask],
            ledger.y_scaler.transform(ledger.y_all_raw[ledger.val_mask].reshape(-1, 1)).astype(np.float32),
        ),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        callbacks=callbacks,
        verbose=1,
    )
    _ensure_parent_dir(model_path)
    model.save(model_path)
    return model


def _predict_baseline_raw(model: keras.Model, ledger: BaselineLedger, batch_size: int) -> np.ndarray:
    pred_scaled = model.predict(ledger.X_all, batch_size=batch_size, verbose=0).reshape(-1, 1)
    pred_raw = ledger.y_scaler.inverse_transform(np.clip(pred_scaled, 0.0, 1.0)).reshape(-1)
    return np.clip(pred_raw, 0.0, None)


def _build_residual_frame(df: pd.DataFrame, ledger: BaselineLedger, baseline_pred_raw: np.ndarray) -> pd.DataFrame:
    frame = df.iloc[ledger.target_indices].copy().reset_index(drop=True)
    frame = frame.rename(columns={"ghi": "actual_ghi"})
    frame["baseline_pred"] = baseline_pred_raw
    frame["residual_target"] = frame["actual_ghi"] - frame["baseline_pred"]

    frame["solar_elevation"] = 90.0 - frame["solar_zenith_angle"].clip(0.0, 180.0)
    frame["cos_zenith"] = np.cos(np.deg2rad(frame["solar_zenith_angle"].clip(0.0, 180.0))).clip(0.0, 1.0)
    frame["air_mass"] = _compute_air_mass(frame["solar_zenith_angle"])
    frame["clear_sky_ghi_est"] = _compute_clear_sky_ghi(frame["solar_zenith_angle"])
    frame["baseline_clear_sky_index"] = np.where(
        frame["clear_sky_ghi_est"].to_numpy(dtype=np.float32) > 20.0,
        frame["baseline_pred"].to_numpy(dtype=np.float32) / np.clip(frame["clear_sky_ghi_est"].to_numpy(dtype=np.float32), 20.0, None),
        0.0,
    )
    frame["cloud_cover_proxy"] = np.clip(1.0 - np.clip(frame["baseline_clear_sky_index"], 0.0, 1.0), 0.0, 1.0)
    frame["dew_point"] = _compute_dew_point(frame["temperature"], frame["relative_humidity"])
    frame["hour_sin"] = np.sin(2.0 * np.pi * frame["hour"].astype(float) / 24.0)
    frame["hour_cos"] = np.cos(2.0 * np.pi * frame["hour"].astype(float) / 24.0)
    frame["doy_sin"] = np.sin(2.0 * np.pi * frame["timestamp"].dt.dayofyear.astype(float) / 365.25)
    frame["doy_cos"] = np.cos(2.0 * np.pi * frame["timestamp"].dt.dayofyear.astype(float) / 365.25)
    frame["regime_label_true"] = _regime_from_clear_sky_index(
        frame["actual_ghi"] / np.clip(frame["clear_sky_ghi_est"], 1.0, None),
        frame["is_daylight"],
    )

    residual = frame["residual_target"]
    baseline_pred = frame["baseline_pred"]

    for lag in [1, 2, 3, 6, 12, 24]:
        frame[f"residual_lag_{lag}"] = residual.shift(lag)
        frame[f"baseline_pred_lag_{lag}"] = baseline_pred.shift(lag)

    residual_past = residual.shift(1)
    baseline_error_past = residual.shift(1)
    for window in [3, 6, 12, 24]:
        frame[f"residual_roll_mean_{window}"] = residual_past.rolling(window).mean()
        frame[f"residual_roll_std_{window}"] = residual_past.rolling(window).std()
        frame[f"residual_roll_max_{window}"] = residual_past.rolling(window).max()
        frame[f"residual_roll_min_{window}"] = residual_past.rolling(window).min()
        frame[f"baseline_error_roll_mean_{window}"] = baseline_error_past.rolling(window).mean()
        frame[f"baseline_error_roll_std_{window}"] = baseline_error_past.rolling(window).std()

    frame["baseline_error_lag_1"] = residual.shift(1)
    frame["baseline_error_lag_3"] = residual.shift(3)
    frame["baseline_error_lag_6"] = residual.shift(6)
    frame["baseline_error_lag_24"] = residual.shift(24)
    frame["baseline_error_trend_3"] = residual.shift(1) - residual.shift(4)
    frame["baseline_error_trend_6"] = residual.shift(1) - residual.shift(7)
    frame["baseline_error_abs_lag_1"] = residual.shift(1).abs()
    frame["baseline_error_abs_roll_24"] = residual.shift(1).abs().rolling(24).mean()

    frame = frame.dropna().reset_index(drop=True)
    return frame


def _split_frame(frame: pd.DataFrame, train_ratio: float, val_ratio: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    idx = np.arange(n)
    train_mask = idx < train_end
    val_mask = (idx >= train_end) & (idx < val_end)
    test_mask = idx >= val_end
    return train_mask, val_mask, test_mask


def _make_class_weights(y: np.ndarray) -> np.ndarray:
    counts = np.bincount(y, minlength=len(REGIME_NAMES)).astype(np.float64)
    counts[counts == 0.0] = 1.0
    total = counts.sum()
    weights = total / (len(REGIME_NAMES) * counts)
    return weights[y]


def _train_regime_classifier(
    frame: pd.DataFrame,
    clf_features: List[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    seed: int,
    artifacts: HybridArtifacts,
) -> lgb.LGBMClassifier:
    classifier = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(REGIME_NAMES),
        learning_rate=0.05,
        n_estimators=1500,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    x_train = frame.loc[train_mask, clf_features]
    y_train = frame.loc[train_mask, "regime_label_true"].to_numpy(dtype=np.int64)
    x_val = frame.loc[val_mask, clf_features]
    y_val = frame.loc[val_mask, "regime_label_true"].to_numpy(dtype=np.int64)
    sample_weight = _make_class_weights(y_train)

    classifier.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_val, y_val)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(75, verbose=False)],
    )

    _ensure_parent_dir(artifacts.classifier_model_path)
    classifier.booster_.save_model(artifacts.classifier_model_path)
    return classifier


def _append_regime_probabilities(
    frame: pd.DataFrame,
    classifier: lgb.LGBMClassifier,
    clf_features: List[str],
) -> pd.DataFrame:
    probs = classifier.predict_proba(frame[clf_features])
    frame = frame.copy()
    frame["regime_prob_clear"] = probs[:, 0]
    frame["regime_prob_partly_cloudy"] = probs[:, 1]
    frame["regime_prob_cloudy"] = probs[:, 2]
    frame["regime_label_pred"] = np.argmax(probs, axis=1)
    frame["cloud_cover_proxy"] = np.clip(frame["regime_prob_cloudy"], 0.0, 1.0)
    return frame


def _walk_forward_candidates() -> List[Dict[str, float]]:
    return [
        {
            "learning_rate": 0.03,
            "num_leaves": 31,
            "min_child_samples": 20,
            "feature_fraction": 0.85,
            "subsample": 0.85,
            "reg_alpha": 0.0,
            "reg_lambda": 0.1,
        },
        {
            "learning_rate": 0.02,
            "num_leaves": 63,
            "min_child_samples": 15,
            "feature_fraction": 0.9,
            "subsample": 0.9,
            "reg_alpha": 0.0,
            "reg_lambda": 0.2,
        },
        {
            "learning_rate": 0.015,
            "num_leaves": 127,
            "min_child_samples": 30,
            "feature_fraction": 0.8,
            "subsample": 0.8,
            "reg_alpha": 0.05,
            "reg_lambda": 0.3,
        },
    ]


def _sample_weights(frame: pd.DataFrame, train_mask: np.ndarray, peak_threshold: float) -> np.ndarray:
    weights = np.ones(len(frame), dtype=np.float32)
    weights += 0.25 * frame["is_daylight"].to_numpy(dtype=np.float32)
    weights += 1.0 * (frame["regime_label_true"].to_numpy(dtype=np.int64) == REGIME_TO_ID["cloudy"]).astype(np.float32)
    weights += 1.5 * (frame["actual_ghi"].to_numpy(dtype=np.float32) >= peak_threshold).astype(np.float32)
    weights *= train_mask.astype(np.float32)
    return weights


def _score_predictions(y_true: np.ndarray, y_pred: np.ndarray, peak_threshold: float) -> Dict[str, float]:
    daylight_mask = y_true > 10.0
    peak_mask = y_true >= peak_threshold
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "day_mae": float(mean_absolute_error(y_true[daylight_mask], y_pred[daylight_mask])) if np.any(daylight_mask) else float("nan"),
        "peak_mae": float(mean_absolute_error(y_true[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else float("nan"),
    }


def _fit_residual_model(
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_val: pd.DataFrame,
    y_val: np.ndarray,
    sample_weight: Optional[np.ndarray],
    params: Dict[str, float],
    seed: int,
) -> lgb.LGBMRegressor:
    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=3000,
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        feature_fraction=float(params["feature_fraction"]),
        subsample=float(params["subsample"]),
        subsample_freq=1,
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        x_train,
        y_train,
        sample_weight=sample_weight,
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    return model


def _walk_forward_search(
    frame: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    peak_threshold: float,
    seed: int,
    n_splits: int,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    train_indices = np.where(train_mask)[0]
    if len(train_indices) < n_splits + 20:
        raise ValueError("Not enough train samples for walk-forward validation.")

    splitter = TimeSeriesSplit(n_splits=n_splits)
    results: List[Dict[str, float]] = []
    best_score = float("inf")
    best_params: Dict[str, float] = {}

    for candidate_idx, params in enumerate(_walk_forward_candidates(), start=1):
        fold_scores: List[float] = []
        fold_rmse: List[float] = []
        fold_peak: List[float] = []
        fold_day: List[float] = []

        for fold_idx, (fold_train_pos, fold_val_pos) in enumerate(splitter.split(train_indices), start=1):
            fold_train = train_indices[fold_train_pos]
            fold_val = train_indices[fold_val_pos]
            x_train = frame.loc[fold_train, feature_cols]
            x_val = frame.loc[fold_val, feature_cols]
            y_train = frame.loc[fold_train, "residual_target"].to_numpy(dtype=np.float32)
            y_val = frame.loc[fold_val, "residual_target"].to_numpy(dtype=np.float32)

            sw = _sample_weights(frame, np.isin(np.arange(len(frame)), fold_train), peak_threshold)[fold_train]
            model = _fit_residual_model(x_train, y_train, x_val, y_val, sw, params, seed + candidate_idx + fold_idx)
            resid_pred = model.predict(x_val)
            hybrid_pred = frame.loc[fold_val, "baseline_pred"].to_numpy(dtype=np.float32) + resid_pred
            metrics = _score_predictions(frame.loc[fold_val, "actual_ghi"].to_numpy(dtype=np.float32), hybrid_pred, peak_threshold)
            score = metrics["rmse"] + 0.15 * metrics["peak_mae"] + 0.05 * metrics["day_mae"]
            fold_scores.append(score)
            fold_rmse.append(metrics["rmse"])
            fold_peak.append(metrics["peak_mae"])
            fold_day.append(metrics["day_mae"])

        row = dict(params)
        row.update(
            {
                "candidate": candidate_idx,
                "cv_score": float(np.mean(fold_scores)),
                "cv_rmse": float(np.mean(fold_rmse)),
                "cv_peak_mae": float(np.mean(fold_peak)),
                "cv_day_mae": float(np.mean(fold_day)),
            }
        )
        results.append(row)
        if row["cv_score"] < best_score:
            best_score = row["cv_score"]
            best_params = params

    return best_params, pd.DataFrame(results).sort_values("cv_score").reset_index(drop=True)


def _train_residual_ensemble(
    frame: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    peak_threshold: float,
    params: Dict[str, float],
    ensemble_size: int,
    seed: int,
) -> List[lgb.LGBMRegressor]:
    x_train = frame.loc[train_mask, feature_cols]
    y_train = frame.loc[train_mask, "residual_target"].to_numpy(dtype=np.float32)
    x_val = frame.loc[val_mask, feature_cols]
    y_val = frame.loc[val_mask, "residual_target"].to_numpy(dtype=np.float32)
    sample_weight = _sample_weights(frame, train_mask, peak_threshold)[train_mask]

    models: List[lgb.LGBMRegressor] = []
    for member in range(ensemble_size):
        model = _fit_residual_model(
            x_train,
            y_train,
            x_val,
            y_val,
            sample_weight,
            params,
            seed + 100 + member,
        )
        models.append(model)
    return models


def _predict_ensemble(models: Sequence[lgb.LGBMRegressor], x: pd.DataFrame) -> np.ndarray:
    if not models:
        return np.zeros(len(x), dtype=np.float32)
    preds = np.column_stack([model.predict(x) for model in models])
    return preds.mean(axis=1)


def _fit_calibrator(
    frame: pd.DataFrame,
    val_mask: np.ndarray,
    residual_pred: np.ndarray,
    peak_threshold: float,
) -> Ridge:
    val_probs = frame.loc[val_mask, ["regime_prob_clear", "regime_prob_partly_cloudy", "regime_prob_cloudy"]].to_numpy(dtype=np.float32)
    val_resid_pred = residual_pred[val_mask]
    val_features = val_resid_pred.reshape(-1, 1) * val_probs
    val_target = frame.loc[val_mask, "residual_target"].to_numpy(dtype=np.float32)
    sample_weight = _sample_weights(frame, val_mask, peak_threshold)[val_mask]
    calibrator = Ridge(alpha=0.5, fit_intercept=True)
    calibrator.fit(val_features, val_target, sample_weight=sample_weight)
    return calibrator


def _apply_calibrator(frame: pd.DataFrame, residual_pred: np.ndarray, calibrator: Ridge) -> np.ndarray:
    probs = frame[["regime_prob_clear", "regime_prob_partly_cloudy", "regime_prob_cloudy"]].to_numpy(dtype=np.float32)
    features = residual_pred.reshape(-1, 1) * probs
    return calibrator.predict(features)


def _compute_regime_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for regime_id, regime_name in enumerate(REGIME_NAMES):
        mask = frame["regime_label_true"].to_numpy(dtype=np.int64) == regime_id
        if not np.any(mask):
            continue
        rows.append(
            {
                "regime": regime_name,
                "rmse": float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                "mae": float(mean_absolute_error(y_true[mask], y_pred[mask])),
                "bias": float(np.mean(y_pred[mask] - y_true[mask])),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _compute_hourly_mae(frame: pd.DataFrame, y_true: np.ndarray, baseline_pred: np.ndarray, hybrid_pred: np.ndarray) -> pd.DataFrame:
    hours = frame["timestamp"].dt.hour.to_numpy(dtype=np.int64)
    rows = []
    for hour in range(24):
        mask = hours == hour
        if not np.any(mask):
            continue
        rows.append(
            {
                "hour": hour,
                "baseline_mae": float(mean_absolute_error(y_true[mask], baseline_pred[mask])),
                "hybrid_mae": float(mean_absolute_error(y_true[mask], hybrid_pred[mask])),
            }
        )
    return pd.DataFrame(rows)


def _compute_peak_hour_metrics(frame: pd.DataFrame, y_true: np.ndarray, baseline_pred: np.ndarray, hybrid_pred: np.ndarray) -> Dict[str, float]:
    hours = frame["timestamp"].dt.hour.to_numpy(dtype=np.int64)
    peak_hour_mask = np.isin(hours, np.array([10, 11, 12, 13, 14, 15], dtype=np.int64))
    return {
        "baseline_peak_hour_mae": float(mean_absolute_error(y_true[peak_hour_mask], baseline_pred[peak_hour_mask])) if np.any(peak_hour_mask) else float("nan"),
        "hybrid_peak_hour_mae": float(mean_absolute_error(y_true[peak_hour_mask], hybrid_pred[peak_hour_mask])) if np.any(peak_hour_mask) else float("nan"),
    }


def _plot_metric_bars(metrics: pd.DataFrame, output_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_names = ["rmse", "mae", "day_mae", "peak_mae"]
    titles = ["RMSE", "MAE", "Day MAE", "Peak MAE"]
    for ax, metric, title in zip(axes.flat, metric_names, titles):
        ax.bar(metrics["model"], metrics[metric], color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Baseline vs Residual Hybrid Metrics")
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(feature_importance: pd.DataFrame, output_path: str) -> None:
    top = feature_importance.sort_values("importance", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(top["feature"], top["importance"], color="#1f77b4")
    ax.set_title("Top Residual Model Feature Importance")
    ax.set_xlabel("Gain Importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_hourly_errors(hourly: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hourly["hour"], hourly["baseline_mae"], marker="o", label="Baseline MAE", linewidth=2)
    ax.plot(hourly["hour"], hourly["hybrid_mae"], marker="o", label="Hybrid MAE", linewidth=2)
    ax.set_title("Hourly Error Profile")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("MAE")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_regime_errors(regime: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(regime))
    width = 0.35
    ax.bar(x - width / 2, regime["mae"].values, width, label="Hybrid MAE", color="#2ca02c")
    ax.bar(x + width / 2, regime["bias"].abs().values, width, label="Abs Bias", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(regime["regime"].tolist())
    ax.set_title("Regime Diagnostics")
    ax.set_ylabel("Error")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_residual_distribution(residual_error: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(residual_error, bins=60, alpha=0.75, color="#9467bd", density=True)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.5)
    ax.set_title("Hybrid Error Distribution")
    ax.set_xlabel("Prediction Error (W/m^2)")
    ax.set_ylabel("Density")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_walk_forward(cv_table: pd.DataFrame, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cv_table["candidate"], cv_table["cv_score"], marker="o", label="CV score")
    ax.plot(cv_table["candidate"], cv_table["cv_rmse"], marker="o", label="CV RMSE")
    ax.plot(cv_table["candidate"], cv_table["cv_peak_mae"], marker="o", label="CV Peak MAE")
    ax.set_xlabel("Candidate")
    ax.set_ylabel("Score")
    ax.set_title("Walk-Forward Validation Search")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_scatter(frame: pd.DataFrame, baseline_pred: np.ndarray, hybrid_pred: np.ndarray, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    y_true = frame["actual_ghi"].to_numpy(dtype=np.float32)
    ax.scatter(y_true, baseline_pred, s=6, alpha=0.25, label="Baseline", color="#1f77b4")
    ax.scatter(y_true, hybrid_pred, s=6, alpha=0.25, label="Hybrid", color="#2ca02c")
    max_val = float(max(np.max(y_true), np.max(baseline_pred), np.max(hybrid_pred)))
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="black")
    ax.set_xlabel("Actual GHI")
    ax.set_ylabel("Predicted GHI")
    ax.set_title("Actual vs Predicted Scatter")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _compare_optional_models(
    frame: pd.DataFrame,
    feature_cols: List[str],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    peak_threshold: float,
    seed: int,
) -> pd.DataFrame:
    results: List[Dict[str, object]] = []
    x_train = frame.loc[train_mask, feature_cols]
    y_train = frame.loc[train_mask, "residual_target"].to_numpy(dtype=np.float32)
    x_val = frame.loc[val_mask, feature_cols]
    y_val = frame.loc[val_mask, "residual_target"].to_numpy(dtype=np.float32)
    x_test = frame.loc[test_mask, feature_cols]
    sample_weight = _sample_weights(frame, train_mask, peak_threshold)[train_mask]

    try:
        import xgboost as xgb  # type: ignore

        model = xgb.XGBRegressor(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=[(x_val, y_val)], verbose=False)
        pred = np.clip(frame.loc[test_mask, "baseline_pred"].to_numpy(dtype=np.float32) + model.predict(x_test), 0.0, None)
        results.append({"model": "xgboost", **_score_predictions(frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32), pred, peak_threshold)})
    except Exception as exc:
        results.append({"model": "xgboost", "status": f"skipped: {exc}"})

    try:
        from catboost import CatBoostRegressor  # type: ignore

        model = CatBoostRegressor(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            loss_function="RMSE",
            random_seed=seed,
            verbose=False,
        )
        model.fit(x_train, y_train, sample_weight=sample_weight, eval_set=(x_val, y_val), use_best_model=True)
        pred = np.clip(frame.loc[test_mask, "baseline_pred"].to_numpy(dtype=np.float32) + model.predict(x_test), 0.0, None)
        results.append({"model": "catboost", **_score_predictions(frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32), pred, peak_threshold)})
    except Exception as exc:
        results.append({"model": "catboost", "status": f"skipped: {exc}"})

    return pd.DataFrame(results)


def run_pipeline(
    data_path: str,
    sequence_length: int = 48,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    baseline_model_path: str = "outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5",
    baseline_learning_rate: float = 1e-3,
    baseline_epochs: int = 80,
    baseline_batch_size: int = 128,
    ensemble_size: int = 3,
    walk_forward_splits: int = 4,
    seed: int = 42,
    retrain_baseline: bool = False,
    compare_optional_models: bool = False,
    no_plot: bool = True,
) -> Dict[str, object]:
    warnings.filterwarnings("ignore", category=FutureWarning)
    set_seed(seed)
    artifacts = HybridArtifacts(baseline_model_path=baseline_model_path)

    raw_df = base.build_feature_table(data_path)
    ledger = build_baseline_ledger(raw_df, sequence_length=sequence_length, train_ratio=train_ratio, val_ratio=val_ratio)
    baseline_model = _load_or_train_baseline_model(
        ledger,
        sequence_length=sequence_length,
        artifacts=artifacts,
        learning_rate=baseline_learning_rate,
        epochs=baseline_epochs,
        batch_size=baseline_batch_size,
        seed=seed,
        retrain=retrain_baseline,
    )
    baseline_pred_raw = _predict_baseline_raw(baseline_model, ledger, batch_size=baseline_batch_size)

    reference_report = _load_reference_report(Path("outputs/reports/FINAL_SPRINT_REPORT.json"))

    frame = _build_residual_frame(raw_df, ledger, baseline_pred_raw)
    train_mask, val_mask, test_mask = _split_frame(frame, train_ratio=train_ratio, val_ratio=val_ratio)

    classifier_feature_cols = _select_columns(frame, extra_exclude=("regime_prob_clear", "regime_prob_partly_cloudy", "regime_prob_cloudy"))
    regime_classifier = _train_regime_classifier(frame, classifier_feature_cols, train_mask, val_mask, seed=seed, artifacts=artifacts)
    frame = _append_regime_probabilities(frame, regime_classifier, classifier_feature_cols)

    feature_cols = _select_columns(frame)

    train_actual = frame.loc[train_mask, "actual_ghi"].to_numpy(dtype=np.float32)
    train_daylight = frame.loc[train_mask, "is_daylight"].to_numpy(dtype=np.float32) > 0.5
    peak_threshold = _safe_quantile(train_actual[train_daylight], 0.90) if np.any(train_daylight) else _safe_quantile(train_actual, 0.90)
    if not np.isfinite(peak_threshold):
        peak_threshold = float(np.percentile(train_actual, 90))

    best_params, cv_table = _walk_forward_search(
        frame,
        feature_cols,
        train_mask=train_mask,
        peak_threshold=float(peak_threshold),
        seed=seed,
        n_splits=walk_forward_splits,
    )

    residual_models = _train_residual_ensemble(
        frame,
        feature_cols,
        train_mask=train_mask,
        val_mask=val_mask,
        peak_threshold=float(peak_threshold),
        params=best_params,
        ensemble_size=ensemble_size,
        seed=seed,
    )

    _ensure_parent_dir(artifacts.regressor_dir)
    os.makedirs(artifacts.regressor_dir, exist_ok=True)
    for idx, model in enumerate(residual_models, start=1):
        model.booster_.save_model(os.path.join(artifacts.regressor_dir, f"residual_model_{idx}.txt"))

    residual_pred = _predict_ensemble(residual_models, frame[feature_cols])
    calibrator = _fit_calibrator(frame, val_mask=val_mask, residual_pred=residual_pred, peak_threshold=float(peak_threshold))
    _ensure_parent_dir(artifacts.calibrator_path)
    joblib.dump(calibrator, artifacts.calibrator_path)

    calibrated_residual = _apply_calibrator(frame, residual_pred, calibrator)
    hybrid_pred = np.clip(frame["baseline_pred"].to_numpy(dtype=np.float32) + calibrated_residual, 0.0, None)

    baseline_test = _score_predictions(frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32), frame.loc[test_mask, "baseline_pred"].to_numpy(dtype=np.float32), float(peak_threshold))
    residual_test = _score_predictions(frame.loc[test_mask, "residual_target"].to_numpy(dtype=np.float32), residual_pred[test_mask], float(peak_threshold))
    hybrid_test = _score_predictions(frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32), hybrid_pred[test_mask], float(peak_threshold))

    hybrid_metrics_table = pd.DataFrame(
        [
            {"model": "Baseline", **baseline_test},
            {"model": "Residual Only", **residual_test},
            {"model": "Hybrid", **hybrid_test},
        ]
    )

    hourly_errors = _compute_hourly_mae(
        frame.loc[test_mask].reset_index(drop=True),
        frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32),
        frame.loc[test_mask, "baseline_pred"].to_numpy(dtype=np.float32),
        hybrid_pred[test_mask],
    )
    regime_metrics = _compute_regime_metrics(
        frame.loc[test_mask].reset_index(drop=True),
        frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32),
        hybrid_pred[test_mask],
    )
    peak_hour_metrics = _compute_peak_hour_metrics(
        frame.loc[test_mask].reset_index(drop=True),
        frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32),
        frame.loc[test_mask, "baseline_pred"].to_numpy(dtype=np.float32),
        hybrid_pred[test_mask],
    )

    feature_importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": np.mean([model.booster_.feature_importance(importance_type="gain") for model in residual_models], axis=0),
        }
    ).sort_values("importance", ascending=False)

    residual_error = frame.loc[test_mask, "actual_ghi"].to_numpy(dtype=np.float32) - hybrid_pred[test_mask]

    optional_models = pd.DataFrame()
    if compare_optional_models:
        optional_models = _compare_optional_models(frame, feature_cols, train_mask, val_mask, test_mask, float(peak_threshold), seed)

    _ensure_parent_dir(artifacts.predictions_csv)
    predictions = frame.copy()
    predictions["residual_pred"] = residual_pred
    predictions["calibrated_residual"] = calibrated_residual
    predictions["hybrid_pred"] = hybrid_pred
    predictions.to_csv(artifacts.predictions_csv, index=False)

    _ensure_parent_dir(artifacts.cv_results_csv)
    cv_table.to_csv(artifacts.cv_results_csv, index=False)
    _ensure_parent_dir(artifacts.feature_importance_csv)
    feature_importance.to_csv(artifacts.feature_importance_csv, index=False)
    if not optional_models.empty:
        _ensure_parent_dir(artifacts.optional_models_csv)
        optional_models.to_csv(artifacts.optional_models_csv, index=False)

    _plot_metric_bars(hybrid_metrics_table, artifacts.metrics_plot)
    _plot_feature_importance(feature_importance, artifacts.feature_plot)
    _plot_hourly_errors(hourly_errors, artifacts.hour_plot)
    _plot_regime_errors(regime_metrics, artifacts.regime_plot)
    _plot_residual_distribution(residual_error, artifacts.residual_plot)
    _plot_walk_forward(cv_table, artifacts.cv_plot)
    _plot_scatter(frame.loc[test_mask].reset_index(drop=True), frame.loc[test_mask, "baseline_pred"].to_numpy(dtype=np.float32), hybrid_pred[test_mask], artifacts.scatter_plot)

    baseline_vs_report = reference_report.get("all_results", {}).get("Baseline (aggressive_peak_2x)") if reference_report else None

    summary = {
        "reference_baseline": baseline_vs_report,
        "baseline_test": baseline_test,
        "residual_only_test": residual_test,
        "hybrid_test": hybrid_test,
        "peak_hour_metrics": peak_hour_metrics,
        "regime_metrics": regime_metrics.to_dict(orient="records"),
        "walk_forward": cv_table.to_dict(orient="records"),
        "best_params": best_params,
        "feature_importance_top": feature_importance.head(20).to_dict(orient="records"),
        "artifacts": asdict(artifacts),
    }

    _ensure_parent_dir(artifacts.report_json)
    with open(artifacts.report_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\nResidual hybrid summary")
    print(f"Baseline RMSE: {baseline_test['rmse']:.2f}, Peak MAE: {baseline_test['peak_mae']:.2f}")
    print(f"Hybrid RMSE:   {hybrid_test['rmse']:.2f}, Peak MAE: {hybrid_test['peak_mae']:.2f}")
    print(f"Hybrid Day MAE: {hybrid_test['day_mae']:.2f}")
    print(f"Peak hour MAE:  {peak_hour_metrics['hybrid_peak_hour_mae']:.2f}")
    print(f"Reports saved to: {artifacts.report_json}")

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual hybrid solar irradiance forecasting pipeline.")
    parser.add_argument("--data-path", type=str, default="dataset", help="Folder or CSV path with NSRDB data.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Baseline sequence length.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Chronological train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Chronological validation split ratio.")
    parser.add_argument("--baseline-model-path", type=str, default="outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5", help="Path to the stage-1 baseline model.")
    parser.add_argument("--baseline-learning-rate", type=float, default=1e-3, help="Fallback baseline training learning rate.")
    parser.add_argument("--baseline-epochs", type=int, default=80, help="Fallback baseline training epochs.")
    parser.add_argument("--baseline-batch-size", type=int, default=128, help="Fallback baseline batch size.")
    parser.add_argument("--ensemble-size", type=int, default=3, help="Number of residual models to average.")
    parser.add_argument("--walk-forward-splits", type=int, default=4, help="Walk-forward CV splits for residual tuning.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--retrain-baseline", action="store_true", help="Retrain the baseline model instead of loading it.")
    parser.add_argument("--compare-optional-models", action="store_true", help="Compare optional XGBoost/CatBoost residual models if installed.")
    parser.add_argument("--no-plot", action="store_true", help="Disable opening plots; plots are still saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        baseline_model_path=args.baseline_model_path,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_epochs=args.baseline_epochs,
        baseline_batch_size=args.baseline_batch_size,
        ensemble_size=args.ensemble_size,
        walk_forward_splits=args.walk_forward_splits,
        seed=args.seed,
        retrain_baseline=args.retrain_baseline,
        compare_optional_models=args.compare_optional_models,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
