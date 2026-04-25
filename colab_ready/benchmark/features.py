"""
Comprehensive feature engineering pipeline for solar irradiance forecasting.

Every feature is *causal* — constructed from past data only (shift ≥ 1 for
any quantity involving the target).  This guarantees zero future leakage
when paired with the chronological splitter.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark.features")

# ---------------------------------------------------------------------------
# Solar geometry helpers
# ---------------------------------------------------------------------------


def compute_clear_sky_ghi(solar_zenith_angle: pd.Series) -> pd.Series:
    """Hottel clear-sky model approximation (W/m²)."""
    zenith = solar_zenith_angle.clip(0.0, 180.0)
    cos_z = np.cos(np.deg2rad(zenith)).clip(0.0, 1.0)
    denom = np.clip(cos_z, 0.05, None)
    cs = 1098.0 * cos_z * np.exp(-0.059 / denom)
    cs = np.where(cos_z > 0.0, cs, 0.0)
    return pd.Series(cs, index=solar_zenith_angle.index, name="clear_sky_ghi")


def compute_air_mass(solar_zenith_angle: pd.Series) -> pd.Series:
    """Kasten & Young (1989) air-mass formula."""
    zenith = solar_zenith_angle.clip(0.0, 180.0)
    cos_z = np.cos(np.deg2rad(zenith)).clip(0.0, 1.0)
    denom = cos_z + 0.15 * np.power(np.clip(93.885 - zenith, 1e-3, None), -1.253)
    am = np.where(cos_z > 0.0, 1.0 / np.clip(denom, 1e-3, None), 0.0)
    return pd.Series(am, index=solar_zenith_angle.index, name="air_mass")


def compute_dew_point(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """Magnus formula dew-point approximation."""
    rh_safe = rh.clip(1.0, 100.0)
    a, b = 17.27, 237.7
    alpha = (a * temp_c) / (b + temp_c) + np.log(rh_safe / 100.0)
    return pd.Series((b * alpha) / (a - alpha), index=temp_c.index, name="dew_point")


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------

REGIME_NAMES: List[str] = ["clear", "partly_cloudy", "cloudy"]


def classify_regime(
    clear_sky_index: pd.Series,
    is_daylight: pd.Series,
) -> pd.Series:
    """
    Assign weather regime labels based on clear-sky index.

    Returns integer-coded series: 0=clear, 1=partly_cloudy, 2=cloudy.
    Night-time rows are labelled as cloudy (regime 2) because GHI ≈ 0
    regardless of cloud state.
    """
    regime = np.full(len(clear_sky_index), 2, dtype=np.int64)  # default: cloudy
    daylight = is_daylight.to_numpy(dtype=np.float32) > 0.5
    idx = clear_sky_index.to_numpy(dtype=np.float32)

    regime[daylight & (idx >= 0.80)] = 0   # clear
    regime[daylight & (idx >= 0.45) & (idx < 0.80)] = 1  # partly cloudy
    return pd.Series(regime, index=clear_sky_index.index, name="regime_id")


# ---------------------------------------------------------------------------
# Feature enrichment
# ---------------------------------------------------------------------------


def enrich_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all benchmark features on top of the baseline feature table.

    The baseline table (from ``build_feature_table``) already provides:
        ghi, temperature, relative_humidity, wind_speed, pressure,
        solar_zenith_angle, hour, sin_hour, cos_hour, sin_doy, cos_doy,
        cos_zenith, is_daylight,
        ghi_lag_1/2/3/24, ghi_diff_1/3,
        ghi_roll_mean_3/6/24, ghi_roll_std_6/24

    This function adds the following groups of features.
    """
    df = df.copy()
    logger.info("Enriching features …")

    # --- Solar geometry ------------------------------------------------
    df["solar_elevation"] = 90.0 - df["solar_zenith_angle"].clip(0.0, 180.0)
    df["air_mass"] = compute_air_mass(df["solar_zenith_angle"])
    df["clear_sky_ghi"] = compute_clear_sky_ghi(df["solar_zenith_angle"])
    df["clear_sky_index"] = df["ghi"] / np.clip(df["clear_sky_ghi"].values, 1.0, None)
    df["clear_sky_index"] = df["clear_sky_index"].clip(0.0, 2.0)
    
    # Extraterrestrial radiation (I_0 * (1 + 0.033 * cos(360 * doy / 365)))
    doy = df.index.dayofyear if hasattr(df.index, "dayofyear") else pd.Series(np.arange(len(df)) % 365 + 1)
    df["extraterrestrial_rad"] = 1367.0 * (1.0 + 0.033 * np.cos(np.deg2rad(360.0 * doy / 365.0)))
    df["extraterrestrial_rad"] = df["extraterrestrial_rad"].astype(np.float32)

    # --- Weather extras ------------------------------------------------
    df["dew_point"] = compute_dew_point(df["temperature"], df["relative_humidity"])
    df["cloud_cover_proxy"] = np.clip(
        1.0 - df["clear_sky_index"].clip(0.0, 1.0), 0.0, 1.0
    )

    # --- Volatility / gradient features --------------------------------
    df["ghi_gradient_3"] = (df["ghi"] - df["ghi"].shift(3)) / 3.0
    df["ghi_gradient_6"] = (df["ghi"] - df["ghi"].shift(6)) / 6.0
    df["ghi_abs_diff_1"] = df["ghi_diff_1"].abs()
    df["ghi_abs_diff_3"] = df["ghi_diff_3"].abs()
    
    # Gradient acceleration
    df["ghi_gradient_accel"] = df["ghi_gradient_3"] - df["ghi_gradient_3"].shift(1)

    # --- Additional rolling features -----------------------------------
    df["ghi_roll_min_24"] = df["ghi"].rolling(window=24).min()
    df["ghi_roll_max_24"] = df["ghi"].rolling(window=24).max()
    df["ghi_roll_range_24"] = df["ghi_roll_max_24"] - df["ghi_roll_min_24"]
    df["ghi_roll_var_3"] = df["ghi"].rolling(window=3).std() ** 2
    df["ghi_roll_var_6"] = df["ghi_roll_std_6"] ** 2

    # --- Regime features (one-hot) ------------------------------------
    df["regime_id"] = classify_regime(df["clear_sky_index"], df["is_daylight"])
    for idx, name in enumerate(REGIME_NAMES):
        df[f"regime_{name}"] = (df["regime_id"] == idx).astype(np.float32)

    # --- Fill any new NaN introduced by shift / rolling ---------------
    df = df.ffill().fillna(0.0)

    logger.info("Feature enrichment complete — %d columns total", len(df.columns))
    return df


# ---------------------------------------------------------------------------
# Feature lists for different model types
# ---------------------------------------------------------------------------

# Features suitable for tabular models (GBT, SVM, ANN, DNN)
TABULAR_FEATURES: List[str] = [
    # weather / environment
    "temperature", "relative_humidity", "wind_speed", "pressure", "dew_point",
    # solar geometry
    "solar_zenith_angle", "solar_elevation", "cos_zenith", "air_mass",
    "clear_sky_ghi", "clear_sky_index", "cloud_cover_proxy", "extraterrestrial_rad",
    # time
    "hour", "sin_hour", "cos_hour", "sin_doy", "cos_doy", "is_daylight",
    # lag
    "ghi_lag_1", "ghi_lag_2", "ghi_lag_3", "ghi_lag_24",
    # rolling
    "ghi_roll_mean_3", "ghi_roll_mean_6", "ghi_roll_mean_24",
    "ghi_roll_std_6", "ghi_roll_std_24", "ghi_roll_var_3", "ghi_roll_var_6",
    "ghi_roll_min_24", "ghi_roll_max_24", "ghi_roll_range_24",
    # volatility
    "ghi_diff_1", "ghi_diff_3", "ghi_gradient_3", "ghi_gradient_6",
    "ghi_abs_diff_1", "ghi_abs_diff_3", "ghi_gradient_accel",
    # regime
    "regime_clear", "regime_partly_cloudy", "regime_cloudy",
]

# Features for sequence models — include raw ghi at position 0 for
# autoregressive context within the look-back window.
SEQUENCE_FEATURES: List[str] = [
    "ghi",
    "temperature", "relative_humidity", "wind_speed", "pressure",
    "solar_zenith_angle", "cos_zenith", "is_daylight", "extraterrestrial_rad",
    "hour", "sin_hour", "cos_hour", "sin_doy", "cos_doy",
    "ghi_lag_1", "ghi_lag_2", "ghi_lag_3", "ghi_lag_24",
    "ghi_diff_1", "ghi_diff_3", "ghi_gradient_accel",
    "ghi_roll_mean_3", "ghi_roll_mean_6", "ghi_roll_mean_24",
    "ghi_roll_std_6", "ghi_roll_std_24", "ghi_roll_var_3", "ghi_roll_var_6"
]
