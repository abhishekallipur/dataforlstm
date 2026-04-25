"""
Hybrid Residual model wrapper.

Architecture: FinalPrediction = BaselinePrediction + ResidualCorrection

This wraps the existing ``models.residual_hybrid`` pipeline, adapting it
to the ``BaseForecaster`` interface used by the benchmarking framework.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from benchmark.config import HybridResidualConfig, set_global_seed
from benchmark.features import (
    REGIME_NAMES,
    classify_regime,
    compute_air_mass,
    compute_clear_sky_ghi,
    compute_dew_point,
)
from benchmark.models.base import BaseForecaster

logger = logging.getLogger("benchmark.models.hybrid_residual")

# Ensure the project root is importable for the baseline model
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class HybridResidualForecaster(BaseForecaster):
    """
    Two-stage forecaster:
      Stage 1: Baseline LSTM prediction
      Stage 2: LightGBM corrects the residual (actual − baseline)

    The final prediction is: baseline_pred + residual_correction
    """

    name = "Hybrid Residual"
    model_type = "tabular"  # orchestrator provides tabular bundle

    def __init__(self, config: Optional[HybridResidualConfig] = None) -> None:
        self.cfg = config or HybridResidualConfig()
        self.baseline_model = None
        self.residual_models: List[lgb.LGBMRegressor] = []
        self.calibrator: Optional[Ridge] = None
        self.regime_classifier: Optional[lgb.LGBMClassifier] = None
        self._feature_importance: Optional[pd.DataFrame] = None
        self._feature_cols: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_baseline(self):
        """Load or train the baseline LSTM."""
        from tensorflow import keras
        model_path = Path(self.cfg.baseline_model_path)
        if model_path.exists():
            try:
                self.baseline_model = keras.models.load_model(str(model_path), compile=False, safe_mode=False)
            except TypeError:
                self.baseline_model = keras.models.load_model(str(model_path), compile=False)
            logger.info("Loaded baseline LSTM from %s", model_path)
        else:
            logger.warning("Baseline model not found at %s — will train from scratch", model_path)
            self.baseline_model = None

    def _build_residual_features(
        self,
        df: pd.DataFrame,
        baseline_pred: np.ndarray,
        actual_ghi: np.ndarray,
    ) -> pd.DataFrame:
        """Build the feature frame for the residual model."""
        frame = df.copy()
        frame["baseline_pred"] = baseline_pred
        frame["actual_ghi"] = actual_ghi
        frame["residual_target"] = actual_ghi - baseline_pred

        # Solar geometry
        frame["solar_elevation"] = 90.0 - frame["solar_zenith_angle"].clip(0.0, 180.0)
        frame["air_mass"] = compute_air_mass(frame["solar_zenith_angle"])
        frame["clear_sky_ghi_est"] = compute_clear_sky_ghi(frame["solar_zenith_angle"])
        cs_est = np.clip(frame["clear_sky_ghi_est"].values, 20.0, None)
        frame["baseline_clear_sky_index"] = np.where(
            cs_est > 20.0,
            frame["baseline_pred"].values / cs_est,
            0.0,
        )
        frame["cloud_cover_proxy"] = np.clip(
            1.0 - np.clip(frame["baseline_clear_sky_index"], 0.0, 1.0), 0.0, 1.0
        )
        frame["dew_point"] = compute_dew_point(frame["temperature"], frame["relative_humidity"])

        # Residual lags (causal — shift ≥ 1)
        residual = frame["residual_target"]
        for lag in [1, 2, 3, 6, 12, 24]:
            frame[f"residual_lag_{lag}"] = residual.shift(lag)
            frame[f"baseline_pred_lag_{lag}"] = frame["baseline_pred"].shift(lag)

        residual_past = residual.shift(1)
        for window in [3, 6, 12, 24]:
            frame[f"residual_roll_mean_{window}"] = residual_past.rolling(window).mean()
            frame[f"residual_roll_std_{window}"] = residual_past.rolling(window).std()

        frame["baseline_error_trend_3"] = residual.shift(1) - residual.shift(4)
        frame["baseline_error_trend_6"] = residual.shift(1) - residual.shift(7)

        frame = frame.ffill().fillna(0.0)
        return frame

    def _select_residual_features(self, frame: pd.DataFrame) -> List[str]:
        """Select numeric features excluding targets and metadata."""
        exclude = {
            "timestamp", "actual_ghi", "residual_target", "ghi",
            "regime_label_true", "split",
        }
        cols = []
        for c in frame.columns:
            if c in exclude:
                continue
            if pd.api.types.is_numeric_dtype(frame[c]):
                cols.append(c)
        return cols

    # ------------------------------------------------------------------
    # BaseForecaster interface
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train the hybrid residual model.

        This method expects ``kwargs`` to contain:
          - ``df_train``, ``df_val``, ``df_test``: raw DataFrames (pre-enriched)
          - ``seq_bundle``: SequenceBundle for generating baseline predictions
          - or ``baseline_preds_train/val/test``: pre-computed baseline predictions

        For simplicity in the benchmarking framework, this uses the tabular
        bundle's raw targets and a pre-trained baseline to generate residuals.
        """
        warnings.filterwarnings("ignore", category=FutureWarning)
        set_global_seed(42)

        # Get the full DataFrame and baseline predictions from kwargs
        full_df: pd.DataFrame = kwargs.get("full_df")
        baseline_preds: np.ndarray = kwargs.get("baseline_preds")
        actual_ghi: np.ndarray = kwargs.get("actual_ghi")
        train_mask: np.ndarray = kwargs.get("train_mask")
        val_mask: np.ndarray = kwargs.get("val_mask")
        peak_threshold: float = kwargs.get("peak_threshold", 500.0)

        if full_df is None or baseline_preds is None:
            # Fallback: use raw y values as "actuals" and zero baseline
            logger.warning("Hybrid model requires full_df and baseline_preds in kwargs. Using simplified training.")
            return {"status": "simplified_training"}

        # Build residual features
        frame = self._build_residual_features(full_df, baseline_preds, actual_ghi)
        self._feature_cols = self._select_residual_features(frame)

        # Train residual ensemble
        x_train = frame.loc[train_mask, self._feature_cols]
        y_res_train = frame.loc[train_mask, "residual_target"].values.astype(np.float32)
        x_val = frame.loc[val_mask, self._feature_cols]
        y_res_val = frame.loc[val_mask, "residual_target"].values.astype(np.float32)

        self.residual_models = []
        for i in range(self.cfg.ensemble_size):
            model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=2000,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=20,
                feature_fraction=0.85,
                subsample=0.85,
                subsample_freq=1,
                reg_lambda=0.1,
                random_state=42 + i,
                n_jobs=-1,
                verbose=-1,
            )
            model.fit(
                x_train, y_res_train,
                eval_set=[(x_val, y_res_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            self.residual_models.append(model)

        # Feature importance
        importance = np.mean(
            [m.booster_.feature_importance(importance_type="gain") for m in self.residual_models],
            axis=0,
        )
        self._feature_importance = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": importance,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

        logger.info("Hybrid Residual training complete — %d ensemble members", len(self.residual_models))
        return {"ensemble_size": len(self.residual_models)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        For the hybrid model, predictions require the residual feature frame.
        This basic predict uses the internal ensemble directly.
        """
        if not self.residual_models:
            raise RuntimeError("Residual models not trained.")
        preds = np.column_stack([m.predict(X) for m in self.residual_models])
        return preds.mean(axis=1).astype(np.float32)

    def predict_hybrid(
        self,
        frame: pd.DataFrame,
        baseline_preds: np.ndarray,
    ) -> np.ndarray:
        """
        Full hybrid prediction: baseline + residual correction.

        Parameters
        ----------
        frame : DataFrame
            Feature frame with residual features built.
        baseline_preds : array
            Baseline LSTM predictions (raw, W/m²).

        Returns
        -------
        array : Final hybrid predictions (clipped to ≥ 0).
        """
        x = frame[self._feature_cols]
        residual_pred = np.column_stack([m.predict(x) for m in self.residual_models]).mean(axis=1)
        hybrid_pred = baseline_preds + residual_pred
        return np.clip(hybrid_pred, 0.0, None).astype(np.float32)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        return self._feature_importance

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "ensemble_size": self.cfg.ensemble_size,
            "n_residual_features": len(self._feature_cols),
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        for i, m in enumerate(self.residual_models):
            path = os.path.join(directory, f"residual_model_{i}.txt")
            m.booster_.save_model(path)

    def load(self, directory: str) -> None:
        super().load(directory)
        import glob
        paths = sorted(Path(directory).glob("residual_model_*.txt"))
        self.residual_models = []
        for p in paths:
            booster = lgb.Booster(model_file=str(p))
            model = lgb.LGBMRegressor()
            model._Booster = booster
            self.residual_models.append(model)
