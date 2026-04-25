"""
Regime-Specific Ensemble Forecaster

Splits the data based on `clear_sky_index` regime states (Clear, Partly Cloudy, Cloudy)
and trains a specialized LightGBM model for each, alongside Quantile models for
uncertainty (prediction interval) bounds.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from benchmark.models.base import BaseForecaster
from benchmark.features import REGIME_NAMES

logger = logging.getLogger("benchmark.models.regime_ensemble")


class RegimeEnsembleForecaster(BaseForecaster):
    """
    Trains distinct models for distinct weather regimes to prevent under-fitting
    extreme cloud events. Also generates 10th and 90th percentile bounds.
    """

    def __init__(self, features: List[str]):
        self.name = "Regime Ensemble"
        self.features = features
        self.model_type = "tabular"
        self.regime_models: Dict[int, LGBMRegressor] = {}
        self.quantile_lower: Dict[int, LGBMRegressor] = {}
        self.quantile_upper: Dict[int, LGBMRegressor] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        Expects `X_train` to have regime flags in the features. We must extract 
        the regime to split the datasets internally. Since X is a numpy array, 
        we rely on the feature names list to find the `regime_{name}` column index.
        """
        logger.info(f"Training {self.name} with Regime Specialization and Uncertainty Bounds")
        
        # Find index of the regime columns
        try:
            regime_idxs = {
                idx: self.features.index(f"regime_{rname}")
                for idx, rname in enumerate(REGIME_NAMES)
            }
        except ValueError:
            logger.error("Regime features missing from feature list. Cannot train ensemble.")
            return

        for idx, rname in enumerate(REGIME_NAMES):
            col_idx = regime_idxs[idx]
            
            # Mask
            train_mask = X_train[:, col_idx] == 1.0
            x_sub = X_train[train_mask]
            y_sub = y_train[train_mask]
            
            if len(y_sub) < 50:
                logger.warning(f"Not enough data for regime {rname}. Fallback to generic.")
                x_sub = X_train
                y_sub = y_train
                
            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42 + idx,
                n_jobs=-1
            )
            model.fit(x_sub, y_sub)
            self.regime_models[idx] = model
            
            # Uncertainty models
            lower = LGBMRegressor(n_estimators=150, objective='quantile', alpha=0.1, n_jobs=-1, random_state=42)
            upper = LGBMRegressor(n_estimators=150, objective='quantile', alpha=0.9, n_jobs=-1, random_state=42)
            
            lower.fit(x_sub, y_sub)
            upper.fit(x_sub, y_sub)
            
            self.quantile_lower[idx] = lower
            self.quantile_upper[idx] = upper

        logger.info(f"{self.name} training complete.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Routes predictions to the specific regime model.
        """
        out = np.zeros(len(X), dtype=np.float32)
        
        try:
            regime_idxs = {
                idx: self.features.index(f"regime_{rname}")
                for idx, rname in enumerate(REGIME_NAMES)
            }
        except ValueError:
            return out

        for idx, model in self.regime_models.items():
            if model is None:
                continue
            col_idx = regime_idxs[idx]
            mask = X[:, col_idx] == 1.0
            if mask.any():
                out[mask] = model.predict(X[mask])
                
        # Clip
        out = np.clip(out, 0.0, None)
        return out
        
    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (y_pred, y_lower_p10, y_upper_p90).
        """
        y_pred = self.predict(X)
        y_low = np.zeros_like(y_pred)
        y_high = np.zeros_like(y_pred)
        
        regime_idxs = {idx: self.features.index(f"regime_{rname}") for idx, rname in enumerate(REGIME_NAMES)}
        for idx in range(3):
            mask = X[:, regime_idxs[idx]] == 1.0
            if mask.any():
                y_low[mask] = self.quantile_lower[idx].predict(X[mask])
                y_high[mask] = self.quantile_upper[idx].predict(X[mask])
                
        return y_pred, np.clip(y_low, 0.0, None), np.clip(y_high, 0.0, None)
