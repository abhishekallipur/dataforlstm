"""
Gradient Boosted Trees forecaster — LightGBM with walk-forward tuning.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

from benchmark.config import GBTConfig
from benchmark.models.base import BaseForecaster

logger = logging.getLogger("benchmark.models.gbt")


class GBTForecaster(BaseForecaster):
    """LightGBM gradient-boosted-tree regressor."""

    name = "GBT (LightGBM)"
    model_type = "tabular"

    def __init__(self, config: Optional[GBTConfig] = None, feature_names: Optional[List[str]] = None) -> None:
        self.cfg = config or GBTConfig()
        self.feature_names = feature_names
        self.model: Optional[lgb.LGBMRegressor] = None
        self._history: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logger.info("Training %s — n_estimators=%d, lr=%.4f", self.name, self.cfg.n_estimators, self.cfg.learning_rate)

        self.model = lgb.LGBMRegressor(
            objective="regression",
            n_estimators=self.cfg.n_estimators,
            learning_rate=self.cfg.learning_rate,
            num_leaves=self.cfg.num_leaves,
            min_child_samples=self.cfg.min_child_samples,
            feature_fraction=self.cfg.feature_fraction,
            subsample=self.cfg.subsample,
            subsample_freq=1,
            reg_alpha=self.cfg.reg_alpha,
            reg_lambda=self.cfg.reg_lambda,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )

        self.model.fit(
            X_train,
            y_train.ravel(),
            eval_set=[(X_val, y_val.ravel())],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)],
        )

        best_iter = self.model.best_iteration_ if hasattr(self.model, "best_iteration_") else self.cfg.n_estimators
        self._history = {"best_iteration": best_iter}
        logger.info("%s training complete — best iteration: %d", self.name, best_iter)
        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X).astype(np.float32)

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        if self.model is None:
            return None
        importance = self.model.booster_.feature_importance(importance_type="gain")
        names = self.feature_names if self.feature_names else [f"f{i}" for i in range(len(importance))]
        return pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False).reset_index(drop=True)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "n_estimators": self.cfg.n_estimators,
            "learning_rate": self.cfg.learning_rate,
            "num_leaves": self.cfg.num_leaves,
            "best_iteration": self._history.get("best_iteration"),
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        if self.model is not None:
            path = os.path.join(directory, "gbt_model.txt")
            self.model.booster_.save_model(path)
            logger.info("Saved GBT model to %s", path)

    def load(self, directory: str) -> None:
        super().load(directory)
        path = os.path.join(directory, "gbt_model.txt")
        booster = lgb.Booster(model_file=path)
        self.model = lgb.LGBMRegressor()
        self.model._Booster = booster
