"""
Support Vector Regression forecaster with stratified subsampling.
"""

from __future__ import annotations

import logging
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.svm import SVR

from benchmark.config import SVMConfig
from benchmark.models.base import BaseForecaster

logger = logging.getLogger("benchmark.models.svm")


def _stratified_subsample(
    X: np.ndarray,
    y: np.ndarray,
    max_samples: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample while preserving the distribution of target values.

    Strategy: keep all peak-hour and cloud-event samples, downsample
    the abundant clear-sky / night samples.
    """
    n = len(y)
    if n <= max_samples:
        return X, y

    rng = np.random.RandomState(seed)

    # Define strata by target value quartiles
    quartiles = np.percentile(y, [25, 50, 75, 90])
    strata = np.digitize(y, quartiles)

    indices: list = []
    unique_strata = np.unique(strata)
    per_stratum = max_samples // len(unique_strata)

    for s in unique_strata:
        s_idx = np.where(strata == s)[0]
        # Keep all rare high-value samples
        if s == len(quartiles):  # top stratum (peak)
            indices.extend(s_idx.tolist())
        else:
            take = min(len(s_idx), per_stratum)
            chosen = rng.choice(s_idx, size=take, replace=False)
            indices.extend(chosen.tolist())

    indices = sorted(set(indices))
    logger.info("SVR subsampled %d → %d samples (max_samples=%d)", n, len(indices), max_samples)
    return X[indices], y[indices]


class SVMForecaster(BaseForecaster):
    """Support Vector Regression with RBF kernel."""

    name = "SVM (SVR-RBF)"
    model_type = "tabular"

    def __init__(self, config: Optional[SVMConfig] = None) -> None:
        self.cfg = config or SVMConfig()
        self.model: Optional[SVR] = None
        self._history: Dict[str, Any] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        logger.info(
            "Training %s — kernel=%s, C=%.1f, epsilon=%.3f, max_samples=%d",
            self.name, self.cfg.kernel, self.cfg.C, self.cfg.epsilon, self.cfg.max_train_samples,
        )

        # Subsample to keep SVR tractable
        X_sub, y_sub = _stratified_subsample(
            X_train, y_train.ravel(), self.cfg.max_train_samples
        )

        self.model = SVR(
            kernel=self.cfg.kernel,
            C=self.cfg.C,
            epsilon=self.cfg.epsilon,
            gamma=self.cfg.gamma,
            cache_size=1000,  # MB
        )
        self.model.fit(X_sub, y_sub)

        n_sv = self.model.n_support_ if hasattr(self.model, "n_support_") else 0
        self._history = {
            "train_samples_used": len(X_sub),
            "n_support_vectors": int(np.sum(n_sv)) if isinstance(n_sv, np.ndarray) else int(n_sv),
        }
        logger.info(
            "%s training complete — %d support vectors from %d samples",
            self.name, self._history["n_support_vectors"], self._history["train_samples_used"],
        )
        return self._history

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X).astype(np.float32)

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "kernel": self.cfg.kernel,
            "C": self.cfg.C,
            "epsilon": self.cfg.epsilon,
            **self._history,
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        if self.model is not None:
            path = os.path.join(directory, "svm_model.pkl")
            with open(path, "wb") as fh:
                pickle.dump(self.model, fh)
            logger.info("Saved SVM model to %s", path)

    def load(self, directory: str) -> None:
        super().load(directory)
        path = os.path.join(directory, "svm_model.pkl")
        with open(path, "rb") as fh:
            self.model = pickle.load(fh)
