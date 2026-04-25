"""
Abstract base class for all forecasting models in the benchmark.

Every model — tabular or sequence-based — implements this interface
so that the orchestrator can treat them uniformly.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark.models.base")


class BaseForecaster(ABC):
    """
    Contract that every benchmarked model must satisfy.

    Attributes
    ----------
    name : str
        Human-readable model name (used in reports and plots).
    model_type : str
        Either ``"tabular"`` or ``"sequence"``.  Determines which bundle
        the orchestrator prepares.
    """

    name: str = "BaseForecaster"
    model_type: str = "tabular"  # or "sequence"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Train the model.

        Parameters
        ----------
        X_train, y_train : arrays
            Training data (scaled).
        X_val, y_val : arrays
            Validation data (scaled) for early stopping / tuning.

        Returns
        -------
        dict
            Training history or metadata (e.g. ``{"epochs": 50, "best_val_loss": 0.01}``).
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate predictions (scaled space).

        Returns a 1-D float32 array of length ``len(X)``.
        """

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Return a DataFrame with columns ``["feature", "importance"]``.

        Not all models support this — default returns ``None``.
        """
        return None

    def get_params(self) -> Dict[str, Any]:
        """Return a JSON-serializable dict of the model's key parameters."""
        return {"name": self.name, "model_type": self.model_type}

    def get_training_history(self) -> Optional[Dict[str, list]]:
        """Return training curves (loss, val_loss, etc.) if available."""
        return None

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, directory: str) -> None:
        """Save the model to *directory*.  Sub-classes may override."""
        os.makedirs(directory, exist_ok=True)
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.get_params(), fh, indent=2)
        logger.info("Saved %s metadata to %s", self.name, meta_path)

    def load(self, directory: str) -> None:
        """Load the model from *directory*.  Sub-classes may override."""
        logger.info("Loading %s from %s", self.name, directory)
