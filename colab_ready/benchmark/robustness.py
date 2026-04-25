"""
Prediction robustness: NaN detection, exploding-value guards, outlier
clipping, and persistence-based fallback logic.

Every model's raw predictions pass through ``PredictionSanitizer`` before
evaluation so that numerical instabilities can never poison the metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from benchmark.config import GHI_CEILING, GHI_FLOOR

logger = logging.getLogger("benchmark.robustness")


@dataclass
class SanityReport:
    """Records every correction applied to a prediction array."""
    model_name: str = ""
    n_predictions: int = 0
    n_nan: int = 0
    n_negative: int = 0
    n_exploding: int = 0
    n_outlier: int = 0
    n_total_corrected: int = 0
    applied_fallback: bool = False
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "n_predictions": self.n_predictions,
            "n_nan": self.n_nan,
            "n_negative": self.n_negative,
            "n_exploding": self.n_exploding,
            "n_outlier": self.n_outlier,
            "n_total_corrected": self.n_total_corrected,
            "applied_fallback": self.applied_fallback,
            "notes": self.notes,
        }


class PredictionSanitizer:
    """
    Post-prediction validation and correction.

    Corrections applied (in order):
      1. Replace NaN with persistence fallback.
      2. Clip negatives to 0.
      3. Clip values > GHI_CEILING (1500 W/m²).
      4. Flag statistical outliers (> 4σ from training mean).
    """

    def __init__(
        self,
        train_mean: float,
        train_std: float,
        max_observed: float,
        outlier_sigma: float = 4.0,
    ) -> None:
        self.train_mean = train_mean
        self.train_std = train_std
        self.max_observed = max_observed
        self.outlier_sigma = outlier_sigma
        self.ceiling = min(GHI_CEILING, max_observed * 1.3)

    # ------------------------------------------------------------------

    def check(self, predictions: np.ndarray, model_name: str = "") -> SanityReport:
        """Analyse predictions WITHOUT modifying them."""
        report = SanityReport(model_name=model_name, n_predictions=len(predictions))
        report.n_nan = int(np.isnan(predictions).sum())
        report.n_negative = int((predictions < GHI_FLOOR).sum())
        report.n_exploding = int((predictions > GHI_CEILING).sum())

        upper = self.train_mean + self.outlier_sigma * self.train_std
        report.n_outlier = int((predictions > upper).sum()) - report.n_exploding
        report.n_outlier = max(report.n_outlier, 0)

        report.n_total_corrected = (
            report.n_nan + report.n_negative + report.n_exploding + report.n_outlier
        )
        return report

    def sanitize(
        self,
        predictions: np.ndarray,
        model_name: str = "",
        last_known_good: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, SanityReport]:
        """
        Return a corrected copy of *predictions* and the sanity report.

        Parameters
        ----------
        predictions : array
            Raw model output.
        model_name : str
            For logging and report labelling.
        last_known_good : array, optional
            Same-length array of persistence values to use as NaN fallback.
        """
        preds = predictions.copy().astype(np.float64)
        report = self.check(preds, model_name)

        # 1. NaN → fallback
        nan_mask = np.isnan(preds)
        if nan_mask.any():
            if last_known_good is not None and len(last_known_good) == len(preds):
                preds[nan_mask] = last_known_good[nan_mask]
                report.applied_fallback = True
                report.notes.append(f"Replaced {report.n_nan} NaN with persistence fallback.")
            else:
                preds[nan_mask] = 0.0
                report.notes.append(f"Replaced {report.n_nan} NaN with zero (no fallback available).")
            logger.warning("%s: %d NaN predictions replaced", model_name, report.n_nan)

        # 2. Negative → 0
        neg_mask = preds < GHI_FLOOR
        if neg_mask.any():
            preds[neg_mask] = GHI_FLOOR
            report.notes.append(f"Clipped {report.n_negative} negative values to {GHI_FLOOR}.")

        # 3. Exploding → ceiling
        exp_mask = preds > self.ceiling
        if exp_mask.any():
            preds[exp_mask] = self.ceiling
            report.notes.append(f"Clipped {report.n_exploding} exploding values to {self.ceiling:.1f}.")
            logger.warning("%s: %d exploding predictions clipped", model_name, report.n_exploding)

        if report.n_total_corrected > 0:
            logger.info(
                "%s: sanitized %d / %d predictions  (NaN=%d, neg=%d, exp=%d, outlier=%d)",
                model_name,
                report.n_total_corrected,
                report.n_predictions,
                report.n_nan,
                report.n_negative,
                report.n_exploding,
                report.n_outlier,
            )

        return preds.astype(np.float32), report

    # ------------------------------------------------------------------

    @staticmethod
    def build_from_training(y_train_raw: np.ndarray) -> "PredictionSanitizer":
        """Factory: derive thresholds from the training target distribution."""
        return PredictionSanitizer(
            train_mean=float(np.nanmean(y_train_raw)),
            train_std=float(np.nanstd(y_train_raw)),
            max_observed=float(np.nanmax(y_train_raw)),
        )
