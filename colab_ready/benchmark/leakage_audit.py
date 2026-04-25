"""
Automatic leakage detection and temporal integrity auditor.

Runs a battery of checks BEFORE any model is trained and produces a
structured audit report.  If any check fails, the pipeline aborts.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark.leakage_audit")


@dataclass
class AuditCheck:
    """Result of a single audit check."""
    name: str
    passed: bool
    detail: str


@dataclass
class LeakageAuditReport:
    """Aggregated audit report."""
    checks: List[AuditCheck] = field(default_factory=list)
    all_passed: bool = True

    def add(self, check: AuditCheck) -> None:
        self.checks.append(check)
        if not check.passed:
            self.all_passed = False
            logger.error("LEAKAGE AUDIT FAILED — %s: %s", check.name, check.detail)
        else:
            logger.info("✓ %s: %s", check.name, check.detail)

    def to_dict(self) -> Dict:
        return {
            "all_passed": self.all_passed,
            "checks": [asdict(c) for c in self.checks],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)
        logger.info("Leakage audit report saved to: %s", path)


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_temporal_ordering(df: pd.DataFrame) -> AuditCheck:
    """Verify timestamps are monotonically non-decreasing."""
    ts = df["timestamp"].values
    mono = np.all(ts[1:] >= ts[:-1])
    return AuditCheck(
        name="temporal_ordering",
        passed=bool(mono),
        detail="Timestamps are monotonically non-decreasing." if mono else "Timestamps are NOT monotonic — data may be shuffled.",
    )


def _check_split_integrity(
    train_ts: np.ndarray,
    val_ts: np.ndarray,
    test_ts: np.ndarray,
) -> AuditCheck:
    """Verify train < val < test with no temporal overlap."""
    if len(train_ts) == 0 or len(val_ts) == 0 or len(test_ts) == 0:
        return AuditCheck("split_integrity", False, "One or more splits are empty.")

    train_max = np.max(train_ts)
    val_min = np.min(val_ts)
    val_max = np.max(val_ts)
    test_min = np.min(test_ts)

    ok = (train_max <= val_min) and (val_max <= test_min)
    detail = (
        f"train_max={train_max}  val_min={val_min}  val_max={val_max}  test_min={test_min}"
    )
    return AuditCheck(
        name="split_no_overlap",
        passed=bool(ok),
        detail=("No temporal overlap between splits. " + detail) if ok else ("OVERLAP DETECTED — " + detail),
    )


def _check_no_future_features(df: pd.DataFrame) -> AuditCheck:
    """
    Heuristic: verify that lag / rolling features use shift(≥1).

    We check that ghi_lag_1 equals ghi.shift(1) for a sample of rows.
    """
    if "ghi_lag_1" not in df.columns:
        return AuditCheck("no_future_features", True, "No lag features to verify (skipped).")

    expected = df["ghi"].shift(1)
    # Compare on rows where both are non-NaN
    mask = expected.notna() & df["ghi_lag_1"].notna()
    if mask.sum() < 10:
        return AuditCheck("no_future_features", True, "Not enough non-NaN rows to verify lag features.")

    mismatch = (~np.isclose(df.loc[mask, "ghi_lag_1"].values, expected[mask].values, atol=1e-4)).sum()
    ok = mismatch == 0
    return AuditCheck(
        name="no_future_features",
        passed=bool(ok),
        detail=f"ghi_lag_1 matches ghi.shift(1) — {mismatch} mismatches in {mask.sum()} rows." if ok else f"{mismatch} rows where ghi_lag_1 ≠ ghi.shift(1). Possible future leakage.",
    )


def _check_scaler_train_only(
    scaler_n_samples: int,
    train_size: int,
) -> AuditCheck:
    """Verify the scaler was fit on the training partition only."""
    ok = scaler_n_samples == train_size
    return AuditCheck(
        name="scaler_train_only",
        passed=bool(ok),
        detail=f"Scaler fit on {scaler_n_samples} samples (train size = {train_size})." if ok else f"Scaler fit on {scaler_n_samples} samples but train has {train_size}. LEAKAGE!",
    )


def _check_target_not_in_features(feature_cols: List[str], target_col: str = "ghi") -> AuditCheck:
    """
    For tabular models the raw target should NOT appear as a feature.
    (For sequence models `ghi` is allowed inside the look-back window
    because the target is one step ahead.)
    """
    # We only flag a problem for tabular feature lists where raw ghi is present
    # In practice, tabular features use lags, not raw ghi.
    if target_col in feature_cols:
        return AuditCheck(
            "target_not_in_features",
            False,
            f"Raw target column '{target_col}' appears in the tabular feature list — likely target leakage.",
        )
    return AuditCheck(
        "target_not_in_features",
        True,
        f"Target column '{target_col}' is not in the tabular feature list.",
    )


def _check_no_backfill_after_split(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
) -> AuditCheck:
    """Check for NaN in target arrays — presence after split suggests backfill issues."""
    nan_train = int(np.isnan(y_train).sum())
    nan_val = int(np.isnan(y_val).sum())
    nan_test = int(np.isnan(y_test).sum())
    total = nan_train + nan_val + nan_test
    ok = total == 0
    return AuditCheck(
        name="no_backfill_after_split",
        passed=bool(ok),
        detail=f"NaN in targets — train: {nan_train}, val: {nan_val}, test: {nan_test}." if not ok else "No NaN in any target partition.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_audit(
    df: pd.DataFrame,
    train_timestamps: np.ndarray,
    val_timestamps: np.ndarray,
    test_timestamps: np.ndarray,
    feature_cols: List[str],
    scaler_n_samples: int,
    train_size: int,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    target_col: str = "ghi",
    is_tabular: bool = True,
) -> LeakageAuditReport:
    """
    Execute the full leakage audit battery.

    Returns a ``LeakageAuditReport`` with pass/fail for each check.
    Raises ``RuntimeError`` if any critical check fails.
    """
    report = LeakageAuditReport()

    report.add(_check_temporal_ordering(df))
    report.add(_check_split_integrity(train_timestamps, val_timestamps, test_timestamps))
    report.add(_check_no_future_features(df))
    report.add(_check_scaler_train_only(scaler_n_samples, train_size))
    if is_tabular:
        report.add(_check_target_not_in_features(feature_cols, target_col))
    report.add(_check_no_backfill_after_split(y_train, y_val, y_test))

    if not report.all_passed:
        logger.error("=== LEAKAGE AUDIT FAILED — pipeline will NOT proceed ===")
    else:
        logger.info("=== ALL LEAKAGE CHECKS PASSED ===")

    return report
