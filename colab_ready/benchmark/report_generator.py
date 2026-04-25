"""
Report generator — produces the final markdown benchmark report,
CSV summary tables, and JSON audit outputs.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("benchmark.report_generator")


def _ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_comparison_table(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(path)
    df.to_csv(path, index=False)
    logger.info("Saved comparison table: %s", path)


def save_regime_metrics(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(path)
    df.to_csv(path, index=False)
    logger.info("Saved regime metrics: %s", path)


def save_robustness_report(df: pd.DataFrame, path: str) -> None:
    _ensure_dir(path)
    df.to_csv(path, index=False)
    logger.info("Saved robustness report: %s", path)


def save_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    path: str,
) -> None:
    """Save all model predictions to a single CSV."""
    _ensure_dir(path)
    data = {"timestamp": pd.to_datetime(timestamps), "actual_ghi": y_true}
    for model_name, preds in predictions.items():
        safe_col = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        data[safe_col] = preds
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)
    logger.info("Saved predictions CSV: %s", path)


def save_experiment_log(
    config_dict: Dict[str, Any],
    all_metrics: Dict[str, Dict[str, float]],
    sanity_reports: Dict[str, dict],
    leakage_audit: dict,
    path: str,
) -> None:
    """Save a JSON experiment log for reproducibility."""
    _ensure_dir(path)

    def _clean(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    log = {
        "timestamp": datetime.now().isoformat(),
        "config": _clean(config_dict),
        "metrics": _clean(all_metrics),
        "sanity_reports": _clean(sanity_reports),
        "leakage_audit": _clean(leakage_audit),
    }

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(log, fh, indent=2, default=str)
    logger.info("Saved experiment log: %s", path)


def generate_markdown_report(
    comparison_df: pd.DataFrame,
    regime_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    leakage_audit: dict,
    model_params: Dict[str, dict],
    plot_paths: List[str],
    output_path: str,
) -> None:
    """Generate a comprehensive markdown benchmark report."""
    _ensure_dir(output_path)
    lines: List[str] = []

    lines.append("# Solar Irradiance Forecasting — Benchmark Report")
    lines.append("")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # --- Executive Summary ---
    lines.append("## Executive Summary")
    lines.append("")
    if not comparison_df.empty:
        best = comparison_df.iloc[0]
        lines.append(f"**Best Model**: {best['model']} (Composite Score: {best['composite_score']:.2f})")
        lines.append(f"- RMSE: {best['rmse']:.2f} W/m²")
        lines.append(f"- MAE: {best['mae']:.2f} W/m²")
        lines.append(f"- R²: {best['r2']:.4f}")
        lines.append(f"- Peak MAE: {best.get('peak_mae', float('nan')):.2f} W/m²")
    lines.append("")

    # --- Leakage Audit ---
    lines.append("## Leakage Audit Report")
    lines.append("")
    audit_passed = leakage_audit.get("all_passed", False)
    lines.append(f"**Overall**: {'✅ ALL CHECKS PASSED' if audit_passed else '❌ LEAKAGE DETECTED'}")
    lines.append("")
    for check in leakage_audit.get("checks", []):
        status = "✅" if check["passed"] else "❌"
        lines.append(f"- {status} **{check['name']}**: {check['detail']}")
    lines.append("")

    # --- Model Comparison ---
    lines.append("## Model Comparison")
    lines.append("")
    if not comparison_df.empty:
        lines.append(comparison_df.to_markdown(index=False))
    lines.append("")

    # --- Regime-Wise Performance ---
    lines.append("## Regime-Wise Performance")
    lines.append("")
    if not regime_df.empty:
        lines.append(regime_df.to_markdown(index=False))
    lines.append("")

    # --- Robustness Analysis ---
    lines.append("## Robustness Analysis")
    lines.append("")
    if not robustness_df.empty:
        lines.append(robustness_df.to_markdown(index=False))
    lines.append("")

    # --- Model Architectures ---
    lines.append("## Model Architectures")
    lines.append("")
    for model_name, params in model_params.items():
        lines.append(f"### {model_name}")
        lines.append("```json")
        lines.append(json.dumps(params, indent=2, default=str))
        lines.append("```")
        lines.append("")

    # --- Validation Statement ---
    lines.append("## Validation Statement")
    lines.append("")
    lines.append("This benchmark employs the following safeguards to ensure scientific validity:")
    lines.append("")
    lines.append("1. **Chronological splitting only** — no random shuffling at any stage.")
    lines.append("2. **Scalers fit on training data only** — preventing information leakage from validation/test.")
    lines.append("3. **Causal features only** — all lag and rolling features use `shift(≥1)`.")
    lines.append("4. **Fixed future holdout** — the test set (final 15%) is never used during model selection.")
    lines.append("5. **Automated leakage audit** — 6 checks run before training with hard failure on violation.")
    lines.append("6. **Prediction sanitization** — NaN, negative, and exploding predictions are logged and corrected.")
    lines.append("7. **Deterministic seeds** — all random sources seeded for full reproducibility.")
    lines.append("")

    # --- Plots ---
    lines.append("## Generated Plots")
    lines.append("")
    for p in plot_paths:
        basename = os.path.basename(p)
        lines.append(f"- `{basename}`")
    lines.append("")

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    logger.info("Saved benchmark report: %s", output_path)
