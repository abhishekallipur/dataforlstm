"""
Main orchestrator for the benchmarking framework.

Usage:
    python -m benchmark.run [--data-path dataset] [--quick] [--models gbt,lstm,...]

This script:
  1. Loads and feature-engineers the NSRDB data.
  2. Creates leakage-safe chronological splits.
  3. Runs the automated leakage audit.
  4. Trains all selected models.
  5. Sanitizes predictions via the robustness module.
  6. Evaluates all models with the unified metric suite.
  7. Generates all research-quality plots.
  8. Produces the final benchmark report.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from benchmark.config import (
    ExperimentConfig,
    ensure_dirs,
    set_global_seed,
    setup_logging,
    ARTIFACTS_DIR,
    PLOTS_DIR,
    PREDICTIONS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
)
from benchmark.data_loader import load_benchmark_data
from benchmark.features import TABULAR_FEATURES, SEQUENCE_FEATURES
from benchmark.splitter import build_tabular_bundle, build_sequence_bundle
from benchmark.leakage_audit import run_audit
from benchmark.robustness import PredictionSanitizer
from benchmark.evaluation import (
    compute_metrics,
    compute_regime_metrics,
    build_comparison_table,
    build_robustness_ranking,
)
from benchmark.visualizations import generate_all_plots
from benchmark.report_generator import (
    generate_markdown_report,
    save_comparison_table,
    save_regime_metrics,
    save_robustness_report,
    save_predictions,
    save_experiment_log,
)

# Model imports
from benchmark.models.base import BaseForecaster
from benchmark.models.gbt import GBTForecaster
from benchmark.models.svm import SVMForecaster
from benchmark.models.ann import ANNForecaster
from benchmark.models.dnn import DNNForecaster
from benchmark.models.lstm import LSTMForecaster
from benchmark.models.cnn_dnn import CNNDNNForecaster
from benchmark.models.cnn_lstm import CNNLSTMForecaster
from benchmark.models.cnn_attention_lstm import CNNAttentionLSTMForecaster
from benchmark.models.hybrid_residual import HybridResidualForecaster

logger = logging.getLogger("benchmark.run")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

ALL_MODELS: Dict[str, type] = {
    "gbt": GBTForecaster,
    "svm": SVMForecaster,
    "ann": ANNForecaster,
    "dnn": DNNForecaster,
    "lstm": LSTMForecaster,
    "cnn_dnn": CNNDNNForecaster,
    "cnn_lstm": CNNLSTMForecaster,
    "cnn_a_lstm": CNNAttentionLSTMForecaster,
    "hybrid_residual": HybridResidualForecaster,
}


def _instantiate_models(config: ExperimentConfig) -> List[BaseForecaster]:
    """Create model instances from the config."""
    selected = config.selected_models or list(ALL_MODELS.keys())
    models: List[BaseForecaster] = []

    config_map = {
        "gbt": lambda: GBTForecaster(config.gbt, feature_names=TABULAR_FEATURES),
        "svm": lambda: SVMForecaster(config.svm),
        "ann": lambda: ANNForecaster(config.ann),
        "dnn": lambda: DNNForecaster(config.dnn),
        "lstm": lambda: LSTMForecaster(config.lstm),
        "cnn_dnn": lambda: CNNDNNForecaster(config.cnn_dnn),
        "cnn_lstm": lambda: CNNLSTMForecaster(config.cnn_lstm),
        "cnn_a_lstm": lambda: CNNAttentionLSTMForecaster(config.cnn_attention_lstm),
        "hybrid_residual": lambda: HybridResidualForecaster(config.hybrid_residual),
    }

    for key in selected:
        key_lower = key.lower().replace("-", "_")
        if key_lower in config_map:
            models.append(config_map[key_lower]())
            logger.info("Registered model: %s", models[-1].name)
        else:
            logger.warning("Unknown model key: %s — skipping", key)

    return models


# ---------------------------------------------------------------------------
# Training + prediction pipeline
# ---------------------------------------------------------------------------


def _train_and_predict_tabular(
    model: BaseForecaster,
    tab_bundle,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train a tabular model and return (raw_predictions, history)."""
    history = model.fit(
        tab_bundle.X_train, tab_bundle.y_train,
        tab_bundle.X_val, tab_bundle.y_val,
    )
    preds_scaled = model.predict(tab_bundle.X_test)
    preds_raw = tab_bundle.target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(-1)
    return preds_raw.astype(np.float32), history


def _train_and_predict_sequence(
    model: BaseForecaster,
    seq_bundle,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train a sequence model and return (raw_predictions, history)."""
    history = model.fit(
        seq_bundle.X_train, seq_bundle.y_train,
        seq_bundle.X_val, seq_bundle.y_val,
    )
    preds_scaled = model.predict(seq_bundle.X_test)
    preds_raw = seq_bundle.target_scaler.inverse_transform(
        preds_scaled.reshape(-1, 1)
    ).reshape(-1)
    return preds_raw.astype(np.float32), history


def _train_and_predict_hybrid(
    model: HybridResidualForecaster,
    tab_bundle,
    seq_bundle,
    df: pd.DataFrame,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Train the hybrid residual model.

    This model needs baseline predictions, so we train a quick baseline LSTM
    internally and then correct its residuals.
    """
    # First, generate baseline predictions using a simple LSTM
    from benchmark.models.lstm import LSTMForecaster
    from benchmark.config import LSTMConfig

    baseline_cfg = LSTMConfig(epochs=config.hybrid_residual.baseline_epochs)
    if config.quick_mode:
        baseline_cfg.epochs = 10
    baseline = LSTMForecaster(baseline_cfg)

    baseline.fit(
        seq_bundle.X_train, seq_bundle.y_train,
        seq_bundle.X_val, seq_bundle.y_val,
    )

    # Get baseline predictions for all splits
    all_X = np.concatenate([seq_bundle.X_train, seq_bundle.X_val, seq_bundle.X_test], axis=0)
    all_preds_scaled = baseline.predict(all_X)
    all_preds_raw = seq_bundle.target_scaler.inverse_transform(
        all_preds_scaled.reshape(-1, 1)
    ).reshape(-1)
    all_preds_raw = np.clip(all_preds_raw, 0.0, None)

    all_y_raw = np.concatenate([seq_bundle.y_train_raw, seq_bundle.y_val_raw, seq_bundle.y_test_raw])
    all_ts = np.concatenate([seq_bundle.train_timestamps, seq_bundle.val_timestamps, seq_bundle.test_timestamps])

    n_train = len(seq_bundle.y_train_raw)
    n_val = len(seq_bundle.y_val_raw)
    n_total = len(all_y_raw)

    train_mask = np.zeros(n_total, dtype=bool)
    train_mask[:n_train] = True
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[n_train:n_train + n_val] = True
    test_mask = np.zeros(n_total, dtype=bool)
    test_mask[n_train + n_val:] = True

    # Build a minimal DataFrame for residual features
    # We need the columns the hybrid model's feature builder expects
    ts_series = pd.to_datetime(all_ts)
    mini_df = pd.DataFrame({
        "timestamp": ts_series,
        "solar_zenith_angle": 45.0,  # placeholder — will be overridden
        "temperature": 20.0,
        "relative_humidity": 50.0,
        "wind_speed": 3.0,
        "pressure": 1013.0,
        "is_daylight": (all_y_raw > 10).astype(float),
        "hour": ts_series.hour.astype(float),
        "sin_hour": np.sin(2 * np.pi * ts_series.hour / 24.0),
        "cos_hour": np.cos(2 * np.pi * ts_series.hour / 24.0),
    })

    # Enrich from the original df if shapes align
    if len(df) >= n_total:
        # Try to use real data for key columns
        for col in ["solar_zenith_angle", "temperature", "relative_humidity", "wind_speed", "pressure"]:
            if col in df.columns:
                vals = df[col].values
                # Use the tail of the df matching the sequence outputs
                offset = len(df) - n_total
                if offset >= 0:
                    mini_df[col] = vals[offset:offset + n_total]

    history = model.fit(
        tab_bundle.X_train, tab_bundle.y_train,
        tab_bundle.X_val, tab_bundle.y_val,
        full_df=mini_df,
        baseline_preds=all_preds_raw,
        actual_ghi=all_y_raw,
        train_mask=train_mask,
        val_mask=val_mask,
        peak_threshold=seq_bundle.peak_threshold_raw,
    )

    # Build the full feature set so predict_hybrid has all the lagged residual columns
    full_hybrid_features = model._build_residual_features(mini_df, all_preds_raw, all_y_raw)
    
    # Generate final hybrid predictions on the test set
    hybrid_preds = model.predict_hybrid(
        full_hybrid_features.iloc[n_train + n_val:].reset_index(drop=True),
        all_preds_raw[n_train + n_val:],
    )
    return hybrid_preds, history


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_benchmark(config: ExperimentConfig) -> Dict[str, Any]:
    """Execute the full benchmarking pipeline."""
    total_start = time.time()
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Setup
    config.output_dir = config.output_dir
    ensure_dirs()
    setup_logging(LOGS_DIR)
    set_global_seed(config.seed)
    config.apply_quick_mode()

    logger.info("=" * 70)
    logger.info("SOLAR IRRADIANCE BENCHMARKING FRAMEWORK v1.0")
    logger.info("=" * 70)
    logger.info("Data path: %s", config.data_path)
    logger.info("Output dir: %s", config.output_dir)
    logger.info("Quick mode: %s", config.quick_mode)
    logger.info("Seed: %d", config.seed)

    # ================================================================
    # STEP 1: Load and feature-engineer data
    # ================================================================
    logger.info("\n>>> STEP 1: Loading data and engineering features")
    df = load_benchmark_data(config.data_path)
    logger.info("Dataset: %d rows × %d columns", len(df), len(df.columns))

    # ================================================================
    # STEP 2: Create splits
    # ================================================================
    logger.info("\n>>> STEP 2: Creating chronological splits")

    # Tabular bundle (for GBT, SVM, ANN, DNN)
    tab_features = [f for f in TABULAR_FEATURES if f in df.columns]
    tab_bundle = build_tabular_bundle(
        df, tab_features,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    # Sequence bundle (for LSTM, CNN variants)
    seq_features = [f for f in SEQUENCE_FEATURES if f in df.columns]
    seq_bundle = build_sequence_bundle(
        df, seq_features,
        sequence_length=config.sequence_length,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
    )

    # ================================================================
    # STEP 3: Leakage audit
    # ================================================================
    logger.info("\n>>> STEP 3: Running leakage audit")

    audit_report = run_audit(
        df=df,
        train_timestamps=tab_bundle.train_timestamps,
        val_timestamps=tab_bundle.val_timestamps,
        test_timestamps=tab_bundle.test_timestamps,
        feature_cols=tab_features,
        scaler_n_samples=tab_bundle.feature_scaler.n_samples_seen_,
        train_size=len(tab_bundle.y_train),
        y_train=tab_bundle.y_train_raw,
        y_val=tab_bundle.y_val_raw,
        y_test=tab_bundle.y_test_raw,
        is_tabular=True,
    )
    audit_path = os.path.join(REPORTS_DIR, "leakage_audit.json")
    audit_report.save(audit_path)

    if not audit_report.all_passed:
        logger.error("ABORTING: Leakage audit failed. Fix issues before proceeding.")
        return {"status": "LEAKAGE_AUDIT_FAILED", "audit": audit_report.to_dict()}

    # ================================================================
    # STEP 4: Instantiate models
    # ================================================================
    logger.info("\n>>> STEP 4: Instantiating models")
    models = _instantiate_models(config)
    logger.info("Models to train: %s", [m.name for m in models])

    # ================================================================
    # STEP 5: Train all models and collect predictions
    # ================================================================
    logger.info("\n>>> STEP 5: Training models")

    sanitizer = PredictionSanitizer.build_from_training(tab_bundle.y_train_raw)

    all_predictions: Dict[str, np.ndarray] = {}
    all_metrics: Dict[str, Dict[str, float]] = {}
    all_sanity: Dict[str, dict] = {}
    all_histories: Dict[str, Dict[str, list]] = {}
    all_importances: Dict[str, pd.DataFrame] = {}
    all_params: Dict[str, dict] = {}
    all_regime_metrics: List[pd.DataFrame] = []
    attention_weights_arr: Optional[np.ndarray] = None

    for model in models:
        model_start = time.time()
        logger.info("\n--- Training: %s ---", model.name)

        try:
            if isinstance(model, HybridResidualForecaster):
                preds_raw, history = _train_and_predict_hybrid(
                    model, tab_bundle, seq_bundle, df, config,
                )
                # Use sequence bundle test data for hybrid
                test_y_raw = seq_bundle.y_test_raw
                test_ts = seq_bundle.test_timestamps
                test_daylight = seq_bundle.is_daylight_test
                test_regime = seq_bundle.regime_id_test
                peak_threshold = seq_bundle.peak_threshold_raw
            elif model.model_type == "tabular":
                preds_raw, history = _train_and_predict_tabular(model, tab_bundle, config)
                test_y_raw = tab_bundle.y_test_raw
                test_ts = tab_bundle.test_timestamps
                test_daylight = tab_bundle.is_daylight_test
                test_regime = tab_bundle.regime_id_test
                peak_threshold = tab_bundle.peak_threshold_raw
            else:  # sequence
                preds_raw, history = _train_and_predict_sequence(model, seq_bundle, config)
                test_y_raw = seq_bundle.y_test_raw
                test_ts = seq_bundle.test_timestamps
                test_daylight = seq_bundle.is_daylight_test
                test_regime = seq_bundle.regime_id_test
                peak_threshold = seq_bundle.peak_threshold_raw

            # Sanitize predictions
            preds_clean, sanity = sanitizer.sanitize(preds_raw, model.name)

            # Ensure prediction length matches test set
            min_len = min(len(preds_clean), len(test_y_raw))
            preds_clean = preds_clean[:min_len]
            test_y = test_y_raw[:min_len]
            ts = test_ts[:min_len]
            daylight = test_daylight[:min_len]
            regime = test_regime[:min_len]

            # Compute metrics
            metrics = compute_metrics(
                test_y, preds_clean, peak_threshold,
                timestamps=ts,
                is_daylight=daylight,
                regime_ids=regime,
            )

            # Regime-specific metrics
            regime_met = compute_regime_metrics(test_y, preds_clean, regime, peak_threshold)
            regime_met["model"] = model.name
            all_regime_metrics.append(regime_met)

            # Store results
            all_predictions[model.name] = preds_clean
            all_metrics[model.name] = metrics
            all_sanity[model.name] = sanity.to_dict()
            all_params[model.name] = model.get_params()

            # Training history
            hist = model.get_training_history()
            if hist:
                all_histories[model.name] = hist

            # Feature importance
            imp = model.get_feature_importance()
            if imp is not None:
                all_importances[model.name] = imp

            # Attention weights
            if isinstance(model, CNNAttentionLSTMForecaster) and hasattr(model, "get_attention_weights"):
                try:
                    sample = seq_bundle.X_test[:100]
                    attention_weights_arr = model.get_attention_weights(sample)
                except Exception as e:
                    logger.warning("Could not extract attention weights: %s", e)

            # Save model (non-fatal — predictions are already recorded)
            try:
                model_dir = os.path.join(ARTIFACTS_DIR, model.name.lower().replace(" ", "_").replace("(", "").replace(")", ""))
                model.save(model_dir)
            except Exception as save_exc:
                logger.warning("Could not save %s model artifact: %s", model.name, save_exc)

            elapsed = time.time() - model_start
            logger.info(
                "%s done in %.1fs — RMSE: %.2f, MAE: %.2f, R²: %.4f, Peak MAE: %.2f",
                model.name, elapsed,
                metrics["rmse"], metrics["mae"], metrics["r2"],
                metrics.get("peak_mae", float("nan")),
            )

        except Exception as exc:
            logger.error("FAILED to train %s: %s", model.name, exc, exc_info=True)
            continue

    # ================================================================
    # STEP 6: Build comparison table and rankings
    # ================================================================
    logger.info("\n>>> STEP 6: Building comparison tables")

    comparison_df = build_comparison_table(all_metrics, config.evaluation)
    robustness_df = build_robustness_ranking(all_sanity)

    # Combine regime metrics
    if all_regime_metrics:
        regime_df = pd.concat(all_regime_metrics, ignore_index=True)
    else:
        regime_df = pd.DataFrame()

    # ================================================================
    # STEP 7: Generate plots
    # ================================================================
    logger.info("\n>>> STEP 7: Generating plots")

    # Use the common test set (tabular bundle's) for consistent plots
    # But need to align prediction lengths
    common_ts = tab_bundle.test_timestamps
    common_y = tab_bundle.y_test_raw
    common_regime = tab_bundle.regime_id_test

    # For models with different test set sizes, truncate to common length
    plot_preds: Dict[str, np.ndarray] = {}
    min_test_len = len(common_y)
    for name, preds in all_predictions.items():
        if len(preds) == min_test_len:
            plot_preds[name] = preds
        elif len(preds) < min_test_len:
            # Pad with last value
            padded = np.full(min_test_len, preds[-1] if len(preds) > 0 else 0.0, dtype=np.float32)
            padded[:len(preds)] = preds
            plot_preds[name] = padded
        else:
            plot_preds[name] = preds[:min_test_len]

    plot_paths = generate_all_plots(
        timestamps=common_ts,
        y_true=common_y,
        predictions=plot_preds,
        regime_ids=common_regime,
        peak_threshold=tab_bundle.peak_threshold_raw,
        training_histories=all_histories,
        feature_importances=all_importances,
        comparison_df=comparison_df,
        attention_weights=attention_weights_arr,
        plots_dir=PLOTS_DIR,
    )

    # ================================================================
    # STEP 8: Generate reports
    # ================================================================
    logger.info("\n>>> STEP 8: Generating reports")

    save_comparison_table(comparison_df, os.path.join(REPORTS_DIR, "model_comparison.csv"))
    save_regime_metrics(regime_df, os.path.join(REPORTS_DIR, "regime_performance.csv"))
    save_robustness_report(robustness_df, os.path.join(REPORTS_DIR, "robustness_analysis.csv"))
    save_predictions(common_ts, common_y, plot_preds, os.path.join(PREDICTIONS_DIR, "all_predictions.csv"))

    save_experiment_log(
        config_dict=asdict(config),
        all_metrics=all_metrics,
        sanity_reports=all_sanity,
        leakage_audit=audit_report.to_dict(),
        path=os.path.join(LOGS_DIR, "experiment_log.json"),
    )

    generate_markdown_report(
        comparison_df=comparison_df,
        regime_df=regime_df,
        robustness_df=robustness_df,
        leakage_audit=audit_report.to_dict(),
        model_params=all_params,
        plot_paths=plot_paths,
        output_path=os.path.join(REPORTS_DIR, "benchmark_report.md"),
    )

    # Feature importance report
    if all_importances:
        combined_imp = []
        for model_name, imp_df in all_importances.items():
            imp_copy = imp_df.copy()
            imp_copy["model"] = model_name
            combined_imp.append(imp_copy)
        pd.concat(combined_imp, ignore_index=True).to_csv(
            os.path.join(REPORTS_DIR, "feature_importance.csv"), index=False,
        )

    # ================================================================
    # Summary
    # ================================================================
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 70)
    logger.info("BENCHMARK COMPLETE — %.1f minutes", total_elapsed / 60)
    logger.info("=" * 70)

    if not comparison_df.empty:
        logger.info("\nFinal Rankings:")
        for _, row in comparison_df.iterrows():
            logger.info(
                "  #%d  %-20s  RMSE=%.2f  MAE=%.2f  R²=%.4f  Peak=%.2f  Score=%.2f",
                row["rank"], row["model"],
                row["rmse"], row["mae"], row["r2"],
                row.get("peak_mae", float("nan")),
                row["composite_score"],
            )

    logger.info("\nOutputs saved to: %s", config.output_dir)
    logger.info("  Reports: %s", REPORTS_DIR)
    logger.info("  Plots:   %s", PLOTS_DIR)
    logger.info("  Models:  %s", ARTIFACTS_DIR)
    logger.info("  Logs:    %s", LOGS_DIR)

    return {
        "status": "SUCCESS",
        "elapsed_minutes": total_elapsed / 60,
        "comparison": comparison_df.to_dict(orient="records"),
        "audit": audit_report.to_dict(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solar Irradiance Forecasting — Research Benchmarking Framework",
    )
    parser.add_argument("--data-path", type=str, default="dataset", help="NSRDB dataset path.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory.")
    parser.add_argument(
        "--models", type=str, default=None,
        help="Comma-separated list of models to run (e.g. 'gbt,lstm,cnn_a_lstm'). Default: all.",
    )
    parser.add_argument("--quick", action="store_true", help="Quick mode — reduced epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--sequence-length", type=int, default=48, help="Lookback window for sequence models.")
    parser.add_argument("--train-ratio", type=float, default=0.70, help="Training split ratio.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        quick_mode=args.quick,
    )

    if args.output_dir:
        config.output_dir = args.output_dir

    if args.models:
        config.selected_models = [m.strip() for m in args.models.split(",")]

    run_benchmark(config)


if __name__ == "__main__":
    main()
