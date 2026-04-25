"""
Central configuration for the benchmarking framework.

All hyperparameters, paths, splits, and experiment settings are defined
here so that every module draws from a single source of truth.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

GLOBAL_SEED: int = 42


def set_global_seed(seed: int = GLOBAL_SEED) -> None:
    """Set deterministic seeds for all frameworks."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "dataset")
OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "benchmark")

# Sub-directories created at runtime
ARTIFACTS_DIR = os.path.join(OUTPUT_DIR, "artifacts")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, "predictions")


def ensure_dirs() -> None:
    """Create all output directories."""
    for d in (ARTIFACTS_DIR, PLOTS_DIR, REPORTS_DIR, LOGS_DIR, PREDICTIONS_DIR):
        os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Data splits  (chronological, no shuffling)
# ---------------------------------------------------------------------------

TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15  # implied: 1 - TRAIN - VAL

SEQUENCE_LENGTH: int = 48  # hours look-back for sequence models

# Physical constants for sanity checks
GHI_FLOOR: float = 0.0        # W/m² — irradiance cannot be negative
GHI_CEILING: float = 1500.0   # W/m² — above any terrestrial measurement
GHI_DAYLIGHT_THRESHOLD: float = 10.0  # W/m² — below is "night"


# ---------------------------------------------------------------------------
# Model-specific hyper-parameter dataclasses
# ---------------------------------------------------------------------------

@dataclass
class GBTConfig:
    """Gradient Boosted Trees hyperparameters."""
    n_estimators: int = 2000
    learning_rate: float = 0.03
    num_leaves: int = 63
    min_child_samples: int = 20
    feature_fraction: float = 0.85
    subsample: float = 0.85
    reg_alpha: float = 0.0
    reg_lambda: float = 0.1
    early_stopping_rounds: int = 100


@dataclass
class SVMConfig:
    """Support Vector Regression hyperparameters."""
    kernel: str = "rbf"
    C: float = 100.0
    epsilon: float = 0.1
    gamma: str = "scale"
    max_train_samples: int = 8000  # subsample for O(n²) SVR


@dataclass
class ANNConfig:
    """Shallow Artificial Neural Network hyperparameters."""
    hidden_units: List[int] = field(default_factory=lambda: [128, 64])
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 128
    patience: int = 15


@dataclass
class DNNConfig:
    """Deep Neural Network hyperparameters."""
    hidden_units: List[int] = field(default_factory=lambda: [256, 128, 64, 32])
    dropout_rates: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2, 0.1])
    use_batch_norm: bool = True
    learning_rate: float = 1e-3
    epochs: int = 120
    batch_size: int = 128
    patience: int = 15


@dataclass
class LSTMConfig:
    """LSTM hyperparameters."""
    units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.2
    dense_units: int = 16
    learning_rate: float = 1e-3
    epochs: int = 80
    batch_size: int = 128
    patience: int = 12


@dataclass
class CNNDNNConfig:
    """CNN-DNN hyperparameters."""
    conv_filters: List[int] = field(default_factory=lambda: [64, 32])
    kernel_size: int = 3
    dense_units: List[int] = field(default_factory=lambda: [64, 32])
    dropout_rate: float = 0.3
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 128
    patience: int = 12


@dataclass
class CNNLSTMConfig:
    """CNN-LSTM hyperparameters."""
    conv_filters: List[int] = field(default_factory=lambda: [64, 32])
    kernel_size: int = 3
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    dense_units: int = 16
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 128
    patience: int = 12


@dataclass
class CNNAttentionLSTMConfig:
    """CNN + Attention + LSTM hyperparameters."""
    conv_filters: List[int] = field(default_factory=lambda: [64, 32])
    kernel_size: int = 3
    lstm_units: List[int] = field(default_factory=lambda: [64, 32])
    attention_units: int = 32
    dense_units: int = 32
    dropout_rate: float = 0.2
    learning_rate: float = 1e-3
    epochs: int = 100
    batch_size: int = 128
    patience: int = 12


@dataclass
class HybridResidualConfig:
    """Hybrid Residual model hyperparameters."""
    baseline_model_path: str = "outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5"
    ensemble_size: int = 3
    walk_forward_splits: int = 4
    baseline_epochs: int = 80
    baseline_batch_size: int = 128
    baseline_learning_rate: float = 1e-3


# ---------------------------------------------------------------------------
# Evaluation settings
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation and ranking configuration."""
    # Composite score weights for final ranking
    weight_rmse: float = 0.30
    weight_mae: float = 0.20
    weight_peak_mae: float = 0.20
    weight_cloud_mae: float = 0.15
    weight_transition_mae: float = 0.15
    # Walk-forward validation
    walk_forward_splits: int = 5
    # Sunrise/sunset transition window (hours before/after)
    transition_window_hours: int = 2


# ---------------------------------------------------------------------------
# Master experiment config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Top-level experiment configuration combining all settings."""
    seed: int = GLOBAL_SEED
    data_path: str = DATA_PATH
    output_dir: str = OUTPUT_DIR
    train_ratio: float = TRAIN_RATIO
    val_ratio: float = VAL_RATIO
    sequence_length: int = SEQUENCE_LENGTH
    quick_mode: bool = False  # reduced epochs for fast iteration

    # Per-model configs
    gbt: GBTConfig = field(default_factory=GBTConfig)
    svm: SVMConfig = field(default_factory=SVMConfig)
    ann: ANNConfig = field(default_factory=ANNConfig)
    dnn: DNNConfig = field(default_factory=DNNConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    cnn_dnn: CNNDNNConfig = field(default_factory=CNNDNNConfig)
    cnn_lstm: CNNLSTMConfig = field(default_factory=CNNLSTMConfig)
    cnn_attention_lstm: CNNAttentionLSTMConfig = field(default_factory=CNNAttentionLSTMConfig)
    hybrid_residual: HybridResidualConfig = field(default_factory=HybridResidualConfig)

    # Evaluation
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    # List of models to run (None = all)
    selected_models: Optional[List[str]] = None

    def apply_quick_mode(self) -> None:
        """Reduce epochs for fast iteration / debugging."""
        if not self.quick_mode:
            return
        self.ann.epochs = 10
        self.dnn.epochs = 10
        self.lstm.epochs = 10
        self.cnn_dnn.epochs = 10
        self.cnn_lstm.epochs = 10
        self.cnn_attention_lstm.epochs = 10
        self.hybrid_residual.baseline_epochs = 10
        self.gbt.n_estimators = 200
        self.svm.max_train_samples = 2000


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(log_dir: str = LOGS_DIR, level: int = logging.INFO) -> logging.Logger:
    """Configure project-wide logging with file and console handlers."""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("benchmark")
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(os.path.join(log_dir, "benchmark.log"), encoding="utf-8")
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
