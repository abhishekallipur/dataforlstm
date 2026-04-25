# =========================================
# SHARED HELPERS (RUN FIRST)
# =========================================

## Required preprocessing dependencies
```python
# Colab cell: dependencies
!pip -q install lightgbm scikit-learn pandas numpy matplotlib tensorflow
```

## Model code
```python
# Colab cell: imports and shared helper functions
import os
import sys
import numpy as np
import pandas as pd

# Set this to your uploaded project root in Colab.
# Example after upload: /content/code
PROJECT_ROOT = "/content/code"
DATA_PATH = f"{PROJECT_ROOT}/dataset"

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.config import (
    set_global_seed,
    ExperimentConfig,
    GBTConfig,
    SVMConfig,
    ANNConfig,
    DNNConfig,
    LSTMConfig,
    CNNDNNConfig,
    CNNLSTMConfig,
    CNNAttentionLSTMConfig,
)
from benchmark.data_loader import load_benchmark_data
from benchmark.features import TABULAR_FEATURES, SEQUENCE_FEATURES
from benchmark.splitter import build_tabular_bundle, build_sequence_bundle
from benchmark.evaluation import compute_metrics, compute_regime_metrics

# Model classes (exact implementations from local project)
from benchmark.models.gbt import GBTForecaster
from benchmark.models.svm import SVMForecaster
from benchmark.models.ann import ANNForecaster
from benchmark.models.dnn import DNNForecaster
from benchmark.models.lstm import LSTMForecaster
from benchmark.models.cnn_dnn import CNNDNNForecaster
from benchmark.models.cnn_lstm import CNNLSTMForecaster
from benchmark.models.cnn_attention_lstm import CNNAttentionLSTMForecaster

set_global_seed(42)


def prepare_shared_bundles(data_path=DATA_PATH, train_ratio=0.70, val_ratio=0.15, sequence_length=48):
    df = load_benchmark_data(data_path, extra_features=True)

    tab_features = [f for f in TABULAR_FEATURES if f in df.columns]
    seq_features = [f for f in SEQUENCE_FEATURES if f in df.columns]

    tab_bundle = build_tabular_bundle(
        df,
        tab_features,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    seq_bundle = build_sequence_bundle(
        df,
        seq_features,
        sequence_length=sequence_length,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )

    return df, tab_bundle, seq_bundle, tab_features, seq_features


def evaluate_on_tabular_test(y_true_raw, y_pred_raw, bundle):
    return compute_metrics(
        y_true=y_true_raw,
        y_pred=y_pred_raw,
        peak_threshold=bundle.peak_threshold_raw,
        timestamps=bundle.test_timestamps,
        is_daylight=bundle.is_daylight_test,
        regime_ids=bundle.regime_id_test,
    )


def evaluate_on_sequence_test(y_true_raw, y_pred_raw, bundle):
    return compute_metrics(
        y_true=y_true_raw,
        y_pred=y_pred_raw,
        peak_threshold=bundle.peak_threshold_raw,
        timestamps=bundle.test_timestamps,
        is_daylight=bundle.is_daylight_test,
        regime_ids=bundle.regime_id_test,
    )


def regime_metrics_on_tabular_test(y_true_raw, y_pred_raw, bundle):
    return compute_regime_metrics(
        y_true=y_true_raw,
        y_pred=y_pred_raw,
        regime_ids=bundle.regime_id_test,
        peak_threshold=bundle.peak_threshold_raw,
        is_daylight=bundle.is_daylight_test,
    )


def regime_metrics_on_sequence_test(y_true_raw, y_pred_raw, bundle):
    return compute_regime_metrics(
        y_true=y_true_raw,
        y_pred=y_pred_raw,
        regime_ids=bundle.regime_id_test,
        peak_threshold=bundle.peak_threshold_raw,
        is_daylight=bundle.is_daylight_test,
    )
```

## Training code
```python
# Colab cell: build shared data once
DF, TAB_BUNDLE, SEQ_BUNDLE, TAB_FEATURES_ACTIVE, SEQ_FEATURES_ACTIVE = prepare_shared_bundles(
    data_path=DATA_PATH,
    train_ratio=0.70,
    val_ratio=0.15,
    sequence_length=48,
)
print("Tabular train/val/test:", TAB_BUNDLE.X_train.shape, TAB_BUNDLE.X_val.shape, TAB_BUNDLE.X_test.shape)
print("Sequence train/val/test:", SEQ_BUNDLE.X_train.shape, SEQ_BUNDLE.X_val.shape, SEQ_BUNDLE.X_test.shape)
```

## Prediction code
```python
# Shared helper section does not produce predictions directly.
# Predictions are generated per-model below.
```

## Evaluation code
```python
# Shared helper section does not produce evaluation directly.
# Evaluation is generated per-model below.
```

# =========================================
# MODEL 1 — GBT
# =========================================

## Required preprocessing dependencies
```python
# Uses TAB_BUNDLE from shared helper section.
```

## Model code
```python
gbt_model = GBTForecaster(
    config=GBTConfig(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=20,
        feature_fraction=0.85,
        subsample=0.85,
        reg_alpha=0.0,
        reg_lambda=0.1,
        early_stopping_rounds=100,
    ),
    feature_names=TAB_FEATURES_ACTIVE,
)
```

## Training code
```python
gbt_history = gbt_model.fit(
    TAB_BUNDLE.X_train,
    TAB_BUNDLE.y_train,
    TAB_BUNDLE.X_val,
    TAB_BUNDLE.y_val,
)
gbt_history
```

## Prediction code
```python
gbt_pred_scaled = gbt_model.predict(TAB_BUNDLE.X_test)
gbt_pred_raw = TAB_BUNDLE.target_scaler.inverse_transform(gbt_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
gbt_metrics = evaluate_on_tabular_test(TAB_BUNDLE.y_test_raw, gbt_pred_raw, TAB_BUNDLE)
gbt_regime_metrics = regime_metrics_on_tabular_test(TAB_BUNDLE.y_test_raw, gbt_pred_raw, TAB_BUNDLE)
print(gbt_metrics)
gbt_regime_metrics
```

# =========================================
# MODEL 2 — SVM
# =========================================

## Required preprocessing dependencies
```python
# Uses TAB_BUNDLE from shared helper section (leakage-safe scaling preserved by splitter).
```

## Model code
```python
svm_model = SVMForecaster(
    config=SVMConfig(
        kernel="rbf",
        C=100.0,
        epsilon=0.1,
        gamma="scale",
        max_train_samples=8000,
    )
)
```

## Training code
```python
svm_history = svm_model.fit(
    TAB_BUNDLE.X_train,
    TAB_BUNDLE.y_train,
    TAB_BUNDLE.X_val,
    TAB_BUNDLE.y_val,
)
svm_history
```

## Prediction code
```python
svm_pred_scaled = svm_model.predict(TAB_BUNDLE.X_test)
svm_pred_raw = TAB_BUNDLE.target_scaler.inverse_transform(svm_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
svm_metrics = evaluate_on_tabular_test(TAB_BUNDLE.y_test_raw, svm_pred_raw, TAB_BUNDLE)
svm_regime_metrics = regime_metrics_on_tabular_test(TAB_BUNDLE.y_test_raw, svm_pred_raw, TAB_BUNDLE)
print(svm_metrics)
svm_regime_metrics
```

# =========================================
# MODEL 3 — ANN
# =========================================

## Required preprocessing dependencies
```python
# Uses TAB_BUNDLE from shared helper section.
```

## Model code
```python
ann_model = ANNForecaster(
    config=ANNConfig(
        hidden_units=[128, 64],
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=100,
        batch_size=128,
        patience=15,
    )
)
```

## Training code
```python
ann_history = ann_model.fit(
    TAB_BUNDLE.X_train,
    TAB_BUNDLE.y_train,
    TAB_BUNDLE.X_val,
    TAB_BUNDLE.y_val,
)
ann_history
```

## Prediction code
```python
ann_pred_scaled = ann_model.predict(TAB_BUNDLE.X_test)
ann_pred_raw = TAB_BUNDLE.target_scaler.inverse_transform(ann_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
ann_metrics = evaluate_on_tabular_test(TAB_BUNDLE.y_test_raw, ann_pred_raw, TAB_BUNDLE)
ann_regime_metrics = regime_metrics_on_tabular_test(TAB_BUNDLE.y_test_raw, ann_pred_raw, TAB_BUNDLE)
print(ann_metrics)
ann_regime_metrics
```

# =========================================
# MODEL 4 — DNN
# =========================================

## Required preprocessing dependencies
```python
# Uses TAB_BUNDLE from shared helper section.
```

## Model code
```python
dnn_model = DNNForecaster(
    config=DNNConfig(
        hidden_units=[256, 128, 64, 32],
        dropout_rates=[0.3, 0.3, 0.2, 0.1],
        use_batch_norm=True,
        learning_rate=1e-3,
        epochs=120,
        batch_size=128,
        patience=15,
    )
)
```

## Training code
```python
dnn_history = dnn_model.fit(
    TAB_BUNDLE.X_train,
    TAB_BUNDLE.y_train,
    TAB_BUNDLE.X_val,
    TAB_BUNDLE.y_val,
)
dnn_history
```

## Prediction code
```python
dnn_pred_scaled = dnn_model.predict(TAB_BUNDLE.X_test)
dnn_pred_raw = TAB_BUNDLE.target_scaler.inverse_transform(dnn_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
dnn_metrics = evaluate_on_tabular_test(TAB_BUNDLE.y_test_raw, dnn_pred_raw, TAB_BUNDLE)
dnn_regime_metrics = regime_metrics_on_tabular_test(TAB_BUNDLE.y_test_raw, dnn_pred_raw, TAB_BUNDLE)
print(dnn_metrics)
dnn_regime_metrics
```

# =========================================
# MODEL 5 — LSTM
# =========================================

## Required preprocessing dependencies
```python
# Uses SEQ_BUNDLE from shared helper section (sequence generation + lookback preserved).
```

## Model code
```python
lstm_model = LSTMForecaster(
    config=LSTMConfig(
        units=[64, 32],
        dropout_rate=0.2,
        dense_units=16,
        learning_rate=1e-3,
        epochs=80,
        batch_size=128,
        patience=12,
    )
)
```

## Training code
```python
lstm_history = lstm_model.fit(
    SEQ_BUNDLE.X_train,
    SEQ_BUNDLE.y_train,
    SEQ_BUNDLE.X_val,
    SEQ_BUNDLE.y_val,
)
lstm_history
```

## Prediction code
```python
lstm_pred_scaled = lstm_model.predict(SEQ_BUNDLE.X_test)
lstm_pred_raw = SEQ_BUNDLE.target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
lstm_metrics = evaluate_on_sequence_test(SEQ_BUNDLE.y_test_raw, lstm_pred_raw, SEQ_BUNDLE)
lstm_regime_metrics = regime_metrics_on_sequence_test(SEQ_BUNDLE.y_test_raw, lstm_pred_raw, SEQ_BUNDLE)
print(lstm_metrics)
lstm_regime_metrics
```

# =========================================
# MODEL 6 — CNN-DNN
# =========================================

## Required preprocessing dependencies
```python
# Uses SEQ_BUNDLE from shared helper section.
```

## Model code
```python
cnn_dnn_model = CNNDNNForecaster(
    config=CNNDNNConfig(
        conv_filters=[64, 32],
        kernel_size=3,
        dense_units=[64, 32],
        dropout_rate=0.3,
        learning_rate=1e-3,
        epochs=100,
        batch_size=128,
        patience=12,
    )
)
```

## Training code
```python
cnn_dnn_history = cnn_dnn_model.fit(
    SEQ_BUNDLE.X_train,
    SEQ_BUNDLE.y_train,
    SEQ_BUNDLE.X_val,
    SEQ_BUNDLE.y_val,
)
cnn_dnn_history
```

## Prediction code
```python
cnn_dnn_pred_scaled = cnn_dnn_model.predict(SEQ_BUNDLE.X_test)
cnn_dnn_pred_raw = SEQ_BUNDLE.target_scaler.inverse_transform(cnn_dnn_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
cnn_dnn_metrics = evaluate_on_sequence_test(SEQ_BUNDLE.y_test_raw, cnn_dnn_pred_raw, SEQ_BUNDLE)
cnn_dnn_regime_metrics = regime_metrics_on_sequence_test(SEQ_BUNDLE.y_test_raw, cnn_dnn_pred_raw, SEQ_BUNDLE)
print(cnn_dnn_metrics)
cnn_dnn_regime_metrics
```

# =========================================
# MODEL 7 — CNN-LSTM
# =========================================

## Required preprocessing dependencies
```python
# Uses SEQ_BUNDLE from shared helper section.
```

## Model code
```python
cnn_lstm_model = CNNLSTMForecaster(
    config=CNNLSTMConfig(
        conv_filters=[64, 32],
        kernel_size=3,
        lstm_units=[64, 32],
        dense_units=16,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=100,
        batch_size=128,
        patience=12,
    )
)
```

## Training code
```python
cnn_lstm_history = cnn_lstm_model.fit(
    SEQ_BUNDLE.X_train,
    SEQ_BUNDLE.y_train,
    SEQ_BUNDLE.X_val,
    SEQ_BUNDLE.y_val,
)
cnn_lstm_history
```

## Prediction code
```python
cnn_lstm_pred_scaled = cnn_lstm_model.predict(SEQ_BUNDLE.X_test)
cnn_lstm_pred_raw = SEQ_BUNDLE.target_scaler.inverse_transform(cnn_lstm_pred_scaled.reshape(-1, 1)).reshape(-1)
```

## Evaluation code
```python
cnn_lstm_metrics = evaluate_on_sequence_test(SEQ_BUNDLE.y_test_raw, cnn_lstm_pred_raw, SEQ_BUNDLE)
cnn_lstm_regime_metrics = regime_metrics_on_sequence_test(SEQ_BUNDLE.y_test_raw, cnn_lstm_pred_raw, SEQ_BUNDLE)
print(cnn_lstm_metrics)
cnn_lstm_regime_metrics
```

# =========================================
# MODEL 8 — CNN-A-LSTM
# =========================================

## Required preprocessing dependencies
```python
# Uses SEQ_BUNDLE from shared helper section.
```

## Model code
```python
cnn_a_lstm_model = CNNAttentionLSTMForecaster(
    config=CNNAttentionLSTMConfig(
        conv_filters=[64, 32],
        kernel_size=3,
        lstm_units=[64, 32],
        attention_units=32,
        dense_units=32,
        dropout_rate=0.2,
        learning_rate=1e-3,
        epochs=100,
        batch_size=128,
        patience=12,
    )
)
```

## Training code
```python
cnn_a_lstm_history = cnn_a_lstm_model.fit(
    SEQ_BUNDLE.X_train,
    SEQ_BUNDLE.y_train,
    SEQ_BUNDLE.X_val,
    SEQ_BUNDLE.y_val,
)
cnn_a_lstm_history
```

## Prediction code
```python
cnn_a_lstm_pred_scaled = cnn_a_lstm_model.predict(SEQ_BUNDLE.X_test)
cnn_a_lstm_pred_raw = SEQ_BUNDLE.target_scaler.inverse_transform(cnn_a_lstm_pred_scaled.reshape(-1, 1)).reshape(-1)

# Attention weights extraction (exact model capability preserved)
attention_weights = cnn_a_lstm_model.get_attention_weights(SEQ_BUNDLE.X_test[:100])
print("attention_weights shape:", attention_weights.shape)
```

## Evaluation code
```python
cnn_a_lstm_metrics = evaluate_on_sequence_test(SEQ_BUNDLE.y_test_raw, cnn_a_lstm_pred_raw, SEQ_BUNDLE)
cnn_a_lstm_regime_metrics = regime_metrics_on_sequence_test(SEQ_BUNDLE.y_test_raw, cnn_a_lstm_pred_raw, SEQ_BUNDLE)
print(cnn_a_lstm_metrics)
cnn_a_lstm_regime_metrics
```

# =========================================
# MODEL 9 — HYBRID RESIDUAL MODEL
# =========================================

## Required preprocessing dependencies
```python
# Uses exact production pipeline from models.residual_hybrid.model
# which internally preserves:
# baseline prediction pipeline, residual computation, regime-aware logic,
# walk-forward search, LightGBM residual correction,
# causal preprocessing, leakage-safe feature generation,
# and calibration step.
```

## Model code
```python
from models.residual_hybrid.model import run_pipeline as run_hybrid_residual_pipeline
```

## Training code
```python
hybrid_summary = run_hybrid_residual_pipeline(
    data_path=DATA_PATH,
    sequence_length=48,
    train_ratio=0.70,
    val_ratio=0.15,
    baseline_model_path=f"{PROJECT_ROOT}/outputs/artifacts/task_d_baseline_aggressive_peak_2x.h5",
    baseline_learning_rate=1e-3,
    baseline_epochs=80,
    baseline_batch_size=128,
    ensemble_size=3,
    walk_forward_splits=4,
    seed=42,
    retrain_baseline=False,
    compare_optional_models=False,
    no_plot=True,
)
```

## Prediction code
```python
# Exact hybrid pipeline writes final predictions to:
# outputs/reports/residual_hybrid_predictions.csv
import pandas as pd
hybrid_predictions = pd.read_csv(f"{PROJECT_ROOT}/outputs/reports/residual_hybrid_predictions.csv")
hybrid_predictions[["timestamp", "actual_ghi", "baseline_pred", "residual_pred", "calibrated_residual", "hybrid_pred"]].head()
```

## Evaluation code
```python
# Exact hybrid evaluation is returned in summary + saved report JSON.
hybrid_summary["baseline_test"], hybrid_summary["hybrid_test"], hybrid_summary["peak_hour_metrics"]
```
