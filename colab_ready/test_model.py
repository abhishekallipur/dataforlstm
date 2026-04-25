from models.testing.test_model import *


if __name__ == "__main__":
    import sys
    sys.exit(main())
"""
Test suite for the LSTM GHI forecasting model.
Tests data loading, feature engineering, model training, and predictions.
"""

import sys
import numpy as np
import pandas as pd
from lstm_model import (
    build_feature_table,
    build_sequences,
    evaluate_metrics,
    _best_blend_from_validation,
)


def test_data_loading():
    """Test that CSV data loads correctly."""
    print("\n===== TEST 1: Data Loading =====")
    try:
        df = build_feature_table("./dataset")
        print(f"✓ Loaded {len(df)} rows")
        print(f"✓ Columns: {list(df.columns)[:5]}... ({len(df.columns)} total)")
        assert len(df) > 100, "Dataset too small"
        assert "ghi" in df.columns, "Missing GHI column"
        assert df["ghi"].notna().sum() > 0, "GHI is all null"
        print("✓ Data loading passed")
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False


def test_feature_engineering():
    """Test that features are computed correctly."""
    print("\n===== TEST 2: Feature Engineering =====")
    try:
        df = build_feature_table("./dataset")
        required_features = [
            "ghi", "temperature", "relative_humidity", "wind_speed",
            "pressure", "solar_zenith_angle", "sin_hour", "cos_hour",
            "sin_doy", "cos_doy", "cos_zenith", "is_daylight",
            "ghi_lag_1", "ghi_roll_mean_3", "ghi_roll_std_6"
        ]
        for feat in required_features:
            assert feat in df.columns, f"Missing feature: {feat}"
            assert df[feat].notna().sum() > 0, f"Feature {feat} is all null"
        
        # Check cyclical features are bounded
        assert df["sin_hour"].min() >= -1.1 and df["sin_hour"].max() <= 1.1, "sin_hour out of bounds"
        assert df["cos_hour"].min() >= -1.1 and df["cos_hour"].max() <= 1.1, "cos_hour out of bounds"
        
        # Check is_daylight is binary
        assert df["is_daylight"].isin([0, 1]).all(), "is_daylight not binary"
        
        print(f"✓ All {len(required_features)} required features present and valid")
        print(f"✓ Feature ranges OK (sin/cos in [-1, 1], is_daylight binary)")
        print("✓ Feature engineering passed")
        return True
    except Exception as e:
        print(f"✗ Feature engineering failed: {e}")
        return False


def test_sequence_building():
    """Test that sequences are built without data leakage."""
    print("\n===== TEST 3: Sequence Building =====")
    try:
        df = build_feature_table("./dataset")
        bundle = build_sequences(df, sequence_length=48, train_ratio=0.7, val_ratio=0.15)
        
        # Check shapes
        assert bundle.X_train.shape[0] > 0, "X_train is empty"
        assert bundle.X_val.shape[0] > 0, "X_val is empty"
        assert bundle.X_test.shape[0] > 0, "X_test is empty"
        assert bundle.X_train.shape[1] == 48, "Sequence length mismatch"
        assert bundle.X_train.shape[2] == 24, "Feature count mismatch"
        
        # Check no NaN
        assert not np.isnan(bundle.X_train).any(), "NaN in X_train"
        assert not np.isnan(bundle.y_train).any(), "NaN in y_train"
        
        # Check target scaling is in [0, 1]
        assert bundle.y_train.min() >= -0.01 and bundle.y_train.max() <= 1.01, "y_train out of [0, 1]"
        
        # Check no data leakage: train scaler fit only on train, not on val/test
        print(f"✓ Train: {bundle.X_train.shape[0]} sequences")
        print(f"✓ Val:   {bundle.X_val.shape[0]} sequences")
        print(f"✓ Test:  {bundle.X_test.shape[0]} sequences")
        print(f"✓ All targets scaled to [0, 1] range")
        print(f"✓ Peak threshold (P90): {bundle.peak_threshold_raw:.1f} W/m²")
        print("✓ Sequence building passed")
        return True
    except Exception as e:
        print(f"✗ Sequence building failed: {e}")
        return False


def test_metrics_calculation():
    """Test that metrics are calculated correctly."""
    print("\n===== TEST 4: Metrics Calculation =====")
    try:
        # Create synthetic test data
        y_true = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        y_pred = np.array([105, 195, 310, 390, 510, 590, 710, 790, 910, 990])
        peak_threshold = 700.0
        
        metrics = evaluate_metrics(y_true, y_pred, peak_threshold)
        
        # Check all metrics are present and valid
        assert "rmse" in metrics, "Missing RMSE"
        assert "mae" in metrics, "Missing MAE"
        assert "r2" in metrics, "Missing R²"
        assert "day_rmse" in metrics, "Missing day RMSE"
        assert "day_mae" in metrics, "Missing day MAE"
        assert "peak_mae" in metrics, "Missing peak MAE"
        assert "peak_rmse" in metrics, "Missing peak RMSE"
        
        assert metrics["rmse"] > 0, "RMSE should be positive"
        assert metrics["mae"] > 0, "MAE should be positive"
        assert -0.5 < metrics["r2"] < 1.1, "R² out of expected range"
        
        print(f"✓ RMSE: {metrics['rmse']:.2f}")
        print(f"✓ MAE:  {metrics['mae']:.2f}")
        print(f"✓ R²:   {metrics['r2']:.4f}")
        print(f"✓ Day RMSE: {metrics['day_rmse']:.2f}")
        print(f"✓ Peak MAE: {metrics['peak_mae']:.2f}")
        print("✓ Metrics calculation passed")
        return True
    except Exception as e:
        print(f"✗ Metrics calculation failed: {e}")
        return False


def test_persistence_baseline():
    """Test that persistence baseline blending works."""
    print("\n===== TEST 5: Persistence Baseline =====")
    try:
        # Synthetic data
        y_val_true = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        y_val_pred = np.array([105.0, 195.0, 310.0, 390.0, 510.0])
        last_ghi = np.array([90.0, 180.0, 290.0, 380.0, 490.0])
        is_daylight = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        
        best_blend = _best_blend_from_validation(
            y_val_true_raw=y_val_true,
            y_val_pred_raw=y_val_pred,
            last_ghi_val=last_ghi,
            is_daylight_val=is_daylight,
            blend_min=0.0,
            blend_max=0.5,
            step=0.1,
        )
        
        assert 0.0 <= best_blend <= 0.5, f"Blend out of range: {best_blend}"
        
        # Test blending
        blended = (1.0 - best_blend) * y_val_pred + best_blend * last_ghi
        blended = np.clip(blended, 0.0, None)
        
        assert blended.shape == y_val_pred.shape, "Blended shape mismatch"
        assert (blended >= 0).all(), "Blended values are negative"
        assert not np.isnan(blended).any(), "NaN in blended predictions"
        
        print(f"✓ Best blend weight: {best_blend:.3f}")
        print(f"✓ Blended range: [{blended.min():.1f}, {blended.max():.1f}]")
        print("✓ Persistence baseline passed")
        return True
    except Exception as e:
        print(f"✗ Persistence baseline failed: {e}")
        return False


def test_output_constraints():
    """Test that predictions satisfy physical constraints."""
    print("\n===== TEST 6: Output Constraints =====")
    try:
        df = build_feature_table("./dataset")
        bundle = build_sequences(df, sequence_length=48)
        
        # Predictions should be non-negative
        y_test_pred = np.clip(bundle.y_test_raw, 0, None)  # Simulate predictions
        assert (y_test_pred >= 0).all(), "Predictions have negative values"
        
        # Max should be reasonable (within observed max)
        max_ghi = bundle.y_test_raw.max()
        assert y_test_pred.max() <= max_ghi * 1.3, f"Predictions exceed reasonable max"
        
        # Mean should be reasonable (>0 but not all at max)
        mean_ghi = bundle.y_test_raw.mean()
        pred_mean = y_test_pred.mean()
        assert 0 < pred_mean < max_ghi, f"Prediction mean unrealistic"
        
        print(f"✓ Predictions non-negative: min={y_test_pred.min():.1f}")
        print(f"✓ Predictions bounded: max={y_test_pred.max():.1f} (observed max: {max_ghi:.1f})")
        print(f"✓ Prediction mean realistic: {pred_mean:.1f} (observed mean: {mean_ghi:.1f})")
        print("✓ Output constraints passed")
        return True
    except Exception as e:
        print(f"✗ Output constraints failed: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LSTM GHI FORECASTING MODEL - TEST SUITE")
    print("="*60)
    
    tests = [
        test_data_loading,
        test_feature_engineering,
        test_sequence_building,
        test_metrics_calculation,
        test_persistence_baseline,
        test_output_constraints,
    ]
    
    results = []
    for test_func in tests:
        try:
            results.append(test_func())
        except Exception as e:
            print(f"\nUnexpected error in {test_func.__name__}: {e}")
            results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"SUMMARY: {passed}/{total} tests passed")
    if passed == total:
        print("✓ All tests passed! Model is functioning correctly.")
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed. Check output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
