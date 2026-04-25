"""Quick verification that the data pipeline works end-to-end."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.data_loader import load_benchmark_data
from benchmark.features import TABULAR_FEATURES, SEQUENCE_FEATURES
from benchmark.splitter import build_tabular_bundle, build_sequence_bundle
from benchmark.leakage_audit import run_audit
from benchmark.robustness import PredictionSanitizer
import numpy as np

# Load data
df = load_benchmark_data("dataset")
print(f"Data loaded: {len(df)} rows x {len(df.columns)} cols")

# Tabular split
tab_feats = [f for f in TABULAR_FEATURES if f in df.columns]
print(f"\nTabular features available: {len(tab_feats)} / {len(TABULAR_FEATURES)}")
tab = build_tabular_bundle(df, tab_feats)
print(f"Train: {tab.X_train.shape}, Val: {tab.X_val.shape}, Test: {tab.X_test.shape}")
print(f"Peak threshold: {tab.peak_threshold_raw:.1f} W/m2")

# Sequence split
seq_feats = [f for f in SEQUENCE_FEATURES if f in df.columns]
print(f"\nSequence features available: {len(seq_feats)} / {len(SEQUENCE_FEATURES)}")
seq = build_sequence_bundle(df, seq_feats, sequence_length=48)
print(f"Train: {seq.X_train.shape}, Val: {seq.X_val.shape}, Test: {seq.X_test.shape}")

# Leakage audit
audit = run_audit(
    df=df,
    train_timestamps=tab.train_timestamps,
    val_timestamps=tab.val_timestamps,
    test_timestamps=tab.test_timestamps,
    feature_cols=tab_feats,
    scaler_n_samples=tab.feature_scaler.n_samples_seen_,
    train_size=len(tab.y_train),
    y_train=tab.y_train_raw,
    y_val=tab.y_val_raw,
    y_test=tab.y_test_raw,
)
status = "ALL PASSED" if audit.all_passed else "FAILED"
print(f"\nLeakage audit: {status} ({len(audit.checks)} checks)")
for c in audit.checks:
    mark = "PASS" if c.passed else "FAIL"
    print(f"  [{mark}] {c.name}: {c.detail}")

# Sanitizer test
sanitizer = PredictionSanitizer.build_from_training(tab.y_train_raw)
fake_preds = np.random.rand(len(tab.y_test_raw)) * 800
fake_preds[0] = np.nan
fake_preds[1] = -50
fake_preds[2] = 2000
cleaned, report = sanitizer.sanitize(fake_preds, "test")
print(f"\nSanitizer: {report.n_total_corrected} corrections on {report.n_predictions} predictions")

print("\n=== DATA PIPELINE VERIFICATION COMPLETE ===")
