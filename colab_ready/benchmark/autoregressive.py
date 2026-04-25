"""
Autoregressive forecasting mechanism for testing true deployment capabilities
without Teacher Forcing.

By default, sequence and tabular evaluations provide the model with the 
ACTUAL ground-truth of `ghi_lag_1`, which creates an artificially optimistic
"Teacher Forcing" evaluation. This script implements functions to iteratively
feed the model's own predictions forward for a horizon H, calculating the
compounding error.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple

logger = logging.getLogger("benchmark.autoregressive")

def simulate_recursive_predictions(
    model, 
    X_test_baseline: np.ndarray, 
    horizon: int = 24, 
    is_sequence: bool = False,
    ghi_idx: int = 0,
    roll_mean_6_idx: int = -1,  # Placeholder, must resolve feature idx
):
    """
    Simulates recursive forecasting over a 24-hour horizon.
    
    Rather than rebuilding the entire Pandas DataFrame per hour (which is 
    extremely slow), we use the pre-built X_test matrix (which has perfect
    future exogenous variables like hour/sun position). We just overwrite
    the historical 'ghi' dependencies dynamically using a numpy circular buffer.
    """
    preds_recursive = np.zeros(len(X_test_baseline), dtype=np.float32)
    
    # In a full production setup, this loops via Pandas
    pass

def simulate_recursive_day_ahead(
    model, 
    full_df: pd.DataFrame, 
    start_idx: int, 
    horizon: int = 24,
    feature_cols: list = None,
    is_sequence: bool = False,
    seq_len: int = 48,
    feature_scaler = None,
    target_scaler = None
) -> np.ndarray:
    """
    Simulates a true deployment scenario where we only know GHI up to start_idx.
    We predict H steps forward, using our PREDICTIONS as the past GHI to compute
    future lags and rolling means, blocking all ground-truth leakage.
    """
    from benchmark.features import enrich_features
    
    # Create an isolated workspace up to the start_idx + horizon
    # Ensure we only use GHI up to start_idx
    work_df = full_df.iloc[:start_idx + horizon].copy()
    work_df.loc[work_df.index[start_idx:], "ghi"] = np.nan
    
    preds = []
    
    # Step forward one hour at a time
    for i in range(horizon):
        current_idx = start_idx + i
        
        # We must re-enrich features dynamically based on our forecasted GHI
        enriched = enrich_features(work_df.iloc[:current_idx + 1])
        
        # Format the input for the model
        if is_sequence:
            x_input = enriched[feature_cols].values[-seq_len:]
            if feature_scaler:
                x_input = feature_scaler.transform(x_input)
            x_input = np.expand_dims(x_input, axis=0) # shape (1, seq_len, features)
        else:
            x_input = enriched[feature_cols].values[-1:]
            if feature_scaler:
                x_input = feature_scaler.transform(x_input)
            
        # Predict
        pred_scaled = model.predict(x_input)[0]
        # In case the model returns an array
        if isinstance(pred_scaled, np.ndarray):
            pred_scaled = pred_scaled[0]
            
        if target_scaler:
            pred_ghi = target_scaler.inverse_transform(np.array([[pred_scaled]]))[0][0]
        else:
            pred_ghi = pred_scaled
            
        pred_ghi = max(0.0, float(pred_ghi)) # No negative GHI
        preds.append(pred_ghi)
        
        # Inject the prediction back into the timeline
        work_df.loc[work_df.index[current_idx], "ghi"] = pred_ghi
        
    return np.array(preds)

    
