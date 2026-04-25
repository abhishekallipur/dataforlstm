"""
Ablation study module for validating feature importances mathematically.

Iteratively strips features out of the dataset (e.g., lag features, cloud boundaries, 
regimes) and measures the resulting performance degradation on models to prove their
efficacy.
"""

import logging
import re
import numpy as np
import pandas as pd
from typing import List, Callable

logger = logging.getLogger("benchmark.ablation")

ABLATION_GROUPS = {
    "No Solar Physics": ["solar_zenith_angle", "solar_elevation", "air_mass", "extraterrestrial_rad"],
    "No Lag Features": [r"ghi_lag_\d+"],
    "No Volatility/Gradients": [r"ghi_diff_\d+", r"ghi_gradient_.*", r"ghi_abs_diff_\d+"],
    "No Regimes": [r"regime_.*"],
    "No Cloud Proxy": ["cloud_cover_proxy", "clear_sky_index", "clear_sky_ghi"],
    "Base Variables Only": [
        r"ghi_lag_\d+", r"ghi_diff_\d+", r"ghi_gradient_.*", r"ghi_abs_diff_\d+",
        r"regime_.*", "cloud_cover_proxy", "clear_sky_index", "clear_sky_ghi",
        "solar_zenith_angle", "solar_elevation", "air_mass", "extraterrestrial_rad",
        r"ghi_roll_.*"
    ]
}

def remove_features_by_pattern(features: List[str], patterns: List[str]) -> List[str]:
    """Drops any feature matching any of the regex patterns."""
    survivors = []
    for f in features:
        drop = False
        for p in patterns:
            # Exact match or regex
            if f == p or re.match(f"^{p}$", f):
                drop = True
                break
        if not drop:
            survivors.append(f)
    return survivors

def run_ablation_study(
    base_features: List[str],
    train_fn: Callable[[List[str]], None],  # Function that builds/runs model with exact features and returns RMSE
    evaluate_fn: Callable, # Returns metrics dict
) -> pd.DataFrame:
    """
    Executes multiple runs removing groups of features to test stability.
    """
    logger.info("Starting Ablation Study...")
    
    results = []
    
    # 1. Baseline Full Features
    logger.info("Evaluating [Baseline Full Features]")
    baseline_metrics = evaluate_fn(base_features)
    baseline_rmse = baseline_metrics['rmse']
    baseline_peak = baseline_metrics['peak_mae']
    
    results.append({
        "ablated_group": "None (Full Features)",
        "features_count": len(base_features),
        "rmse": baseline_rmse,
        "peak_mae": baseline_peak,
        "rmse_degradation": "0.0%",
        "peak_degradation": "0.0%"
    })
    
    # 2. Iterate Ablation
    for group_name, patterns in ABLATION_GROUPS.items():
        surviving_features = remove_features_by_pattern(base_features, patterns)
        
        if len(surviving_features) == len(base_features):
            continue # None removed
            
        logger.info(f"Evaluating [- {group_name}] ({len(surviving_features)} features left)")
        
        try:
            metrics = evaluate_fn(surviving_features)
            rmse = metrics['rmse']
            peak = metrics['peak_mae']
            
            rmse_deg = ((rmse - baseline_rmse) / baseline_rmse) * 100
            peak_deg = ((peak - baseline_peak) / baseline_peak) * 100
            
            results.append({
                "ablated_group": f"- {group_name}",
                "features_count": len(surviving_features),
                "rmse": rmse,
                "peak_mae": peak,
                "rmse_degradation": f"+{rmse_deg:.2f}%",
                "peak_degradation": f"+{peak_deg:.2f}%"
            })
            
        except Exception as e:
            logger.error(f"Failed ablation on {group_name}: {e}")
            
    df = pd.DataFrame(results)
    logger.info("\n" + df.to_markdown(index=False))
    return df
