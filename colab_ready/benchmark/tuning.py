"""
Bayesian Hyperparameter Optimization via Optuna for tabular models (LightGBM).
"""

import logging
import numpy as np

logger = logging.getLogger("benchmark.tuning")

def tune_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, n_trials: int = 30) -> dict:
    """
    Sweeps LightGBM hyperparameters optimizing for both RMSE and Peak MAE.
    """
    try:
        import optuna
        from lightgbm import LGBMRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error
    except ImportError:
        logger.warning("Optuna is not installed. Returning default parameters.")
        return {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 7
        }

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "random_state": 42,
            "n_jobs": -1
        }
        
        model = LGBMRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        # Calculate Peak MAE penalty
        peak_mask = y_val > 500.0  # arbitrary threshold for penalization during tuning
        if peak_mask.any():
            peak_mae = mean_absolute_error(y_val[peak_mask], preds[peak_mask])
        else:
            peak_mae = 0.0
            
        # Composite objective
        return rmse + (0.5 * peak_mae)

    # Suppress optuna logging for cleaner terminal output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=1) # n_jobs=1 because LGBM is already multi-threaded
    
    logger.info(f"Tuning complete. Best Composite Loss: {study.best_value:.2f}")
    return study.best_params
