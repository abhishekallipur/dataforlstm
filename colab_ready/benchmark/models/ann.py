"""
Shallow Artificial Neural Network (ANN) forecaster.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from benchmark.config import ANNConfig, set_global_seed
from benchmark.models.base import BaseForecaster

logger = logging.getLogger("benchmark.models.ann")


class ANNForecaster(BaseForecaster):
    """Two-layer dense neural network for tabular regression."""

    name = "ANN"
    model_type = "tabular"

    def __init__(self, config: Optional[ANNConfig] = None) -> None:
        self.cfg = config or ANNConfig()
        self.model = None
        self._history_dict: Optional[Dict[str, list]] = None

    def _build(self, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        set_global_seed(42)
        tf.keras.backend.clear_session()

        inputs = tf.keras.Input(shape=(n_features,), name="input")
        x = inputs
        for units in self.cfg.hidden_units:
            x = layers.Dense(units, activation="relu")(x)
            x = layers.Dropout(self.cfg.dropout_rate)(x)
        outputs = layers.Dense(1, name="output")(x)

        model = Model(inputs, outputs, name="ann_ghi")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss="huber",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )
        return model

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import tensorflow as tf

        logger.info("Training %s — layers=%s, epochs=%d", self.name, self.cfg.hidden_units, self.cfg.epochs)
        self.model = self._build(X_train.shape[-1])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.cfg.patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6, verbose=1,
            ),
        ]

        history = self.model.fit(
            X_train, y_train.ravel(),
            validation_data=(X_val, y_val.ravel()),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            callbacks=callbacks,
            verbose=0,
        )

        self._history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
        best_epoch = int(np.argmin(history.history["val_loss"])) + 1
        logger.info("%s training complete — best epoch: %d", self.name, best_epoch)
        return {"best_epoch": best_epoch}

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained yet.")
        return self.model.predict(X, verbose=0).reshape(-1).astype(np.float32)

    def get_training_history(self) -> Optional[Dict[str, list]]:
        return self._history_dict

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "hidden_units": self.cfg.hidden_units,
            "dropout_rate": self.cfg.dropout_rate,
            "epochs": self.cfg.epochs,
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        if self.model is not None:
            path = os.path.join(directory, "ann_model.h5")
            self.model.save(path)
            logger.info("Saved ANN model to %s", path)

    def load(self, directory: str) -> None:
        super().load(directory)
        import tensorflow as tf
        path = os.path.join(directory, "ann_model.h5")
        self.model = tf.keras.models.load_model(path, compile=False)
