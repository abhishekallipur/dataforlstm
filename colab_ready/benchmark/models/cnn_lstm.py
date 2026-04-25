"""
CNN-LSTM forecaster: CNN temporal feature extraction + LSTM sequence modelling.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from benchmark.config import CNNLSTMConfig, set_global_seed
from benchmark.models.base import BaseForecaster

logger = logging.getLogger("benchmark.models.cnn_lstm")


class CNNLSTMForecaster(BaseForecaster):
    """1-D CNN feature extraction followed by LSTM temporal learning."""

    name = "CNN-LSTM"
    model_type = "sequence"

    def __init__(self, config: Optional[CNNLSTMConfig] = None) -> None:
        self.cfg = config or CNNLSTMConfig()
        self.model = None
        self._history_dict: Optional[Dict[str, list]] = None

    def _build(self, sequence_length: int, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        set_global_seed(42)
        tf.keras.backend.clear_session()

        inputs = tf.keras.Input(shape=(sequence_length, n_features), name="seq_input")
        x = inputs

        # CNN feature extraction
        for filters in self.cfg.conv_filters:
            x = layers.Conv1D(filters, kernel_size=self.cfg.kernel_size, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)

        # LSTM temporal learning
        for i, units in enumerate(self.cfg.lstm_units):
            return_seq = i < len(self.cfg.lstm_units) - 1
            x = layers.LSTM(units, return_sequences=return_seq)(x)
            x = layers.Dropout(self.cfg.dropout_rate)(x)

        x = layers.Dense(self.cfg.dense_units, activation="relu")(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs, outputs, name="cnn_lstm_ghi")
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

        seq_len, n_feat = X_train.shape[1], X_train.shape[2]
        logger.info("Training %s — filters=%s, lstm=%s, epochs=%d", self.name, self.cfg.conv_filters, self.cfg.lstm_units, self.cfg.epochs)
        self.model = self._build(seq_len, n_feat)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.cfg.patience,
                restore_best_weights=True, verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1,
            ),
        ]

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            shuffle=True,
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
            "conv_filters": self.cfg.conv_filters,
            "lstm_units": self.cfg.lstm_units,
            "epochs": self.cfg.epochs,
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        if self.model is not None:
            self.model.save(os.path.join(directory, "cnn_lstm_model.h5"))

    def load(self, directory: str) -> None:
        super().load(directory)
        import tensorflow as tf
        self.model = tf.keras.models.load_model(os.path.join(directory, "cnn_lstm_model.h5"), compile=False)
