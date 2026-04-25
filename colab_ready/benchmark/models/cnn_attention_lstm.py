"""
CNN + Attention + LSTM forecaster (CNN-A-LSTM) with attention visualization.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import numpy as np

from benchmark.config import CNNAttentionLSTMConfig, set_global_seed
from benchmark.models.base import BaseForecaster

logger = logging.getLogger("benchmark.models.cnn_attention_lstm")


# Serializable layer to replace Lambda(tf.reduce_sum)
try:
    import tensorflow as tf

    @tf.keras.utils.register_keras_serializable(package="benchmark")
    class ReduceSumLayer(tf.keras.layers.Layer):
        """Sum over the time axis (axis=1) — serializable alternative to Lambda."""
        def call(self, x):
            return tf.reduce_sum(x, axis=1)

        def get_config(self):
            return super().get_config()
except ImportError:
    pass


class CNNAttentionLSTMForecaster(BaseForecaster):
    """
    CNN → LSTM (return_sequences) → Attention → Dense.

    The attention mechanism computes a weighted sum of LSTM hidden states,
    allowing the model to focus on the most informative timesteps.
    Attention weights are extractable for visualization.
    """

    name = "CNN-A-LSTM"
    model_type = "sequence"

    def __init__(self, config: Optional[CNNAttentionLSTMConfig] = None) -> None:
        self.cfg = config or CNNAttentionLSTMConfig()
        self.model = None
        self._attention_model = None
        self._history_dict: Optional[Dict[str, list]] = None

    def _build(self, sequence_length: int, n_features: int):
        import tensorflow as tf
        from tensorflow.keras import layers, Model

        set_global_seed(42)
        tf.keras.backend.clear_session()

        inputs = tf.keras.Input(shape=(sequence_length, n_features), name="seq_input")
        x = inputs

        # --- CNN feature extraction ---
        for filters in self.cfg.conv_filters:
            x = layers.Conv1D(filters, kernel_size=self.cfg.kernel_size, padding="same", activation="relu")(x)
            x = layers.BatchNormalization()(x)

        # --- LSTM (return sequences for attention) ---
        for i, units in enumerate(self.cfg.lstm_units):
            x = layers.LSTM(units, return_sequences=True, name=f"lstm_{i}")(x)
            x = layers.Dropout(self.cfg.dropout_rate)(x)

        # --- Attention mechanism ---
        # Score each timestep
        score = layers.Dense(1, activation="tanh", name="attention_score")(x)
        attention_weights = layers.Softmax(axis=1, name="attention_weights")(score)

        # Weighted sum
        context = layers.Multiply(name="weighted_seq")([x, attention_weights])
        context = ReduceSumLayer(name="context_vector")(context)

        # --- Dense head ---
        x = layers.Dense(self.cfg.dense_units, activation="relu", name="dense")(context)
        x = layers.Dropout(self.cfg.dropout_rate)(x)
        outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

        model = Model(inputs, outputs, name="cnn_attention_lstm_ghi")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.cfg.learning_rate),
            loss="huber",
            metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
        )

        # Separate model for extracting attention weights
        attention_model = Model(inputs, attention_weights, name="attention_extractor")

        return model, attention_model

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
        logger.info(
            "Training %s — filters=%s, lstm=%s, attn_units=%d, epochs=%d",
            self.name, self.cfg.conv_filters, self.cfg.lstm_units,
            self.cfg.attention_units, self.cfg.epochs,
        )
        self.model, self._attention_model = self._build(seq_len, n_feat)

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

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Extract attention weights for a batch of inputs.

        Returns shape (batch, timesteps, 1).
        """
        if self._attention_model is None:
            raise RuntimeError("Attention model not built.")
        return self._attention_model.predict(X, verbose=0)

    def get_training_history(self) -> Optional[Dict[str, list]]:
        return self._history_dict

    def get_params(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model_type": self.model_type,
            "conv_filters": self.cfg.conv_filters,
            "lstm_units": self.cfg.lstm_units,
            "attention_units": self.cfg.attention_units,
            "epochs": self.cfg.epochs,
        }

    def save(self, directory: str) -> None:
        super().save(directory)
        if self.model is not None:
            path = os.path.join(directory, "cnn_attn_lstm_model.keras")
            self.model.save(path)
            logger.info("Saved CNN-A-LSTM model to %s", path)

    def load(self, directory: str) -> None:
        super().load(directory)
        import tensorflow as tf
        self.model = tf.keras.models.load_model(
            os.path.join(directory, "cnn_attn_lstm_model.keras"),
            compile=False,
            custom_objects={"ReduceSumLayer": ReduceSumLayer},
        )
