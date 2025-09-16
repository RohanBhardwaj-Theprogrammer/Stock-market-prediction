from typing import List, Tuple
import numpy as np
import tensorflow as tf


def build_lstm_model(n_lags: int, number_nodes: int, learning_rate: float) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input((n_lags, 1)),
        tf.keras.layers.LSTM(number_nodes, input_shape=(n_lags, 1)),
        tf.keras.layers.Dense(units=number_nodes, activation="relu"),
        tf.keras.layers.Dense(units=number_nodes, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate), metrics=["mean_absolute_error"])
    return model


def train_lstm(model: tf.keras.Model, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Tuple[tf.keras.callbacks.History, List[float]]:
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    preds = model.predict(X, verbose=0).flatten().tolist()
    return history, preds


def roll_forecast(model: tf.keras.Model, last_seq: np.ndarray, test_values: np.ndarray, days: int) -> List[float]:
    preds = []
    seq = last_seq.copy()
    for i in range(days + 1):
        next_pred = model.predict(seq, verbose=0).flatten()[0]
        preds.append(float(next_pred))
        new_val = test_values[i] if i < len(test_values) else next_pred
        seq = np.append(seq[:, 1:, :], np.array([[[new_val]]]), axis=1)
    return preds
