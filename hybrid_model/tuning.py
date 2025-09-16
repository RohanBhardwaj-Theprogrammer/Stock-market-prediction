import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .components.data import split_train_test, to_lagged_sequences
from .components.lstm import build_lstm_model, train_lstm


@dataclass
class SearchSpace:
    n_lags: Tuple[int, int] = (15, 60)  # integer range
    epochs: Tuple[int, int] = (5, 20)
    learning_rate: Tuple[float, float] = (1e-4, 5e-3)  # log-uniform-ish via sampling exp
    batch_size_choices: Tuple[int, ...] = (16, 32, 64)
    number_nodes_choices: Tuple[int, ...] = (32, 64, 128)


def _sample(space: SearchSpace) -> Dict:
    n_lags = random.randint(space.n_lags[0], space.n_lags[1])
    epochs = random.randint(space.epochs[0], space.epochs[1])
    lr = 10 ** random.uniform(np.log10(space.learning_rate[0]), np.log10(space.learning_rate[1]))
    batch_size = random.choice(space.batch_size_choices)
    number_nodes = random.choice(space.number_nodes_choices)
    return {
        "n_lags": n_lags,
        "epochs": epochs,
        "learning_rate": float(lr),
        "batch_size": batch_size,
        "number_nodes": number_nodes,
    }


def time_series_validation(series: pd.Series, params: Dict) -> float:
    days = max(10, int(len(series) * 0.2))  # 20% horizon minimum 10
    train, val = split_train_test(series, days)

    X, y = to_lagged_sequences(train.values, params["n_lags"])
    if len(X) < 5:
        return float("inf")

    model = build_lstm_model(params["n_lags"], params["number_nodes"], params["learning_rate"])
    history, _ = train_lstm(model, X, y, params["epochs"], params["batch_size"])

    # roll forward to validation horizon
    last_seq = train.values[-params["n_lags"] :].reshape((1, params["n_lags"], 1))
    preds = []
    seq = last_seq.copy()
    for i in range(days):
        next_pred = model.predict(seq, verbose=0).flatten()[0]
        preds.append(float(next_pred))
        new_val = val.values[i]  # teacher forcing on validation
        seq = np.append(seq[:, 1:, :], np.array([[[new_val]]]), axis=1)

    # metric: RMSE on validation horizon
    rmse = float(np.sqrt(np.mean((np.array(preds) - val.values[: days]) ** 2)))
    return rmse


def random_search(series: pd.Series, trials: int, space: SearchSpace) -> Dict:
    best = None
    for t in range(1, trials + 1):
        cand = _sample(space)
        score = time_series_validation(series, cand)
        if best is None or score < best["score"]:
            best = {"params": cand, "score": score}
    return best or {"params": {}, "score": float("inf")}
