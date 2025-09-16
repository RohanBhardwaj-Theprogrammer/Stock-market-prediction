from typing import Tuple
import numpy as np
import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    df.columns = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
    df = df.dropna()
    return df


def split_train_test(series: pd.Series, days: int) -> Tuple[pd.Series, pd.Series]:
    train, test = series.iloc[:-days], series.iloc[-days:]
    return train, test


def to_lagged_sequences(values: np.ndarray, n_lags: int):
    X, y = [], []
    for i in range(n_lags, len(values)):
        X.append(values[i - n_lags : i])
        y.append(values[i])
    X = np.array(X).reshape((-1, n_lags, 1))
    y = np.array(y)
    return X, y
