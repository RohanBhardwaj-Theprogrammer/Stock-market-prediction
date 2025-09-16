from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .components.data import split_train_test, to_lagged_sequences
from .components.lstm import build_lstm_model, train_lstm, roll_forecast
from .components.arima import fit_error_arima, predict_error_future
from .components.plots import plot_series, plot_two_series, plot_bar, plot_errors


@dataclass
class PipelineParams:
    n_lags: int = 30
    days: int = 30
    epochs: int = 10
    learning_rate: float = 0.001
    batch_size: int = 32
    number_nodes: int = 64
    acf_pacf_lags: int = 30


def metrics(y_true: List[float], y_pred: List[float]):
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    return float(mse), float(rmse), float(mae)


class HybridPipeline:
    def __init__(self, series: pd.Series, output_dir: str, params: PipelineParams):
        self.series = series
        self.output_dir = output_dir
        self.p = params

    def run(self):
        p = self.p
        # split
        train, test = split_train_test(self.series, p.days)

        # plots
        plot_series(self.series.index, self.series.values, "Raw Time Series Data", "Close Price", self.output_dir)
        plot_two_series(
            list(train.index) + list(test.index),
            list(train.values) + [np.nan] * len(test),
            [np.nan] * len(train) + list(test.values),
            "Train and Test Data",
            self.output_dir,
            filename="Train and Test Data.jpg",
        )

        # LSTM setup
        X, y = to_lagged_sequences(train.values, p.n_lags)
        model = build_lstm_model(p.n_lags, p.number_nodes, p.learning_rate)
        history, train_preds = train_lstm(model, X, y, p.epochs, p.batch_size)

        plot_two_series(train.index[p.n_lags :], y.tolist(), train_preds, "LSTM PREDICTIONS VS ACTUAL Values For TRAIN Data Set", self.output_dir)

        last_seq = train.values[-p.n_lags :].reshape((1, p.n_lags, 1))
        preds_lstm = roll_forecast(model, last_seq, test.values, p.days)
        plot_two_series(test.index, test.values, preds_lstm[:-1], "LSTM Predictions VS Actual Values", self.output_dir)

        # errors + arima
        errs = [true - pred for true, pred in zip(y.tolist(), train_preds)]
        plot_errors(errs, self.output_dir)
        order, _, preds_err_full = fit_error_arima(errs, p.acf_pacf_lags, self.output_dir)
        preds_err_future = predict_error_future(errs, order, len(test))

        # metrics
        mse, rmse, mae = metrics(test.values[: p.days], preds_lstm[: p.days])
        plot_bar(["MSE", "RMSE", "MAE"], [mse, rmse, mae], "Model Accuracy Metrics", self.output_dir, "Model Accuracy Metrics.jpg")
        arima_mse, arima_rmse, arima_mae = metrics(errs, preds_err_full)
        plot_bar(["MSE", "RMSE", "MAE"], [arima_mse, arima_rmse, arima_mae], "ARIMA Model Accuracy Metrics", self.output_dir, "ARIMA Model Accuracy Metrics.jpg")

        # final
        final_preds = [e + y for e, y in zip(preds_err_future[: p.days], preds_lstm[: p.days])]
        plot_two_series(test.index[: p.days], test.values[: p.days], final_preds[: p.days], "Final Predictions with Error Correction", self.output_dir)

        final_next = float(preds_lstm[p.days] + preds_err_future[p.days])

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "arima_mse": arima_mse,
            "arima_rmse": arima_rmse,
            "arima_mae": arima_mae,
            "final_forecast_next": final_next,
            "predictions_lstm": preds_lstm,
            "predictions_arima_err": preds_err_future,
            "predictions_final": final_preds,
        }
