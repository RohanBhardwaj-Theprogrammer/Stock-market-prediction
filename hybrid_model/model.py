import os
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .pipeline import HybridPipeline, PipelineParams


@dataclass
class ModelParams:
    n_lags: int = 30
    days: int = 30
    epochs: int = 20
    learning_rate: float = 0.001
    batch_size: int = 32
    number_nodes: int = 64
    acf_pacf_lags: int = 30


@dataclass
class RunResult:
    mse: float
    rmse: float
    mae: float
    arima_mse: float
    arima_rmse: float
    arima_mae: float
    final_forecast_next: float
    output_dir: str
    days: int
    predictions_lstm: List[float]
    predictions_arima_err: List[float]
    predictions_final: List[float]


class HybridForecaster:
    def __init__(self, df: pd.DataFrame, close_col: str = "Adj_Close", output_dir: str = "Output", params: Optional[ModelParams] = None):
        self.df = df
        self.close_col = close_col
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.params = params or ModelParams()

    def run(self) -> RunResult:
        p = self.params
        series = self.df[self.close_col].dropna()
        pipeline = HybridPipeline(series, self.output_dir, PipelineParams(
            n_lags=p.n_lags,
            days=p.days,
            epochs=p.epochs,
            learning_rate=p.learning_rate,
            batch_size=p.batch_size,
            number_nodes=p.number_nodes,
            acf_pacf_lags=p.acf_pacf_lags,
        ))
        res = pipeline.run()
        return RunResult(
            mse=res["mse"],
            rmse=res["rmse"],
            mae=res["mae"],
            arima_mse=res["arima_mse"],
            arima_rmse=res["arima_rmse"],
            arima_mae=res["arima_mae"],
            final_forecast_next=res["final_forecast_next"],
            output_dir=self.output_dir,
            days=p.days,
            predictions_lstm=res["predictions_lstm"],
            predictions_arima_err=res["predictions_arima_err"],
            predictions_final=res["predictions_final"],
        )


def load_dataset(csv_path: str) -> pd.DataFrame:
    from .components.data import load_dataset as _load
    return _load(csv_path)
