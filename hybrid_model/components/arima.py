import os
from typing import List, Tuple


def fit_error_arima(errors: List[float], acf_pacf_lags: int, output_dir: str) -> Tuple[Tuple[int, int, int], List[float], List[float]]:
    import matplotlib.pyplot as plt  # lazy import
    from pmdarima import auto_arima  # lazy import
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # lazy import
    from statsmodels.tsa.arima.model import ARIMA  # lazy import

    plot_acf(errors, lags=acf_pacf_lags)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "ACF.jpg"))
    plt.close()
    plot_pacf(errors, lags=acf_pacf_lags)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "PACF.jpg"))
    plt.close()

    finding = auto_arima(errors, trace=False, suppress_warnings=True)
    order = finding.order
    arima = ARIMA(errors, order=order).fit()

    preds_future = arima.predict(start=len(errors), end=len(errors), typ="levels").tolist()  # placeholder, caller can extend
    preds_full = arima.predict(start=0, end=len(errors) - 1, typ="levels").tolist()
    return order, preds_future, preds_full


def predict_error_future(errors: List[float], order: Tuple[int, int, int], horizon: int) -> List[float]:
    from statsmodels.tsa.arima.model import ARIMA  # lazy import
    arima = ARIMA(errors, order=order).fit()
    preds = arima.predict(start=len(errors), end=len(errors) + horizon, typ="levels").tolist()
    return preds
