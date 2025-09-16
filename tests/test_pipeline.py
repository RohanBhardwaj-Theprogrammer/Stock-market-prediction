import numpy as np
from hybrid_model.pipeline import HybridPipeline, PipelineParams


class DummyModel:
    def __init__(self, val=1.0):
        self.val = val
    def predict(self, X, verbose=0):
        import numpy as np
        return np.array([[self.val]])


def test_pipeline_runs_with_monkeypatch(monkeypatch, small_series, tmp_output_dir):
    # Monkeypatch LSTM builder to use a constant predictor and fast return
    from hybrid_model.components import lstm as lstm_mod
    def fake_build(n_lags, number_nodes, lr):
        return DummyModel(val=small_series.mean())
    def fake_train(model, X, y, epochs, batch_size):
        # pretend perfect fit
        preds = y.tolist()
        class H: pass
        return H(), preds
    def fake_roll(model, last_seq, test_values, days):
        return [float(small_series.mean())] * (days + 1)
    monkeypatch.setattr(lstm_mod, 'build_lstm_model', fake_build)
    monkeypatch.setattr(lstm_mod, 'train_lstm', fake_train)
    monkeypatch.setattr(lstm_mod, 'roll_forecast', fake_roll)

    # Monkeypatch ARIMA to skip heavy fit
    from hybrid_model.components import arima as arima_mod
    def fake_fit_err(errors, acf_pacf_lags, output_dir):
        return (1,0,0), [0.0], [0.0]*len(errors)
    def fake_predict_err_future(errors, order, horizon):
        return [0.0] * (horizon + 1)
    monkeypatch.setattr(arima_mod, 'fit_error_arima', fake_fit_err)
    monkeypatch.setattr(arima_mod, 'predict_error_future', fake_predict_err_future)

    # Monkeypatch plotting to no-op (avoid requiring matplotlib)
    from hybrid_model.components import plots as plots_mod
    monkeypatch.setattr(plots_mod, 'plot_series', lambda *a, **k: None)
    monkeypatch.setattr(plots_mod, 'plot_two_series', lambda *a, **k: None)
    monkeypatch.setattr(plots_mod, 'plot_bar', lambda *a, **k: None)
    monkeypatch.setattr(plots_mod, 'plot_errors', lambda *a, **k: None)

    params = PipelineParams(n_lags=10, days=30, epochs=1, learning_rate=0.001, batch_size=8, number_nodes=16)
    pipe = HybridPipeline(small_series, tmp_output_dir, params)
    res = pipe.run()
    assert 'mse' in res and 'final_forecast_next' in res
    assert isinstance(res['predictions_final'], list)
    assert len(res['predictions_final']) == params.days
