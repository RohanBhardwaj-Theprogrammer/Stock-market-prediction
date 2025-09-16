import numpy as np
from hybrid_model.components.data import to_lagged_sequences


def test_to_lagged_sequences_shapes():
    values = np.arange(20, dtype=float)
    X, y = to_lagged_sequences(values, n_lags=5)
    assert X.shape == (15, 5, 1)
    assert y.shape == (15,)
    # first target equals values[5]
    assert y[0] == 5.0
