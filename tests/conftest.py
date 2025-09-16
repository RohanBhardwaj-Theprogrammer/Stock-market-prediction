import os
import tempfile
import numpy as np
import pandas as pd
import pytest

@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d

@pytest.fixture
def small_series():
    # simple increasing sequence with noise
    idx = pd.date_range('2020-01-01', periods=120, freq='D')
    values = np.linspace(100, 150, 120) + np.random.normal(0, 1, 120)
    s = pd.Series(values, index=idx, name='Adj_Close')
    return s
