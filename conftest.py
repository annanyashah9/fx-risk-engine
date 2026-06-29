"""Shared pytest fixtures and import-path setup for the FX risk engine test suite.

Placing this at the repo root ensures `import src...` resolves when running `pytest` from
anywhere in the project. Fixtures here are available to all tests.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(__file__))

# Portfolio order used throughout the engine.
PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]


@pytest.fixture
def daily_cov():
    """A realistic, positive-definite 3x3 daily-return covariance (EUR/GBP +, both vs JPY -)."""
    return np.array([
        [1.0e-4, 0.55e-4, -0.35e-4],
        [0.55e-4, 1.3e-4, -0.25e-4],
        [-0.35e-4, -0.25e-4, 2.0e-4],
    ])


@pytest.fixture
def synthetic_returns(daily_cov):
    """1,000 days of correlated, zero-mean log returns with `daily_cov` as covariance.

    Columns are in portfolio order so they can be passed straight to `run_simulation`.
    """
    rng = np.random.default_rng(0)
    L = np.linalg.cholesky(daily_cov)
    data = rng.standard_normal((1000, 3)) @ L.T
    return pd.DataFrame(data, columns=PAIRS)
