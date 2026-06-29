"""Tests for moment estimation, horizon scaling, PSD repair, the Gaussian generator,
and the run_simulation pipeline."""

import numpy as np
import pytest

from src.portfolio import default_portfolio, SPOT_RATES
from src.simulation import (
    estimate_moments, scale_to_horizon, ensure_psd, gaussian_returns, run_simulation,
)


def test_scale_to_horizon_is_linear_in_time(daily_cov):
    mean = np.array([1e-4, 2e-4, -1e-4])
    mh, ch = scale_to_horizon(mean, daily_cov, horizon_days=10)
    assert np.allclose(mh, 10 * mean)
    assert np.allclose(ch, 10 * daily_cov)


def test_estimate_moments_shapes(synthetic_returns):
    mean, cov = estimate_moments(synthetic_returns)
    assert mean.shape == (3,)
    assert cov.shape == (3, 3)
    assert np.allclose(cov, cov.T)               # symmetric


def test_ensure_psd_passes_through_valid_matrix(daily_cov):
    cov, diag = ensure_psd(daily_cov)
    assert diag.repaired is False
    assert diag.min_eigenvalue > 0
    np.linalg.cholesky(cov)                       # must not raise


def test_ensure_psd_repairs_indefinite_matrix():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])      # eigenvalues 3 and -1 (indefinite)
    cov, diag = ensure_psd(bad)
    assert diag.repaired is True
    assert diag.min_eigenvalue == pytest.approx(-1.0)
    assert np.linalg.eigvalsh(cov).min() >= -1e-12   # repaired to PSD
    np.linalg.cholesky(cov)                       # must not raise


def test_gaussian_returns_reproduces_target_moments(daily_cov):
    rng = np.random.default_rng(7)
    mean = np.array([1e-4, 0.0, -2e-4])
    R = gaussian_returns(mean, daily_cov, 300_000, rng)
    assert R.shape == (300_000, 3)
    assert np.allclose(R.mean(axis=0), mean, atol=5e-5)
    assert np.allclose(np.cov(R, rowvar=False), daily_cov, rtol=0.05, atol=1e-6)


def test_run_simulation_shape_and_reproducibility(synthetic_returns):
    pf = default_portfolio()
    kw = dict(portfolio=pf, spot_rates=SPOT_RATES, returns=synthetic_returns,
              n_sims=5_000, seed=42)
    r1 = run_simulation(**kw)
    r2 = run_simulation(**kw)
    assert r1.pnl.shape == (5_000,)
    assert np.array_equal(r1.pnl, r2.pnl)        # same seed -> identical
    assert r1.psd.repaired is False


def test_run_simulation_different_seed_differs(synthetic_returns):
    pf = default_portfolio()
    base = dict(portfolio=pf, spot_rates=SPOT_RATES, returns=synthetic_returns, n_sims=5_000)
    a = run_simulation(seed=1, **base).pnl
    b = run_simulation(seed=2, **base).pnl
    assert not np.array_equal(a, b)
