"""Tests for the fat-tailed generators: df estimation, covariance/correlation preservation,
and that tails are genuinely heavier than Gaussian."""

import numpy as np
import pytest
from scipy import stats

from src.fat_tails import (
    corr_from_cov, fit_marginal_dfs, multivariate_t_returns, t_copula_returns,
)


def test_corr_from_cov(daily_cov):
    corr, sigma = corr_from_cov(daily_cov)
    assert np.allclose(np.diag(corr), 1.0)
    assert np.allclose(sigma, np.sqrt(np.diag(daily_cov)))
    # EUR/GBP positive, both vs JPY negative (the inverted-quote sign structure).
    assert corr[0, 1] > 0 and corr[0, 2] < 0 and corr[1, 2] < 0


def test_fit_marginal_dfs_ordering_and_bounds():
    """Method-of-moments df should DECREASE as tails get fatter, and stay >= the floor.

    Heavier tails (smaller true df) -> larger kurtosis -> smaller estimated df.
    """
    rng = np.random.default_rng(0)
    n = 400_000
    cols = np.column_stack([
        rng.standard_t(15, n),   # mild tails -> larger df
        rng.standard_t(6, n),    # medium
        rng.standard_t(5, n),    # heavier -> smaller df
    ])
    dfs = fit_marginal_dfs(cols)
    assert dfs[0] > dfs[1] > dfs[2]              # ordered by tail-heaviness
    assert np.all(dfs >= 4.5)                    # floored for finite, stable variance


def test_fit_marginal_dfs_near_normal_hits_cap():
    rng = np.random.default_rng(1)
    near_normal = rng.standard_normal((200_000, 1))
    df = fit_marginal_dfs(near_normal)[0]
    assert df > 20                               # ~0 excess kurtosis -> very large df (capped)


def test_multivariate_t_preserves_covariance(daily_cov):
    rng = np.random.default_rng(4)
    mean = np.zeros(3)
    R = multivariate_t_returns(mean, daily_cov, 400_000, rng, df=8)
    assert R.shape == (400_000, 3)
    assert np.allclose(np.cov(R, rowvar=False), daily_cov, rtol=0.12, atol=1e-6)


def test_multivariate_t_requires_df_above_two(daily_cov):
    rng = np.random.default_rng(5)
    with pytest.raises(ValueError):
        multivariate_t_returns(np.zeros(3), daily_cov, 10, rng, df=2.0)


def test_t_copula_preserves_correlation(daily_cov):
    rng = np.random.default_rng(6)
    target_corr, _ = corr_from_cov(daily_cov)
    R = t_copula_returns(np.zeros(3), daily_cov, 300_000, rng,
                         copula_df=8, marginal_dfs=np.array([8.0, 8.0, 8.0]))
    emp_corr, _ = corr_from_cov(np.cov(R, rowvar=False))
    assert np.allclose(emp_corr, target_corr, atol=0.05)


def test_t_copula_has_fatter_marginal_tails(daily_cov):
    rng = np.random.default_rng(8)
    R = t_copula_returns(np.zeros(3), daily_cov, 300_000, rng,
                         copula_df=5, marginal_dfs=np.array([5.0, 5.0, 5.0]))
    # Each marginal should show clear positive excess kurtosis (Gaussian -> ~0).
    for j in range(3):
        assert stats.kurtosis(R[:, j]) > 1.0
