"""Fat-tailed shock generators for the FX risk scenario engine (Phase 3).

Real FX returns have heavier tails than a normal distribution. These generators replace
the Gaussian shock from Phase 2 while PRESERVING the Phase-2 correlation structure (both
reuse the same Cholesky step). They are signature-compatible with
`simulation.gaussian_returns` — `(mean_h, cov_h, n_sims, rng, **params)` -> (n_sims, dim) —
so `run_simulation()`, `mtm()`, and `risk.var_es()` are reused UNCHANGED (extra params are
bound via functools.partial by the caller).

Two models (see plan):
  * multivariate_t_returns : elliptical Student-t, ONE shared df. A single common chi-square
    mixing variable scales the whole return vector -> all currencies share the same tail
    heaviness AND extremes are forced to occur jointly. Pearson correlation preserved exactly.
  * t_copula_returns (HEADLINE) : separates dependence (a t-copula with df `copula_df`) from
    marginal tail fatness (per-currency `marginal_dfs`). Preserves the rank-correlation /
    copula correlation exactly; Pearson preserved approximately (nonlinear margins).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def corr_from_cov(cov: np.ndarray):
    """Return (correlation_matrix, sigma_vector) from a covariance matrix."""
    cov = np.asarray(cov, dtype=float)
    sigma = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sigma, sigma)
    corr = (corr + corr.T) / 2.0          # symmetrize against fp noise
    np.fill_diagonal(corr, 1.0)
    return corr, sigma


def fit_marginal_dfs(returns, df_floor: float = 4.5, df_cap: float = 50.0) -> np.ndarray:
    """Per-column Student-t degrees of freedom by matching excess kurtosis (method of moments).

    For a Student-t with df=nu (>4), the excess kurtosis is k = 6/(nu-4), so nu = 4 + 6/k.
    We use this rather than `scipy.stats.t.fit` (MLE) because the MLE is DEGENERATE near the
    variance boundary nu->2: on this data it returns df=2.07 for EURUSD even though EURUSD has
    the LOWEST kurtosis of the three pairs (a known instability). At df~2 the variance is so
    outlier-dominated that the standardized 1% quantile becomes THINNER than normal, which is
    nonsense for a "fat tail" model. The kurtosis match is robust, correctly ordered with each
    series' tail-heaviness, and keeps variance finite.

    `returns` is (n_obs, dim) array-like. df is floored (variance/kurtosis well-defined and
    stable) and capped (near-Gaussian series -> very large df, avoid numeric blowup).
    """
    arr = np.asarray(returns, dtype=float)
    dfs = []
    for j in range(arr.shape[1]):
        k = stats.kurtosis(arr[:, j])           # excess (Fisher); normal -> 0
        nu = 4.0 + 6.0 / k if k > 1e-6 else df_cap
        dfs.append(float(np.clip(nu, df_floor, df_cap)))
    return np.array(dfs, dtype=float)


def fit_joint_df(returns, grid=None, df_floor: float = 2.05) -> float:
    """Shared elliptical/copula df via 1-D grid MLE on the multivariate-t log-likelihood.

    For each candidate nu, the scale matrix is set to sample_cov*(nu-2)/nu so the implied
    covariance equals the sample covariance, isolating nu as the tail parameter. Returns
    the nu maximizing the summed `multivariate_t.logpdf`.
    """
    arr = np.asarray(returns, dtype=float)
    if grid is None:
        grid = np.arange(3, 16)  # 3..15
    mean = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False, ddof=1)

    best_nu, best_ll = None, -np.inf
    for nu in grid:
        nu = float(nu)
        if nu <= 2:
            continue
        shape = cov * (nu - 2.0) / nu
        ll = stats.multivariate_t.logpdf(arr, loc=mean, shape=shape, df=nu).sum()
        if ll > best_ll:
            best_ll, best_nu = ll, nu
    return max(best_nu, df_floor)


# ---------------------------------------------------------------------------
# Generators (the Phase-2 swap point; same signature as gaussian_returns)
# ---------------------------------------------------------------------------
def multivariate_t_returns(mean_h, cov_h, n_sims, rng, *, df) -> np.ndarray:
    """Correlated multivariate Student-t horizon returns, shape (n_sims, dim).

    X = mean_h + (Z @ Lᵀ) * sqrt(nu / W),  Z~N(0,I), W~chi2(nu),
    with scale = cov_h*(nu-2)/nu so Cov(X) = cov_h (correlation preserved EXACTLY).
    """
    nu = float(df)
    if nu <= 2:
        raise ValueError(f"multivariate-t df must be > 2 for finite variance; got {nu}")
    mean_h = np.asarray(mean_h, dtype=float)
    dim = mean_h.size

    scale = np.asarray(cov_h, dtype=float) * (nu - 2.0) / nu
    L = np.linalg.cholesky(scale)
    Z = rng.standard_normal((n_sims, dim))
    G = Z @ L.T                               # correlated normal, cov = scale
    W = rng.chisquare(nu, size=(n_sims, 1))   # ONE shared mixing var per scenario row
    return mean_h + G * np.sqrt(nu / W)


def t_copula_returns(mean_h, cov_h, n_sims, rng, *, copula_df, marginal_dfs) -> np.ndarray:
    """t-copula returns: t-copula dependence + per-currency Student-t marginals.

    Steps:
      1. Correlated multivariate-t SCORES with df=copula_df and correlation R=corr(cov_h):
         T = (Z @ L_Rᵀ) * sqrt(nu_c / W).
      2. Map to uniforms via the t CDF: U = t_cdf(T; nu_c)  -> a t-copula.
      3. Invert each currency's OWN standardized Student-t marginal (df=marginal_dfs[j]),
         scaled to horizon vol: R_j = mean_h_j + sigma_j * t_ppf(U_j; nu_j)*sqrt((nu_j-2)/nu_j).

    copula_df controls joint tail dependence; marginal_dfs control each marginal's tail
    fatness — independently. Correlation (copula/rank) preserved via chol(R).
    """
    nu_c = float(copula_df)
    if nu_c <= 0:
        raise ValueError(f"copula_df must be > 0; got {nu_c}")
    mean_h = np.asarray(mean_h, dtype=float)
    marginal_dfs = np.asarray(marginal_dfs, dtype=float)
    dim = mean_h.size

    corr, sigma = corr_from_cov(cov_h)
    L_R = np.linalg.cholesky(corr)

    # 1. correlated t-scores
    Z = rng.standard_normal((n_sims, dim))
    G = Z @ L_R.T
    W = rng.chisquare(nu_c, size=(n_sims, 1))
    T = G * np.sqrt(nu_c / W)

    # 2. to uniforms via the copula df's t CDF
    U = stats.t.cdf(T, df=nu_c)

    # 3. per-currency standardized-t marginals, scaled to horizon vol
    out = np.empty_like(U)
    for j in range(dim):
        nu_j = marginal_dfs[j]
        if nu_j <= 2:
            raise ValueError(f"marginal df must be > 2; got {nu_j} for column {j}")
        std_t = stats.t.ppf(U[:, j], df=nu_j) * np.sqrt((nu_j - 2.0) / nu_j)  # unit variance
        out[:, j] = mean_h[j] + sigma[j] * std_t
    return out
