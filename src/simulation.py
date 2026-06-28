"""Correlated Monte Carlo simulation for the FX risk scenario engine (Phase 2).

Estimates the covariance/drift of daily FX log returns, scales them to a risk horizon,
generates correlated return shocks via Cholesky, converts to simulated spot rates, and
revalues the portfolio with the Phase-1 `mtm()` function (unchanged, one matrix call).

DESIGN SEAM FOR PHASE 3 (fat tails)
-----------------------------------
The random draw is isolated behind a `return_generator(mean_h, cov_h, n_sims, rng)`
callable. Phase 2 supplies `gaussian_returns`. Phase 3 will pass a Student-t generator
that reuses the SAME Cholesky correlation but a heavier-tailed draw — nothing downstream
(spot conversion, `mtm()`, risk metrics) changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio import FXForward
from .mtm import mtm, rates_to_array

# ---------------------------------------------------------------------------
# Risk-horizon constant. 10 trading days = Basel-style 10-day VaR.
# Change this one line for a different horizon (e.g. ~21 for a 1-month VaR).
# ---------------------------------------------------------------------------
HORIZON_DAYS = 10

SEED = 42
N_SIMS = 10_000


# ---------------------------------------------------------------------------
# 1. Moment estimation
# ---------------------------------------------------------------------------
def estimate_moments(returns: pd.DataFrame):
    """Daily mean vector and covariance matrix of log returns.

    Returns (mean_daily (d,), cov_daily (d, d)), columns ordered as `returns.columns`.
    Uses the sample covariance (ddof=1).
    """
    mean_daily = returns.mean(axis=0).to_numpy(dtype=float)
    cov_daily = np.cov(returns.to_numpy(dtype=float), rowvar=False, ddof=1)
    return mean_daily, cov_daily


def scale_to_horizon(mean_daily, cov_daily, horizon_days: int = HORIZON_DAYS):
    """Scale daily moments to the risk horizon.

    Under i.i.d. daily log returns, the H-day log return is the sum of H daily returns,
    so BOTH the mean and the covariance scale LINEARLY in time:
        mean_H = H * mean_daily
        cov_H  = H * cov_daily          (variance ~ time, volatility ~ sqrt(time))
    This is HORIZON scaling, not annualization (annualizing would be the special case
    H = 252, which we deliberately do not use here).
    """
    mean_h = horizon_days * np.asarray(mean_daily, dtype=float)
    cov_h = horizon_days * np.asarray(cov_daily, dtype=float)
    return mean_h, cov_h


# ---------------------------------------------------------------------------
# 2. Positive semi-definiteness
# ---------------------------------------------------------------------------
@dataclass
class PSDDiagnostics:
    min_eigenvalue: float
    repaired: bool
    method: str


def ensure_psd(cov, tol: float = 1e-12, ridge: float = 1e-12):
    """Return a (symmetric, positive-definite) covariance plus diagnostics.

    Steps:
      1. Symmetrize.
      2. Eigendecompose (eigh); record the minimum eigenvalue.
      3. If min eigenvalue < tol: clip negative/tiny eigenvalues up to a small positive
         floor and reconstruct (nearest-PSD via eigenvalue clipping; Higham-lite).
      4. Final guard: if Cholesky still fails, add a tiny diagonal ridge and retry.

    For three real FX series over years of data this reports repaired=False; the repair
    path exists for the degenerate edge case so Cholesky cannot blow up.
    """
    cov = np.asarray(cov, dtype=float)
    cov = (cov + cov.T) / 2.0

    eigvals = np.linalg.eigvalsh(cov)
    min_eig = float(eigvals.min())

    repaired = False
    method = "none"

    if min_eig < tol:
        # Eigenvalue clipping: floor eigenvalues at a small positive value.
        vals, vecs = np.linalg.eigh(cov)
        floor = max(tol, 0.0)
        vals_clipped = np.clip(vals, floor, None)
        cov = (vecs * vals_clipped) @ vecs.T
        cov = (cov + cov.T) / 2.0
        repaired = True
        method = "eigenvalue-clip"

    # Final guard: ensure Cholesky succeeds, adding a scaled ridge if needed.
    try:
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        scale = float(np.trace(cov)) / cov.shape[0]
        cov = cov + ridge * scale * np.eye(cov.shape[0])
        repaired = True
        method = method + "+ridge" if method != "none" else "ridge"

    return cov, PSDDiagnostics(min_eigenvalue=min_eig, repaired=repaired, method=method)


# ---------------------------------------------------------------------------
# 3. Return generators (THE PHASE-3 SWAP POINT)
# ---------------------------------------------------------------------------
def gaussian_returns(mean_h, cov_h, n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Correlated multivariate-normal horizon returns, shape (n_sims, dim).

    Cholesky factor L (cov_h = L Lᵀ) turns iid standard normals into correlated draws:
        R = mean_h + Z @ Lᵀ,   Z ~ N(0, I).
    Phase 3 replaces this function (same signature) with a heavier-tailed generator.
    """
    mean_h = np.asarray(mean_h, dtype=float)
    L = np.linalg.cholesky(cov_h)            # lower-triangular
    Z = rng.standard_normal((n_sims, mean_h.size))
    return mean_h + Z @ L.T


# ---------------------------------------------------------------------------
# 4. Spot conversion + full simulation
# ---------------------------------------------------------------------------
def simulate_spots(base_spot: np.ndarray, returns: np.ndarray) -> np.ndarray:
    """Convert simulated log returns to simulated spot rates: S = S_base * exp(R).

    base_spot: (dim,) current spot in portfolio order. returns: (n_sims, dim).
    """
    return base_spot * np.exp(returns)


@dataclass
class SimulationResult:
    pnl: np.ndarray              # (n_sims,) signed USD P&L; negative = loss
    base_mtm: float
    sim_spots: np.ndarray        # (n_sims, dim)
    sim_returns: np.ndarray      # (n_sims, dim)
    mean_h: np.ndarray
    cov_h: np.ndarray
    psd: PSDDiagnostics
    horizon_days: int
    n_sims: int
    seed: int


def run_simulation(
    portfolio: list[FXForward],
    spot_rates: dict[str, float],
    returns: pd.DataFrame,
    horizon_days: int = HORIZON_DAYS,
    n_sims: int = N_SIMS,
    seed: int = SEED,
    include_drift: bool = True,
    return_generator=gaussian_returns,
) -> SimulationResult:
    """Run the correlated Monte Carlo and return the horizon P&L distribution.

    `returns` columns MUST be ordered like `portfolio` (caller ensures this; the engine's
    portfolio order is EURUSD, GBPUSD, USDJPY, matching the FRED data columns).

    P&L convention: pnl = mtm(simulated spot) - mtm(current spot). Negative = loss.
    """
    base_spot = rates_to_array(spot_rates, portfolio)  # (dim,), portfolio order

    mean_daily, cov_daily = estimate_moments(returns)
    mean_h, cov_h = scale_to_horizon(mean_daily, cov_daily, horizon_days)
    if not include_drift:
        mean_h = np.zeros_like(mean_h)

    cov_h, psd = ensure_psd(cov_h)

    rng = np.random.default_rng(seed)
    sim_returns = return_generator(mean_h, cov_h, n_sims, rng)
    sim_spots = simulate_spots(base_spot, sim_returns)

    base_mtm = mtm(base_spot, portfolio)
    pnl = mtm(sim_spots, portfolio) - base_mtm   # one vectorized matrix call

    return SimulationResult(
        pnl=pnl,
        base_mtm=base_mtm,
        sim_spots=sim_spots,
        sim_returns=sim_returns,
        mean_h=mean_h,
        cov_h=cov_h,
        psd=psd,
        horizon_days=horizon_days,
        n_sims=n_sims,
        seed=seed,
    )
