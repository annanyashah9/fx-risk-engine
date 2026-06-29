"""Jump risk and scheduled-event risk for the FX risk scenario engine (Phase 4).

Two SEPARATE, independently toggleable mechanisms that sit ON TOP of the Phase-3 fat-tailed
engine as ADDITIVE log-return components (returns add in log space, so
`S_sim = S_base * exp(R_base + J_A + J_B)`):

  Mechanism A — Merton jump-diffusion (UNSCHEDULED). Jumps arrive at random times via a
    Poisson process with random magnitudes. Timing is unknowable — there is no calendar
    date. Models shocks like the SNB franc de-peg (Jan 2015).

  Mechanism B — Scheduled binary event (the Brexit overlay). The opposite structure: the
    DATE is known, the OUTCOME is not. A GENUINE MIXTURE at the horizon: with prob p a large
    directional jump hits a target pair and spills to correlated pairs; with prob 1-p nothing
    happens. This is NOT inflated everyday volatility — it puts a point mass at "no event"
    against a displaced blob, producing a distinct second lump in the loss tail.

Crisis correlations differ from normal times: during events (and during jumps) currencies
co-move more violently, so normal diversification partly evaporates. `crisis_correlation`
builds an elevated regime used for both mechanisms.

The composite `full_model_returns` is signature-compatible with the Phase-2/3 generator seam
`(mean_h, cov_h, n_sims, rng)`, so `run_simulation()`, `mtm()`, and `var_es()` are reused
UNCHANGED (extra params bound via functools.partial by the caller).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fat_tails import corr_from_cov
from .simulation import ensure_psd


# ---------------------------------------------------------------------------
# Crisis correlation regime
# ---------------------------------------------------------------------------
def crisis_correlation(normal_corr: np.ndarray, rho_crisis: float) -> np.ndarray:
    """Elevated-correlation regime: push |off-diagonals| up to rho_crisis, PRESERVING sign.

    During stress, co-movement intensifies but the SIGN structure holds (e.g. the negative
    EUR/JPY relation from the inverted USD/JPY quote stays negative, just larger). The result
    is run through `ensure_psd` so Cholesky is safe.
    """
    normal_corr = np.asarray(normal_corr, dtype=float)
    R = np.sign(normal_corr) * rho_crisis
    np.fill_diagonal(R, 1.0)
    R, _ = ensure_psd(R)
    return R


def _cov_from_vol_corr(vol: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """Covariance matrix from a per-asset volatility vector and a correlation matrix."""
    vol = np.asarray(vol, dtype=float)
    cov = corr * np.outer(vol, vol)
    cov, _ = ensure_psd(cov)
    return cov


# ---------------------------------------------------------------------------
# Parameter containers
# ---------------------------------------------------------------------------
@dataclass
class JumpParams:
    """Mechanism A — Merton jump-diffusion parameters (per currency, log-return space)."""
    lam_annual: float = 3.0          # Poisson intensity: expected jumps per year
    horizon_days: int = 10           # risk horizon (matches the engine)
    jump_mean: np.ndarray = field(   # per-ccy mean jump size (0 = symmetric)
        default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    jump_vol: np.ndarray = field(    # per-ccy jump size volatility (2.5%)
        default_factory=lambda: np.array([0.025, 0.025, 0.025]))
    crisis_rho: float = 0.80         # jumps co-move under crisis correlation


@dataclass
class EventParams:
    """Mechanism B — scheduled binary event (Brexit overlay) parameters (log-return space).

    `event_mean` is the directional outcome IF the event fires: GBP/USD ~ -10% lead, with
    spillover to EUR/USD and USD/JPY. Order is [EURUSD, GBPUSD, USDJPY] (portfolio order).
    """
    prob: float = 0.30               # probability the event resolves badly
    event_mean: np.ndarray = field(  # directional log-move if fired
        default_factory=lambda: np.array([-0.05, np.log(0.90), -0.03]))
    event_vol: np.ndarray = field(   # dispersion around the directional mean
        default_factory=lambda: np.array([0.03, 0.03, 0.03]))
    crisis_rho: float | None = 0.70  # crisis correlation for dispersion;
    #                                  None -> use the actual calm-period correlation


# ---------------------------------------------------------------------------
# Mechanism A — unscheduled Merton jumps
# ---------------------------------------------------------------------------
def mechanism_a_jumps(n_sims, dim, rng, normal_corr, params: JumpParams) -> np.ndarray:
    """Aggregate jump log-returns over the horizon, shape (n_sims, dim).

    N_i ~ Poisson(lam_h); the aggregate of N_i iid correlated normal jumps is
    MVN(N_i*mu_J, N_i*Sigma_J). Vectorized as
        J = N[:,None]*mu_J + sqrt(N)[:,None]*(Z @ L_Jᵀ).
    N_i = 0 yields an all-zero row (no jump). Sigma_J uses the crisis correlation regime.
    """
    lam_h = params.lam_annual * params.horizon_days / 252.0
    N = rng.poisson(lam_h, size=n_sims).astype(float)        # jumps per scenario

    crisis = crisis_correlation(normal_corr, params.crisis_rho)
    Sigma_J = _cov_from_vol_corr(params.jump_vol, crisis)
    L = np.linalg.cholesky(Sigma_J)

    Z = rng.standard_normal((n_sims, dim))
    mu = np.asarray(params.jump_mean, dtype=float)
    return N[:, None] * mu + np.sqrt(N)[:, None] * (Z @ L.T)


# ---------------------------------------------------------------------------
# Mechanism B — scheduled binary event (genuine mixture)
# ---------------------------------------------------------------------------
def mechanism_b_event(n_sims, dim, rng, normal_corr, params: EventParams) -> np.ndarray:
    """Scheduled-event log-returns, shape (n_sims, dim). True mixture, not inflated vol.

    B_i ~ Bernoulli(p). Added term = B_i * (event_mean + Z @ L_Eᵀ): with prob 1-p exactly
    zero (no event), with prob p a large directional correlated shock. Sigma_E uses the
    crisis correlation regime.
    """
    fired = rng.random(n_sims) < params.prob                 # Bernoulli(p)

    # crisis_rho=None -> use the actual calm-period correlation (the "normal regime"
    # baseline); otherwise elevate to the crisis regime.
    corr = normal_corr if params.crisis_rho is None \
        else crisis_correlation(normal_corr, params.crisis_rho)
    Sigma_E = _cov_from_vol_corr(params.event_vol, corr)
    L = np.linalg.cholesky(Sigma_E)

    Z = rng.standard_normal((n_sims, dim))
    mean = np.asarray(params.event_mean, dtype=float)
    shock = mean + Z @ L.T                                    # directional + dispersion
    return fired[:, None] * shock                            # zero where not fired


# ---------------------------------------------------------------------------
# Composite generator (the Phase-2/3 seam; independently toggleable components)
# ---------------------------------------------------------------------------
def full_model_returns(
    mean_h, cov_h, n_sims, rng, *,
    base_generator,
    enable_jumps: bool = True,
    jump_params: JumpParams | None = None,
    enable_event: bool = True,
    event_params: EventParams | None = None,
) -> np.ndarray:
    """Base (diffusion + fat tails) + optional Mechanism A + optional Mechanism B.

    `base_generator` is any seam-compatible generator (e.g. a partial of `t_copula_returns`).
    A and B are additive in log-return space and each gated by its own flag, so the caller can
    run base-only / A-only / B-only / A+B and attribute the tail to the right source. The
    crisis correlation is derived from cov_h's correlation (the calm-period structure) and
    then elevated inside each mechanism.

    INDEPENDENT RNG SUBSTREAMS: base, A, and B each draw from their own spawned child
    Generator, so toggling one component does NOT change another's realizations. This makes
    attribution clean and apples-to-apples — base-only / A-only / B-only / full all share the
    SAME base draws, and full = base + A + B pointwise. Reproducible from the parent seed.
    """
    base_rng, jump_rng, event_rng = rng.spawn(3)

    R = base_generator(mean_h, cov_h, n_sims, base_rng)
    dim = R.shape[1]
    normal_corr, _ = corr_from_cov(cov_h)

    if enable_jumps:
        R = R + mechanism_a_jumps(n_sims, dim, jump_rng, normal_corr,
                                  jump_params or JumpParams())
    if enable_event:
        R = R + mechanism_b_event(n_sims, dim, event_rng, normal_corr,
                                  event_params or EventParams())
    return R
