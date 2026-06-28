"""Mark-to-market (MTM) revaluation engine for the FX risk scenario engine.

This is the single most reused piece of the whole project. Phase 2 (Monte Carlo) will
call `mtm()` thousands of times on simulated rate vectors, so its interface is fixed
now and it is fully vectorized: it accepts one scenario or a whole matrix of scenarios
and returns the portfolio's USD value for each.

DEFINITIONS
-----------
For a forward, the mark-to-market *value* is the undiscounted unrealized P&L versus the
contracted rate. (Discounting is OMITTED in Phase 1 — see note below.) For one contract:

    usd_value(r) = notional * r        if quote == USD_PER_FOREIGN   (EUR/USD, GBP/USD)
    usd_value(r) = notional / r        if quote == FOREIGN_PER_USD   (USD/JPY)

    pnl = sign * (usd_value(spot) - usd_value(contracted))

where `sign` is +1 if long the notional currency and -1 if short it (see portfolio.py).
The portfolio value is the sum of per-contract pnl.

DISCOUNTING: omitted in Phase 1. `mtm()` returns the undiscounted USD P&L. A discount
factor per contract (from `horizon_days`) can be multiplied in later without changing
this signature.
"""

from __future__ import annotations

import numpy as np

from .portfolio import FXForward, FOREIGN_PER_USD


def rates_to_array(rates, portfolio: list[FXForward]) -> np.ndarray:
    """Normalize a `rates` argument into an ndarray aligned to `portfolio` order.

    Accepts:
      * a dict {pair: rate}            -> 1-D array (n_contracts,)
      * a 1-D array (n_contracts,)     -> returned as float ndarray
      * a 2-D array (n_scenarios, n_contracts) -> returned as float ndarray

    For array inputs the columns MUST already be in the same order as `portfolio`.
    """
    if isinstance(rates, dict):
        return np.array([rates[c.pair] for c in portfolio], dtype=float)
    arr = np.asarray(rates, dtype=float)
    if arr.shape[-1] != len(portfolio):
        raise ValueError(
            f"rates last axis = {arr.shape[-1]} does not match portfolio size "
            f"{len(portfolio)}; columns must be ordered as the portfolio."
        )
    return arr


def _usd_value(rate: np.ndarray, notional: np.ndarray, invert: np.ndarray) -> np.ndarray:
    """USD value of each notional at `rate`, vectorized over the contract axis.

    `invert` is a boolean mask (True for FOREIGN_PER_USD contracts) broadcast against
    `rate`, which may be shape (n_contracts,) or (n_scenarios, n_contracts).
    """
    return np.where(invert, notional / rate, notional * rate)


def mtm(rates, portfolio: list[FXForward]):
    """Portfolio mark-to-market value in USD.

    Parameters
    ----------
    rates:
        Current spot rates. One of:
          * dict {pair: rate}
          * 1-D array (n_contracts,) ordered like `portfolio`  -> single scenario
          * 2-D array (n_scenarios, n_contracts)               -> many scenarios
    portfolio:
        List of `FXForward` contracts.

    Returns
    -------
    float for a single scenario, or a 1-D ndarray (n_scenarios,) for a matrix input.

    Notes
    -----
    Fully vectorized. Per-contract parameters are gathered once into arrays, then the
    whole computation broadcasts over an optional leading scenario axis. This is the
    hot path Phase 2 reuses unchanged.
    """
    spot = rates_to_array(rates, portfolio)

    # Per-contract parameter vectors, shape (n_contracts,).
    notional = np.array([c.notional for c in portfolio], dtype=float)
    contracted = np.array([c.contracted_rate for c in portfolio], dtype=float)
    sign = np.array([c.sign for c in portfolio], dtype=float)
    invert = np.array([c.quote == FOREIGN_PER_USD for c in portfolio], dtype=bool)

    # Work in 2-D (n_scenarios, n_contracts) so one code path handles 1 or many
    # scenarios; remember whether to squeeze the result back to a scalar.
    scalar_input = spot.ndim == 1
    spot2d = np.atleast_2d(spot)  # (n_scenarios, n_contracts)

    value_now = _usd_value(spot2d, notional, invert)         # (n_scenarios, n_contracts)
    value_contracted = _usd_value(contracted, notional, invert)  # (n_contracts,)

    pnl = sign * (value_now - value_contracted)              # broadcasts over scenarios
    total = pnl.sum(axis=1)                                   # (n_scenarios,)

    return float(total[0]) if scalar_input else total


def mtm_breakdown(rates, portfolio: list[FXForward]) -> dict[str, float]:
    """Per-contract USD P&L for a SINGLE scenario, keyed by pair. For reporting/auditing.

    Mirrors `mtm()` exactly but keeps contracts separate so the contribution of each
    leg is visible. Only accepts a single scenario (dict or 1-D array).
    """
    spot = rates_to_array(rates, portfolio)
    if spot.ndim != 1:
        raise ValueError("mtm_breakdown expects a single scenario (dict or 1-D array).")

    breakdown: dict[str, float] = {}
    for contract, r in zip(portfolio, spot):
        if contract.quote == FOREIGN_PER_USD:
            value_now = contract.notional / r
            value_contracted = contract.notional / contract.contracted_rate
        else:
            value_now = contract.notional * r
            value_contracted = contract.notional * contract.contracted_rate
        breakdown[contract.pair] = contract.sign * (value_now - value_contracted)
    return breakdown
