"""Tests for the mark-to-market engine — the project's core correctness invariant.

The single most important thing to get right is the sign/quote convention, especially the
inverted USD/JPY. These tests pin all three pairs, the vectorization contract, and the
dict/array input equivalence.
"""

import numpy as np
import pytest

from src.portfolio import default_portfolio, SPOT_RATES, CONTRACTED_RATES
from src.mtm import mtm, mtm_breakdown, rates_to_array

# Hand-derived per-leg P&L for the default portfolio at SPOT vs CONTRACTED.
EXP_EUR = 10_000_000 * (1.10 - 1.08)                       # +200,000
EXP_GBP = 5_000_000 * (1.27 - 1.25)                        # +100,000
EXP_JPY = 800_000_000 * (1 / 145.0 - 1 / 150.0)           # +183,908.05 (long USD, inverted)
EXP_TOTAL = EXP_EUR + EXP_GBP + EXP_JPY


def test_base_case_total_matches_hand_derivation():
    pf = default_portfolio()
    assert mtm(SPOT_RATES, pf) == pytest.approx(EXP_TOTAL)
    assert mtm(SPOT_RATES, pf) == pytest.approx(483_908.0459770, rel=1e-9)


def test_per_leg_breakdown():
    bd = mtm_breakdown(SPOT_RATES, default_portfolio())
    assert bd["EURUSD"] == pytest.approx(EXP_EUR)
    assert bd["GBPUSD"] == pytest.approx(EXP_GBP)
    assert bd["USDJPY"] == pytest.approx(EXP_JPY)


def test_usdjpy_inverted_sign_direction():
    """Long USD/JPY must GAIN when USD/JPY rises (USD strengthens) and LOSE when it falls.

    This is the inverted-quote trap: it needs both usd_value=N/r AND sign=-1.
    """
    pf = default_portfolio()
    base = rates_to_array(SPOT_RATES, pf)
    i = [c.pair for c in pf].index("USDJPY")

    up = base.copy(); up[i] *= 1.10            # USD/JPY up 10% -> USD stronger
    down = base.copy(); down[i] *= 0.90        # USD/JPY down 10%

    assert mtm(up, pf) > mtm(base, pf)         # long USD gains when USD/JPY rises
    assert mtm(down, pf) < mtm(base, pf)       # and loses when it falls


def test_eurusd_gbpusd_direction():
    """Long EUR/USD and GBP/USD gain when their rate rises."""
    pf = default_portfolio()
    base = rates_to_array(SPOT_RATES, pf)
    for pair in ("EURUSD", "GBPUSD"):
        i = [c.pair for c in pf].index(pair)
        up = base.copy(); up[i] *= 1.10
        assert mtm(up, pf) > mtm(base, pf)


def test_dict_and_array_inputs_agree():
    pf = default_portfolio()
    arr = rates_to_array(SPOT_RATES, pf)
    assert mtm(SPOT_RATES, pf) == pytest.approx(mtm(arr, pf))


def test_rates_to_array_respects_portfolio_order():
    pf = default_portfolio()
    arr = rates_to_array(SPOT_RATES, pf)
    assert list(arr) == [SPOT_RATES[c.pair] for c in pf]


def test_vectorization_scalar_vs_matrix():
    """Scalar, 1-D, and a stacked matrix must all agree."""
    pf = default_portfolio()
    vec = rates_to_array(SPOT_RATES, pf)

    scalar = mtm(vec, pf)
    assert np.isscalar(scalar) or np.ndim(scalar) == 0

    matrix = np.tile(vec, (50, 1))             # 50 identical scenarios
    out = mtm(matrix, pf)
    assert out.shape == (50,)
    assert np.allclose(out, scalar)


def test_matrix_rows_are_independent():
    pf = default_portfolio()
    base = rates_to_array(SPOT_RATES, pf)
    contracted = rates_to_array(CONTRACTED_RATES, pf)
    matrix = np.vstack([base, contracted])     # row 0 = spot, row 1 = at contracted
    out = mtm(matrix, pf)
    assert out[0] == pytest.approx(EXP_TOTAL)
    assert out[1] == pytest.approx(0.0, abs=1e-6)   # MTM at contracted rate is zero


def test_wrong_width_raises():
    pf = default_portfolio()
    with pytest.raises(ValueError):
        mtm(np.ones((10, 2)), pf)               # 2 cols != 3 contracts
