"""Tests for VaR / Expected Shortfall metrics and their sign convention."""

import numpy as np
import pytest

from src.risk import var_es


def test_var_es_match_definition():
    """var = -quantile(pnl, 1-c); es = -mean of the tail at/under that quantile."""
    rng = np.random.default_rng(1)
    pnl = rng.normal(0, 1_000_000, size=100_000)
    df = var_es(pnl, confidences=(0.95, 0.99))

    for c in (0.95, 0.99):
        row = df.loc[df.confidence == c].iloc[0]
        q = np.quantile(pnl, 1 - c)
        assert row["var"] == pytest.approx(-q)
        assert row["es"] == pytest.approx(-pnl[pnl <= q].mean())


def test_es_at_least_var_and_99_at_least_95():
    rng = np.random.default_rng(2)
    pnl = rng.normal(50_000, 800_000, size=50_000)
    df = var_es(pnl).set_index("confidence")
    assert df.loc[0.99, "var"] >= df.loc[0.95, "var"]
    assert df.loc[0.99, "es"] >= df.loc[0.99, "var"]
    assert df.loc[0.95, "es"] >= df.loc[0.95, "var"]


def test_losses_are_positive_magnitudes():
    """A distribution centered above zero with a heavy loss tail -> positive VaR/ES.

    Use a 3% loss block so the 1% quantile lands firmly inside the loss region (a 1% block
    would have the quantile interpolate across the cliff).
    """
    pnl = np.concatenate([np.full(9_700, 100_000.0), np.full(300, -5_000_000.0)])
    df = var_es(pnl).set_index("confidence")
    assert df.loc[0.99, "var"] > 0
    assert df.loc[0.99, "es"] == pytest.approx(5_000_000.0)   # worst 1% is all the big loss


def test_mean_and_std_reported():
    rng = np.random.default_rng(3)
    pnl = rng.normal(0, 1, size=10_000)
    df = var_es(pnl)
    assert df["pnl_mean"].iloc[0] == pytest.approx(pnl.mean())
    assert df["pnl_std"].iloc[0] == pytest.approx(pnl.std(ddof=1))
