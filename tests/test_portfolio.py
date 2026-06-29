"""Tests for portfolio definitions and their conventions."""

from src.portfolio import (
    default_portfolio, brexit_portfolio, BREXIT_SPOT_RATES,
    USD_PER_FOREIGN, FOREIGN_PER_USD,
)


def test_default_portfolio_conventions():
    pf = default_portfolio()
    assert [c.pair for c in pf] == ["EURUSD", "GBPUSD", "USDJPY"]

    by_pair = {c.pair: c for c in pf}
    # EUR/USD and GBP/USD: quoted USD-per-foreign, long the foreign currency.
    for pair in ("EURUSD", "GBPUSD"):
        assert by_pair[pair].quote == USD_PER_FOREIGN
        assert by_pair[pair].sign == +1
    # USD/JPY: inverted quote AND short the JPY notional (long USD).
    assert by_pair["USDJPY"].quote == FOREIGN_PER_USD
    assert by_pair["USDJPY"].sign == -1


def test_brexit_portfolio_is_gbp_dominant_and_atm():
    pf = brexit_portfolio()
    by_pair = {c.pair: c for c in pf}

    # GBP exposure (in USD terms) is the largest leg.
    usd_sizes = {c.pair: c.notional * (BREXIT_SPOT_RATES[c.pair]
                 if c.quote == USD_PER_FOREIGN else 1 / BREXIT_SPOT_RATES[c.pair])
                 for c in pf}
    assert max(usd_sizes, key=usd_sizes.get) == "GBPUSD"

    # At-the-money forwards: contracted == current spot -> base MTM ~ 0.
    for c in pf:
        assert c.contracted_rate == BREXIT_SPOT_RATES[c.pair]

    # USD/JPY convention carried over unchanged.
    assert by_pair["USDJPY"].quote == FOREIGN_PER_USD
    assert by_pair["USDJPY"].sign == -1
