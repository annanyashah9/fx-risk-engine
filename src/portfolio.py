"""Portfolio definition for the FX risk scenario engine (Phase 1).

A portfolio is a list of FX forward contracts. Each contract carries everything the
mark-to-market engine (`src/mtm.py`) needs to revalue it in USD given a current spot
rate, with the *quote convention* and *direction sign* made fully explicit so that no
later phase has to re-derive them.

KEY CONVENTION (see README / plan for the full derivation):
    "long PAIR" means long the FIRST-named (base) currency of the pair.
        EUR/USD -> long EUR    GBP/USD -> long GBP    USD/JPY -> long USD

This matters most for USD/JPY, whose quote is inverted relative to the other two:
    - EUR/USD, GBP/USD are quoted as USD per 1 unit of foreign currency.
    - USD/JPY is quoted as JPY per 1 USD.
The MTM engine handles the inversion via the `quote` field, and the long/short
direction via the `sign` field. See `FXForward` below.
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Quote-convention constants
# ---------------------------------------------------------------------------
# How a contract's rate is quoted, which determines how a notional in the
# notional currency is converted to USD:
#   USD_PER_FOREIGN : rate = USD per 1 unit of the (foreign) notional currency
#                     -> usd_value(N units) = N * rate          (e.g. EUR/USD, GBP/USD)
#   FOREIGN_PER_USD : rate = units of the (foreign) notional currency per 1 USD
#                     -> usd_value(N units) = N / rate          (e.g. USD/JPY)
USD_PER_FOREIGN = "USD_PER_FOREIGN"
FOREIGN_PER_USD = "FOREIGN_PER_USD"


# ---------------------------------------------------------------------------
# Market parameters — clearly labeled, edit here.
# ---------------------------------------------------------------------------
# Contracted forward rates: the rate locked in at trade inception. The MTM of a
# forward is its gain/loss versus this rate.
CONTRACTED_RATES = {
    "EURUSD": 1.0800,   # USD per EUR, locked
    "GBPUSD": 1.2500,   # USD per GBP, locked
    "USDJPY": 145.00,   # JPY per USD, locked
}

# Current spot rates: the base case used to revalue the portfolio today. Phase 1's
# static stress shocks these by +/-10%; Phase 2 will replace them with simulated paths.
SPOT_RATES = {
    "EURUSD": 1.1000,   # USD per EUR, now
    "GBPUSD": 1.2700,   # USD per GBP, now
    "USDJPY": 150.00,   # JPY per USD, now (USD has strengthened vs JPY: 145 -> 150)
}


@dataclass(frozen=True)
class FXForward:
    """A single FX forward contract.

    Attributes
    ----------
    pair:
        Currency pair, e.g. "EURUSD". The first-named currency is the one you are
        long/short (see `sign`).
    notional:
        Size in the notional currency (a positive number; direction lives in `sign`).
    notional_ccy:
        Currency the `notional` is denominated in (EUR, GBP, JPY).
    contracted_rate:
        Forward rate locked at inception, in the pair's quote convention.
    horizon_days:
        Settlement horizon in days. Carried for completeness; NOT used in Phase 1
        (discounting is omitted — see `src/mtm.py`). A discounting hook in a later
        phase will use it without changing the MTM interface.
    quote:
        One of USD_PER_FOREIGN / FOREIGN_PER_USD — how `notional` converts to USD.
    sign:
        +1 if long the notional currency, -1 if short it. NOTE this is the sign on the
        *notional currency*, not on the pair label. For USD/JPY we are long USD, which
        means SHORT the JPY notional, hence sign = -1.
    """

    pair: str
    notional: float
    notional_ccy: str
    contracted_rate: float
    horizon_days: int
    quote: str
    sign: int


def default_portfolio() -> list[FXForward]:
    """The default Phase 1 portfolio: three long forwards.

    Direction recap (long the first-named currency of each pair):
      * Long EUR/USD  : long EUR.  Profits when EUR/USD RISES.   sign = +1 (long EUR notional)
      * Long GBP/USD  : long GBP.  Profits when GBP/USD RISES.   sign = +1 (long GBP notional)
      * Long USD/JPY  : long USD.  Profits when USD/JPY RISES.   sign = -1 (SHORT the JPY notional)

    The USD/JPY contract is the trap: its quote is inverted (FOREIGN_PER_USD) AND its
    sign is negative (long USD == short the JPY notional). Both are required for the
    P&L to come out right; getting either alone wrong flips the sign.
    """
    return [
        FXForward(
            pair="EURUSD",
            notional=10_000_000,          # EUR 10,000,000
            notional_ccy="EUR",
            contracted_rate=CONTRACTED_RATES["EURUSD"],
            horizon_days=90,
            quote=USD_PER_FOREIGN,
            sign=+1,                       # long EUR
        ),
        FXForward(
            pair="GBPUSD",
            notional=5_000_000,           # GBP 5,000,000
            notional_ccy="GBP",
            contracted_rate=CONTRACTED_RATES["GBPUSD"],
            horizon_days=90,
            quote=USD_PER_FOREIGN,
            sign=+1,                       # long GBP
        ),
        FXForward(
            pair="USDJPY",
            notional=800_000_000,         # JPY 800,000,000
            notional_ccy="JPY",
            contracted_rate=CONTRACTED_RATES["USDJPY"],
            horizon_days=90,
            quote=FOREIGN_PER_USD,         # JPY per USD -> usd_value = N / rate
            sign=-1,                       # long USD == short the JPY notional
        ),
    ]


# ---------------------------------------------------------------------------
# Brexit capstone configuration (June 2016 pre-referendum, GBP-emphasized book)
# ---------------------------------------------------------------------------
# Pre-referendum spot levels (approx. 23 June 2016). Contracted = spot (at-the-money
# forwards struck at inception), so the base MTM is ~0 and the simulated P&L is the pure
# horizon move — exactly what we want for the event study.
BREXIT_SPOT_RATES = {
    "EURUSD": 1.13,    # USD per EUR
    "GBPUSD": 1.48,    # USD per GBP (pre-vote)
    "USDJPY": 106.0,   # JPY per USD
}


def brexit_portfolio() -> list[FXForward]:
    """GBP-emphasized pre-referendum book for the Brexit capstone.

    GBP/USD is the dominant leg (the position most exposed to a Leave outcome); EUR/USD and
    USD/JPY are kept small to carry the correlated spillover. Same conventions as
    `default_portfolio` (incl. the inverted USD/JPY: long USD == sign -1). At-the-money
    forwards (contracted = current spot) so base MTM ~ 0.
    """
    return [
        FXForward(
            pair="EURUSD",
            notional=5_000_000,            # EUR 5,000,000 (spillover leg)
            notional_ccy="EUR",
            contracted_rate=BREXIT_SPOT_RATES["EURUSD"],
            horizon_days=10,
            quote=USD_PER_FOREIGN,
            sign=+1,                        # long EUR
        ),
        FXForward(
            pair="GBPUSD",
            notional=25_000_000,           # GBP 25,000,000 (DOMINANT GBP exposure)
            notional_ccy="GBP",
            contracted_rate=BREXIT_SPOT_RATES["GBPUSD"],
            horizon_days=10,
            quote=USD_PER_FOREIGN,
            sign=+1,                        # long GBP
        ),
        FXForward(
            pair="USDJPY",
            notional=200_000_000,          # JPY 200,000,000 (small spillover leg)
            notional_ccy="JPY",
            contracted_rate=BREXIT_SPOT_RATES["USDJPY"],
            horizon_days=10,
            quote=FOREIGN_PER_USD,
            sign=-1,                        # long USD == short the JPY notional
        ),
    ]
