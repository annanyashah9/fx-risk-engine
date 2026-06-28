"""FX risk scenario engine — Phase 1 entry point.

Prints the base-case portfolio MTM and the static +/-10% scenario table, and saves the
table to results/static_stress.csv.

------------------------------------------------------------------------------------
WHAT THIS STATIC +/-10% METHOD CANNOT CAPTURE  (this is the whole point of the project)
------------------------------------------------------------------------------------
  * NO PROBABILITIES. A +/-10% move is asserted, never assigned a likelihood. We cannot
    say how probable a given loss is, nor compute VaR / expected shortfall. (-> Phase 2:
    Monte Carlo gives a full P&L distribution.)
  * NO CORRELATION between rates. Shocking one rate at a time assumes the others are
    frozen; the "all -10%/+10%" rows assume perfect, hand-picked co-movement. Real FX
    rates move together with a correlation structure that this ignores. (-> Phase 2/3:
    correlated draws.)
  * NO FAT TAILS. A flat +/-10% is a thin, symmetric, bounded view. Real FX returns are
    leptokurtic — extreme moves are far more likely than a normal/bounded view implies,
    and 10% may badly understate the tail. (-> Phase 3: fat-tailed distributions.)
  * NO EVENT JUMPS. Devaluations, pegs breaking, SNB-style shocks, and central-bank
    surprises produce discontinuous gaps that no smooth +/-10% grid represents.
    (-> Phase 4: jump risk.)
------------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from src.portfolio import default_portfolio, SPOT_RATES, CONTRACTED_RATES
from src.mtm import mtm, mtm_breakdown
from src.stress import run_static_stress, save_table


def _fmt_usd(x: float) -> str:
    return f"${x:>15,.2f}"


def main() -> None:
    portfolio = default_portfolio()

    # ---------------------------------------------------------------- base case
    base_mtm = mtm(SPOT_RATES, portfolio)
    breakdown = mtm_breakdown(SPOT_RATES, portfolio)

    print("=" * 72)
    print("FX RISK SCENARIO ENGINE — Phase 1")
    print("=" * 72)
    print("\nBase-case spot rates:")
    for pair, rate in SPOT_RATES.items():
        print(f"  {pair}:  spot {rate:>10}   contracted {CONTRACTED_RATES[pair]:>10}")

    print("\nBase-case MTM (undiscounted unrealized P&L vs contracted, USD):")
    for contract in portfolio:
        # Direction note: long the first-named currency of the pair.
        print(
            f"  {contract.pair}  (long {contract.pair[:3]}):"
            f"  {_fmt_usd(breakdown[contract.pair])}"
        )
    print("  " + "-" * 40)
    print(f"  TOTAL portfolio MTM:        {_fmt_usd(base_mtm)}")

    # ----------------------------------------------------- sign-convention guard
    # Cheap regression guard: re-derive the three legs by hand and assert the engine
    # agrees. Protects every later phase against a silent sign/quote regression.
    eur = 10_000_000 * (SPOT_RATES["EURUSD"] - CONTRACTED_RATES["EURUSD"])
    gbp = 5_000_000 * (SPOT_RATES["GBPUSD"] - CONTRACTED_RATES["GBPUSD"])
    jpy = 800_000_000 * (1 / CONTRACTED_RATES["USDJPY"] - 1 / SPOT_RATES["USDJPY"])
    expected = eur + gbp + jpy
    assert abs(base_mtm - expected) < 1e-6, (
        f"MTM sign/quote regression! engine={base_mtm} hand-derived={expected}"
    )
    print(
        f"\n  [sign check OK] hand-derived total = {_fmt_usd(expected)} "
        f"(EUR {_fmt_usd(eur)}, GBP {_fmt_usd(gbp)}, USDJPY {_fmt_usd(jpy)})"
    )

    # -------------------------------------------------------- static +/-10% stress
    df = run_static_stress(portfolio, SPOT_RATES)
    path = save_table(df)

    print("\n" + "=" * 72)
    print("STATIC +/-10% SCENARIO STRESS")
    print("=" * 72)
    with pd.option_context(
        "display.float_format", lambda v: f"{v:,.2f}", "display.width", 120
    ):
        print(df.to_string(index=False))
    print(f"\nSaved table -> {path}")

    print(
        "\nDirection sanity: USDJPY +10% (USD stronger) is a GAIN for the long-USD leg; "
        "USDJPY -10% is a LOSS. EUR/GBP +10% are gains, -10% are losses."
    )


if __name__ == "__main__":
    main()
