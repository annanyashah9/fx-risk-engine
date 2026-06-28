"""FX risk scenario engine — Phase 2 entry point.

Correlated Monte Carlo -> full P&L distribution -> VaR & Expected Shortfall.

Pipeline: load FRED FX history -> daily log returns -> estimate daily moments ->
scale to the 10-day risk horizon -> ensure PSD -> Cholesky correlated Gaussian draws ->
simulated spots -> revalue with the Phase-1 mtm() (one matrix call) -> VaR/ES.

Outputs:
  data/fx_rates.csv            raw merged FX levels (date range printed below)
  figures/mc_pnl_distribution.png  histogram with VaR/ES marked
  results/mc_risk_summary.csv  VaR95/ES95/VaR99/ES99 + mean/std + run metadata

------------------------------------------------------------------------------------
HOW THIS DIFFERS FROM PHASE 1 (the point of the leap)
------------------------------------------------------------------------------------
  * PROBABILITIES. Phase 1 asserted a +/-10% move with no likelihood. Here every loss
    sits on a distribution: VaR/ES are quantiles ("the 1-in-20 / 1-in-100 loss"), which
    a static grid simply cannot express.
  * CORRELATION. Phase 1 shocked one rate at a time (or all by a hand-picked +/-10%).
    Here the three rates co-move through the estimated covariance matrix (Cholesky), so
    the portfolio's diversification/concentration is captured — the reason to treat
    these hedges as one book rather than three independent bets.
Still ASSUMED-AWAY (Phases 3-4): fat tails (Gaussian under-weights extremes) and event
jumps. Those come next.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.portfolio import default_portfolio, SPOT_RATES
from src.data import get_fx_history, log_returns
from src.simulation import run_simulation, HORIZON_DAYS, N_SIMS, SEED
from src.risk import var_es
from src.plots import plot_pnl_distribution

# Phase 1 reference point (static "ALL rates -10%" P&L) for the comparison note.
PHASE1_ALL_DOWN_10 = -2_327_592.59


def _fmt_usd(x: float) -> str:
    return f"${x:,.2f}"


def main() -> None:
    portfolio = default_portfolio()

    print("=" * 72)
    print("FX RISK SCENARIO ENGINE — Phase 2: Correlated Monte Carlo (VaR / ES)")
    print("=" * 72)

    # ---------------------------------------------------------------- data
    levels = get_fx_history()
    # Order columns to match the portfolio so simulated columns line up with mtm().
    levels = levels[[c.pair for c in portfolio]]
    rets = log_returns(levels)
    print(
        f"[data] Date range: {levels.index.min().date()} -> {levels.index.max().date()} "
        f"({len(levels)} daily levels, {len(rets)} return obs)"
    )

    # ---------------------------------------------------------- simulation
    result = run_simulation(
        portfolio=portfolio,
        spot_rates=SPOT_RATES,
        returns=rets,
        horizon_days=HORIZON_DAYS,
        n_sims=N_SIMS,
        seed=SEED,
        include_drift=True,
    )

    # Covariance / horizon-scaling diagnostics.
    daily_corr = rets.corr().to_numpy()
    print(f"\n[covariance] Horizon: {result.horizon_days} trading days "
          f"(cov_H = {result.horizon_days} x cov_daily; horizon scaling, not annualization).")
    print(f"[covariance] include_drift=True  (mean_H = {result.horizon_days} x mean_daily)")
    print(f"[PSD] min eigenvalue = {result.psd.min_eigenvalue:.3e}  "
          f"repaired = {result.psd.repaired}  method = {result.psd.method}")
    print("[correlation] daily log-return correlation matrix:")
    print(pd.DataFrame(daily_corr, index=[c.pair for c in portfolio],
                       columns=[c.pair for c in portfolio]).round(3).to_string())

    # Cholesky correctness: empirical cov of simulated returns ~ cov_H.
    emp_cov = np.cov(result.sim_returns, rowvar=False, ddof=1)
    rel_err = np.linalg.norm(emp_cov - result.cov_h) / np.linalg.norm(result.cov_h)
    print(f"[check] ||cov(sim_returns) - cov_H|| / ||cov_H|| = {rel_err:.4f} "
          f"(should be small -> Cholesky correlation correct)")

    # --------------------------------------------------------- risk metrics
    metrics = var_es(result.pnl, confidences=(0.95, 0.99))

    print("\n" + "=" * 72)
    print(f"RISK METRICS  ({result.n_sims:,} scenarios, seed={result.seed}, "
          f"{result.horizon_days}-day horizon)")
    print("=" * 72)
    print("Sign convention: P&L negative = loss; VaR/ES reported as positive loss magnitudes.\n")
    print(f"  P&L mean : {_fmt_usd(metrics['pnl_mean'].iloc[0])}")
    print(f"  P&L std  : {_fmt_usd(metrics['pnl_std'].iloc[0])}")
    print(f"  base MTM : {_fmt_usd(result.base_mtm)}  (today's mark; P&L is change vs this)\n")
    for _, row in metrics.iterrows():
        c = int(row["confidence"] * 100)
        print(f"  {c}%  VaR = {_fmt_usd(row['var']):>18}     ES = {_fmt_usd(row['es']):>18}")

    # Internal consistency sanity checks.
    v95, v99 = metrics["var"].to_numpy()
    e95, e99 = metrics["es"].to_numpy()
    assert v99 > v95, "VaR99 should exceed VaR95"
    assert e95 >= v95 and e99 >= v99, "ES should be >= VaR at each level"

    # --------------------------------------------------------------- outputs
    summary = metrics.copy()
    summary["horizon_days"] = result.horizon_days
    summary["n_sims"] = result.n_sims
    summary["seed"] = result.seed
    summary["date_start"] = str(levels.index.min().date())
    summary["date_end"] = str(levels.index.max().date())
    summary["base_mtm"] = result.base_mtm
    table_path = "results/mc_risk_summary.csv"
    summary.to_csv(table_path, index=False)

    fig_path = plot_pnl_distribution(result.pnl, metrics)

    print(f"\nSaved table  -> {table_path}")
    print(f"Saved figure -> {fig_path}")

    # ----------------------------------------------------- Phase 1 comparison
    print("\n" + "=" * 72)
    print("COMPARISON vs PHASE 1 (static +/-10%)")
    print("=" * 72)
    print(f"  Phase 1 'ALL rates -10%' P&L (asserted, no probability): "
          f"{_fmt_usd(PHASE1_ALL_DOWN_10)}")
    print(f"  Phase 2 99% VaR (1-in-100 loss, {result.horizon_days}-day): "
          f"{_fmt_usd(v99)}")
    print(f"  Phase 2 99% ES  (avg loss beyond 99% VaR):                 "
          f"{_fmt_usd(e99)}")
    print(
        "\n  Phase 2 attaches a LIKELIHOOD to each loss and lets the three rates move\n"
        "  together via the estimated covariance. The static figure was a single\n"
        "  hand-picked co-move with no probability; this is a full distribution."
    )


if __name__ == "__main__":
    main()
