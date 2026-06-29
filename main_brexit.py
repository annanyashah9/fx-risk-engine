"""FX risk scenario engine — Brexit capstone (application of the Phase 1-4 engine).

Replays the June 2016 Brexit referendum as a worked example. This is NOT a prediction and
makes NO claim to have foreseen the outcome. It is a PROBABILITY SWEEP: across a plausible
range of Leave-outcome probabilities p (which has no true value), it shows what the existing
engine would have reported as tail risk ahead of a dated event, versus the static +/-10%
method and versus what actually happened.

Honesty measures on hindsight:
  * Ex-ante covariance: correlation/df inputs estimated ONLY from data ending 23 June 2016.
  * The ~10% GBP/USD Leave-move is a plausible pre-event estimate (ex ante it would come from
    option-implied vols / risk reversals), and is SWEPT (6/8/10/12%) to show the structure
    holds regardless of the exact figure.
  * The realized 23->24 June 2016 move is shown for CONTEXT ONLY, clearly labeled historical.

Reuses the engine unchanged: full_model_returns (t-copula + Mechanism A jumps + Mechanism B
event), run_simulation, mtm, var_es, run_static_stress, fit_marginal_dfs/fit_joint_df.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from src.portfolio import brexit_portfolio, BREXIT_SPOT_RATES
from src.data import get_fx_history, log_returns
from src.simulation import run_simulation, HORIZON_DAYS, N_SIMS, SEED
from src.mtm import mtm, rates_to_array
from src.fat_tails import fit_marginal_dfs, fit_joint_df, t_copula_returns
from src.events import full_model_returns, JumpParams, EventParams
from src.stress import run_static_stress
from src.risk import var_es
from src.plots import plot_brexit_sweep, plot_brexit_distribution

REFERENDUM = "2016-06-23"          # vote day (data up to & incl. this date is ex-ante)
RESULT_DAY = "2016-06-24"          # result known; markets moved
DATA_START = "2012-01-01"          # ex-ante estimation window start
CRISIS_RHO = 0.70                  # Phase-4 moderate crisis-correlation regime

P_TABLE = [0.2, 0.3, 0.4, 0.5]     # requested table grid
P_FIGURE = np.round(np.linspace(0.0, 0.5, 11), 3)   # fine grid for the sweep figure
JUMP_SIZES = [0.06, 0.08, 0.10, 0.12]               # GBP Leave-move sweep
HEADLINE_SIZE = 0.10               # representative GBP move for the table / distribution
HEADLINE_P = 0.40                  # representative p for the distribution figure


def _fmt(x) -> str:
    return f"${x/1e6:,.2f}M"


def _event_params(p: float, s: float) -> EventParams:
    """Brexit event targeting GBP/USD: GBP ln(1-s); EUR/JPY spillover fractions of s.
    Order = [EURUSD, GBPUSD, USDJPY] (portfolio order)."""
    event_mean = np.array([np.log(1 - 0.4 * s), np.log(1 - s), np.log(1 - 0.3 * s)])
    return EventParams(prob=p, event_mean=event_mean,
                       event_vol=np.array([0.03, 0.03, 0.03]), crisis_rho=CRISIS_RHO)


def _full_gen(p: float, s: float, base_gen):
    return partial(full_model_returns, base_generator=base_gen,
                   enable_jumps=True, jump_params=JumpParams(),
                   enable_event=True, event_params=_event_params(p, s))


def main() -> None:
    portfolio = brexit_portfolio()
    pairs = [c.pair for c in portfolio]

    print("=" * 80)
    print("FX RISK SCENARIO ENGINE — Brexit Capstone (June 2016)")
    print("A PROBABILITY SWEEP, not a prediction. No claim to have foreseen the outcome.")
    print("=" * 80)

    # ----------------------------------------------- ex-ante data (<= vote day)
    levels_all = get_fx_history(start=DATA_START, save_path="data/fx_rates_brexit.csv")[pairs]
    pre = levels_all.loc[levels_all.index <= pd.Timestamp(REFERENDUM)]
    rets = log_returns(pre)
    print(f"\n[ex-ante] Covariance/df estimated from {pre.index.min().date()} -> "
          f"{pre.index.max().date()} ({len(rets)} obs) — NO post-vote data leaks in.")

    marginal_dfs = fit_marginal_dfs(rets)
    joint_df = fit_joint_df(rets)
    base_gen = partial(t_copula_returns, copula_df=joint_df, marginal_dfs=marginal_dfs)
    print(f"[ex-ante] marginal df = {np.round(marginal_dfs,2)}, copula df = {joint_df:.2f}")

    print("\n[portfolio] GBP-emphasized pre-referendum book (at-the-money forwards):")
    for c in portfolio:
        print(f"    long {c.notional:>14,.0f} {c.notional_ccy:<3}  {c.pair}  "
              f"@ {BREXIT_SPOT_RATES[c.pair]}")

    # --------------------------------------------------- static +/-10% baseline
    static_df = run_static_stress(portfolio, BREXIT_SPOT_RATES)
    static_gbp = float(static_df.loc[static_df.scenario == "GBPUSD -10%", "scenario_pnl_usd"].iloc[0])
    static_all = float(static_df["scenario_pnl_usd"].min())
    print(f"\n[static] GBP/USD -10% single-shock loss = {_fmt(static_gbp)}  "
          f"(no probability attached)")
    print(f"[static] worst aggregate (ALL -10%)       = {_fmt(static_all)}")

    # ------------------------------------------- realized move (context only)
    r23 = levels_all.loc[pd.Timestamp(REFERENDUM)]
    r24 = levels_all.loc[pd.Timestamp(RESULT_DAY)]
    realized_factor = (r24 / r23).reindex(pairs).to_numpy()
    base_spot = rates_to_array(BREXIT_SPOT_RATES, portfolio)
    realized_spot = base_spot * realized_factor
    realized_loss = float(mtm(realized_spot, portfolio) - mtm(base_spot, portfolio))
    gbp_move = realized_factor[pairs.index("GBPUSD")] - 1.0
    print(f"\n[realized — CONTEXT ONLY] {REFERENDUM} -> {RESULT_DAY} (historical FRED):")
    print(f"    GBP/USD {r23['GBPUSD']:.4f} -> {r24['GBPUSD']:.4f}  ({gbp_move:+.2%})")
    print(f"    implied portfolio loss = {_fmt(realized_loss)}")

    # ------------------------------------------------ sweep: ES99 vs p by size
    print("\n[sweep] Running engine across p and GBP jump size (seed=42, "
          f"{N_SIMS:,} sims each)...")
    es99_by_size = {}
    for s in JUMP_SIZES:
        es = []
        for p in P_FIGURE:
            _, m = _run_engine(portfolio, rets, _full_gen(p, s, base_gen))
            es.append(float(m.loc[m.confidence == 0.99, "es"].iloc[0]))
        es99_by_size[f"{s:.0%}"] = np.array(es)

    # ------------------------------------------- table at headline jump size
    rows = []
    for p in P_TABLE:
        _, m = _run_engine(portfolio, rets, _full_gen(p, HEADLINE_SIZE, base_gen))
        rows.append({
            "p_leave": p,
            "VaR95": m.loc[m.confidence == 0.95, "var"].iloc[0],
            "VaR99": m.loc[m.confidence == 0.99, "var"].iloc[0],
            "ES95": m.loc[m.confidence == 0.95, "es"].iloc[0],
            "ES99": m.loc[m.confidence == 0.99, "es"].iloc[0],
        })
    table = pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print(f"ENGINE TAIL RISK vs LEAVE PROBABILITY  (GBP move = {HEADLINE_SIZE:.0%}, "
          f"crisis rho={CRISIS_RHO})")
    print("=" * 80)
    print("Sign: losses are positive magnitudes. p is SWEPT — it has no true value.\n")
    show = table.copy()
    for col in ["VaR95", "VaR99", "ES95", "ES99"]:
        show[col] = show[col].map(_fmt)
    print(show.to_string(index=False))
    print(f"\n  Static GBP -10% (one unconditional number): {_fmt(static_gbp)}")
    print(f"  Realized Jun-2016 move implied loss        : {_fmt(realized_loss)} (context)")

    # ------------------------------------------------------- sanity checks
    es_at_p04 = {k: v[list(P_FIGURE).index(0.4)] for k, v in es99_by_size.items()}
    assert es_at_p04["12%"] > es_at_p04["6%"], "ES99 should rise with assumed jump size"
    es10 = es99_by_size["10%"]
    assert es10[list(P_FIGURE).index(0.5)] > es10[0], "event should raise ES99 above the p=0 baseline"
    assert (table["ES99"] > -static_gbp).all(), \
        "engine 99% ES across the swept p should exceed the static GBP -10% number"
    assert (table["ES99"] >= table["VaR99"]).all(), "ES >= VaR"

    # --------------------------------------------- representative distribution
    pnl_rep, m_rep = _run_engine(portfolio, rets, _full_gen(HEADLINE_P, HEADLINE_SIZE, base_gen))

    # --------------------------------------------------------------- outputs
    out = table.copy()
    out["static_gbp_minus10"] = static_gbp
    out["static_all_minus10"] = static_all
    out["realized_loss"] = realized_loss
    out["gbp_jump_size"] = HEADLINE_SIZE
    out["crisis_rho"] = CRISIS_RHO
    out["seed"] = SEED
    table_path = "results/brexit_summary.csv"
    out.to_csv(table_path, index=False)

    fig1 = plot_brexit_sweep(P_FIGURE, es99_by_size, static_gbp, realized_loss)
    fig2 = plot_brexit_distribution(pnl_rep, m_rep, static_gbp, realized_loss,
                                    p_label=f"p={HEADLINE_P:.0%}, GBP move {HEADLINE_SIZE:.0%}")

    print(f"\nSaved table  -> {table_path}")
    print(f"Saved figure -> {fig1}")
    print(f"Saved figure -> {fig2}")

    # ------------------------------------------------------- conclusion text
    es99_headline = float(table.loc[table.p_leave == HEADLINE_P, "ES99"].iloc[0])
    print("\n" + "=" * 80)
    print("CONCLUSION (framing: a sweep, not a prediction)")
    print("=" * 80)
    print(
        "  The static +/-10% method returns ONE unconditional number "
        f"({_fmt(static_gbp)} for a\n"
        "  GBP -10% shock) with NO probability and NO awareness of a dated event. The engine\n"
        "  instead traces tail risk across a plausible RANGE of Leave probabilities: at "
        f"p={HEADLINE_P:.0%}\n"
        f"  it flags a 99% ES of {_fmt(es99_headline)} — materially larger than the static "
        "figure and\n"
        "  the realized move's implied loss — risk the static method was structurally blind\n"
        "  to. This is NOT a claim to have predicted Brexit; p is a swept sensitivity input.")


def _run_engine(portfolio, rets, generator):
    res = run_simulation(
        portfolio=portfolio, spot_rates=BREXIT_SPOT_RATES, returns=rets,
        horizon_days=HORIZON_DAYS, n_sims=N_SIMS, seed=SEED,
        include_drift=True, return_generator=generator,
    )
    return res.pnl, var_es(res.pnl, confidences=(0.95, 0.99))


if __name__ == "__main__":
    main()
