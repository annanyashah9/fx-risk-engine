"""FX risk scenario engine — Phase 4 entry point (jumps + scheduled event).

Adds two SEPARATE, independently toggleable mechanisms on top of the Phase-3 fat-tailed
engine and produces the headline four-method comparison:
  (1) static +/-10%  (2) Gaussian MC  (3) fat-tailed MC  (4) full jump+event MC.

Reuses `mtm()`, `var_es()`, `run_simulation()`, `t_copula_returns()`, and the Phase-1 static
stress UNCHANGED. The full model is just a composite generator behind the same seam.

------------------------------------------------------------------------------------
WHAT EACH EARLIER METHOD MISSED (the whole project in one screen)
------------------------------------------------------------------------------------
  * Static +/-10%   : no probabilities, no correlation, no fat tails, no events.
  * Gaussian MC     : adds probabilities + correlation, but thin tails understate extremes.
  * Fat-tailed MC   : adds heavier marginals/tail dependence (t-copula).
  * Full jump+event : adds UNSCHEDULED Merton jumps (timing unknowable) AND a SCHEDULED
                      binary event (date known, outcome unknown) as a genuine mixture, with
                      crisis correlations that erode diversification in stress.
The headline quantifies how much tail risk static and Gaussian hid versus the full model.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from src.portfolio import default_portfolio, SPOT_RATES
from src.data import get_fx_history, log_returns
from src.simulation import run_simulation, gaussian_returns, HORIZON_DAYS, N_SIMS, SEED
from src.fat_tails import fit_marginal_dfs, fit_joint_df, t_copula_returns
from src.events import full_model_returns, JumpParams, EventParams
from src.stress import run_static_stress
from src.risk import var_es
from src.plots import plot_four_method_overlay, plot_crisis_regimes


def _fmt_usd(x) -> str:
    return "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else f"${x:,.0f}"


def _run(portfolio, rets, generator):
    """Run one model through the shared pipeline; return (pnl, metrics DataFrame)."""
    res = run_simulation(
        portfolio=portfolio, spot_rates=SPOT_RATES, returns=rets,
        horizon_days=HORIZON_DAYS, n_sims=N_SIMS, seed=SEED,
        include_drift=True, return_generator=generator,
    )
    return res.pnl, var_es(res.pnl, confidences=(0.95, 0.99))


def _metrics_row(name, pnl, m, static_loss=None):
    worst = -float(np.min(pnl)) if pnl is not None else None
    return {
        "method": name,
        "VaR95": None if m is None else m.loc[m.confidence == 0.95, "var"].iloc[0],
        "VaR99": None if m is None else m.loc[m.confidence == 0.99, "var"].iloc[0],
        "ES95": None if m is None else m.loc[m.confidence == 0.95, "es"].iloc[0],
        "ES99": None if m is None else m.loc[m.confidence == 0.99, "es"].iloc[0],
        "worst_loss": worst,
    }


def main() -> None:
    portfolio = default_portfolio()
    pairs = [c.pair for c in portfolio]

    print("=" * 80)
    print("FX RISK SCENARIO ENGINE — Phase 4: Jumps + Scheduled Event (final phase)")
    print("=" * 80)

    # ------------------------------------------------------------------ data
    levels = get_fx_history()[pairs]
    rets = log_returns(levels)
    print(f"[data] {levels.index.min().date()} -> {levels.index.max().date()} "
          f"({len(rets)} return obs)")

    # ------------------------------------------------------ fit fat-tail dfs
    marginal_dfs = fit_marginal_dfs(rets)
    joint_df = fit_joint_df(rets)
    base_gen = partial(t_copula_returns, copula_df=joint_df, marginal_dfs=marginal_dfs)
    print(f"[fit] marginal df = {np.round(marginal_dfs, 2)}, copula df = {joint_df:.2f}")

    # ------------------------------------------------------- mechanism params
    jump_params = JumpParams()                 # lam=3/yr, sym, vol 2.5%, crisis rho 0.80
    event_params = EventParams()               # p=0.30, GBP -10% lead, crisis rho 0.70
    print(f"[A] Merton jumps: lambda={jump_params.lam_annual}/yr, jump_vol="
          f"{jump_params.jump_vol[0]:.1%}, crisis_rho={jump_params.crisis_rho}")
    print(f"[B] Scheduled event: p={event_params.prob}, "
          f"GBPUSD mean={np.expm1(event_params.event_mean[1]):.1%}, "
          f"crisis_rho={event_params.crisis_rho}")

    # ----------------------------------------------- static +/-10% reference
    static_df = run_static_stress(portfolio, SPOT_RATES)
    static_loss = float(static_df["scenario_pnl_usd"].min())   # worst "ALL -10%"
    print(f"[static] worst +/-10% scenario P&L = {_fmt_usd(static_loss)}")

    # --------------------------------------------- the four headline methods
    print("\n[run] Same seed/pipeline for every model; only the shock generator changes.")
    print("      (Fat-tailed/A/B/full share base draws via independent RNG substreams,")
    print("       so attribution is apples-to-apples: full = base + A + B pointwise.)")
    pnl_gauss, m_gauss = _run(portfolio, rets, gaussian_returns)

    # Fat-tailed, A-only, B-only, and full all go through the composite so they share the
    # SAME base (and, where enabled, the SAME A / B) realizations.
    fat_gen = partial(full_model_returns, base_generator=base_gen,
                      enable_jumps=False, enable_event=False)
    a_only_gen = partial(full_model_returns, base_generator=base_gen,
                         enable_jumps=True, jump_params=jump_params, enable_event=False)
    b_only_gen = partial(full_model_returns, base_generator=base_gen,
                         enable_jumps=False, enable_event=True, event_params=event_params)
    full_gen = partial(full_model_returns, base_generator=base_gen,
                       enable_jumps=True, jump_params=jump_params,
                       enable_event=True, event_params=event_params)
    pnl_fat, m_fat = _run(portfolio, rets, fat_gen)
    pnl_a, m_a = _run(portfolio, rets, a_only_gen)
    pnl_b, m_b = _run(portfolio, rets, b_only_gen)
    pnl_full, m_full = _run(portfolio, rets, full_gen)

    # ------------------------------------------------------------- table
    rows = [
        _metrics_row("Static ±10%", None, None),          # no probabilities
        _metrics_row("Gaussian MC", pnl_gauss, m_gauss),
        _metrics_row("Fat-tailed MC", pnl_fat, m_fat),
        _metrics_row("Jumps only (A)", pnl_a, m_a),
        _metrics_row("Event only (B)", pnl_b, m_b),
        _metrics_row("Full jump+event MC", pnl_full, m_full),
    ]
    table = pd.DataFrame(rows)
    # static row: record the worst-case stress loss in worst_loss for context
    table.loc[table.method == "Static ±10%", "worst_loss"] = -static_loss

    print("\n" + "=" * 80)
    print(f"METHOD COMPARISON  ({N_SIMS:,} scenarios, seed={SEED}, {HORIZON_DAYS}-day horizon)")
    print("=" * 80)
    print("Sign convention: P&L negative = loss; VaR/ES/worst are positive loss magnitudes.\n")
    show = table.copy()
    for col in ["VaR95", "VaR99", "ES95", "ES99", "worst_loss"]:
        show[col] = show[col].map(_fmt_usd)
    print(show.to_string(index=False))

    # ------------------------------------------------------- sanity checks
    es99 = {r["method"]: r["ES99"] for r in rows if r["ES99"] is not None}
    assert es99["Gaussian MC"] < es99["Fat-tailed MC"] < es99["Full jump+event MC"], \
        "99% ES should escalate Gaussian < fat-tailed < full"
    assert es99["Jumps only (A)"] > es99["Fat-tailed MC"], "jumps should raise the tail"
    assert es99["Event only (B)"] > es99["Fat-tailed MC"], "event should raise the tail"
    assert es99["Full jump+event MC"] >= es99["Jumps only (A)"], "full should be >= A alone"
    # The scheduled event dominates the 99% tail (p=0.30 >> 1%), so adding rare, mean-zero
    # jumps barely moves ES99 there: full ~= B alone at this quantile (jumps show up more in
    # the extreme worst case and in everyday-tail widening, see Jumps-only vs Fat-tailed).
    assert abs(es99["Full jump+event MC"] / es99["Event only (B)"] - 1.0) < 0.05, \
        "with the event present, full 99% ES should be within MC noise of B alone"
    # Mixture signature: fraction of event-affected scenarios ~ p (B-only minus base mass).
    # Approx check: P(B-only loss beyond fat-tailed 99% VaR) should clearly exceed 1%.
    frac_tail_b = float(np.mean(pnl_b <= -m_fat.loc[m_fat.confidence == 0.99, "var"].iloc[0]))
    assert frac_tail_b > 0.05, f"event mixture should thicken the tail; got {frac_tail_b:.3f}"

    # --------------------------------------------------------- headline
    full_es99 = es99["Full jump+event MC"]
    gauss_es99 = es99["Gaussian MC"]
    print("\n" + "=" * 80)
    print("HEADLINE — how much tail risk the earlier methods MISSED")
    print("=" * 80)
    print(f"  Full-model 99% ES                 : {_fmt_usd(full_es99)}")
    print(f"  vs Gaussian MC 99% ES             : {_fmt_usd(gauss_es99)}"
          f"   ->  {full_es99 / gauss_es99:.2f}x")
    print(f"  vs Static ±10% worst-case loss    : {_fmt_usd(-static_loss)}"
          f"   ->  {full_es99 / -static_loss:.2f}x")
    print(f"  Full-model worst single scenario  : {_fmt_usd(-float(np.min(pnl_full)))}")
    print("  The full model attaches a PROBABILITY to losses the static grid only asserted,\n"
          "  and reaches deep losses the Gaussian/fat-tailed models could not — the jump and\n"
          "  scheduled-event tail static and smooth MC both miss.")

    # ----------------------------------- Mechanism-B crisis-correlation study
    print("\n" + "=" * 80)
    print("MECHANISM B — crisis-correlation regime study (diversification in stress)")
    print("=" * 80)
    regimes = {
        "Event, normal corr": None,
        "Event, moderate crisis (rho=0.70)": 0.70,
        "Event, severe crisis (rho=0.90)": 0.90,
    }
    pnl_by_regime, metrics_by_regime, regime_rows = {}, {}, []
    for name, rho in regimes.items():
        ep = EventParams(crisis_rho=rho)
        gen = partial(full_model_returns, base_generator=base_gen,
                      enable_jumps=False, enable_event=True, event_params=ep)
        pnl_r, m_r = _run(portfolio, rets, gen)
        pnl_by_regime[name] = pnl_r
        metrics_by_regime[name] = m_r
        regime_rows.append({
            "regime": name,
            "VaR99": m_r.loc[m_r.confidence == 0.99, "var"].iloc[0],
            "ES99": m_r.loc[m_r.confidence == 0.99, "es"].iloc[0],
        })
    regime_tbl = pd.DataFrame(regime_rows)
    show_r = regime_tbl.copy()
    for c in ["VaR99", "ES99"]:
        show_r[c] = show_r[c].map(_fmt_usd)
    print(show_r.to_string(index=False))
    es_norm = regime_tbl.loc[0, "ES99"]
    es_sev = regime_tbl.loc[2, "ES99"]
    # The correlation regime must MATTER; we do not hard-code a direction because, for THIS
    # portfolio, the honest result runs opposite to the textbook "diversification evaporates".
    assert abs(es_sev / es_norm - 1.0) > 0.01, "crisis-correlation regime should move the tail"
    direction = "REDUCES" if es_sev < es_norm else "INCREASES"
    print(f"\n  Severe-crisis 99% ES is {es_sev / es_norm:.2f}x the normal-correlation case "
          f"-> rising crisis correlation {direction} this portfolio's event tail.")
    print(
        "  WHY (and it's the inverted-USD/JPY through-line again): the book is long EUR/USD\n"
        "  and long GBP/USD (both effectively SHORT USD) but long USD/JPY (LONG USD). The\n"
        "  USD/JPY leg is NEGATIVELY correlated with the EUR/GBP legs, so it is a partial USD\n"
        "  hedge. A USD-driven crisis STRENGTHENS that negative correlation, so diversification\n"
        "  here improves rather than evaporates. 'Crisis correlation' cuts the opposite way for\n"
        "  a naturally-hedged book than for a same-signed one — a portfolio-specific result."
    )

    # --------------------------------------------------------------- outputs
    table_path = "results/phase4_method_comparison.csv"
    out = table.copy()
    out["full_es99_over_gaussian"] = full_es99 / gauss_es99
    out["full_es99_over_static"] = full_es99 / -static_loss
    out["horizon_days"] = HORIZON_DAYS
    out["n_sims"] = N_SIMS
    out["seed"] = SEED
    out.to_csv(table_path, index=False)

    overlay = {
        "Gaussian MC": pnl_gauss,
        "Fat-tailed MC": pnl_fat,
        "Full jump+event MC": pnl_full,
    }
    overlay_metrics = {
        "Gaussian MC": m_gauss, "Fat-tailed MC": m_fat, "Full jump+event MC": m_full,
    }
    fig1 = plot_four_method_overlay(overlay, overlay_metrics, static_loss)
    fig2 = plot_crisis_regimes(
        pnl_by_regime, metrics_by_regime,
        title="Scheduled event under rising crisis correlation "
              "(this USD-hedged book: tail shrinks, not grows)",
    )

    print(f"\nSaved table  -> {table_path}")
    print(f"Saved figure -> {fig1}")
    print(f"Saved figure -> {fig2}")


if __name__ == "__main__":
    main()
