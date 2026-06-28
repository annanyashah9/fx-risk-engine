"""FX risk scenario engine — Phase 3 entry point (fat tails).

Swaps ONLY the random-shock generator: Gaussian (Phase 2) vs multivariate Student-t vs
t-copula (headline). Everything else — moment estimation, horizon scaling, PSD, Cholesky
correlation, `mtm()` revaluation, and `var_es()` — is reused unchanged via
`run_simulation(..., return_generator=...)`.

Headline result: real FX tails are fatter than normal, so the Gaussian VaR/ES from Phase 2
UNDERSTATES tail risk. We quantify the understatement as the ratio of the t-copula's 99% ES
to the Gaussian's 99% ES.

------------------------------------------------------------------------------------
WHY t-COPULA IS THE HEADLINE (vs naive multivariate-t)
------------------------------------------------------------------------------------
  * Multivariate-t ties every currency's tail heaviness AND their joint extremity to ONE
    shared df via a single common chi-square mixing variable — one knob for everything.
  * t-copula separates them: each currency gets its OWN marginal df (how fat is EUR vs GBP
    vs JPY) while a copula df governs how jointly extreme crises are. For a 3-currency book
    with different tail behavior and the negative EUR/JPY relation, that separation gives a
    more defensible tail estimate.
Still ASSUMED-AWAY (Phase 4): discrete jumps / a scheduled binary event.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd

from src.portfolio import default_portfolio, SPOT_RATES
from src.data import get_fx_history, log_returns
from src.simulation import run_simulation, gaussian_returns, HORIZON_DAYS, N_SIMS, SEED
from src.fat_tails import (
    fit_marginal_dfs,
    fit_joint_df,
    multivariate_t_returns,
    t_copula_returns,
    corr_from_cov,
)
from src.risk import var_es
from src.plots import plot_tail_comparison

DF_SWEEP = (3, 5, 10)


def _fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def main() -> None:
    portfolio = default_portfolio()
    pairs = [c.pair for c in portfolio]

    print("=" * 78)
    print("FX RISK SCENARIO ENGINE — Phase 3: Fat Tails (multivariate-t & t-copula)")
    print("=" * 78)

    # ---------------------------------------------------------------- data
    levels = get_fx_history()[pairs]
    rets = log_returns(levels)
    print(f"[data] {levels.index.min().date()} -> {levels.index.max().date()} "
          f"({len(rets)} return obs)")

    # ------------------------------------------------ fit degrees of freedom
    marginal_dfs = fit_marginal_dfs(rets)
    joint_df = fit_joint_df(rets)
    print("\n[fit] Per-marginal Student-t df (method-of-moments via excess kurtosis,\n"
          "      df = 4 + 6/kurtosis; lower = fatter tails):")
    for pair, nu in zip(pairs, marginal_dfs):
        print(f"        {pair}: df = {nu:.2f}")
    print(f"[fit] Joint/copula df (grid MLE): df = {joint_df:.2f}")

    # ------------------------------------------------------- model registry
    # Each entry is a return_generator with its tail params bound; identical otherwise.
    models = {
        "Gaussian": gaussian_returns,
        f"Multivariate-t (df={joint_df:.0f})": partial(multivariate_t_returns, df=joint_df),
    }
    for d in DF_SWEEP:
        models[f"Multivariate-t (df={d})"] = partial(multivariate_t_returns, df=d)
    models["t-copula (HEADLINE)"] = partial(
        t_copula_returns, copula_df=joint_df, marginal_dfs=marginal_dfs
    )

    # ----------------------------------------------- run each model + metrics
    target_corr, _ = corr_from_cov(
        # cov_H correlation == cov_daily correlation; use daily for the target.
        np.cov(rets.to_numpy(), rowvar=False, ddof=1)
    )

    pnl_by_model: dict[str, np.ndarray] = {}
    metrics_by_model: dict[str, pd.DataFrame] = {}
    rows = []
    print("\n[run] Same seed/pipeline for every model; only the shock generator changes.")
    for name, generator in models.items():
        res = run_simulation(
            portfolio=portfolio,
            spot_rates=SPOT_RATES,
            returns=rets,
            horizon_days=HORIZON_DAYS,
            n_sims=N_SIMS,
            seed=SEED,
            include_drift=True,
            return_generator=generator,
        )
        m = var_es(res.pnl, confidences=(0.95, 0.99))
        pnl_by_model[name] = res.pnl
        metrics_by_model[name] = m

        # Correlation-preservation check vs target.
        emp_corr, _ = corr_from_cov(np.cov(res.sim_returns, rowvar=False, ddof=1))
        corr_relerr = np.linalg.norm(emp_corr - target_corr) / np.linalg.norm(target_corr)

        rows.append({
            "model": name,
            "VaR95": m.loc[m.confidence == 0.95, "var"].iloc[0],
            "VaR99": m.loc[m.confidence == 0.99, "var"].iloc[0],
            "ES95": m.loc[m.confidence == 0.95, "es"].iloc[0],
            "ES99": m.loc[m.confidence == 0.99, "es"].iloc[0],
            "pnl_mean": m["pnl_mean"].iloc[0],
            "pnl_std": m["pnl_std"].iloc[0],
            "corr_relerr": corr_relerr,
        })

    table = pd.DataFrame(rows)

    # --------------------------------------------------------------- report
    print("\n" + "=" * 78)
    print(f"RISK COMPARISON  ({N_SIMS:,} scenarios, seed={SEED}, {HORIZON_DAYS}-day horizon)")
    print("=" * 78)
    print("Sign convention: P&L negative = loss; VaR/ES are positive loss magnitudes.\n")
    show = table.copy()
    for col in ["VaR95", "VaR99", "ES95", "ES99", "pnl_std"]:
        show[col] = show[col].map(_fmt_usd)
    show["corr_relerr"] = table["corr_relerr"].map(lambda v: f"{v:.4f}")
    show = show.drop(columns=["pnl_mean"])
    print(show.to_string(index=False))

    # ---------------------------------------------------------- sanity checks
    g = table.set_index("model")
    gauss99_es = g.loc["Gaussian", "ES99"]
    gauss99_var = g.loc["Gaussian", "VaR99"]
    copula = g.loc["t-copula (HEADLINE)"]
    # Fat-tailed models must not understate the Gaussian at 99%.
    assert copula["ES99"] >= gauss99_es and copula["VaR99"] >= gauss99_var, \
        "t-copula should be at least as heavy as Gaussian at 99%"
    # df sweep monotonicity: lower df => larger 99% ES.
    sweep = [g.loc[f"Multivariate-t (df={d})", "ES99"] for d in DF_SWEEP]
    assert sweep[0] >= sweep[1] >= sweep[2], f"sweep not monotone in df: {sweep}"
    # Correlation preserved.
    assert g.loc["Multivariate-t (df=%d)" % DF_SWEEP[1], "corr_relerr"] < 0.05
    assert copula["corr_relerr"] < 0.10, "t-copula Pearson correlation drifted too far"

    # ----------------------------------------------------------- understatement
    es_ratio = copula["ES99"] / gauss99_es
    var_ratio = copula["VaR99"] / gauss99_var
    print("\n" + "=" * 78)
    print("HEADLINE — how much the Gaussian model HID")
    print("=" * 78)
    print(f"  Gaussian 99% ES : {_fmt_usd(gauss99_es)}")
    print(f"  t-copula 99% ES : {_fmt_usd(copula['ES99'])}")
    print(f"  -> the Gaussian UNDERSTATED the 1-in-100 expected loss by {es_ratio:.2f}x "
          f"(99% VaR: {var_ratio:.2f}x).")
    print("  That gap is pure tail risk the normal distribution could not see.")

    # --------------------------------------------------------------- outputs
    table_path = "results/phase3_risk_comparison.csv"
    meta = table.copy()
    meta["horizon_days"] = HORIZON_DAYS
    meta["n_sims"] = N_SIMS
    meta["seed"] = SEED
    meta["joint_df"] = joint_df
    for pair, nu in zip(pairs, marginal_dfs):
        meta[f"marg_df_{pair}"] = nu
    meta.to_csv(table_path, index=False)

    # Plot a clean subset (Gaussian, fitted multivariate-t, t-copula) for clarity.
    plot_subset = {
        "Gaussian": pnl_by_model["Gaussian"],
        f"Multivariate-t (df={joint_df:.0f})": pnl_by_model[f"Multivariate-t (df={joint_df:.0f})"],
        "t-copula (HEADLINE)": pnl_by_model["t-copula (HEADLINE)"],
    }
    metric_subset = {k: metrics_by_model[k] for k in plot_subset}
    fig_path = plot_tail_comparison(plot_subset, metric_subset)

    print(f"\nSaved table  -> {table_path}")
    print(f"Saved figure -> {fig_path}")


if __name__ == "__main__":
    main()
