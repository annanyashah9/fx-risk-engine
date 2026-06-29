"""FX risk scenario engine.

Phase 1 public API: portfolio definition, MTM revaluation, and static +/-10% stress.
"""

from .portfolio import (
    FXForward,
    default_portfolio,
    brexit_portfolio,
    CONTRACTED_RATES,
    SPOT_RATES,
    BREXIT_SPOT_RATES,
    USD_PER_FOREIGN,
    FOREIGN_PER_USD,
)
from .mtm import mtm, mtm_breakdown, rates_to_array
from .stress import run_static_stress, save_table, build_scenarios
from .data import get_fx_history, log_returns, PAIRS
from .simulation import (
    run_simulation,
    estimate_moments,
    scale_to_horizon,
    ensure_psd,
    gaussian_returns,
    simulate_spots,
    SimulationResult,
    HORIZON_DAYS,
    N_SIMS,
    SEED,
)
from .risk import var_es
from .plots import (
    plot_pnl_distribution,
    plot_tail_comparison,
    plot_four_method_overlay,
    plot_crisis_regimes,
    plot_brexit_sweep,
    plot_brexit_distribution,
)
from .fat_tails import (
    corr_from_cov,
    fit_marginal_dfs,
    fit_joint_df,
    multivariate_t_returns,
    t_copula_returns,
)
from .events import (
    crisis_correlation,
    mechanism_a_jumps,
    mechanism_b_event,
    full_model_returns,
    JumpParams,
    EventParams,
)

__all__ = [
    "FXForward",
    "default_portfolio",
    "brexit_portfolio",
    "CONTRACTED_RATES",
    "SPOT_RATES",
    "BREXIT_SPOT_RATES",
    "USD_PER_FOREIGN",
    "FOREIGN_PER_USD",
    "mtm",
    "mtm_breakdown",
    "rates_to_array",
    "run_static_stress",
    "save_table",
    "build_scenarios",
    # Phase 2
    "get_fx_history",
    "log_returns",
    "PAIRS",
    "run_simulation",
    "estimate_moments",
    "scale_to_horizon",
    "ensure_psd",
    "gaussian_returns",
    "simulate_spots",
    "SimulationResult",
    "HORIZON_DAYS",
    "N_SIMS",
    "SEED",
    "var_es",
    "plot_pnl_distribution",
    "plot_tail_comparison",
    "plot_four_method_overlay",
    "plot_crisis_regimes",
    "plot_brexit_sweep",
    "plot_brexit_distribution",
    # Phase 3
    "corr_from_cov",
    "fit_marginal_dfs",
    "fit_joint_df",
    "multivariate_t_returns",
    "t_copula_returns",
    # Phase 4
    "crisis_correlation",
    "mechanism_a_jumps",
    "mechanism_b_event",
    "full_model_returns",
    "JumpParams",
    "EventParams",
]
