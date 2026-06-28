"""FX risk scenario engine.

Phase 1 public API: portfolio definition, MTM revaluation, and static +/-10% stress.
"""

from .portfolio import (
    FXForward,
    default_portfolio,
    CONTRACTED_RATES,
    SPOT_RATES,
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
from .plots import plot_pnl_distribution

__all__ = [
    "FXForward",
    "default_portfolio",
    "CONTRACTED_RATES",
    "SPOT_RATES",
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
]
