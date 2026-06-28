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
]
