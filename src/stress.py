"""Static +/-10% scenario stress for the FX risk scenario engine (Phase 1).

This is the *baseline* method the rest of the project argues against: shock each spot
rate by a fixed +/-10%, one at a time (and all together), and tabulate the resulting
portfolio P&L. It has no notion of probability, correlation, fat tails, or jumps — see
the comment block in `main.py`. Those gaps motivate Phases 2-4.

The scenarios are stacked into one (n_scenarios, n_contracts) matrix and revalued with
a SINGLE vectorized `mtm()` call, demonstrating the interface Phase 2 depends on.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .portfolio import FXForward
from .mtm import mtm, rates_to_array

SHOCK = 0.10  # +/-10% static shock


def build_scenarios(portfolio: list[FXForward], base_rates: dict[str, float]):
    """Construct the static +/-10% scenario set.

    Returns
    -------
    labels : list[str]
        Human-readable scenario name per row.
    shocked_pairs : list[str | None]
        Which pair was shocked (None for the all-rates aggregate scenarios).
    matrix : np.ndarray, shape (n_scenarios, n_contracts)
        Shocked rate vectors, columns ordered like `portfolio`.

    Scenarios: for each rate, -10% then +10% (one at a time, others at base), then an
    all-rates -10% and an all-rates +10% aggregate.
    """
    base = rates_to_array(base_rates, portfolio)  # (n_contracts,), portfolio order

    labels: list[str] = []
    shocked_pairs: list[str | None] = []
    rows: list[np.ndarray] = []

    # One rate at a time.
    for i, contract in enumerate(portfolio):
        for direction, factor in (("-10%", 1 - SHOCK), ("+10%", 1 + SHOCK)):
            row = base.copy()
            row[i] = base[i] * factor
            labels.append(f"{contract.pair} {direction}")
            shocked_pairs.append(contract.pair)
            rows.append(row)

    # All rates together.
    labels.append("ALL rates -10%")
    shocked_pairs.append(None)
    rows.append(base * (1 - SHOCK))

    labels.append("ALL rates +10%")
    shocked_pairs.append(None)
    rows.append(base * (1 + SHOCK))

    return labels, shocked_pairs, np.vstack(rows)


def run_static_stress(
    portfolio: list[FXForward],
    base_rates: dict[str, float],
) -> pd.DataFrame:
    """Run the static stress and return a tidy results table.

    Columns: scenario, shocked_pair, shocked_rate (the changed rate value, or NaN for
    aggregate scenarios), scenario_mtm_usd, scenario_pnl_usd (vs base MTM).
    """
    base_mtm = mtm(base_rates, portfolio)

    labels, shocked_pairs, matrix = build_scenarios(portfolio, base_rates)

    # Single vectorized revaluation of all scenarios at once.
    scenario_mtm = mtm(matrix, portfolio)
    scenario_pnl = scenario_mtm - base_mtm

    # For single-rate scenarios, surface the shocked rate's new value for readability.
    pair_to_col = {c.pair: i for i, c in enumerate(portfolio)}
    shocked_rate = [
        matrix[row, pair_to_col[p]] if p is not None else np.nan
        for row, p in enumerate(shocked_pairs)
    ]

    return pd.DataFrame(
        {
            "scenario": labels,
            "shocked_pair": [p if p is not None else "ALL" for p in shocked_pairs],
            "shocked_rate": shocked_rate,
            "scenario_mtm_usd": scenario_mtm,
            "scenario_pnl_usd": scenario_pnl,
        }
    )


def save_table(df: pd.DataFrame, path: str = "results/static_stress.csv") -> str:
    """Write the results table to CSV, returning the path. Creates the dir if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path
