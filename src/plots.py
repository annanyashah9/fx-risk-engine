"""Plotting for the FX risk scenario engine (Phase 2). Reused by Phase 3.

Renders the simulated P&L distribution as a histogram with VaR and ES marked. Uses a
non-interactive matplotlib backend so it works headless.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def plot_pnl_distribution(
    pnl: np.ndarray,
    metrics: pd.DataFrame,
    path: str = "figures/mc_pnl_distribution.png",
    title: str = "Monte Carlo portfolio P&L distribution (10-day horizon)",
) -> str:
    """Histogram of `pnl` with VaR/ES marked as losses (drawn at -VaR, -ES on the P&L axis).

    `metrics` is the DataFrame from `risk.var_es` (columns: confidence, var, es).
    Returns the saved path.
    """
    pnl = np.asarray(pnl, dtype=float)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(pnl, bins=80, color="#4C72B0", alpha=0.75, edgecolor="white", linewidth=0.3)

    ax.axvline(0.0, color="black", linewidth=1.0, linestyle="-", alpha=0.6)

    # VaR/ES are positive loss magnitudes; plot them on the loss (negative P&L) side.
    colors = {0.95: "#DD8452", 0.99: "#C44E52"}
    for _, row in metrics.iterrows():
        c = row["confidence"]
        color = colors.get(c, "#555555")
        ax.axvline(
            -row["var"], color=color, linestyle="--", linewidth=1.6,
            label=f"VaR {int(c*100)}% = ${row['var']:,.0f}",
        )
        ax.axvline(
            -row["es"], color=color, linestyle=":", linewidth=1.6,
            label=f"ES {int(c*100)}%  = ${row['es']:,.0f}",
        )

    ax.set_title(title)
    ax.set_xlabel("Portfolio P&L over horizon (USD)   |   negative = loss")
    ax.set_ylabel("Frequency (count of scenarios)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
