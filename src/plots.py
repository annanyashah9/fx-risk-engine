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


def plot_tail_comparison(
    pnl_by_model: dict,
    metrics_by_model: dict,
    path: str = "figures/phase3_tail_comparison.png",
    title: str = "Fat tails vs Gaussian — portfolio P&L (10-day horizon)",
) -> str:
    """Compare model P&L distributions: heavier tails should be visually obvious.

    Panel (a): overlaid density histograms (step outlines).
    Panel (b): left-tail loss exceedance P(P&L <= x) on a log-y axis, zoomed on the loss
    side — a fatter tail sits visibly ABOVE the Gaussian curve out in the losses.

    `pnl_by_model`: {model_name: pnl_array}. `metrics_by_model`: {model_name: var_es DataFrame}
    (used only to mark each model's 99% VaR on panel b).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors = ["#4C72B0", "#DD8452", "#C44E52", "#55A868", "#8172B3", "#937860"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    all_pnl = np.concatenate([np.asarray(v) for v in pnl_by_model.values()])
    lo, hi = np.percentile(all_pnl, [0.1, 99.9])
    bins = np.linspace(lo, hi, 90)

    for (name, pnl), color in zip(pnl_by_model.items(), colors):
        pnl = np.asarray(pnl, dtype=float)
        ax1.hist(pnl, bins=bins, histtype="step", density=True, linewidth=1.6,
                 color=color, label=name)
    ax1.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    ax1.set_title("(a) P&L densities (overlaid)")
    ax1.set_xlabel("Portfolio P&L (USD)   |   negative = loss")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=9)
    ax1.ticklabel_format(style="plain", axis="x")

    # Panel (b): empirical loss-exceedance on the loss side, log-y.
    for (name, pnl), color in zip(pnl_by_model.items(), colors):
        pnl = np.sort(np.asarray(pnl, dtype=float))
        exceed = np.arange(1, pnl.size + 1) / pnl.size  # P(P&L <= x)
        mask = pnl <= np.percentile(pnl, 25)            # zoom on the loss side
        ax2.semilogy(pnl[mask], exceed[mask], color=color, linewidth=1.8, label=name)
        m = metrics_by_model.get(name)
        if m is not None:
            v99 = float(m.loc[m["confidence"] == 0.99, "var"].iloc[0])
            ax2.axvline(-v99, color=color, linestyle=":", linewidth=1.2, alpha=0.7)
    ax2.set_title("(b) Left-tail loss exceedance  P(P&L ≤ x)  [log scale]\n"
                  "higher curve = fatter loss tail; dotted = each model's 99% VaR")
    ax2.set_xlabel("Portfolio P&L (USD)   |   loss side")
    ax2.set_ylabel("P(P&L ≤ x)  (log)")
    ax2.legend(fontsize=9)
    ax2.ticklabel_format(style="plain", axis="x")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
