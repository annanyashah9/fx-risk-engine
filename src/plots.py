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


def plot_four_method_overlay(
    pnl_by_method: dict,
    metrics_by_method: dict,
    static_loss: float,
    path: str = "figures/phase4_four_method_overlay.png",
    title: str = "Escalating tail risk: static vs Gaussian vs fat-tailed vs full jump+event",
) -> str:
    """THE headline figure. Overlay the MC method densities (Gaussian, fat-tailed, full),
    mark the static +/-10% worst loss as a reference line, and mark each model's 99% VaR/ES.

    Panel (a): densities (full-model event lump should show as a second bump on the loss side).
    Panel (b): left-tail loss exceedance on log-y so the escalating tail is unmistakable.

    `static_loss` is a negative P&L (e.g. Phase-1 "ALL -10%").
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors = {"Gaussian MC": "#4C72B0", "Fat-tailed MC": "#DD8452",
              "Full jump+event MC": "#C44E52"}
    default_cycle = ["#4C72B0", "#DD8452", "#C44E52", "#55A868"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    all_pnl = np.concatenate([np.asarray(v) for v in pnl_by_method.values()])
    lo = min(np.percentile(all_pnl, 0.1), static_loss * 1.05)
    hi = np.percentile(all_pnl, 99.9)
    bins = np.linspace(lo, hi, 100)

    def color_for(name, i):
        return colors.get(name, default_cycle[i % len(default_cycle)])

    # Panel (a): densities
    for i, (name, pnl) in enumerate(pnl_by_method.items()):
        ax1.hist(np.asarray(pnl, float), bins=bins, histtype="step", density=True,
                 linewidth=1.7, color=color_for(name, i), label=name)
    ax1.axvline(static_loss, color="black", linestyle="-", linewidth=2.0,
                label=f"Static ±10% worst = ${static_loss:,.0f}")
    ax1.axvline(0.0, color="grey", linewidth=0.8, alpha=0.5)
    ax1.set_title("(a) P&L densities — full model shows a second 'event' lump in the loss tail")
    ax1.set_xlabel("Portfolio P&L over horizon (USD)   |   negative = loss")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=8.5, loc="upper left")
    ax1.ticklabel_format(style="plain", axis="x")

    # Panel (b): left-tail loss exceedance (log-y), with 99% VaR/ES marks
    for i, (name, pnl) in enumerate(pnl_by_method.items()):
        c = color_for(name, i)
        s = np.sort(np.asarray(pnl, float))
        exceed = np.arange(1, s.size + 1) / s.size
        mask = s <= np.percentile(s, 30)
        ax2.semilogy(s[mask], exceed[mask], color=c, linewidth=1.9, label=name)
        m = metrics_by_method.get(name)
        if m is not None:
            v99 = float(m.loc[m["confidence"] == 0.99, "var"].iloc[0])
            es99 = float(m.loc[m["confidence"] == 0.99, "es"].iloc[0])
            ax2.axvline(-v99, color=c, linestyle="--", linewidth=1.0, alpha=0.6)
            ax2.axvline(-es99, color=c, linestyle=":", linewidth=1.0, alpha=0.6)
    ax2.axvline(static_loss, color="black", linewidth=2.0, label="Static ±10% worst")
    ax2.set_title("(b) Left-tail loss exceedance  P(P&L ≤ x)  [log]\n"
                  "dashed = 99% VaR, dotted = 99% ES per model")
    ax2.set_xlabel("Portfolio P&L (USD)   |   loss side")
    ax2.set_ylabel("P(P&L ≤ x)  (log)")
    ax2.legend(fontsize=8.5, loc="lower right")
    ax2.ticklabel_format(style="plain", axis="x")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_crisis_regimes(
    pnl_by_regime: dict,
    metrics_by_regime: dict,
    path: str = "figures/phase4_crisis_regimes.png",
    title: str = "Diversification erosion: scheduled event under rising crisis correlation",
) -> str:
    """Full model + event under normal / moderate / severe event correlation regimes.

    Shows how the loss tail widens as crisis correlation rises (diversification evaporates).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors = ["#55A868", "#DD8452", "#C44E52", "#8172B3"]

    fig, ax = plt.subplots(figsize=(10, 6.5))
    for (name, pnl), c in zip(pnl_by_regime.items(), colors):
        s = np.sort(np.asarray(pnl, float))
        exceed = np.arange(1, s.size + 1) / s.size
        mask = s <= np.percentile(s, 30)
        ax.semilogy(s[mask], exceed[mask], color=c, linewidth=2.0, label=name)
        m = metrics_by_regime.get(name)
        if m is not None:
            es99 = float(m.loc[m["confidence"] == 0.99, "es"].iloc[0])
            ax.axvline(-es99, color=c, linestyle=":", linewidth=1.1, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Portfolio P&L (USD)   |   loss side")
    ax.set_ylabel("P(P&L ≤ x)  (log)   |   dotted = 99% ES")
    ax.legend(fontsize=9)
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_brexit_sweep(
    p_grid,
    es99_by_size: dict,
    static_loss: float,
    realized_loss: float,
    path: str = "figures/brexit_es_vs_p.png",
    title: str = "Brexit capstone: 99% ES vs Leave probability (a sweep, NOT a prediction)",
) -> str:
    """Headline figure. 99% ES as a function of the (swept) Leave probability p, one line per
    assumed GBP jump size, with the static ±10% result as a FLAT reference and the realized
    historical loss marked. Makes visually obvious that static returns ONE number with no
    probability while the engine traces a risk profile across p.

    `es99_by_size`: {jump_size_label: array of 99% ES aligned to p_grid}. `static_loss` and
    `realized_loss` are negative P&Ls (plotted as positive loss magnitudes).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    colors = ["#8172B3", "#DD8452", "#C44E52", "#55A868", "#4C72B0"]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    for (label, es), c in zip(es99_by_size.items(), colors):
        ax.plot(p_grid, np.asarray(es) / 1e6, color=c, linewidth=2.0, marker="o",
                markersize=3, label=f"Engine 99% ES — GBP jump {label}")

    ax.axhline(-static_loss / 1e6, color="black", linestyle="--", linewidth=1.8,
               label=f"Static ±10% (GBP −10%) = ${-static_loss/1e6:,.2f}M  (no probability)")
    ax.axhline(-realized_loss / 1e6, color="grey", linestyle=":", linewidth=1.8,
               label=f"Realized 23→24 Jun 2016 loss = ${-realized_loss/1e6:,.2f}M  (context only)")

    ax.set_title(title)
    ax.set_xlabel("Assumed Leave-outcome probability  p   (swept — no true value)")
    ax.set_ylabel("99% Expected Shortfall (USD millions)")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_brexit_distribution(
    pnl: np.ndarray,
    metrics,
    static_loss: float,
    realized_loss: float,
    p_label: str,
    path: str = "figures/brexit_distribution.png",
    title: str = "Brexit capstone: portfolio P&L distribution at a representative p",
) -> str:
    """P&L distribution at a representative p, showing the distinct event-driven second lump,
    with 99% VaR/ES, the static reference, and the realized historical loss marked.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pnl = np.asarray(pnl, float)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    lo = min(np.percentile(pnl, 0.1), static_loss * 1.05, realized_loss * 1.05)
    hi = np.percentile(pnl, 99.9)
    ax.hist(pnl, bins=np.linspace(lo, hi, 100), color="#C44E52", alpha=0.75,
            edgecolor="white", linewidth=0.3, label=f"Full engine MC ({p_label})")

    v99 = float(metrics.loc[metrics["confidence"] == 0.99, "var"].iloc[0])
    es99 = float(metrics.loc[metrics["confidence"] == 0.99, "es"].iloc[0])
    ax.axvline(0.0, color="grey", linewidth=0.8, alpha=0.5)
    ax.axvline(-v99, color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"Engine 99% VaR = ${v99/1e6:,.2f}M")
    ax.axvline(-es99, color="#C44E52", linestyle=":", linewidth=1.6,
               label=f"Engine 99% ES  = ${es99/1e6:,.2f}M")
    ax.axvline(static_loss, color="black", linestyle="--", linewidth=1.8,
               label=f"Static GBP −10% = ${static_loss/1e6:,.2f}M")
    ax.axvline(realized_loss, color="dimgrey", linestyle="-", linewidth=2.0,
               label=f"Realized Jun-2016 loss = ${realized_loss/1e6:,.2f}M (context)")

    ax.set_title(title + "\n(the second lump on the loss side IS the scheduled-event mixture)")
    ax.set_xlabel("Portfolio P&L over horizon (USD)   |   negative = loss")
    ax.set_ylabel("Frequency (count of scenarios)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.ticklabel_format(style="plain", axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
