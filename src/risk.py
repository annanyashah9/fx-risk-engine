"""Risk metrics (VaR, Expected Shortfall) for the FX risk scenario engine (Phase 2).

SIGN CONVENTION (stated everywhere, must stay consistent across all phases):
    `pnl` is the signed USD change in portfolio value; NEGATIVE = loss.
    VaR and ES are reported as POSITIVE loss magnitudes (a $1m loss -> VaR = +1,000,000).

Definitions at confidence c (e.g. 0.95):
    VaR_c = -quantile(pnl, 1 - c)
            the loss not exceeded with probability c; the (1-c) lower-tail quantile, sign-flipped.
    ES_c  = -mean(pnl | pnl <= quantile(pnl, 1 - c))
            the average loss GIVEN we are in the worst (1-c) tail (>= VaR_c by construction).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def var_es(pnl: np.ndarray, confidences=(0.95, 0.99)) -> pd.DataFrame:
    """Compute VaR and ES (positive loss magnitudes) at each confidence level.

    Returns a tidy DataFrame with columns: confidence, var, es, plus repeated
    distribution-level mean/std for convenience.
    """
    pnl = np.asarray(pnl, dtype=float)
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))

    rows = []
    for c in confidences:
        q = np.quantile(pnl, 1.0 - c)          # lower-tail quantile of P&L (a loss -> negative)
        tail = pnl[pnl <= q]
        var = -q
        es = -tail.mean() if tail.size else float("nan")
        rows.append(
            {
                "confidence": c,
                "var": var,
                "es": es,
                "pnl_mean": mean,
                "pnl_std": std,
            }
        )
    return pd.DataFrame(rows)
