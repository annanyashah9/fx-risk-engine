"""Historical FX data handling for Phase 2 (correlated Monte Carlo).

Pulls daily FX rates for EUR/USD, GBP/USD, USD/JPY from FRED (no API key needed) and
computes daily log returns. Falls back to a local CSV in `data/` if the live pull fails.

FRED series are chosen to match the engine's quote conventions EXACTLY (see portfolio.py):
    DEXUSEU  = US dollars per 1 euro          -> EURUSD  (USD per EUR)
    DEXUSUK  = US dollars per 1 UK pound       -> GBPUSD  (USD per GBP)
    DEXJPUS  = Japanese yen per 1 US dollar     -> USDJPY  (JPY per USD)
So a FRED column maps directly onto a portfolio pair with no inversion. Missing
observations (weekends/holidays) are marked "." in FRED CSVs and parsed as NaN.

NOTE on data vs revaluation: this historical data is used ONLY to estimate the
covariance/drift of returns. The portfolio is revalued at the *current* spot
(`SPOT_RATES` in portfolio.py), not at the last historical level.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

# FRED series id -> engine pair name (order here defines the CSV column order).
FRED_SERIES = {
    "DEXUSEU": "EURUSD",
    "DEXUSUK": "GBPUSD",
    "DEXJPUS": "USDJPY",
}

PAIRS = list(FRED_SERIES.values())  # ["EURUSD", "GBPUSD", "USDJPY"]

DEFAULT_START = "2019-01-01"
RAW_CSV_PATH = "data/fx_rates.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"


def _fetch_fred_series(series_id: str) -> pd.Series:
    """Download one FRED series as a date-indexed float Series (NaN for '.')."""
    df = pd.read_csv(
        FRED_URL.format(series=series_id),
        na_values=".",
        parse_dates=["observation_date"],
    )
    df = df.set_index("observation_date")
    return df[series_id].astype(float)


def fetch_fx_history(start: str = DEFAULT_START, save_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Fetch and merge the three FX series from FRED into one levels DataFrame.

    Columns are the engine pair names (EURUSD, GBPUSD, USDJPY). Rows are inner-joined on
    date and any row with a missing value is dropped, so all three series are aligned.
    The merged levels are saved to `save_path`. Raises on network failure (caller may
    fall back to `load_fx_csv`).
    """
    series = {}
    for series_id, pair in FRED_SERIES.items():
        s = _fetch_fred_series(series_id)
        series[pair] = s
    df = pd.DataFrame(series)
    df = df.loc[df.index >= pd.Timestamp(start)]
    df = df.dropna(how="any").sort_index()
    df.index.name = "date"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    return df


def load_fx_csv(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Load previously-saved merged FX levels from a local CSV (offline fallback)."""
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date")
    return df[PAIRS].dropna(how="any").sort_index()


def get_fx_history(start: str = DEFAULT_START, save_path: str = RAW_CSV_PATH) -> pd.DataFrame:
    """Fetch from FRED, falling back to the local CSV if the live pull fails."""
    try:
        df = fetch_fx_history(start=start, save_path=save_path)
        print(f"[data] Fetched FX history from FRED: {len(df)} rows.")
        return df
    except Exception as exc:  # network / parse failure -> offline fallback
        if os.path.exists(save_path):
            print(f"[data] FRED fetch failed ({exc!r}); loading local {save_path}.")
            return load_fx_csv(save_path)
        raise RuntimeError(
            f"FRED fetch failed and no local fallback at {save_path}: {exc!r}"
        ) from exc


def log_returns(levels: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns r_t = ln(P_t / P_{t-1}) for each pair, NaNs dropped."""
    return np.log(levels / levels.shift(1)).dropna(how="any")
