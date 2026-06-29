# FX Risk Scenario Engine

Stress-tests a portfolio of FX forward hedges against correlated, fat-tailed, jump-prone
currency moves — and shows it produces far more realistic tail-risk estimates than the
static "up 10% / down 10%" scenario analysis most corporate treasury teams rely on.

The engine is built in four phases on a single reusable revaluation function, then applied
to a historical worked example (the June 2016 Brexit referendum).

## Conventions (the correctness backbone)

- **Quote/sign:** "long PAIR" = long the first-named currency. EUR/USD and GBP/USD are
  quoted USD-per-foreign (`usd_value = N·rate`); **USD/JPY is inverted** — quoted JPY-per-USD
  (`usd_value = N/rate`) and long USD/JPY means **short the JPY notional** (`sign = -1`). Both
  the inverted value function and the negative sign are required; either alone flips the P&L.
- **Risk horizon:** 10 trading days. Covariance is **horizon-scaled** (`cov_H = H·cov_daily`),
  not annualized.
- **Sign of risk numbers:** P&L negative = loss; VaR/ES are reported as positive loss
  magnitudes. Seed=42, 10,000 scenarios throughout (reproducible).
- **Data:** daily FX from FRED (`DEXUSEU`→EUR/USD, `DEXUSUK`→GBP/USD, `DEXJPUS`→USD/JPY).

## Phases

| Phase | What it adds | Entry point |
|-------|--------------|-------------|
| 1 | Portfolio + mark-to-market + static ±10% baseline | `main.py` |
| 2 | Correlated **Gaussian Monte Carlo** (Cholesky), VaR & Expected Shortfall | `main_phase2.py` |
| 3 | **Fat tails** — multivariate-t and (headline) **t-copula** | `main_phase3.py` |
| 4 | **Jumps** (Merton, unscheduled) + **scheduled binary event** (mixture) | `main_phase4.py` |
| Capstone | **Brexit** application — probability sweep of a dated event | `main_brexit.py` |

The mark-to-market function (`src/mtm.py`) and risk metrics (`src/risk.py`) are written once
and reused unchanged by every later phase. Each phase only swaps the **return generator**
behind a stable seam `(mean_h, cov_h, n_sims, rng) -> returns`.

### Headline results (default portfolio, 10-day, 99% ES)

- Gaussian MC understated the 99% ES vs the **fat-tailed** t-copula by ~**1.26×**.
- Adding the **full jump + scheduled-event** model raised 99% ES to ~**4×** the Gaussian.
- A subtle, correct twist: for this book, rising **crisis correlation** *reduces* the event
  tail, because the long-USD/JPY leg is a partial USD hedge against the (short-USD) EUR/GBP
  legs and that hedge strengthens in a USD-driven crisis — the opposite of the textbook
  "diversification evaporates," and a direct consequence of the inverted USD/JPY convention.

## Brexit capstone — conclusion

**This is a probability sweep, not a prediction. It makes no claim to have foreseen the
outcome.** It replays June 2016 to compare what the engine would have reported as tail risk
ahead of a dated event against the static method and the realized move.

Setup (honest about hindsight): correlation/df inputs are estimated **only from data ending
23 June 2016** (genuinely ex-ante); the assumed GBP Leave-move (~10%) is a plausible
pre-event estimate that a real exercise would take from option-implied vols / risk reversals,
and it is **swept** (6–12%) to show the conclusion does not depend on the exact figure; the
Leave-outcome probability `p` has no true value and is **swept** (0–50%). The realized move
(GBP/USD **−7.84%** close-to-close, 23→24 June 2016) is shown for context only.

Result: the static ±10% method returns **one unconditional number** (≈ **−$3.70M** for a
GBP −10% shock) with no probability attached and no awareness of a dated event. The engine
instead **traces tail risk across the plausible range of Leave probabilities** — at p = 40%
(GBP move 10%) it flags a **99% ES of ≈ $6.8M**, roughly **1.8× the static figure** and
**2.2× the realized move's implied loss (−$3.09M)**. The P&L distribution is visibly
**bimodal** — a distinct second lump on the loss side that is the scheduled-event mixture, a
calendar cliff-edge that smoothing everyday volatility can never reproduce.

The takeaway: the static method was *structurally* blind to a dated, binary event; the engine
would have given a treasury a materially larger, probability-aware warning — **without
predicting the result.**

See `figures/brexit_es_vs_p.png` (the sweep vs the flat static line) and
`figures/brexit_distribution.png` (the event lump), with numbers in
`results/brexit_summary.csv`.

## Running

```bash
pip install -r requirements.txt
python main.py            # Phase 1: static baseline
python main_phase2.py     # Phase 2: Gaussian MC, VaR/ES
python main_phase3.py     # Phase 3: fat tails
python main_phase4.py     # Phase 4: jumps + scheduled event
python main_brexit.py     # Capstone: Brexit probability sweep
```

Outputs are written to `results/` (tables) and `figures/` (plots); raw FX data is cached in
`data/`. All three are gitignored.

## Tests

```bash
pip install -r requirements-dev.txt   # adds pytest
pytest                                 # 35 tests, deterministic, no network
```

The suite (`tests/`) pins the correctness backbone: MTM sign/quote conventions incl. the
inverted USD/JPY and the vectorization contract (`test_mtm.py`), VaR/ES definitions and sign
(`test_risk.py`), horizon scaling + PSD repair + reproducibility (`test_simulation.py`),
fat-tail df estimation and covariance/correlation preservation (`test_fat_tails.py`), and the
Phase-4 mixture, crisis-correlation, and independent-substream toggles (`test_events.py`).
Tests use synthetic data only — no FRED calls.
