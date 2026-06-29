"""Tests for Phase-4 jump and scheduled-event mechanisms, the crisis-correlation regime,
and the composite generator's toggles + independent RNG substreams."""

import numpy as np
import pytest

from src.fat_tails import corr_from_cov
from src.simulation import gaussian_returns
from src.events import (
    crisis_correlation, mechanism_a_jumps, mechanism_b_event, full_model_returns,
    JumpParams, EventParams,
)


def test_crisis_correlation_preserves_sign_raises_magnitude(daily_cov):
    normal, _ = corr_from_cov(daily_cov)
    crisis = crisis_correlation(normal, rho_crisis=0.9)
    assert np.allclose(np.diag(crisis), 1.0)
    # Sign preserved, off-diagonal magnitude raised to ~0.9.
    for i, j in [(0, 1), (0, 2), (1, 2)]:
        assert np.sign(crisis[i, j]) == np.sign(normal[i, j])
        assert abs(crisis[i, j]) == pytest.approx(0.9, abs=1e-9)
    np.linalg.cholesky(crisis)                   # PSD -> Cholesky must succeed


def test_mechanism_a_zero_intensity_gives_no_jumps(daily_cov):
    normal, _ = corr_from_cov(daily_cov)
    rng = np.random.default_rng(0)
    jp = JumpParams(lam_annual=0.0)
    J = mechanism_a_jumps(10_000, 3, rng, normal, jp)
    assert J.shape == (10_000, 3)
    assert np.allclose(J, 0.0)                    # Poisson(0) -> always zero jumps


def test_mechanism_b_is_a_genuine_mixture(daily_cov):
    """p=0 -> never fires; p=1 -> always fires; otherwise a point-mass-at-0 mixture."""
    normal, _ = corr_from_cov(daily_cov)

    none = mechanism_b_event(10_000, 3, np.random.default_rng(1), normal,
                             EventParams(prob=0.0))
    assert np.allclose(none, 0.0)

    allf = mechanism_b_event(10_000, 3, np.random.default_rng(2), normal,
                             EventParams(prob=1.0))
    assert np.all(np.abs(allf).sum(axis=1) > 0)  # every row fired

    # Mixture mass: fraction of all-zero (no-event) rows ~ 1 - p.
    p = 0.3
    mix = mechanism_b_event(200_000, 3, np.random.default_rng(3), normal,
                            EventParams(prob=p))
    frac_no_event = np.mean(np.all(mix == 0.0, axis=1))
    assert frac_no_event == pytest.approx(1 - p, abs=0.01)


def test_mechanism_b_event_is_directional(daily_cov):
    """With GBP/USD targeted down, fired rows should be losses on the GBP column (index 1)."""
    normal, _ = corr_from_cov(daily_cov)
    ep = EventParams(prob=1.0, event_mean=np.array([np.log(0.96), np.log(0.90), np.log(0.97)]),
                     event_vol=np.array([0.0, 0.0, 0.0]))  # ~no dispersion -> pure direction
    shock = mechanism_b_event(1_000, 3, np.random.default_rng(4), normal, ep)
    # GBP leg sits at the directional drop (tiny residual is the PSD ridge on a 0-vol cov).
    assert shock[:, 1].mean() == pytest.approx(np.log(0.90), abs=1e-3)
    assert shock[:, 1].std() < 1e-3
    assert shock[:, 1].mean() < shock[:, 0].mean()   # GBP falls more than the EUR spillover


def _base_gen(mean_h, cov_h, n, rng):
    return gaussian_returns(mean_h, cov_h, n, rng)


def test_full_model_toggles_and_independent_substreams(daily_cov):
    """base/A/B draw from independent spawned substreams, so toggling one does not perturb
    another. Therefore the A-component is identical with or without B (and vice versa):
        (A_only - base_only) == (full - B_only)
        (B_only - base_only) == (full - A_only)
    """
    mean = np.zeros(3)
    n = 20_000

    def run(jumps, event):
        return full_model_returns(
            mean, daily_cov, n, np.random.default_rng(123),
            base_generator=_base_gen,
            enable_jumps=jumps, jump_params=JumpParams(lam_annual=5.0),
            enable_event=event, event_params=EventParams(prob=0.4),
        )

    base = run(False, False)
    a_only = run(True, False)
    b_only = run(False, True)
    full = run(True, True)

    assert np.allclose(a_only - base, full - b_only)   # jump component independent of event
    assert np.allclose(b_only - base, full - a_only)   # event component independent of jumps


def test_full_model_both_off_is_just_base(daily_cov):
    mean = np.zeros(3)
    out = full_model_returns(mean, daily_cov, 5_000, np.random.default_rng(9),
                             base_generator=_base_gen, enable_jumps=False, enable_event=False)
    # Same spawned base substream -> identical to calling the base gen on that child rng.
    expected = _base_gen(mean, daily_cov, 5_000, np.random.default_rng(9).spawn(3)[0])
    assert np.allclose(out, expected)
