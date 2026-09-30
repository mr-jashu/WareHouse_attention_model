"""Tests aimed at subtle bugs (leakage, off-by-one, wrong statistics), not just coverage."""
import dataclasses

import numpy as np
import pytest

from src.data_generator import (
    REGIME_A, REGIME_B, SPLIT_DAYS, WINDOW, build_all, daily_shape,
    event_multiplier, generate_series, make_windows, train_stats,
)


# ---------------------------------------------------------------- determinism
def test_same_seed_identical():
    a, b = generate_series(REGIME_A, 30, 7), generate_series(REGIME_A, 30, 7)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_different_seed_differs():
    a, b = generate_series(REGIME_A, 30, 7), generate_series(REGIME_A, 30, 8)
    assert not np.array_equal(a["y"], b["y"])


def test_output_is_nonnegative_integer_of_right_length():
    s = generate_series(REGIME_A, 10, 0)
    assert s["y"].shape == (240,) and s["y"].dtype == np.int64 and (s["y"] >= 0).all()


# ---------------------------------------------------------------- common random numbers
def test_changing_noise_scale_does_not_move_events():
    a = generate_series(REGIME_A, 60, 3)
    b = generate_series(dataclasses.replace(REGIME_A, sigma_eta=0.12), 60, 3)
    assert np.array_equal(a["spike_onset"], b["spike_onset"])
    assert np.array_equal(a["drop_onset"], b["drop_onset"])


def test_changing_noise_scale_scales_innovations_exactly():
    a = generate_series(dataclasses.replace(REGIME_A, p_spike=0, p_drop=0), 60, 3)
    b = generate_series(dataclasses.replace(REGIME_A, p_spike=0, p_drop=0, sigma_eta=0.12), 60, 3)
    assert np.allclose(b["eps"], 2 * a["eps"])   # AR(1) is linear in eta


def test_higher_spike_rate_keeps_lower_rate_spikes():
    a = generate_series(REGIME_A, 90, 5)
    b = generate_series(dataclasses.replace(REGIME_A, p_spike=1 / 36), 90, 5)
    assert (b["spike_onset"] | ~a["spike_onset"]).all()   # onsets(A) subset of onsets(B)


# ---------------------------------------------------------------- deterministic structure
def test_daily_shape_is_night_trough_and_daytime_plateau():
    # Amendment 1 in docs/phase0_design.md: with sigma=2/2.5 the two bumps merge into ONE plateau.
    d = daily_shape()
    assert d.mean() == pytest.approx(1.0)
    assert d.argmin() in (1, 2, 3)                    # overnight trough
    assert d.min() == pytest.approx(0.44, abs=0.02)
    assert d[9:17].min() > 1.4                        # broad 09:00-16:00 plateau
    assert d.max() / d.min() > 3.5


def test_level_matches_formula_by_hand():
    cfg = REGIME_A
    s = generate_series(cfg, 8, 0)
    t = 24 * 5 + 15                                   # day 5 (Saturday), 15:00
    expected = 200 * (1 + 0.0005 * 5) * daily_shape()[15] * 0.70
    assert s["level"][t] == pytest.approx(expected)
    assert (s["dow"][t], s["hour"][t]) == (5, 15)


def test_spike_multiplier_shape():
    n = 30
    onset = np.zeros(n, bool); onset[10] = True
    E, sp, dr = event_multiplier(onset, np.full(n, 1.0), np.zeros(n, bool),
                                 np.zeros(n, int), np.zeros(n), tau=2.0)
    assert E[9] == 1.0
    assert E[10] == pytest.approx(2.0)                # 1 + m at onset
    assert E[12] == pytest.approx(1 + np.exp(-1.0))   # k=2, tau=2
    assert sp[10] and not sp[9] and not dr.any()


def test_drop_multiplier_length_and_depth():
    n = 20
    onset = np.zeros(n, bool); onset[5] = True
    E, sp, dr = event_multiplier(np.zeros(n, bool), np.zeros(n), onset,
                                 np.full(n, 3), np.full(n, 0.8), tau=2.0)
    assert np.allclose(E[5:8], 0.2) and E[4] == 1 and E[8] == 1
    assert dr.sum() == 3


def test_event_near_series_end_does_not_crash():
    n = 5
    onset = np.zeros(n, bool); onset[4] = True
    event_multiplier(onset, np.ones(n), onset, np.full(n, 3), np.full(n, 0.5), 2.0)


# ---------------------------------------------------------------- statistics of the noise
def test_ar1_noise_has_designed_std_and_autocorrelation():
    cfg = dataclasses.replace(REGIME_A, p_spike=0, p_drop=0)
    s = generate_series(cfg, 2000, 11)
    eps = s["eps"]
    assert eps.std() == pytest.approx(0.06 / np.sqrt(1 - 0.25), rel=0.03)   # ~0.0693
    assert np.corrcoef(eps[:-1], eps[1:])[0, 1] == pytest.approx(0.5, abs=0.02)


def test_event_rates_close_to_config():
    s = generate_series(REGIME_A, 3000, 2)
    n = len(s["y"])
    assert s["spike_onset"].mean() == pytest.approx(1 / 72, rel=0.1)
    assert s["drop_onset"].mean() == pytest.approx(1 / 144, rel=0.15)


def test_regime_b_changes_only_what_phase0_says():
    a, b = dataclasses.asdict(REGIME_A), dataclasses.asdict(REGIME_B)
    changed = {k for k in a if a[k] != b[k]}
    assert changed == {"sigma_eta", "p_spike", "spike_m_range"}
    assert b["sigma_eta"] == 2 * a["sigma_eta"] and b["p_spike"] == 2 * a["p_spike"]


# ---------------------------------------------------------------- windows: leakage and off-by-one
@pytest.fixture(scope="module")
def data():
    return build_all(1234, 4321)


def test_window_input_and_target_alignment(data):
    s, mu, sg = data["series"], data["mu"], data["sigma"]
    w = data["val"]
    i = 17
    t = w["t_last"][i]
    assert w["t_target"][i] == t + 1
    assert w["y_next"][i] == s["y"][t + 1]                       # target is NEXT hour
    assert np.allclose(w["y_hist"][i], s["y"][t - 23 : t + 1])   # 24 inputs, ending at t
    assert np.allclose(w["X"][i, :, 0], (s["y"][t - 23 : t + 1] - mu) / sg, atol=1e-5)


def test_target_value_cannot_leak_into_inputs(data):
    """Corrupt the target hour of one sample; that sample's input X must not change."""
    s, mu, sg = data["series"], data["mu"], data["sigma"]
    w = data["val"]
    i = 17
    t = w["t_last"][i]
    s2 = {k: v.copy() for k, v in s.items()}
    s2["y"][t + 1] += 10_000
    w2 = make_windows(s2, mu, sg, SPLIT_DAYS["val"])
    assert np.array_equal(w["X"][i], w2["X"][i])
    assert w2["y_next"][i] == s["y"][t + 1] + 10_000


def test_no_window_straddles_a_split_boundary(data):
    for name, (a, b) in SPLIT_DAYS.items():
        w = data[name]
        first_hour = w["t_last"] - (WINDOW - 1)
        assert first_hour.min() >= 24 * a
        assert w["t_target"].max() <= 24 * b - 1
        assert len(w["y_next"]) == 24 * (b - a) - WINDOW      # count check: no window lost/added


def test_splits_are_chronological_and_disjoint(data):
    tr, va, te = (data[k]["t_target"] for k in ("train", "val", "test"))
    assert tr.max() < va.min() and va.max() < te.min()


def test_normalisation_uses_train_split_only(data):
    s = data["series"]
    assert data["mu"] == pytest.approx(s["y"][: 24 * 120].mean())
    assert data["mu"] != pytest.approx(s["y"].mean(), abs=1e-9)   # not the full-series mean
    mu2, sg2 = train_stats(s)
    assert (mu2, sg2) == (data["mu"], data["sigma"])


def test_feature_encoding_unit_circle(data):
    X = data["train"]["X"]
    assert X.shape[1:] == (24, 5) and X.dtype == np.float32
    assert np.allclose(X[..., 1] ** 2 + X[..., 2] ** 2, 1, atol=1e-5)   # hour
    assert np.allclose(X[..., 3] ** 2 + X[..., 4] ** 2, 1, atol=1e-5)   # dow


def test_shift_sets_are_normalised_with_training_stats(data):
    w = data["shift_full"]
    s = data["shift_full_series"]
    t = w["t_last"][0]
    assert w["X"][0, -1, 0] == pytest.approx((s["y"][t] - data["mu"]) / data["sigma"], abs=1e-4)


def test_shift_full_has_more_extreme_inputs_than_train(data):
    assert np.abs(data["shift_full"]["X"][..., 0]).max() > np.abs(data["train"]["X"][..., 0]).max()
