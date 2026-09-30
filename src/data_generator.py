"""Synthetic warehouse-demand generator.  Implements docs/phase0_design.md section 2.

    L(t) = mu * Trend(day) * Daily(hour) * Weekly(dow)                 deterministic level
    y(t) = round( max(0, L(t) * (1 + eps_t) * E(t)) )                  observed demand
    eps_t = rho * eps_{t-1} + eta_t,   eta_t ~ N(0, sigma_eta^2)       AR(1), relative noise
    E(t)  = product of active spike multipliers * active drop multipliers

Design choices that matter for the experiments
----------------------------------------------
* Every random quantity is drawn from its OWN stream (SeedSequence.spawn) and always as a
  full-length array, whatever the parameters are.  So changing sigma_eta changes ONLY the
  noise scale, and changing the spike rate changes ONLY which hours spike.  This gives common
  random numbers, which makes the B-noise / B-spikes decomposition of the shift experiment clean.
* Day 0 is a Monday.  hour = t % 24, day = t // 24, dow = day % 7.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

WINDOW = 24
N_FEATURES = 5  # z-demand, sin/cos hour, sin/cos dow
SPIKE_TAIL_HOURS = 12  # exp(-12/2) = 0.0025: multiplier is 1 to 3 decimals beyond this


# ----------------------------------------------------------------------------- config
@dataclass(frozen=True)
class RegimeConfig:
    mu: float = 200.0                    # base level, orders/hour
    trend_per_day: float = 0.0005        # Trend(day) = 1 + trend_per_day * day
    weekly: tuple = (1.05, 1.05, 1.00, 1.00, 0.95, 0.70, 0.50)  # Mon..Sun
    rho: float = 0.5                     # AR(1) coefficient of relative noise
    sigma_eta: float = 0.06              # innovation std (relative)
    p_spike: float = 1 / 72              # per-hour spike-onset probability
    spike_m_range: tuple = (0.5, 1.0)    # amplitude m ~ U(lo, hi)
    spike_tau: float = 2.0               # decay time constant, hours
    p_drop: float = 1 / 144              # per-hour drop-onset probability
    drop_len_range: tuple = (1, 3)       # length ~ U{lo..hi} hours
    drop_depth_range: tuple = (0.5, 0.9)  # depth d ~ U(lo, hi); multiplier = 1 - d


REGIME_A = RegimeConfig()  # training regime
REGIME_B = replace(        # full distribution shift (Phase 0 section 6)
    REGIME_A, sigma_eta=0.12, p_spike=1 / 36, spike_m_range=(1.0, 2.0)
)
REGIME_B_NOISE = replace(REGIME_A, sigma_eta=0.12)
REGIME_B_SPIKES = replace(REGIME_A, p_spike=1 / 36, spike_m_range=(1.0, 2.0))

# Chronological split of the 180-day main series (Phase 0 section 2.2)
SPLIT_DAYS = {"train": (0, 120), "val": (120, 150), "test": (150, 180)}


# ----------------------------------------------------------------------------- deterministic parts
def _circ_gauss(h: np.ndarray, centre: float, sigma: float) -> np.ndarray:
    d = np.abs(h - centre)
    d = np.minimum(d, 24 - d)  # circular distance on a 24h clock
    return np.exp(-(d ** 2) / (2 * sigma ** 2))


def daily_shape() -> np.ndarray:
    """Daily(h), h = 0..23, normalised to mean 1."""
    h = np.arange(24, dtype=float)
    raw = 0.35 + 0.9 * _circ_gauss(h, 10, 2.0) + 1.0 * _circ_gauss(h, 15, 2.5)
    return raw / raw.mean()


def event_multiplier(spike_onset, spike_m, drop_onset, drop_len, drop_depth, tau):
    """Turn onset indicators into the multiplicative event term E(t).

    spike at onset s with amplitude m contributes (1 + m*exp(-k/tau)) at hour s+k, k=0..11.
    drop  at onset s with length n, depth d contributes (1 - d) at hours s..s+n-1.
    Overlapping events multiply.
    """
    n = len(spike_onset)
    E = np.ones(n)
    spike_on = np.zeros(n, dtype=bool)
    drop_on = np.zeros(n, dtype=bool)
    for s in np.flatnonzero(spike_onset):
        k = np.arange(min(SPIKE_TAIL_HOURS, n - s))
        E[s + k] *= 1.0 + spike_m[s] * np.exp(-k / tau)
        spike_on[s : s + len(k)] = True
    for s in np.flatnonzero(drop_onset):
        e = min(s + int(drop_len[s]), n)
        E[s:e] *= 1.0 - drop_depth[s]
        drop_on[s:e] = True
    return E, spike_on, drop_on


# ----------------------------------------------------------------------------- generator
def generate_series(cfg: RegimeConfig, n_days: int, seed: int) -> dict:
    """Pure function of (cfg, n_days, seed). Returns a dict of aligned arrays of length 24*n_days."""
    n = 24 * n_days
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(6)]
    u_spike, u_mag, u_drop, u_len, u_depth = (streams[i].random(n) for i in range(5))
    z_noise = streams[5].standard_normal(n)

    t = np.arange(n)
    hour, day = t % 24, t // 24
    dow = day % 7

    level = (
        cfg.mu
        * (1.0 + cfg.trend_per_day * day)
        * daily_shape()[hour]
        * np.asarray(cfg.weekly)[dow]
    )

    # AR(1) relative noise, started from its stationary distribution
    eta = cfg.sigma_eta * z_noise
    eps = np.empty(n)
    eps[0] = eta[0] / np.sqrt(1.0 - cfg.rho ** 2)
    for i in range(1, n):
        eps[i] = cfg.rho * eps[i - 1] + eta[i]

    # events (drawn as full-length arrays; only the thresholds depend on cfg)
    lo, hi = cfg.spike_m_range
    spike_onset = u_spike < cfg.p_spike
    spike_m = lo + (hi - lo) * u_mag
    drop_onset = u_drop < cfg.p_drop
    dlo, dhi = cfg.drop_len_range
    drop_len = dlo + np.floor((dhi - dlo + 1) * u_len).astype(int)
    depth_lo, depth_hi = cfg.drop_depth_range
    drop_depth = depth_lo + (depth_hi - depth_lo) * u_depth
    E, spike_on, drop_on = event_multiplier(
        spike_onset, spike_m, drop_onset, drop_len, drop_depth, cfg.spike_tau
    )

    y = np.rint(np.maximum(0.0, level * (1.0 + eps) * E))
    return {
        "y": y.astype(np.int64), "level": level, "eps": eps, "event_mult": E,
        "hour": hour, "day": day, "dow": dow,
        "spike_onset": spike_onset, "drop_onset": drop_onset,
        "spike_active": spike_on, "drop_active": drop_on,
    }


# ----------------------------------------------------------------------------- windows / features
def train_stats(series: dict, split: str = "train") -> tuple[float, float]:
    """mu, sigma of demand on the TRAINING split only (no val/test information)."""
    a, b = SPLIT_DAYS[split]
    y = series["y"][24 * a : 24 * b].astype(float)
    return float(y.mean()), float(y.std())


def token_features(series: dict, idx: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """(..., 5) features for hours `idx`: z-demand, sin/cos hour, sin/cos dow."""
    h, d = series["hour"][idx], series["dow"][idx]
    z = (series["y"][idx] - mu) / sigma
    return np.stack(
        [z, np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
         np.sin(2 * np.pi * d / 7), np.cos(2 * np.pi * d / 7)], axis=-1
    ).astype(np.float32)


def make_windows(series: dict, mu: float, sigma: float, day_range: tuple | None = None) -> dict:
    """All windows whose 24 inputs AND the target lie inside [24*day_a, 24*day_b).

    A sample at index t uses hours t-23..t as input and hour t+1 as target, so the
    straddling-a-boundary leak described in Phase 0 section 2.2 cannot happen.
    """
    n = len(series["y"])
    a, b = (0, n) if day_range is None else (24 * day_range[0], 24 * day_range[1])
    t = np.arange(a + WINDOW - 1, b - 1)                 # last input hour; target is t+1 <= b-1
    idx = t[:, None] + np.arange(-(WINDOW - 1), 1)[None, :]   # (N, 24)
    tgt = t + 1
    y = series["y"].astype(float)
    return {
        "X": token_features(series, idx, mu, sigma),          # (N, 24, 5)
        "y_next": y[tgt],                                     # (N,)  raw orders/hour
        "y_hist": y[idx],                                     # (N, 24) raw, for baselines
        "t_last": t, "t_target": tgt,
        "target_spike": series["spike_active"][tgt],
        "target_drop": series["drop_active"][tgt],
        "target_spike_onset": series["spike_onset"][tgt],
        "target_drop_onset": series["drop_onset"][tgt],
        "target_hour": series["hour"][tgt], "target_dow": series["dow"][tgt],
        "target_level": series["level"][tgt],
    }


def build_all(data_seed: int = 1234, shift_seed: int = 4321):
    """Main 180-day Regime-A series, its train/val/test windows, and the shifted test sets."""
    main = generate_series(REGIME_A, 180, data_seed)
    mu, sigma = train_stats(main)
    out = {"mu": mu, "sigma": sigma, "series": main}
    for name, dr in SPLIT_DAYS.items():
        out[name] = make_windows(main, mu, sigma, dr)
    for name, cfg in [("shift_full", REGIME_B), ("shift_noise", REGIME_B_NOISE),
                      ("shift_spikes", REGIME_B_SPIKES), ("test_A_fresh", REGIME_A)]:
        s = generate_series(cfg, 30, shift_seed)   # same seed => common random numbers
        out[name] = make_windows(s, mu, sigma)
        out[name + "_series"] = s
    return out


# ----------------------------------------------------------------------------- CLI: regenerate
def main() -> None:
    p = argparse.ArgumentParser(description="Regenerate the synthetic warehouse datasets.")
    p.add_argument("--data-seed", type=int, default=1234)
    p.add_argument("--shift-seed", type=int, default=4321)
    p.add_argument("--out", default="results/data")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = build_all(a.data_seed, a.shift_seed)
    np.savez_compressed(out / "main_series.npz", **d["series"])
    np.savez_compressed(out / "shift_full_series.npz", **d["shift_full_series"])
    summary = {
        "data_seed": a.data_seed, "shift_seed": a.shift_seed,
        "mu_train": d["mu"], "sigma_train": d["sigma"],
        "regime_A": asdict(REGIME_A), "regime_B": asdict(REGIME_B),
        "n_windows": {k: int(len(d[k]["y_next"])) for k in
                      ["train", "val", "test", "shift_full", "shift_noise", "shift_spikes"]},
        "n_spike_onsets_train": int(d["series"]["spike_onset"][: 24 * 120].sum()),
        "n_drop_onsets_train": int(d["series"]["drop_onset"][: 24 * 120].sum()),
    }
    (out / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
