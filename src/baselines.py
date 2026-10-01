"""Simple baselines for warehouse forecasting (Phase 0 section 5).

All baselines operate on raw demand histories (y_hist, shape (N, 24)) so they
need no training. Evaluated on the same test windows as the attention model.
"""
from __future__ import annotations

import numpy as np


def last_value(y_hist: np.ndarray) -> np.ndarray:
    """y^(t+1) = y(t): repeat the most recent hour."""
    return y_hist[:, -1].astype(float)


def moving_average_24(y_hist: np.ndarray) -> np.ndarray:
    """y^(t+1) = mean of the 24h window (ignores time of day)."""
    return y_hist.mean(axis=1)


def seasonal_naive(y_hist: np.ndarray) -> np.ndarray:
    """y^(t+1) = y(t-23): same hour yesterday (position 1 of the window)."""
    return y_hist[:, 0].astype(float)


BASELINES = {
    "last": last_value,
    "ma24": moving_average_24,
    "seasonal": seasonal_naive,
}


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.abs(pred - true).mean())


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(((pred - true) ** 2).mean()))
