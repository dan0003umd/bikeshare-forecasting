"""Shared probabilistic metrics for every forecasting model."""
from __future__ import annotations

import numpy as np

from .config import QUANTILES


def _arrays(y_true, quantile_predictions, quantiles=QUANTILES):
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(quantile_predictions, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if pred.shape != (len(y), len(q)):
        raise ValueError(f"expected predictions shaped ({len(y)}, {len(q)}), got {pred.shape}")
    if not np.all(np.diff(q) > 0) or q[0] <= 0 or q[-1] >= 1:
        raise ValueError("quantiles must be strictly increasing values between zero and one")
    return y, pred, q


def pinball_loss(y_true, quantile_predictions, quantiles=QUANTILES) -> float:
    """Mean pinball loss over observations and the fixed quantile grid."""
    y, pred, q = _arrays(y_true, quantile_predictions, quantiles)
    error = y[:, None] - pred
    return float(np.maximum(q * error, (q - 1) * error).mean())


def crps_from_quantiles(y_true, quantile_predictions, quantiles=QUANTILES) -> float:
    """CRPS approximation: twice the integral of pinball loss over quantiles."""
    y, pred, q = _arrays(y_true, quantile_predictions, quantiles)
    losses = np.maximum(q * (y[:, None] - pred), (q - 1) * (y[:, None] - pred))
    return float(2 * np.trapezoid(losses, q, axis=1).mean())


def interval_coverage_90(y_true, quantile_predictions, quantiles=QUANTILES) -> float:
    """Empirical inclusion rate for the 5th--95th percentile interval."""
    y, pred, q = _arrays(y_true, quantile_predictions, quantiles)
    lo = pred[:, np.where(np.isclose(q, 0.05))[0][0]]
    hi = pred[:, np.where(np.isclose(q, 0.95))[0][0]]
    return float(((y >= lo) & (y <= hi)).mean())


def median_mae(y_true, quantile_predictions, quantiles=QUANTILES) -> float:
    """Mean absolute error of the 0.5 quantile."""
    y, pred, q = _arrays(y_true, quantile_predictions, quantiles)
    median = pred[:, np.where(np.isclose(q, 0.5))[0][0]]
    return float(np.abs(y - median).mean())


def score(y_true, quantile_predictions, quantiles=QUANTILES) -> dict[str, float]:
    """Return the common report schema used by all model evaluators."""
    y, pred, q = _arrays(y_true, quantile_predictions, quantiles)
    return {
        "pinball_loss": pinball_loss(y, pred, q),
        "crps": crps_from_quantiles(y, pred, q),
        "coverage_90": interval_coverage_90(y, pred, q),
        "median_mae": median_mae(y, pred, q),
        "n_observations": int(len(y)),
    }
