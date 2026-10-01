"""Analytic Proposition-2 / Example-1 IVR bounds and a Monte-Carlo checker."""

from __future__ import annotations

import numpy as np
from scipy.special import erf


def ivr_lower_bound(bands: float | np.ndarray, sigma: float) -> float:
    """Return 1/2 E[erf(gamma|Delta|/(sqrt(2)sigma))] for Gaussian margins."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    values = np.asarray(bands, dtype=float)
    return 0.5 * float(np.mean(erf(np.abs(values) / (np.sqrt(2.0) * sigma))))


def ivr_lower_bound_empirical(margins: np.ndarray, bands: np.ndarray) -> float:
    """Empirical form: one half of the near-threshold margin probability."""
    margins, bands = np.asarray(margins, dtype=float), np.asarray(bands, dtype=float)
    if margins.shape != bands.shape:
        raise ValueError("margins and bands must have matching shapes")
    return 0.5 * float(np.mean(np.abs(margins) <= np.abs(bands)))


def simulate_ivr(band: float, sigma: float, n: int = 100_000, seed: int = 0) -> tuple[float, float]:
    """Simulate a symmetric constant-magnitude perturbation and its analytic floor."""
    rng = np.random.default_rng(seed)
    margin = rng.normal(0.0, sigma, size=n)
    displacement = band * rng.choice((-1.0, 1.0), size=n)
    flips = float(np.mean(np.signbit(margin) != np.signbit(margin + displacement)))
    return flips, ivr_lower_bound(np.abs(displacement), sigma)
