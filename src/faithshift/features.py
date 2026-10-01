"""Offline, length-normalised pooling of cached feature grids."""

from __future__ import annotations

import numpy as np

SPAN_NAMES = ("question", "cue", "instruction", "cot", "answer")


def _bands(grid: np.ndarray, count: int = 4) -> list[np.ndarray]:
    edges = np.linspace(0, grid.shape[0], count + 1).astype(int)
    return [grid[edges[i]:max(edges[i] + 1, edges[i + 1])] for i in range(count)]


def _finite(value: float) -> float:
    return float(value) if np.isfinite(value) else 0.0


def sift_features(record: dict) -> np.ndarray:
    """Pool scale-free velocity, drift, and answer-commitment grids into psi(H)."""
    velocity = np.asarray(record["velocity"], dtype=np.float32)
    drift = np.asarray(record["drift"], dtype=np.float32)
    commitment = np.asarray(record["commitment"], dtype=np.float32)
    if min(velocity.ndim, drift.ndim, commitment.ndim) != 2:
        raise ValueError("SIFT grids must be two dimensional [layers, tokens]")
    features: list[float] = []
    for grid in (velocity, drift, commitment):
        for band in _bands(grid):
            features += [_finite(band.mean()), _finite(band.std())]
    half = max(1, commitment.shape[1] // 2)
    for band in _bands(commitment):
        early, late = band[:, :half].mean(), band[:, half:].mean() if commitment.shape[1] > half else band[:, -1:].mean()
        features += [_finite(early), _finite(early - late)]
    profile = commitment.mean(axis=0)
    lo, hi = float(profile.min()), float(profile.max())
    onset = np.flatnonzero(profile >= lo + 0.8 * (hi - lo))
    features += [float(onset[0] / max(profile.size, 1)) if onset.size else 1.0, _finite(hi - lo)]
    return np.asarray(features, dtype=np.float32)


def attribution_features(record: dict) -> np.ndarray:
    """Pool answer-to-span attention rollout into scale-free attribution features."""
    final = np.asarray(record["rollout_final"], dtype=np.float32)
    layers = np.asarray(record["rollout_layers"], dtype=np.float32)
    if final.shape != (len(SPAN_NAMES),) or layers.ndim != 2 or layers.shape[1] != len(SPAN_NAMES):
        raise ValueError("invalid rollout dimensions")
    idx = {name: i for i, name in enumerate(SPAN_NAMES)}
    cue, cot = final[idx["cue"]], final[idx["cot"]]
    prompt = final[idx["question"]] + cue + final[idx["instruction"]]
    features = list(final) + [
        _finite(cot / (cot + prompt + 1e-6)),
        _finite(cue / (cue + cot + 1e-6)),
        _finite(cot - cue),
        _finite(-(np.clip(final, 1e-9, None) * np.log(np.clip(final, 1e-9, None))).sum()),
    ]
    for band in _bands(layers, 3):
        features.extend(_finite(x) for x in band.mean(axis=0))
    return np.asarray(features, dtype=np.float32)


FEATURISERS = {"sift": sift_features, "attribution": attribution_features}
