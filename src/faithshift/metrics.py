"""Metrics for detector quality, invariance, agreement, and selective prediction."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


def auroc(scores, labels) -> float:
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels, dtype=int)
    if scores.size != labels.size or len(np.unique(labels)) < 2:
        return float("nan")
    ranks = stats.rankdata(scores)
    positives = labels == 1
    return float((ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2) / (positives.sum() * (~positives).sum()))


def ivr(base_verdicts, shifted_verdicts) -> float:
    base, shifted = np.asarray(base_verdicts), np.asarray(shifted_verdicts)
    if base.size == 0 or base.shape != shifted.shape:
        raise ValueError("paired non-empty verdict arrays are required")
    return float(np.mean(base != shifted))


def alteration_sensitivity(base_verdicts, altered_verdicts, directions) -> float:
    """Fraction of pairs moving in their predeclared mechanism-altering direction."""
    base, altered, directions = map(np.asarray, (base_verdicts, altered_verdicts, directions))
    if not (base.shape == altered.shape == directions.shape) or base.size == 0:
        raise ValueError("paired non-empty arrays are required")
    return float(np.mean((directions > 0) & (altered > base) | (directions < 0) & (altered < base)))


def mfs(invariance_violation_rate: float, alteration_score: float, coverage: float = 1.0) -> float:
    """Coverage-scaled harmonic mean of invariance (1-IVR) and alteration sensitivity."""
    invariant = max(0.0, min(1.0, 1.0 - invariance_violation_rate))
    altered = max(0.0, min(1.0, alteration_score))
    if invariant + altered == 0:
        return 0.0
    return max(0.0, min(1.0, coverage)) * 2 * invariant * altered / (invariant + altered)


def cohens_kappa(first, second) -> float:
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.size == 0:
        raise ValueError("paired non-empty arrays are required")
    observed = float(np.mean(first == second))
    expected = sum(float(np.mean(first == value) * np.mean(second == value)) for value in np.union1d(first, second))
    return 1.0 if math.isclose(expected, 1.0) and math.isclose(observed, 1.0) else (observed - expected) / (1 - expected)


def sgr_threshold(gate_scores, correct, alpha: float, delta: float = 0.05) -> tuple[float, float, float, float]:
    """Largest SGR-certified coverage using Clopper-Pearson risk bounds."""
    gate, correct = np.asarray(gate_scores, dtype=float), np.asarray(correct, dtype=int)
    if gate.size == 0:
        return float("inf"), 0.0, float("nan"), float("nan")
    order, errors = np.argsort(-gate), None
    ranked_gate, ranked_correct = gate[order], correct[order]
    cumulative_errors = np.cumsum(1 - ranked_correct)
    trials = max(1, int(np.ceil(np.log2(max(gate.size, 2)))))
    best = (float("inf"), 0.0, float("nan"), float("nan"))
    low, high = 1, gate.size
    while low <= high:
        count = (low + high) // 2
        err = int(cumulative_errors[count - 1])
        bound = 1.0 if err == count else float(stats.beta.ppf(1 - delta / trials, err + 1, count - err))
        if bound <= alpha:
            best, low = (float(ranked_gate[count - 1]), count / gate.size, err / count, bound), count + 1
        else:
            high = count - 1
    return best
