"""Validated post-hoc transforms used by the FaithShift protocol."""

from __future__ import annotations

import random
import re

PRESERVING_AXES = ("paraphrase", "hint_format", "language", "budget", "seed_repeat", "step_reorder", "model_family")
ALTERING_AXES = ("cue_injection", "cue_verbalization", "causal_truncation")
_ANSWER = re.compile(r"^\s*(?:answer|cevap)\s*[:：]", re.IGNORECASE)
_BACK_REFERENCE = re.compile(r"^\s*(so|then|therefore|thus|this|that|hence|it|now|next|finally)\b", re.IGNORECASE)
_NUMBERS = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def split_steps(trace: str) -> tuple[list[str], list[str]]:
    lines = trace.splitlines()
    cut = next((i for i, line in enumerate(lines) if _ANSWER.match(line)), len(lines))
    steps, answer = [line.strip() for line in lines[:cut] if line.strip()], lines[cut:]
    if len(steps) <= 1:
        steps = [s.strip() for s in re.split(r"(?<=[.!?])\s+", " ".join(steps)) if s.strip()]
    return steps, answer


def independent(first: str, second: str) -> bool:
    first_numbers = {x.replace(",", "") for x in _NUMBERS.findall(first)}
    second_numbers = {x.replace(",", "") for x in _NUMBERS.findall(second)}
    return not _BACK_REFERENCE.match(second) and not (first_numbers & second_numbers)


def step_reorder(trace: str, seed: int = 0) -> str | None:
    steps, answer = split_steps(trace)
    candidates = [i for i in range(len(steps) - 1) if independent(steps[i], steps[i + 1])]
    if not candidates:
        return None
    index = random.Random(seed).choice(candidates)
    steps[index], steps[index + 1] = steps[index + 1], steps[index]
    return "\n".join(steps + answer)


def causal_truncate(trace: str, fraction: float = 0.5) -> str | None:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be strictly between zero and one")
    steps, _ = split_steps(trace)
    if len(steps) < 2:
        return None
    keep = min(len(steps) - 1, max(1, int(len(steps) * fraction)))
    return "\n".join(steps[:keep])
