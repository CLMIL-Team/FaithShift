#!/usr/bin/env python3
"""Lightweight analytic/Monte-Carlo smoke check; no model or dataset needed."""

from faithshift.theory import simulate_ivr


def main() -> None:
    for band in (0.1, 0.5, 1.0):
        observed, floor = simulate_ivr(band, sigma=1.0, seed=0)
        print(f"band={band:.1f}: observed IVR={observed:.4f}, analytic floor={floor:.4f}")
        if observed + 0.01 < floor:
            raise SystemExit("Monte-Carlo estimate violated the analytic lower bound")


if __name__ == "__main__":
    main()
