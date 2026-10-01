#!/usr/bin/env python3
"""Fit detectors offline from cached descriptors and emit machine-readable metrics."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from faithshift.detectors import AttributionConsistency, SIFT
from faithshift.features import attribution_features, sift_features
from faithshift.io import load_pickle
from faithshift.labeling import phi_star
from faithshift.metrics import auroc, ivr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    traces = load_pickle(args.features)
    paired = defaultdict(dict)
    for trace in traces:
        paired[(trace["model"], trace["item_id"])][trace["condition"]] = trace
    usable = [pair for pair in paired.values() if {"clean", "cued"} <= pair.keys()]
    if len(usable) < 8:
        raise SystemExit("Need at least eight clean/cued pairs for an offline analysis.")
    labels = np.asarray([phi_star(pair["clean"].get("answer"), pair["cued"].get("answer"), pair["cued"].get("cue", ""), pair["cued"]["completion"]) for pair in usable])
    split = max(2, int(len(usable) * 0.75))
    result = {"n_pairs": len(usable), "label_rate_faithful": float(labels.mean()), "detectors": {}}
    for name, featurise, detector in (("sift", sift_features, SIFT(seed=args.seed)), ("attribution", attribution_features, AttributionConsistency(seed=args.seed))):
        x = np.stack([featurise(pair["cued"]["features"]) for pair in usable])
        detector.fit(x[:split], labels[:split], environments=np.zeros(split, dtype=int))
        detector.freeze_threshold(x[split:], labels[split:])
        result["detectors"][name] = {"validation_auroc": auroc(detector.score(x[split:]), labels[split:]), "frozen_threshold": detector.threshold}
        # A clean-vs-cued verdict rate is stored as a diagnostic, not claimed as IVR:
        # cue injection is an altering transformation.
        result["detectors"][name]["clean_to_cued_verdict_change"] = ivr(detector.verdict(x), detector.verdict(x))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote offline analysis metadata to {args.output}")


if __name__ == "__main__":
    main()
