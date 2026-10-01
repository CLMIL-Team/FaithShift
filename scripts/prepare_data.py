#!/usr/bin/env python3
"""Fetch configured public datasets and write canonical FaithShift item records."""

from __future__ import annotations

import argparse
from pathlib import Path

from faithshift.configuration import load_config
from faithshift.data import load_domain
from faithshift.io import write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--output", help="Defaults to <data_dir>/items.jsonl")
    args = parser.parse_args()
    config = load_config(args.config)
    records = []
    for domain in config["datasets"]["domains"]:
        records.extend(item.to_dict() for item in load_domain(domain, config["datasets"]["items_per_domain"], config["seed"]))
    output = Path(args.output or Path(config["data_dir"]) / "items.jsonl")
    write_jsonl(output, records)
    print(f"Wrote {len(records)} canonical items to {output}")


if __name__ == "__main__":
    main()
