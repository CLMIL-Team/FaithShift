"""Configuration loading with project-root-relative paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping and resolve its data/output directories against its root."""
    import yaml

    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be a YAML mapping")
    root = path.parent.parent
    for key in ("data_dir", "output_dir"):
        if key in data:
            value = Path(data[key])
            data[key] = str(value if value.is_absolute() else root / value)
    return data
