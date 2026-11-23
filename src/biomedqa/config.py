"""YAML run configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} must be a YAML mapping")
    return cfg


def resolve(path: str | Path, root: Path = REPO_ROOT) -> Path:
    """Resolve a config path relative to the repo root (absolute paths are kept)."""
    p = Path(path)
    return p if p.is_absolute() else root / p
