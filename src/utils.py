"""Shared utilities: paths, config, reproducibility."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or project_root() / "config.yaml"
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_dirs(cfg: dict[str, Any]) -> dict[str, Path]:
    root = project_root()
    paths = {key: root / rel for key, rel in cfg["paths"].items()}
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int) -> None:
    np.random.seed(seed)


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)
