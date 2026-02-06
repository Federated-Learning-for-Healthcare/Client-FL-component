# will read YAML
# src/config/loader.py
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config into a Python dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping/dictionary.")
    return cfg
