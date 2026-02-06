# will validate required fields
# src/config/schema.py
from __future__ import annotations
from typing import Any, Dict


class ConfigError(ValueError):
    pass


def _require(cfg: Dict[str, Any], path: str) -> Any:
    """Get nested key like 'model.type', raising if missing."""
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            raise ConfigError(f"Missing required config key: '{path}'")
        cur = cur[key]
    return cur


def validate_config(cfg: Dict[str, Any]) -> None:
    # Required blocks
    _require(cfg, "model.type")
    _require(cfg, "data.type")
    _require(cfg, "trainer.type")
    _require(cfg, "privacy.type")
    _require(cfg, "compression.type")
    _require(cfg, "runtime.server_address")
    _require(cfg, "runtime.device")

    # Optional param dicts should exist (we’ll default them if absent in builder)
    # Validate privacy params when gaussian
    privacy_type = _require(cfg, "privacy.type")
    if privacy_type == "gaussian":
        noise = _require(cfg, "privacy.params.noise_multiplier")
        clip = _require(cfg, "privacy.params.clipping_norm")
        if noise < 0:
            raise ConfigError("privacy.params.noise_multiplier must be >= 0")
        if clip <= 0:
            raise ConfigError("privacy.params.clipping_norm must be > 0")

    # Trainer params sanity (if present)
    trainer_params = cfg.get("trainer", {}).get("params", {})
    if isinstance(trainer_params, dict) and "local_epochs" in trainer_params:
        if int(trainer_params["local_epochs"]) < 1:
            raise ConfigError("trainer.params.local_epochs must be >= 1")

    # Data params sanity (if present)
    data_params = cfg.get("data", {}).get("params", {})
    if isinstance(data_params, dict) and "batch_size" in data_params:
        if int(data_params["batch_size"]) < 1:
            raise ConfigError("data.params.batch_size must be >= 1")

    # Runtime device sanity
    device = _require(cfg, "runtime.device")
    if device not in ("cpu", "cuda"):
        raise ConfigError("runtime.device must be 'cpu' or 'cuda'")
