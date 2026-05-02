"""
schema.py — Config validation for the FL client.

Fix: changed val <= min_val to val < min_val so that 0.0 is accepted
for momentum and noise_multiplier. Only negative values are rejected.
"""

from __future__ import annotations
from typing import Any, Dict


class ConfigError(ValueError):
    pass


_REQUIRED_SECTIONS = [
    "model", "data", "trainer", "privacy", "compression", "runtime"
]

# Only these sections require a "type" field
_SECTIONS_WITH_TYPE = ["model", "data", "trainer", "privacy", "compression"]

_VALID_TYPES = {
    "model":       ["kan", "mlp"],
    "data":        ["mnist", "ehr", "ecg"],
    "trainer":     ["standard"],
    "privacy":     ["none", "gaussian", "dpsgd"],
    "compression": ["none", "topk", "quantize"],
}

# (section, param, min_val, max_val)
# Constraint: val must be >= min_val (i.e. reject val < min_val only)
_PARAM_CONSTRAINTS = [
    ("privacy",     "noise_multiplier", 0.0,  None),
    ("privacy",     "clipping_norm",    0.0,  None),
    ("compression", "top_k_ratio",      0.0,  1.0),
    ("trainer",     "lr",               0.0,  None),
    ("trainer",     "momentum",         0.0,  1.0),
]


def validate_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise ConfigError(f"Config must be a dict, got {type(cfg).__name__}")

    for section in _REQUIRED_SECTIONS:
        if section not in cfg:
            raise ConfigError(f"Missing required config section '{section}'")
        if not isinstance(cfg[section], dict):
            raise ConfigError(f"Config section '{section}' must be a dict")

    # Only sections that actually use a type field get this check
    for section in _SECTIONS_WITH_TYPE:
        if "type" not in cfg[section]:
            raise ConfigError(f"Missing 'type' in config section '{section}'")

    for section, valid_values in _VALID_TYPES.items():
        type_val = cfg[section].get("type")
        if type_val not in valid_values:
            raise ConfigError(
                f"Invalid type '{type_val}' for '{section}'. "
                f"Must be one of: {valid_values}"
            )

    if "server_address" not in cfg.get("runtime", {}):
        raise ConfigError("Missing 'server_address' in runtime config")

    for section, param, min_val, max_val in _PARAM_CONSTRAINTS:
        params = cfg.get(section, {}).get("params", {})
        if not isinstance(params, dict) or param not in params:
            continue
        val = params[param]
        if not isinstance(val, (int, float)):
            raise ConfigError(
                f"{section}.params.{param} must be numeric, "
                f"got {type(val).__name__}"
            )
        # Use strict < so that 0.0 is accepted (only negatives are rejected)
        if min_val is not None and val < min_val:
            raise ConfigError(
                f"{section}.params.{param}={val} must be >= {min_val}"
            )
        if max_val is not None and val > max_val:
            raise ConfigError(
                f"{section}.params.{param}={val} must be <= {max_val}"
            )

    if cfg["compression"]["type"] == "quantize":
        bits = cfg["compression"].get("params", {}).get("bits", 16)
        if bits not in (8, 16):
            raise ConfigError(
                f"compression.params.bits must be 8 or 16, got {bits}"
            )

    layers = cfg["model"].get("params", {}).get("layers_hidden")
    if layers is not None:
        if not isinstance(layers, list) or len(layers) < 2:
            raise ConfigError(
                "model.params.layers_hidden must be a list with >= 2 elements"
            )
        if not all(isinstance(d, int) and d > 0 for d in layers):
            raise ConfigError(
                "model.params.layers_hidden must contain positive integers"
            )