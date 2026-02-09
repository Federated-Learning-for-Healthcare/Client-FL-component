# will instantiate modules
# src/config/builder.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch

from src.core.client import ModularFlowerClient
from src.config.registry import DEFAULT_REGISTRY
from src.config.schema import validate_config, ConfigError


@dataclass
class BuiltClient:
    client: ModularFlowerClient
    server_address: str


def _params(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    block = cfg.get(section, {})
    if not isinstance(block, dict):
        return {}
    p = block.get("params", {})
    return p if isinstance(p, dict) else {}


def build_from_config(cfg: Dict[str, Any], status_store=None) -> BuiltClient:
    # Validate first (fail fast)
    validate_config(cfg)

    reg = DEFAULT_REGISTRY

    # Resolve types
    model_type = cfg["model"]["type"]
    data_type = cfg["data"]["type"]
    trainer_type = cfg["trainer"]["type"]
    privacy_type = cfg["privacy"]["type"]
    compression_type = cfg["compression"]["type"]

    if model_type not in reg.models:
        raise ConfigError(f"Unsupported model.type: '{model_type}'")
    if data_type not in reg.data:
        raise ConfigError(f"Unsupported data.type: '{data_type}'")
    if trainer_type not in reg.trainers:
        raise ConfigError(f"Unsupported trainer.type: '{trainer_type}'")
    if privacy_type not in reg.privacy:
        raise ConfigError(f"Unsupported privacy.type: '{privacy_type}'")
    if compression_type not in reg.compression:
        raise ConfigError(f"Unsupported compression.type: '{compression_type}'")

    # Instantiate model
    model_cls = reg.models[model_type]
    model_params = _params(cfg, "model")
    # Your KAN() currently takes no args. If you want config-driven KAN params later,
    # update KAN __init__ to accept them. For now we instantiate without args safely:
    try:
        model = model_cls(**model_params)
    except TypeError:
        model = model_cls()

    # Instantiate data loader
    data_cls = reg.data[data_type]
    data_params = _params(cfg, "data")
    loader = data_cls(**data_params) if data_params else data_cls()
    train_loader, test_loader = loader.load_data()
    print("data loader ")

    # Instantiate modules
    trainer_cls = reg.trainers[trainer_type]
    trainer_params = _params(cfg, "trainer")
    trainer = trainer_cls(**trainer_params) if trainer_params else trainer_cls()
    print("data loader: trainer selected")
    privacy_cls = reg.privacy[privacy_type]
    # privacy_params = _params(cfg, "privacy")
    # privacy = privacy_cls(**privacy_params) if privacy_params else privacy_cls()
    print("data loader: privacy selected")
    compression_cls = reg.compression[compression_type]
    compression_params = _params(cfg, "compression")
    compression = compression_cls(**compression_params) if compression_params else compression_cls()
    print("data loader: compression selected")
    # Runtime
    runtime = cfg["runtime"]
    device = runtime["device"]
    server_address = runtime["server_address"]
    runtime = cfg.get("runtime", {})
    client_name = runtime.get("client_name", "unknown_client")
    print("data loader: runtime loaded")
    # Compose ModularFlowerClient
    modular_client = ModularFlowerClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        trainer=trainer,
        #privacy=privacy,
        compression=compression,
        device=device,
        status_store=status_store,
        client_name=client_name,
    )
    print("done in buider")
    return BuiltClient(client=modular_client, server_address=server_address)
