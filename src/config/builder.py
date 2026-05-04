"""
builder.py — Factory: builds all modules from config dict.

Fix: removed output_dir default parameter. CheckpointStore is only
created here when not supplied AND an explicit checkpoint_dir is given.
When called from _fl_process, checkpoint_store is always passed in
explicitly so the builder never needs to derive a path itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.config.registry import DEFAULT_REGISTRY
from src.config.schema import ConfigError, validate_config
from src.core.client import ModularFlowerClient
from src.core.multimodal_client import MultiModalFlowerClient
from src.models.multimodal import (
    ECGModalityConfig, EHRModalityConfig, MRIModalityConfig, MultiModalModel,
)
from src.observerbility.checkpoint_store import CheckpointStore
from src.observerbility.metrics_store import MetricsStore

logger = logging.getLogger(__name__)

_NO_PARAM_CLASSES = {"none"}


@dataclass
class BuiltClient:
    client:           Any   # ModularFlowerClient or MultiModalFlowerClient
    server_address:   str
    federation_id:    str
    checkpoint_store: Optional[CheckpointStore]


def _instantiate(cls, type_key: str, params: Dict) -> Any:
    if type_key in _NO_PARAM_CLASSES or not params:
        return cls()
    try:
        return cls(**params)
    except TypeError as e:
        raise ConfigError(
            f"Failed to instantiate {cls.__name__} with params {params}: {e}"
        ) from e


def _params(cfg: Dict, section: str) -> Dict:
    block = cfg.get(section, {})
    p = block.get("params", {})
    return p if isinstance(p, dict) else {}


def _build_multimodal_model(model_params: Dict) -> MultiModalModel:
    """
    Construct a MultiModalModel from the model.params config block.

    model.params layout:
      n_classes  : int   (required)
      embed_dim  : int   (default 128)
      modal_mask : list  (optional — derived from enabled modality cfgs if absent)
      ecg:
        mode       : flat | waveform  (default flat)
        n_features : int (flat mode, default 187)
        n_leads    : int (waveform mode, default 1)
        embed_dim  : int (default 128)
      mri:
        in_channels : int (default 1)
        img_size    : int (default 64)
        embed_dim   : int (default 128)
      ehr:
        n_features : int (default 13)
        embed_dim  : int (default 128)

    A modality encoder is instantiated only when its sub-block is present
    in the config — omit the block entirely to leave that encoder as None.
    """
    n_classes        = model_params.get("n_classes")
    embed_dim        = model_params.get("embed_dim", 128)
    modal_mask       = model_params.get("modal_mask", None)
    fusion_head_type = model_params.get("fusion_head_type", "kan")

    if n_classes is None:
        raise ConfigError("model.params.n_classes is required for multimodal model")

    ecg_raw = model_params.get("ecg")
    mri_raw = model_params.get("mri")
    ehr_raw = model_params.get("ehr")

    ecg_cfg = ECGModalityConfig(
        n_features = ecg_raw.get("n_features", 187),
        n_leads    = ecg_raw.get("n_leads", 1),
        mode       = ecg_raw.get("mode", "flat"),
        embed_dim  = ecg_raw.get("embed_dim", embed_dim),
    ) if ecg_raw else None

    mri_cfg = MRIModalityConfig(
        in_channels = mri_raw.get("in_channels", 1),
        img_size    = mri_raw.get("img_size", 64),
        embed_dim   = mri_raw.get("embed_dim", embed_dim),
    ) if mri_raw else None

    ehr_cfg = EHRModalityConfig(
        n_features = ehr_raw.get("n_features", 13),
        embed_dim  = ehr_raw.get("embed_dim", embed_dim),
    ) if ehr_raw else None

    return MultiModalModel(
        ecg_cfg          = ecg_cfg,
        mri_cfg          = mri_cfg,
        ehr_cfg          = ehr_cfg,
        n_classes        = int(n_classes),
        embed_dim        = int(embed_dim),
        modal_mask       = modal_mask,
        fusion_head_type = fusion_head_type,
    )


def build_from_config(
    cfg:              Dict[str, Any],
    status_store=     None,
    metrics_store:    Optional[MetricsStore] = None,
    checkpoint_store: Optional[CheckpointStore] = None,
    federation_id:    str = "default",
) -> BuiltClient:
    """
    Build a complete ModularFlowerClient from a config dict.

    checkpoint_store should always be passed explicitly by the caller
    (FederationManager or _fl_process). If not supplied, checkpointing
    is disabled — no default path is guessed because the caller's cwd
    may not be predictable (especially in subprocesses).
    """
    validate_config(cfg)
    reg = DEFAULT_REGISTRY

    model_type       = cfg["model"]["type"]
    data_type        = cfg["data"]["type"]
    trainer_type     = cfg["trainer"]["type"]
    privacy_type     = cfg["privacy"]["type"]
    compression_type = cfg["compression"]["type"]

    logger.info(
        "[%s] Building — model=%s  data=%s  trainer=%s  privacy=%s  compression=%s",
        federation_id, model_type, data_type,
        trainer_type, privacy_type, compression_type,
    )

    # ── Shared: privacy + compression (same for all model types) ─────────────
    privacy_cls = reg.privacy.get(privacy_type)
    if not privacy_cls:
        raise ConfigError(
            f"Unknown privacy type '{privacy_type}'. Available: {list(reg.privacy)}"
        )
    privacy = _instantiate(privacy_cls, privacy_type, _params(cfg, "privacy"))
    logger.info("[%s] Privacy: %s", federation_id, repr(privacy))

    compression_cls = reg.compression.get(compression_type)
    if not compression_cls:
        raise ConfigError(
            f"Unknown compression type '{compression_type}'. "
            f"Available: {list(reg.compression)}"
        )
    compression = _instantiate(
        compression_cls, compression_type, _params(cfg, "compression")
    )
    logger.info("[%s] Compression: %s", federation_id, repr(compression))

    runtime        = cfg.get("runtime", {})
    device         = runtime.get("device", "cpu")
    server_address = runtime["server_address"]
    client_name    = runtime.get("client_name", federation_id)

    if checkpoint_store is None:
        logger.warning(
            "[%s] No checkpoint_store supplied — checkpointing disabled. "
            "Pass checkpoint_store explicitly to enable.", federation_id
        )

    # ── Multimodal path ───────────────────────────────────────────────────────
    if model_type == "multimodal":
        from src.modules.multimodal_trainer import MultiModalTrainer

        model = _build_multimodal_model(_params(cfg, "model"))
        logger.info("[%s] Model: MultiModalModel  mask=%s", federation_id, model.modal_mask)

        if metrics_store is not None:
            metrics_store.param_count = sum(p.numel() for p in model.parameters())
            metrics_store.model_type  = "multimodal"

        data_cls = reg.data.get(data_type)
        if not data_cls:
            raise ConfigError(
                f"Unknown data type '{data_type}'. Available: {list(reg.data)}"
            )
        loader = _instantiate(data_cls, data_type, _params(cfg, "data"))
        train_loader, test_loader = loader.load_data()
        logger.info("[%s] Data: %s", federation_id, data_cls.__name__)

        trainer_cls = reg.trainers.get(trainer_type)
        if not trainer_cls:
            raise ConfigError(
                f"Unknown trainer type '{trainer_type}'. Available: {list(reg.trainers)}"
            )
        if not issubclass(trainer_cls, MultiModalTrainer):
            raise ConfigError(
                f"Multimodal model requires trainer.type 'multimodal', got '{trainer_type}'"
            )
        trainer = _instantiate(trainer_cls, trainer_type, _params(cfg, "trainer"))
        logger.info("[%s] Trainer: %s", federation_id, trainer_cls.__name__)

        client = MultiModalFlowerClient(
            model            = model,
            modal_mask       = model.modal_mask,
            train_loader     = train_loader,
            test_loader      = test_loader,
            trainer          = trainer,
            privacy          = privacy,
            compression      = compression,
            device           = device,
            status_store     = status_store,
            metrics_store    = metrics_store,
            checkpoint_store = checkpoint_store,
            client_name      = client_name,
        )

        logger.info("[%s] MultiModalFlowerClient built — server=%s  device=%s",
                    federation_id, server_address, device)

        return BuiltClient(
            client           = client,
            server_address   = server_address,
            federation_id    = federation_id,
            checkpoint_store = checkpoint_store,
        )

    # ── Standard single-modality path ─────────────────────────────────────────
    model_cls = reg.models.get(model_type)
    if not model_cls:
        raise ConfigError(
            f"Unknown model type '{model_type}'. Available: {list(reg.models)}"
        )
    model = _instantiate(model_cls, model_type, _params(cfg, "model"))
    logger.info("[%s] Model: %s", federation_id, model.__class__.__name__)

    if metrics_store is not None:
        param_count = sum(p.numel() for p in model.parameters())
        metrics_store.param_count = param_count
        metrics_store.model_type  = model_type
        logger.info("[%s] Model param count: %d", federation_id, param_count)

    data_cls = reg.data.get(data_type)
    if not data_cls:
        raise ConfigError(
            f"Unknown data type '{data_type}'. Available: {list(reg.data)}"
        )
    loader = _instantiate(data_cls, data_type, _params(cfg, "data"))
    train_loader, test_loader = loader.load_data()
    logger.info("[%s] Data: %s", federation_id, data_cls.__name__)

    trainer_cls = reg.trainers.get(trainer_type)
    if not trainer_cls:
        raise ConfigError(
            f"Unknown trainer type '{trainer_type}'. Available: {list(reg.trainers)}"
        )
    trainer = _instantiate(trainer_cls, trainer_type, _params(cfg, "trainer"))
    logger.info("[%s] Trainer: %s", federation_id, trainer_cls.__name__)

    client = ModularFlowerClient(
        model             = model,
        train_loader      = train_loader,
        test_loader       = test_loader,
        trainer           = trainer,
        privacy           = privacy,
        compression       = compression,
        device            = device,
        status_store      = status_store,
        metrics_store     = metrics_store,
        checkpoint_store  = checkpoint_store,
        client_name       = client_name,
    )

    logger.info("[%s] Client built — server=%s  device=%s  checkpointing=%s",
                federation_id, server_address, device, checkpoint_store is not None)

    return BuiltClient(
        client           = client,
        server_address   = server_address,
        federation_id    = federation_id,
        checkpoint_store = checkpoint_store,
    )