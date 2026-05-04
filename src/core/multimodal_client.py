"""
multimodal_client.py — MultiModalFlowerClient

A Flower NumPyClient that is modality-aware:
  - get_parameters() returns only owned-modality weights (not the full state dict).
  - fit() loads only owned weights, trains, then sends back only owned weights.
  - evaluate() caches eval metrics so fit() can write one combined CSV row.

Shared components (modal_attn, fusion_head, null tokens) are always included in
the parameter exchange — every client contributes to these.

Integrates with the existing observability stack:
  StatusStore    — live status.json updated after each round
  MetricsStore   — one CSV row per round (train + eval combined)
  CheckpointStore— versioned model state dicts (full model, not just owned params)

Privacy and compression are applied via the existing module interfaces,
identical to ModularFlowerClient — the compressed payload is a list of numpy
arrays covering only the owned modality components.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from src.core.interfaces import AbstractCompressionModule, AbstractPrivacyModule, AbstractTrainer
from src.models.multimodal import (
    MultiModalModel,
    get_multimodal_parameters,
    set_multimodal_parameters,
)
from src.observerbility.checkpoint_store import CheckpointStore
from src.observerbility.metrics_store import MetricsStore, compute_update_size_bytes
from src.observerbility.status_store import StatusStore

logger = logging.getLogger(__name__)


class MultiModalFlowerClient(fl.client.NumPyClient):
    """
    Flower client for MultiModalModel with modality-selective parameter exchange.

    Args:
        model            : MultiModalModel instance (all encoders instantiated)
        modal_mask       : [has_ecg, has_mri, has_ehr] — owned modalities
        train_loader     : dict-batch DataLoader (from MultiModalDataLoader)
        test_loader      : dict-batch DataLoader
        trainer          : MultiModalTrainer instance
        privacy          : privacy module (NoPrivacy / GaussianPrivacy / DPSGDPrivacy)
        compression      : compression module (NoCompression / TopK / Quantize)
        device           : "cpu" or "cuda"
        status_store     : optional StatusStore for live status.json
        metrics_store    : optional MetricsStore for CSV logging
        checkpoint_store : optional CheckpointStore for model versioning
        client_name      : human-readable identifier for logs
    """

    def __init__(
        self,
        model:            MultiModalModel,
        modal_mask:       List[int],
        train_loader:     Any,
        test_loader:      Any,
        trainer:          AbstractTrainer,
        privacy:          Optional[AbstractPrivacyModule],
        compression:      AbstractCompressionModule,
        device:           str                    = "cpu",
        status_store:     Optional[StatusStore]  = None,
        metrics_store:    Optional[MetricsStore] = None,
        checkpoint_store: Optional[CheckpointStore] = None,
        client_name:      str                    = "unknown_multimodal_client",
    ):
        self.model            = model.to(device)
        self.modal_mask       = modal_mask
        self.train_loader     = train_loader
        self.test_loader      = test_loader
        self.trainer          = trainer
        self.privacy          = privacy
        self.compression      = compression
        self.device           = device
        self.status           = status_store
        self.metrics          = metrics_store
        self.checkpoint_store = checkpoint_store
        self.client_name      = client_name
        self.client_round     = 0

        # Cached eval metrics — fit() writes them into the same CSV row
        self._last_eval_loss:         Optional[float] = None
        self._last_eval_acc:          Optional[float] = None
        self._last_eval_global_round: Optional[int]   = None

        logger.info(
            "MultiModalFlowerClient — client=%s  mask=%s  privacy=%s  "
            "compression=%s  checkpointing=%s",
            client_name, modal_mask, repr(privacy), repr(compression),
            checkpoint_store is not None,
        )

    # ── Flower interface ──────────────────────────────────────────────────────

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return get_multimodal_parameters(self.model, self.modal_mask)

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[Any, int, Dict]:
        self.client_round += 1
        global_round = config.get("round", None)
        epochs       = int(config.get("local_epochs", 1))

        if self.metrics:
            self.metrics.mark_round_start()

        logger.info(
            "fit() — client_round=%d  global_round=%s  epochs=%d  mask=%s",
            self.client_round, global_round, epochs, self.modal_mask,
        )

        # 1) Load global model (owned modalities only)
        raw_params = self.compression.decompress(parameters)
        set_multimodal_parameters(self.model, self.modal_mask, raw_params)

        # 2) Train locally
        train_metrics = self.trainer.train(
            model        = self.model,
            train_loader = self.train_loader,
            epochs       = epochs,
            device       = self.device,
            privacy      = self.privacy,
        ) or {}

        # 3) Extract updated weights (owned only) → privacy sanitise
        updated    = get_multimodal_parameters(self.model, self.modal_mask)
        safe       = (
            self.privacy.sanitize(updated) if self.privacy is not None else updated
        )

        # 4) Compress for transmission
        transport    = self.compression.compress(safe)
        ratio        = self.compression.compression_ratio(safe, transport)
        update_bytes = compute_update_size_bytes(transport)

        train_loss = float(train_metrics.get("train_loss", 0.0))
        train_acc  = float(train_metrics.get("accuracy",   0.0))

        # Privacy budget tracking (DP-SGD only)
        privacy_epsilon: Optional[float] = None
        if hasattr(self.privacy, "epsilon_spent"):
            try:
                dataset_size  = len(self.train_loader.dataset)
                batch_size    = self.train_loader.batch_size or 1
                sampling_rate = batch_size / dataset_size
                total_steps   = self.client_round * epochs * len(self.train_loader)
                privacy_epsilon = self.privacy.epsilon_spent(total_steps, sampling_rate)
                logger.info("fit() — ε=%.4f  steps=%d", privacy_epsilon, total_steps)
            except Exception as exc:
                logger.warning("Could not compute epsilon: %s", exc)

        logger.info(
            "fit() — loss=%.4f  acc=%.4f  ratio=%.2fx  bytes=%d",
            train_loss, train_acc, ratio, update_bytes,
        )

        # 5) Checkpoint — save full model state (including unowned encoders)
        if self.checkpoint_store is not None and global_round is not None:
            try:
                self.checkpoint_store.save(self.model, int(global_round))
            except Exception as exc:
                logger.warning("Checkpoint save failed (round %s): %s", global_round, exc)

        # 6) One CSV row combining train + cached eval metrics
        if self.metrics:
            self.metrics.record_round(
                global_round      = global_round,
                client_round      = self.client_round,
                train_loss        = train_loss,
                train_accuracy    = train_acc,
                eval_loss         = self._last_eval_loss,
                eval_accuracy     = self._last_eval_acc,
                update_size_bytes = update_bytes,
                compression_ratio = float(ratio),
                privacy_epsilon   = privacy_epsilon,
            )

        # 7) Status store
        if self.status:
            self.status.update(
                state             = "TRAINING",
                global_round      = global_round,
                client_round      = self.client_round,
                train_loss        = train_loss,
                train_accuracy    = train_acc,
                compression_ratio = float(ratio),
                update_size_bytes = update_bytes,
                privacy_epsilon   = privacy_epsilon,
                message           = f"Round {global_round} complete",
            )

        train_metrics["client_name"]       = self.client_name
        train_metrics["compression_ratio"] = float(ratio)
        train_metrics["update_size_bytes"] = update_bytes
        train_metrics["modal_mask"]        = str(self.modal_mask)

        return transport, len(self.train_loader.dataset), train_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        raw_params = self.compression.decompress(parameters)
        set_multimodal_parameters(self.model, self.modal_mask, raw_params)

        loss, accuracy = self.trainer.evaluate(self.model, self.test_loader, self.device)
        global_round   = config.get("round", None)

        logger.info(
            "evaluate() — round=%s  loss=%.4f  accuracy=%.4f",
            global_round, loss, accuracy,
        )

        # Cache — fit() includes these in the same CSV row next round
        self._last_eval_loss         = float(loss)
        self._last_eval_acc          = float(accuracy)
        self._last_eval_global_round = global_round

        if self.status:
            self.status.update(
                eval_loss     = float(loss),
                eval_accuracy = float(accuracy),
                message       = "Evaluation complete",
            )

        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}
