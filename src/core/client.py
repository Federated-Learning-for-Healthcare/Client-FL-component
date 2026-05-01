"""
client.py — ModularFlowerClient

Fix: eval metrics are now cached and written into the same CSV row
as train metrics, not a separate disconnected row. Each FL round
produces exactly one CSV row containing both train and eval data.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import flwr as fl
import numpy as np
import torch

from src.core.interfaces import (
    AbstractCompressionModule,
    AbstractPrivacyModule,
    AbstractTrainer,
)
from src.observerbility.checkpoint_store import CheckpointStore
from src.observerbility.metrics_store import MetricsStore, compute_update_size_bytes
from src.observerbility.status_store import StatusStore

logger = logging.getLogger(__name__)


class ModularFlowerClient(fl.client.NumPyClient):

    def __init__(
        self,
        model:             torch.nn.Module,
        train_loader:      Any,
        test_loader:       Any,
        trainer:           AbstractTrainer,
        privacy:           Optional[AbstractPrivacyModule],
        compression:       AbstractCompressionModule,
        device:            str = "cpu",
        status_store:      Optional[StatusStore] = None,
        metrics_store:     Optional[MetricsStore] = None,
        checkpoint_store:  Optional[CheckpointStore] = None,
        client_name:       str = "unknown_client",
    ):
        self.client_round     = 0
        self.model            = model
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
        self.model.to(self.device)

        # Cache latest eval metrics so fit() can write them in the same row
        # evaluate() is called by the server between rounds; fit() is called
        # after, so the cached values correspond to the same global round.
        self._last_eval_loss:        Optional[float] = None
        self._last_eval_acc:         Optional[float] = None
        self._last_eval_global_round: Optional[int]  = None

        logger.info(
            "ModularFlowerClient — client=%s  privacy=%s  compression=%s  "
            "checkpointing=%s",
            client_name, repr(privacy), repr(compression),
            checkpoint_store is not None,
        )

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[Any, int, Dict]:
        self.client_round += 1
        global_round = config.get("round", None)
        epochs       = int(config.get("local_epochs", 1))

        if self.metrics:
            self.metrics.mark_round_start()

        logger.info(
            "fit() — client_round=%d  global_round=%s  epochs=%d",
            self.client_round, global_round, epochs,
        )

        # 1) Load global model
        self._set_parameters(self.compression.decompress(parameters))

        # 2) Train
        train_metrics = self.trainer.train(
            model        = self.model,
            train_loader = self.train_loader,
            epochs       = epochs,
            device       = self.device,
            privacy      = self.privacy,
        ) or {}

        # 3) Post-training sanitisation
        updated_weights = self.get_parameters(config={})
        safe_weights = (
            self.privacy.sanitize(updated_weights)
            if self.privacy is not None else updated_weights
        )

        # 4) Compress
        transport_weights = self.compression.compress(safe_weights)
        ratio             = self.compression.compression_ratio(
            safe_weights, transport_weights
        )
        update_bytes = compute_update_size_bytes(transport_weights)

        train_loss = float(train_metrics.get("train_loss", 0.0))
        train_acc  = float(train_metrics.get("accuracy",   0.0))

        logger.info(
            "fit() — loss=%.4f  acc=%.4f  ratio=%.2fx  bytes=%d",
            train_loss, train_acc, ratio, update_bytes,
        )

        # 5) Save checkpoint
        if self.checkpoint_store is not None and global_round is not None:
            try:
                self.checkpoint_store.save(self.model, int(global_round))
            except Exception as e:
                logger.warning("Checkpoint save failed (round %s): %s", global_round, e)

        # 6) Write ONE CSV row combining train metrics + latest cached eval metrics.
        #    This fixes the two-row-per-round problem: previously fit() wrote a
        #    train-only row and evaluate() wrote a separate eval-only row.
        #    Now fit() writes the complete row including the eval from the
        #    previous evaluate() call (same global round).
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
            )

        # 7) Update status store
        if self.status:
            self.status.update(
                state             = "TRAINING",
                global_round      = global_round,
                client_round      = self.client_round,
                train_loss        = train_loss,
                train_accuracy    = train_acc,
                compression_ratio = float(ratio),
                update_size_bytes = update_bytes,
                message           = f"Round {global_round} complete",
            )

        train_metrics["client_name"]       = self.client_name
        train_metrics["compression_ratio"] = float(ratio)
        train_metrics["update_size_bytes"] = update_bytes

        return transport_weights, len(self.train_loader.dataset), train_metrics

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict
    ) -> Tuple[float, int, Dict]:
        self._set_parameters(self.compression.decompress(parameters))
        loss, accuracy = self.trainer.evaluate(
            self.model, self.test_loader, self.device
        )
        global_round = config.get("round", None)
        logger.info(
            "evaluate() — round=%s  loss=%.4f  accuracy=%.4f",
            global_round, loss, accuracy,
        )

        # Cache eval metrics — fit() will include them in the next CSV row.
        # Do NOT write a separate row here — that was causing two rows per round.
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.tensor(v).to(self.device) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)