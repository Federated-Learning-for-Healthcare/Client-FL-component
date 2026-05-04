"""
multimodal_trainer.py — Local trainer for MultiModalModel.

Differs from StandardPyTorchTrainer in three ways:
  1. Iterates over dict batches {modality: tensor, ..., "label": tensor}
     instead of (inputs, labels) tuples.
  2. Applies modality dropout during training (randomly drops one modality
     per forward pass, forcing the model to be robust to missing modalities).
  3. Optimises only the parameters owned by this client's modal_mask,
     leaving unowned encoder weights frozen.

Privacy integration:
  Weight-level privacy (GaussianPrivacy) — applied post-training by the client,
  not by this trainer. The trainer just calls attach_hooks / remove_hooks for
  hook-based DP-SGD variants (consistent with StandardPyTorchTrainer).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.core.interfaces import AbstractPrivacyModule, AbstractTrainer
from src.models.multimodal import MODALITIES, MultiModalModel, _encoder_for, _null_token_for

logger = logging.getLogger(__name__)


class MultiModalTrainer(AbstractTrainer):
    """
    Local trainer for MultiModalModel with dict-batch DataLoaders.

    Args:
        optimizer    : "sgd" or "adam"
        lr           : learning rate
        momentum     : SGD momentum (ignored for Adam)
        modal_drop_p : per-modality dropout probability during training.
                       0.0 for uni-modal clients, 0.3 for multi-modal clients.
    """

    def __init__(
        self,
        optimizer:    str   = "sgd",
        lr:           float = 0.01,
        momentum:     float = 0.9,
        modal_drop_p: float = 0.0,
    ):
        self.optimizer_name = optimizer.lower()
        self.lr           = lr
        self.momentum     = momentum
        self.modal_drop_p = modal_drop_p

    # ── AbstractTrainer interface ─────────────────────────────────────────────

    def train(
        self,
        model:        nn.Module,
        train_loader: Any,
        epochs:       int,
        device:       str,
        privacy:      Optional[AbstractPrivacyModule] = None,
    ) -> Dict[str, float]:
        """
        Train MultiModalModel for `epochs` local rounds.

        model must be a MultiModalModel instance.
        train_loader must yield dict batches with a "label" key.
        """
        if not isinstance(model, MultiModalModel):
            raise TypeError(
                f"MultiModalTrainer requires a MultiModalModel, got {type(model).__name__}"
            )

        owned_params = self._get_owned_params(model)
        optimizer    = self._build_optimizer(owned_params)
        criterion    = nn.CrossEntropyLoss()
        model.train()

        if privacy is not None:
            privacy.attach_hooks(model)
            logger.debug("MultiModalTrainer: privacy hooks attached — %s", repr(privacy))

        final_loss = 0.0
        correct    = 0
        total      = 0

        try:
            for epoch in range(epochs):
                epoch_loss    = 0.0
                epoch_correct = 0
                epoch_total   = 0

                for batch in train_loader:
                    batch  = {
                        k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    labels = batch["label"]

                    optimizer.zero_grad()
                    logits = model(
                        batch,
                        modal_mask   = model.modal_mask,
                        modal_drop_p = self.modal_drop_p,
                    )
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss    += loss.item()
                    _, predicted   = torch.max(logits.data, 1)
                    epoch_total   += labels.size(0)
                    epoch_correct += (predicted == labels).sum().item()

                final_loss = epoch_loss / max(len(train_loader), 1)
                correct    = epoch_correct
                total      = epoch_total

                logger.debug(
                    "Epoch %d/%d — loss: %.4f  acc: %.4f",
                    epoch + 1, epochs,
                    final_loss,
                    epoch_correct / epoch_total if epoch_total > 0 else 0.0,
                )

        finally:
            if privacy is not None:
                privacy.remove_hooks(model)
                logger.debug("MultiModalTrainer: privacy hooks removed.")

        accuracy = correct / total if total > 0 else 0.0
        return {"train_loss": final_loss, "accuracy": accuracy}

    def evaluate(
        self,
        model:       nn.Module,
        test_loader: Any,
        device:      str,
    ) -> Tuple[float, float]:
        """
        Evaluate MultiModalModel on test_loader.
        Returns (avg_loss, accuracy).
        """
        if not isinstance(model, MultiModalModel):
            raise TypeError(
                f"MultiModalTrainer requires a MultiModalModel, got {type(model).__name__}"
            )

        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct    = 0
        total      = 0
        model.eval()

        with torch.no_grad():
            for batch in test_loader:
                batch  = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                labels = batch["label"]
                logits = model(batch, modal_mask=model.modal_mask, modal_drop_p=0.0)
                total_loss += criterion(logits, labels).item()
                _, predicted = torch.max(logits.data, 1)
                total      += labels.size(0)
                correct    += (predicted == labels).sum().item()

        avg_loss = total_loss / max(len(test_loader), 1)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_owned_params(self, model: MultiModalModel) -> List[torch.nn.Parameter]:
        """
        Return only the parameters this client owns.
        Unowned encoder weights are excluded from the optimizer so they stay frozen.
        """
        params: List[torch.nn.Parameter] = []
        for mod, owned in zip(MODALITIES, model.modal_mask):
            if owned:
                enc = _encoder_for(model, mod)
                if enc is not None:
                    params.extend(enc.parameters())
                params.append(_null_token_for(model, mod))
        params.extend(model.modal_attn.parameters())
        params.extend(model.fusion_head.parameters())
        return params

    def _build_optimizer(
        self, params: List[torch.nn.Parameter]
    ) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=self.lr, momentum=self.momentum)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        raise ValueError(
            f"Unsupported optimizer '{self.optimizer_name}'. Choose 'sgd' or 'adam'."
        )
