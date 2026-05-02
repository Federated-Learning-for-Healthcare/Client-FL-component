"""
training.py — Local training and evaluation for the FL client.

Fix: train_loss now reports the final epoch's loss, not the average
across all epochs. This matches what paper plots should show — the
loss after training completes, not a blended average.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.core.interfaces import AbstractPrivacyModule, AbstractTrainer

logger = logging.getLogger(__name__)


class StandardPyTorchTrainer(AbstractTrainer):

    def __init__(
        self,
        optimizer: str   = "sgd",
        lr:        float = 0.01,
        momentum:  float = 0.9,
    ):
        self.optimizer_name = optimizer.lower()
        self.lr       = lr
        self.momentum = momentum

    def _true_dpsgd_step(
        self,
        model:     nn.Module,
        inputs:    torch.Tensor,
        labels:    torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        C:         float,
        sigma:     float,
    ) -> Tuple[float, torch.Tensor]:
        """
        One true DP-SGD batch step.

        For each sample: compute gradient, clip by global L2 norm C.
        Sum clipped gradients, add Gaussian noise, divide by batch size,
        apply via optimizer.

        Returns (batch_loss, outputs) where outputs is a no_grad forward
        pass used for accuracy tracking.
        """
        trainable   = [p for p in model.parameters() if p.requires_grad]
        accumulated = [torch.zeros_like(p) for p in trainable]

        for i in range(inputs.shape[0]):
            optimizer.zero_grad()
            loss_i = criterion(model(inputs[i:i+1]), labels[i:i+1])
            loss_i.backward()

            # Per-sample global gradient norm across all parameters
            sample_norm = torch.sqrt(sum(
                p.grad.norm(2) ** 2
                for p in trainable if p.grad is not None
            ))
            clip = torch.clamp(
                torch.tensor(C, dtype=sample_norm.dtype, device=sample_norm.device)
                / (sample_norm + 1e-6),
                max=1.0,
            )
            for j, p in enumerate(trainable):
                if p.grad is not None:
                    accumulated[j] += p.grad.detach() * clip

        # Inject noise into the sum, normalise by batch size, apply
        optimizer.zero_grad()
        batch_size = inputs.shape[0]
        for j, p in enumerate(trainable):
            noise  = torch.randn_like(accumulated[j]) * (sigma * C)
            p.grad = (accumulated[j] + noise) / batch_size
        optimizer.step()

        with torch.no_grad():
            outputs    = model(inputs)
            batch_loss = criterion(outputs, labels).item()
        return batch_loss, outputs

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self.optimizer_name == "sgd":
            return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)
        if self.optimizer_name == "adam":
            return torch.optim.Adam(model.parameters(), lr=self.lr)
        raise ValueError(f"Unsupported optimizer '{self.optimizer_name}'. Choose 'sgd' or 'adam'.")

    def train(
        self,
        model:        nn.Module,
        train_loader: Any,
        epochs:       int,
        device:       str,
        privacy:      Optional[AbstractPrivacyModule] = None,
    ) -> Dict[str, float]:
        criterion = nn.CrossEntropyLoss()
        optimizer = self._build_optimizer(model)
        model.train()

        if privacy is not None:
            privacy.attach_hooks(model)
            logger.debug("Trainer: privacy hooks attached — %s", repr(privacy))

        final_epoch_loss = 0.0  # track only the last epoch
        correct = 0
        total   = 0

        try:
            for epoch in range(epochs):
                epoch_loss = 0.0
                epoch_correct = 0
                epoch_total   = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    if getattr(privacy, "requires_per_sample_training", False):
                        # True DP-SGD: per-sample clip → sum → noise → step
                        batch_loss, outputs = self._true_dpsgd_step(
                            model, inputs, labels, criterion, optimizer,
                            privacy.clipping_norm, privacy.noise_multiplier,
                        )
                    else:
                        # Standard path (hooks already attached for approx DPSGD)
                        optimizer.zero_grad()
                        outputs    = model(inputs)
                        loss       = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        batch_loss = loss.item()

                    epoch_loss    += batch_loss
                    _, predicted   = torch.max(outputs.data, 1)
                    epoch_total   += labels.size(0)
                    epoch_correct += (predicted == labels).sum().item()

                # Update final epoch metrics — we report the last epoch only
                final_epoch_loss = epoch_loss / max(len(train_loader), 1)
                correct = epoch_correct
                total   = epoch_total

                logger.debug(
                    "Epoch %d/%d — loss: %.4f  acc: %.4f",
                    epoch + 1, epochs,
                    final_epoch_loss,
                    epoch_correct / epoch_total if epoch_total > 0 else 0.0,
                )

        finally:
            if privacy is not None:
                privacy.remove_hooks(model)
                logger.debug("Trainer: privacy hooks removed.")

        accuracy = correct / total if total > 0 else 0.0
        return {"train_loss": final_epoch_loss, "accuracy": accuracy}

    def evaluate(
        self,
        model:       nn.Module,
        test_loader: Any,
        device:      str,
    ) -> Tuple[float, float]:
        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        correct    = 0
        total      = 0
        model.eval()

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs     = model(inputs)
                total_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total      += labels.size(0)
                correct    += (predicted == labels).sum().item()

        avg_loss = total_loss / max(len(test_loader), 1)
        accuracy = correct / total if total > 0 else 0.0
        return avg_loss, accuracy