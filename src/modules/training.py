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
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss    = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    epoch_loss    += loss.item()
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