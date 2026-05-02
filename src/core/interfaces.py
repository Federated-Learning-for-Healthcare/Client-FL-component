"""
interfaces.py — Abstract base classes for all pluggable FL modules.

Every concrete implementation must subclass the appropriate ABC and
implement all abstract methods. The registry maps config type strings
to these concrete classes.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AbstractDataLoader(ABC):
    """Provides train and test DataLoaders for a given dataset."""

    @abstractmethod
    def load_data(self) -> Tuple[Any, Any]:
        """Return (train_loader, test_loader)."""


class AbstractPrivacyModule(ABC):
    """
    Applies differential privacy to model updates.

    Two-stage interface:
      1. attach_hooks(model) — register backward hooks before training loop
                               used by DP-SGD for gradient-level noise.
      2. sanitize(weights)  — post-training weight-level noise/clipping.
                               Used by GaussianPrivacy. No-op for DP-SGD.
      3. remove_hooks(model) — clean up hooks after training.
    """

    @abstractmethod
    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Apply noise/clipping to extracted weight arrays."""

    @abstractmethod
    def attach_hooks(self, model: nn.Module) -> None:
        """Register backward hooks on model parameters (for DP-SGD)."""

    @abstractmethod
    def remove_hooks(self, model: nn.Module) -> None:
        """Remove previously registered backward hooks."""


class AbstractCompressionModule(ABC):
    """Compresses model updates before transmission and decompresses on receipt."""

    @abstractmethod
    def compress(self, weights: List[np.ndarray]) -> Any:
        """Compress weight arrays into a transmittable payload."""

    @abstractmethod
    def decompress(self, payload: Any) -> List[np.ndarray]:
        """Reconstruct weight arrays from a compressed payload."""

    @abstractmethod
    def compression_ratio(self, original: List[np.ndarray], compressed: Any) -> float:
        """Return compression ratio (original_bytes / compressed_bytes)."""


class AbstractTrainer(ABC):
    """Handles local model training and evaluation."""

    @abstractmethod
    def train(
        self,
        model: nn.Module,
        train_loader: Any,
        epochs: int,
        device: str,
        privacy: Optional[AbstractPrivacyModule] = None,
    ) -> Dict[str, float]:
        """
        Train model for given epochs.

        If privacy is a DP-SGD module, hooks are attached before training
        and removed after. Weight-level modules (GaussianPrivacy) are
        handled by the FL client after this method returns.

        Returns dict with at least: train_loss, accuracy.
        """

    @abstractmethod
    def evaluate(
        self,
        model: nn.Module,
        test_loader: Any,
        device: str,
    ) -> Tuple[float, float]:
        """
        Evaluate model on test_loader.
        Returns (loss, accuracy).
        """