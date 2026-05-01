
"""
privacy.py — Privacy modules for the FL client.

Three implementations:
  - NoPrivacy       : pass-through, no privacy applied
  - GaussianPrivacy : weight-level clipping + Gaussian noise (post-training)
  - DPSGDPrivacy    : proper DP-SGD — per-sample gradient clipping + noise
                      injected via PyTorch backward hooks during training.
                      Follows Abadi et al. (2016).
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.core.interfaces import AbstractPrivacyModule

logger = logging.getLogger(__name__)


class NoPrivacy(AbstractPrivacyModule):
    """Pass-through — no privacy applied."""

    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights

    def attach_hooks(self, model: nn.Module) -> None:
        pass

    def remove_hooks(self, model: nn.Module) -> None:
        pass

    def __repr__(self) -> str:
        return "NoPrivacy()"


class GaussianPrivacy(AbstractPrivacyModule):
    """
    Post-training weight sanitisation.

    Clips each weight tensor by its L2 norm then adds calibrated
    Gaussian noise. Simpler than DP-SGD but provides weaker formal
    guarantees. Suitable as a lightweight baseline.

    Parameters
    ----------
    noise_multiplier : float
        Ratio of noise std to clipping norm.
    clipping_norm : float
        L2 norm bound C for clipping.
    """

    def __init__(self, noise_multiplier: float = 0.5, clipping_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        logger.info(
            "GaussianPrivacy — noise_multiplier=%.3f  clipping_norm=%.3f",
            noise_multiplier, clipping_norm,
        )

    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        sanitized = []
        for w in weights:
            l2 = np.linalg.norm(w.flatten())
            scale = min(1.0, self.clipping_norm / (l2 + 1e-6))
            w_clipped = w * scale
            noise = np.random.normal(
                0, self.clipping_norm * self.noise_multiplier, w.shape
            )
            sanitized.append((w_clipped + noise).astype(w.dtype))
        return sanitized

    def attach_hooks(self, model: nn.Module) -> None:
        pass  # weight-level only — no hooks needed

    def remove_hooks(self, model: nn.Module) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"GaussianPrivacy(noise_multiplier={self.noise_multiplier}, "
            f"clipping_norm={self.clipping_norm})"
        )


class DPSGDPrivacy(AbstractPrivacyModule):
    """
    Differentially Private SGD via per-sample gradient clipping + noise.

    Follows Abadi et al. (2016) 'Deep Learning with Differential Privacy':
      1. After each backward pass, clip each parameter gradient by its
         L2 norm (bound = clipping_norm C).
      2. Add Gaussian noise ~ N(0, (sigma * C)^2) to each clipped gradient.

    Privacy accounting (epsilon/delta) is left to the caller.
    Use Opacus for production-grade (ε,δ)-accounting if needed.

    Parameters
    ----------
    noise_multiplier : float
        sigma = noise_std / clipping_norm. Higher = more privacy, more
        accuracy loss. Typical range: 0.3 – 2.0.
    clipping_norm : float
        Per-sample gradient clipping bound C. Typical: 0.1 – 5.0.
    """

    def __init__(self, noise_multiplier: float = 1.0, clipping_norm: float = 1.0):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm
        self._hooks: list = []
        logger.info(
            "DPSGDPrivacy — noise_multiplier=%.3f  clipping_norm=%.3f",
            noise_multiplier, clipping_norm,
        )

    def attach_hooks(self, model: nn.Module) -> None:
        """Register a backward hook on every trainable parameter."""
        self.remove_hooks(model)  # clear stale hooks first
        self._hooks = []
        for param in model.parameters():
            if param.requires_grad:
                handle = param.register_hook(self._make_hook())
                self._hooks.append(handle)
        logger.debug("DPSGDPrivacy: attached %d gradient hooks.", len(self._hooks))

    def remove_hooks(self, model: nn.Module) -> None:
        """Remove all registered backward hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks = []
        logger.debug("DPSGDPrivacy: removed gradient hooks.")

    def _make_hook(self):
        C = self.clipping_norm
        sigma = self.noise_multiplier

        def hook(grad: torch.Tensor) -> torch.Tensor:
            # 1) Clip by L2 norm
            l2 = grad.norm(2)
            clip_coef = torch.clamp(C / (l2 + 1e-6), max=1.0)
            grad_clipped = grad * clip_coef
            # 2) Add Gaussian noise ~ N(0, (sigma * C)^2)
            noise = torch.randn_like(grad_clipped) * (sigma * C)
            return grad_clipped + noise

        return hook

    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """
        DP-SGD applies noise during training via hooks.
        Post-training sanitisation is a pass-through.
        """
        return weights

    def __repr__(self) -> str:
        return (
            f"DPSGDPrivacy(noise_multiplier={self.noise_multiplier}, "
            f"clipping_norm={self.clipping_norm})"
        )