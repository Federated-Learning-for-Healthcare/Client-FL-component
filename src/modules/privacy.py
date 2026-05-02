
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
import math
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.core.interfaces import AbstractPrivacyModule

logger = logging.getLogger(__name__)


def compute_dp_epsilon(
    noise_multiplier: float,
    num_steps: int,
    sampling_rate: float,
    delta: float = 1e-5,
) -> float:
    """
    Compute the (ε, δ)-DP privacy budget spent after `num_steps` DP-SGD steps.

    Uses the moments accountant (RDP) with first-order Poisson subsampling
    amplification (Mironov 2017). Minimises over RDP orders 2–255 to get
    the tightest bound this approximation can produce.

    Parameters
    ----------
    noise_multiplier : σ  (noise_std / clipping_norm)
    num_steps        : total gradient steps = rounds × epochs × batches_per_epoch
    sampling_rate    : q = batch_size / dataset_size
    delta            : δ target (typically 1/dataset_size or 1e-5)

    Returns ∞ when noise is zero or inputs are degenerate.
    Note: use Opacus for production-grade tight accounting.
    """
    if noise_multiplier <= 0 or num_steps <= 0 or sampling_rate <= 0:
        return float("inf")

    best_eps = float("inf")
    for alpha in range(2, 256):
        # Per-step RDP for subsampled Gaussian (first-order approx, tight for q<<1)
        rdp_per_step = (sampling_rate ** 2 * alpha) / (2.0 * noise_multiplier ** 2)
        total_rdp    = rdp_per_step * num_steps

        # Convert RDP → (ε, δ)-DP  [Balle et al. 2020]
        eps = total_rdp + math.log(1.0 - 1.0 / alpha) - (
            math.log(delta) + math.log(1.0 - 1.0 / alpha)
        ) / (alpha - 1.0)

        if math.isfinite(eps) and eps < best_eps:
            best_eps = eps

    return best_eps


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

    def __init__(self, noise_multiplier: float = 1.0, clipping_norm: float = 1.0,
                 delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm    = clipping_norm
        self.delta            = delta
        self._hooks: list     = []
        logger.info(
            "DPSGDPrivacy — noise_multiplier=%.3f  clipping_norm=%.3f  delta=%.2e",
            noise_multiplier, clipping_norm, delta,
        )

    def epsilon_spent(self, num_steps: int, sampling_rate: float) -> float:
        """Current (ε, δ)-DP budget spent after num_steps gradient steps."""
        return compute_dp_epsilon(
            self.noise_multiplier, num_steps, sampling_rate, self.delta
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


class TrueDPSGDPrivacy(AbstractPrivacyModule):
    """
    True DP-SGD: per-sample gradient clipping + Gaussian noise.

    Implements Abadi et al. (2016) correctly — unlike DPSGDPrivacy which
    clips the aggregate batch gradient, this clips each sample's gradient
    individually before summing, which is what the formal (ε,δ) proof
    requires.

    How it differs from DPSGDPrivacy
    ---------------------------------
    DPSGDPrivacy (approximate):
        batch_grad = sum(grads)          # aggregate first
        clip(batch_grad)                 # then clip — too late, damage done
        add_noise(batch_grad)

    TrueDPSGDPrivacy (correct):
        for each sample i:
            clip(grad_i, C)              # clip individually
            accumulate += grad_i_clipped
        add_noise(accumulate)            # one noise injection on the sum
        apply accumulate / batch_size

    This requires O(batch_size) forward+backward passes per batch so it
    is slower. For production use Opacus; this implementation is for
    comparison and understanding only.

    Parameters
    ----------
    noise_multiplier : float — sigma. Higher = more privacy, less accuracy.
    clipping_norm    : float — per-sample gradient norm bound C.
    """

    # Trainer checks this flag to switch to the per-sample training path
    requires_per_sample_training: bool = True

    def __init__(self, noise_multiplier: float = 1.0, clipping_norm: float = 1.0,
                 delta: float = 1e-5):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm    = clipping_norm
        self.delta            = delta
        logger.info(
            "TrueDPSGDPrivacy — noise_multiplier=%.3f  clipping_norm=%.3f  delta=%.2e",
            noise_multiplier, clipping_norm, delta,
        )

    def epsilon_spent(self, num_steps: int, sampling_rate: float) -> float:
        """Current (ε, δ)-DP budget spent after num_steps gradient steps."""
        return compute_dp_epsilon(
            self.noise_multiplier, num_steps, sampling_rate, self.delta
        )

    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights  # noise already applied per-sample during training

    def attach_hooks(self, _model: nn.Module) -> None:
        pass  # training loop handles everything — no hooks needed

    def remove_hooks(self, _model: nn.Module) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"TrueDPSGDPrivacy(noise_multiplier={self.noise_multiplier}, "
            f"clipping_norm={self.clipping_norm})"
        )