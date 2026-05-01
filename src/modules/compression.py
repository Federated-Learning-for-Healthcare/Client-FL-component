"""
compression.py — Model update compression modules.

All implementations compress and decompress to/from List[np.ndarray]
so Flower's gRPC transport layer always sees plain float32 arrays.
Compression ratio is tracked internally for metrics/logging.

Three implementations:
  - NoCompression         : pass-through
  - TopKCompression       : zeros all but top-K elements (sparse float32)
  - QuantizationCompression : float32 → float16/int8 → float32
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

from src.core.interfaces import AbstractCompressionModule

logger = logging.getLogger(__name__)


class NoCompression(AbstractCompressionModule):
    """Pass-through — no compression applied."""

    def compress(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights

    def decompress(self, payload: List[np.ndarray]) -> List[np.ndarray]:
        return payload

    def compression_ratio(self, original: List[np.ndarray], compressed: List[np.ndarray]) -> float:
        return 1.0

    def __repr__(self) -> str:
        return "NoCompression()"


class TopKCompression(AbstractCompressionModule):
    """
    Top-K sparsification compression.

    Keeps only the top_k_ratio fraction of elements with the largest
    absolute values. All other elements are zeroed. Returns List[np.ndarray]
    of the same shape — fully Flower gRPC compatible.

    Parameters
    ----------
    top_k_ratio : float
        Fraction of elements to keep, in (0, 1]. E.g. 0.1 = top 10%.
    """

    def __init__(self, top_k_ratio: float = 0.1):
        if not (0.0 < top_k_ratio <= 1.0):
            raise ValueError(f"top_k_ratio must be in (0, 1], got {top_k_ratio}")
        self.top_k_ratio = top_k_ratio
        logger.info("TopKCompression — top_k_ratio=%.3f", top_k_ratio)

    def compress(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        compressed = []
        for w in weights:
            flat = w.flatten()
            k = max(1, int(len(flat) * self.top_k_ratio))
            top_idx = np.argpartition(np.abs(flat), -k)[-k:]
            mask = np.zeros_like(flat)
            mask[top_idx] = 1.0
            compressed.append((flat * mask).reshape(w.shape).astype(np.float32))
        ratio = self.compression_ratio(weights, compressed)
        logger.debug("TopKCompression: ratio=%.2fx (keeping %.1f%%)", ratio, self.top_k_ratio * 100)
        return compressed

    def decompress(self, payload: List[np.ndarray]) -> List[np.ndarray]:
        return [w.astype(np.float32) for w in payload]

    def compression_ratio(self, original: List[np.ndarray], compressed: List[np.ndarray]) -> float:
        total   = sum(w.size for w in original)
        nonzero = sum(np.count_nonzero(w) for w in compressed)
        return total / nonzero if nonzero > 0 else 1.0

    def __repr__(self) -> str:
        return f"TopKCompression(top_k_ratio={self.top_k_ratio})"


class QuantizationCompression(AbstractCompressionModule):
    """
    Precision-reduction compression.

    float16: float32 → float16 → float32  (~2x)
    int8:    float32 → uint8  → float32  (~4x, with linear quantisation)

    Returns List[np.ndarray] of float32 — fully Flower gRPC compatible.

    Parameters
    ----------
    bits : int — 16 or 8
    """

    _SUPPORTED = {16, 8}

    def __init__(self, bits: int = 16):
        if bits not in self._SUPPORTED:
            raise ValueError(f"bits must be 16 or 8, got {bits}")
        self.bits = bits
        logger.info("QuantizationCompression — bits=%d", bits)

    def compress(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        if self.bits == 16:
            compressed = [w.astype(np.float16).astype(np.float32) for w in weights]
        else:
            compressed = [self._quantise_int8(w) for w in weights]
        logger.debug("QuantizationCompression: ratio=%.2fx", self.compression_ratio(weights, compressed))
        return compressed

    def decompress(self, payload: List[np.ndarray]) -> List[np.ndarray]:
        return [w.astype(np.float32) for w in payload]

    def _quantise_int8(self, w: np.ndarray) -> np.ndarray:
        w_min, w_max = float(w.min()), float(w.max())
        scale = (w_max - w_min) / 255.0 if w_max != w_min else 1.0
        q = np.clip(np.round((w - w_min) / scale), 0, 255).astype(np.uint8)
        return (q.astype(np.float32) * scale + w_min)

    def compression_ratio(self, original: List[np.ndarray], compressed: List[np.ndarray]) -> float:
        return 32.0 / self.bits

    def __repr__(self) -> str:
        return f"QuantizationCompression(bits={self.bits})"