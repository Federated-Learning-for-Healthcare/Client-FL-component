import numpy as np
from typing import List
from src.core.interfaces import AbstractPrivacyModule

class NoPrivacy(AbstractPrivacyModule):
    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        return weights

class GaussianPrivacy(AbstractPrivacyModule):
    def __init__(self, noise_multiplier: float, clipping_norm: float):
        self.noise_multiplier = noise_multiplier
        self.clipping_norm = clipping_norm

    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        sanitized_weights = []
        for w in weights:
            # 1. Calculate L2 Norm
            w_flat = w.flatten()
            l2_norm = np.linalg.norm(w_flat)
            
            # 2. Clipping (Shrink updates that are too big)
            scale = min(1.0, self.clipping_norm / (l2_norm + 1e-6))
            w_clipped = w * scale
            
            # 3. Add Gaussian Noise
            noise_sigma = self.clipping_norm * self.noise_multiplier
            noise = np.random.normal(0, noise_sigma, w.shape)
            
            sanitized_weights.append(w_clipped + noise)
        return sanitized_weights