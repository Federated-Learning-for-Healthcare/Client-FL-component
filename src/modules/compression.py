from typing import List, Any
import numpy as np
from src.core.interfaces import AbstractCompressionModule

class NoCompression(AbstractCompressionModule):
    def compress(self, weights: List[np.ndarray]) -> Any:
        return weights
    
    def decompress(self, payload: Any) -> List[np.ndarray]:
        return payload