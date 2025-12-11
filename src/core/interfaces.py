from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
import numpy as np
import torch

class AbstractDataLoader(ABC):
    @abstractmethod
    def load_data(self) -> Tuple[Any, Any]:
        """Returns (train_loader, test_loader)."""
        pass

class AbstractPrivacyModule(ABC):
    @abstractmethod
    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        """Apply noise/clipping to weights."""
        pass

class AbstractCompressionModule(ABC):
    @abstractmethod
    def compress(self, weights: List[np.ndarray]) -> Any:
        """Compress before sending."""
        pass
    
    @abstractmethod
    def decompress(self, payload: Any) -> List[np.ndarray]:
        """Decompress after receiving."""
        pass

class AbstractTrainer(ABC):
    @abstractmethod
    def train(self, model: torch.nn.Module, train_loader: Any, epochs: int, device: str) -> Dict[str, float]:
        """Run training loop. Return metrics."""
        pass

    @abstractmethod
    def evaluate(self, model: torch.nn.Module, test_loader: Any, device: str) -> Tuple[float, float]:
        """Run evaluation. Return (loss, accuracy)."""
        pass