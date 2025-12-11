import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from src.core.interfaces import AbstractTrainer, AbstractPrivacyModule, AbstractCompressionModule

# 1. Standard Privacy (Pass-through)
class NoPrivacy(AbstractPrivacyModule):
    def sanitize(self, weights: List[np.ndarray]) -> List[np.ndarray]:
        # Logic: Do nothing. Just return weights as is.
        return weights

# 2. Standard Compression (Pass-through)
class NoCompression(AbstractCompressionModule):
    def compress(self, weights: List[np.ndarray]) -> Any:
        return weights # No compression
    
    def decompress(self, payload: Any) -> List[np.ndarray]:
        return payload # No decompression

# 3. Standard PyTorch Training Loop
class StandardPyTorchTrainer(AbstractTrainer):
    def train(self, model, train_loader, epochs, device) -> Dict[str, float]:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        
        epoch_loss = 0.0
        for _ in range(epochs):
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
        return {"train_loss": epoch_loss / len(train_loader)}

    def evaluate(self, model, test_loader, device) -> Tuple[float, float]:
        criterion = torch.nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        model.eval()
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = correct / total
        return loss / len(test_loader), accuracy