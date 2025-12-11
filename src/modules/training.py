import torch
from typing import Dict, Tuple
from src.core.interfaces import AbstractTrainer

class StandardPyTorchTrainer(AbstractTrainer):
    def train(self, model, train_loader, epochs, device) -> Dict[str, float]:
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        model.train()
        
        # Initialize metrics
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for _ in range(epochs):
            # Reset loss for each epoch (optional, but cleaner)
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                # Track Metrics
                epoch_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        # Return BOTH loss and accuracy
        return {
            "train_loss": epoch_loss / len(train_loader),
            "accuracy": correct / total  # <--- This fixes the KeyError!
        }

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
        
        accuracy = correct / total if total > 0 else 0.0
        return loss / len(test_loader), accuracy