import flwr as fl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

# Framework Components
from src.core.client import ModularFlowerClient
from src.core.interfaces import AbstractDataLoader
from src.modules.training import StandardPyTorchTrainer
from src.modules.privacy import GaussianPrivacy
from src.modules.compression import NoCompression

# --- 1. Define the Model (Simple Neural Network for Digits) ---
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784 pixels
        # Output is 10 classes (digits 0-9)
        self.layer1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.layer1(x)
        x = self.relu(x)
        return self.layer2(x)

# --- 2. Define the Real Data Loader (MNIST) ---
class MNISTDataLoader(AbstractDataLoader):
    def load_data(self):
        print("📥 Downloading/Loading MNIST Data...")
        
        # Transformation: Convert image to Tensor and Normalize
        transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

        # Download training data
        train_data = MNIST(root='./data', train=True, download=True, transform=transform)
        # Download test data
        test_data = MNIST(root='./data', train=False, download=True, transform=transform)

        return DataLoader(train_data, batch_size=32, shuffle=True), DataLoader(test_data, batch_size=32)

# --- Main Execution ---
def main():
    # Setup Model and Data
    model = SimpleMLP()
    loader = MNISTDataLoader()
    train_loader, test_loader = loader.load_data()

    # Initialize Modular Client
    modular_client = ModularFlowerClient(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        trainer=StandardPyTorchTrainer(),
        
        # Privacy is ON (Standard Gaussian)
        privacy=GaussianPrivacy(noise_multiplier=0.5, clipping_norm=1.0),
        
        compression=NoCompression(),
        device="cpu"
    )

    print("🏥 Hospital Client Connecting to Server...")
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=modular_client.to_client()
    )

if __name__ == "__main__":
    main()