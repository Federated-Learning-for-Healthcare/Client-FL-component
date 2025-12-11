import flwr as fl
import torch
from collections import OrderedDict
from typing import Dict, List, Tuple
from .interfaces import AbstractTrainer, AbstractPrivacyModule, AbstractCompressionModule

class ModularFlowerClient(fl.client.NumPyClient):
    def __init__(self, 
                 model: torch.nn.Module, 
                 train_loader, 
                 test_loader, 
                 trainer: AbstractTrainer,
                 privacy: AbstractPrivacyModule, 
                 compression: AbstractCompressionModule,
                 device: str = "cpu"):
        
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = trainer
        self.privacy = privacy
        self.compression = compression
        self.device = device
        self.model.to(self.device)

    def get_parameters(self, config):
        # Extract weights from PyTorch to NumPy
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # 1. Decompress & Load Global Weights
        decompressed_params = self.compression.decompress(parameters)
        self._set_parameters(decompressed_params)
        
        # 2. Train (Delegate to Trainer Module)
        # We get 'local_epochs' from server config (default=1)
        epochs = int(config.get("local_epochs", 1))
        metrics = self.trainer.train(self.model, self.train_loader, epochs=epochs, device=self.device)
        
        # 3. Get Updated Weights
        updated_weights = self.get_parameters(config={})

        # 4. Apply Privacy (Delegate to Privacy Module)
        safe_weights = self.privacy.sanitize(updated_weights)

        # 5. Compress (Delegate to Compression Module)
        final_payload = self.compression.compress(safe_weights)

        # Return: Payload, Dataset Size, Metrics
        return final_payload, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        self._set_parameters(parameters)
        loss, accuracy = self.trainer.evaluate(self.model, self.test_loader, self.device)
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def _set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)