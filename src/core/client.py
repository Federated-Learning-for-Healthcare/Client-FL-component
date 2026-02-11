# import flwr as fl
# import torch
# from collections import OrderedDict
# from typing import Dict, List, Tuple
# from .interfaces import AbstractTrainer, AbstractPrivacyModule, AbstractCompressionModule
# from src.observerbility.status_store import StatusStore

# class ModularFlowerClient(fl.client.NumPyClient):
#     def __init__(self, 
#                  model: torch.nn.Module, 
#                  train_loader, 
#                  test_loader, 
#                  trainer: AbstractTrainer,
#                  privacy: AbstractPrivacyModule, 
#                  compression: AbstractCompressionModule,
#                  device: str = "cpu"
#                  ):
#         self.client_round = 0 #counting rounds for UI
#         self.model = model
#         self.train_loader = train_loader
#         self.test_loader = test_loader
#         self.trainer = trainer
#         self.privacy = privacy
#         self.compression = compression
#         self.device = device
#         self.model.to(self.device)
 

#     def get_parameters(self, config):
#         # Extract weights from PyTorch to NumPy
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def fit(self, parameters, config):
#         self.client_round += 1
#         global_round = config.get("round", None)
#         # 1. Decompress & Load Global Weights
#         decompressed_params = self.compression.decompress(parameters)
#         self._set_parameters(decompressed_params)
        
#         # 2. Train (Delegate to Trainer Module)
#         # We get 'local_epochs' from server config (default=1)
#         epochs = int(config.get("local_epochs", 1))
#         metrics = self.trainer.train(self.model, self.train_loader, epochs=epochs, device=self.device)
        
#         # 3. Get Updated Weights
#         updated_weights = self.get_parameters(config={})

#         # 4. Apply Privacy (Delegate to Privacy Module)
#         safe_weights = self.privacy.sanitize(updated_weights)

#         # 5. Compress (Delegate to Compression Module)
#         final_payload = self.compression.compress(safe_weights)

#         # Return: Payload, Dataset Size, Metrics
#         return final_payload, len(self.train_loader.dataset), metrics

#     # def evaluate(self, parameters, config):
#     #     self._set_parameters(parameters)
#     #     loss, accuracy = self.trainer.evaluate(self.model, self.test_loader, self.device)
#     #     return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

#     def evaluate(self, parameters, config):
#         decompressed_params = self.compression.decompress(parameters)
#         self._set_parameters(decompressed_params)
#         loss, accuracy = self.trainer.evaluate(self.model, self.test_loader, self.device)
#         return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}



#     def _set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

import flwr as fl
import torch
from collections import OrderedDict
from typing import Dict, List, Any
from .interfaces import AbstractTrainer, AbstractPrivacyModule, AbstractCompressionModule
from src.observerbility.status_store import StatusStore


class ModularFlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: Any,
        test_loader: Any,
        trainer: AbstractTrainer,
        #privacy: AbstractPrivacyModule,
        compression: AbstractCompressionModule,
        device: str = "cpu",
        status_store: StatusStore | None = None,
        client_name: str = "unknown_client",
    ):
        self.client_round = 0
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.trainer = trainer
        #self.privacy = privacy
        self.compression = compression
        self.device = device
        self.model.to(self.device)

        self.status = status_store
        self.client_name = client_name

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        # Round bookkeeping
        self.client_round += 1
        global_round = config.get("round", None)

        # if self.status:
        #     self.status.update(
        #         state="TRAINING",
        #         global_round=global_round,
        #         client_round=self.client_round,
        #         message=f"Round started (client_round={self.client_round})",
        #     )

        # if self.status:
        #     self.status.update(
        #         state="TRAINING",
        #         global_round=global_round,
        #         client_round=self.client_round,
        #         message=f"Round started (client_round={self.client_round})",
        #     )



        # 1) Decompress & load
        decompressed_params = self.compression.decompress(parameters)
        self._set_parameters(decompressed_params)

        # 2) Train
        epochs = int(config.get("local_epochs", 1))
        #metrics = self.trainer.train(self.model, self.train_loader, epochs=epochs, device=self.device)
        metrics = (
            self.trainer.train(
                self.model,
                self.train_loader,
                epochs=epochs,
                device=self.device,
            )
            or {}
        )        

        # 3) Get updated weights
        updated_weights = self.get_parameters(config={})

        # 4) Privacy
        #safe_weights = self.privacy.sanitize(updated_weights)

        # 5) Compression
        #final_payload = self.compression.compress(safe_weights)

        # Update status with metrics (end of round)
        if self.status:
            self.status.update(
                state="TRAINING",
                global_round=global_round,
                client_round=self.client_round,
                train_loss=float(metrics.get("train_loss")) if metrics.get("train_loss") is not None else None,
                train_accuracy=float(metrics.get("accuracy")) if metrics.get("accuracy") is not None else None,
                message=f"Round finished (client_round={self.client_round})",
            )
        metrics["client_name"] = self.client_name #server update
        # return final_payload, len(self.train_loader.dataset), metrics
        return updated_weights, len(self.train_loader.dataset), metrics

    def evaluate(self, parameters, config):
        decompressed_params = self.compression.decompress(parameters)
        self._set_parameters(decompressed_params)

        loss, accuracy = self.trainer.evaluate(self.model, self.test_loader, self.device)

        # if self.status:
        #     self.status.update(
        #         eval_loss=float(loss),
        #         accuracy=float(accuracy),
        #         message="Evaluation completed",
        #     )
        if self.status:
            self.status.update(
                eval_loss=float(loss),
                eval_accuracy=float(accuracy),
                message="Evaluation completed",
            )
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

    def _set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v).to(self.device) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
