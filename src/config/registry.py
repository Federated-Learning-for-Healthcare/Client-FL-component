"""
registry.py — Maps config type strings to concrete classes.
"""

from dataclasses import dataclass
from typing import Dict, Type

from src.core.interfaces import (
    AbstractCompressionModule, AbstractDataLoader,
    AbstractPrivacyModule, AbstractTrainer,
)
from src.dataLoaders.mnist_loader import MNISTDataLoader
from src.dataLoaders.ehr_loader import EHRLoader
from src.dataLoaders.ecg_loader import ECGLoader
from src.dataLoaders.multimodal_loader import MultiModalDataLoader
from src.models.kan import KAN
from src.models.mlp import SimpleMLP
from src.models.multimodal import MultiModalModel
from src.modules.compression import NoCompression, QuantizationCompression, TopKCompression
from src.modules.privacy import DPSGDPrivacy, GaussianPrivacy, NoPrivacy, TrueDPSGDPrivacy
from src.modules.training import StandardPyTorchTrainer
from src.modules.multimodal_trainer import MultiModalTrainer


@dataclass(frozen=True)
class Registry:
    models:      Dict[str, Type]
    trainers:    Dict[str, Type[AbstractTrainer]]
    privacy:     Dict[str, Type[AbstractPrivacyModule]]
    compression: Dict[str, Type[AbstractCompressionModule]]
    data:        Dict[str, Type[AbstractDataLoader]]


DEFAULT_REGISTRY = Registry(
    models={
        "kan":        KAN,
        "mlp":        SimpleMLP,
        "multimodal": MultiModalModel,
    },
    trainers={
        "standard":   StandardPyTorchTrainer,
        "multimodal": MultiModalTrainer,
    },
    privacy={
        "none":        NoPrivacy,
        "gaussian":    GaussianPrivacy,
        "dpsgd":       DPSGDPrivacy,
        "true_dpsgd":  TrueDPSGDPrivacy,
    },
    compression={
        "none":     NoCompression,
        "topk":     TopKCompression,
        "quantize": QuantizationCompression,
    },
    data={
        "mnist":      MNISTDataLoader,
        "ehr":        EHRLoader,
        "ecg":        ECGLoader,
        "multimodal": MultiModalDataLoader,
    },
)