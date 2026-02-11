# will whitelist mappings
# src/config/registry.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Type

from src.core.interfaces import (
    AbstractTrainer,
    AbstractPrivacyModule,
    AbstractCompressionModule,
    AbstractDataLoader,
)

# Import concrete implementations (these must exist after Step 01 refactor)
from src.dataLoaders.cardiac_MRI import CardiacMRIDataLoader
from src.modules.training import StandardPyTorchTrainer
from src.modules.privacy import NoPrivacy, GaussianPrivacy
from src.modules.compression import NoCompression

from src.dataLoaders import MNISTDataLoader
from src.models.kan import KAN
from src.models.mlp import SimpleMLP


@dataclass(frozen=True)
class Registry:
    models: Dict[str, Type]
    trainers: Dict[str, Type[AbstractTrainer]]
    privacy: Dict[str, Type[AbstractPrivacyModule]]
    compression: Dict[str, Type[AbstractCompressionModule]]
    data: Dict[str, Type[AbstractDataLoader]]


DEFAULT_REGISTRY = Registry(
    models={
        "kan": KAN,
        "mlp": SimpleMLP,
    },
    trainers={
        "standard": StandardPyTorchTrainer,
    },
    privacy={
        "none": NoPrivacy,
        "gaussian": GaussianPrivacy,
    },
    compression={
        "none": NoCompression,
    },
    data={
        "mnist": MNISTDataLoader,
        "mri": CardiacMRIDataLoader,
    },
)
