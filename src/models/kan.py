"""
kan.py — FastKAN / KAN model for federated learning.

Key fix: KAN class no longer hardcodes [784, 24, 24, 10].
         layers_hidden is now config-driven, defaulting to a
         sensible value if not specified.

Supports any input dimension — MNIST, Cleveland EHR (13 features),
MIT-BIH ECG (187 features), etc.
"""

from __future__ import annotations

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SplineLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, init_scale: float = 0.1, **kw):
        self.init_scale = init_scale
        super().__init__(in_features, out_features, bias=False, **kw)

    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, mean=0, std=self.init_scale)


class RadialBasisFunction(nn.Module):
    def __init__(
        self,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        denominator: float = None,
    ):
        super().__init__()
        grid = torch.linspace(grid_min, grid_max, num_grids)
        self.grid = nn.Parameter(grid, requires_grad=False)
        self.denominator = denominator or (grid_max - grid_min) / (num_grids - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-((x[..., None] - self.grid) / self.denominator) ** 2)


class FastKANLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        use_layernorm: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.layernorm = (
            nn.LayerNorm(input_dim) if use_layernorm and input_dim > 1 else None
        )
        self.rbf = RadialBasisFunction(grid_min, grid_max, num_grids)
        self.spline_linear = SplineLinear(
            input_dim * num_grids, output_dim, spline_weight_init_scale
        )
        self.use_base_update = use_base_update
        if use_base_update:
            self.base_activation = base_activation
            self.base_linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor, use_layernorm: bool = True) -> torch.Tensor:
        spline_basis = self.rbf(
            self.layernorm(x) if self.layernorm and use_layernorm else x
        )
        ret = self.spline_linear(spline_basis.view(*spline_basis.shape[:-2], -1))
        if self.use_base_update:
            ret = ret + self.base_linear(self.base_activation(x))
        return ret


class FastKAN(nn.Module):
    """
    FastKAN — a KAN using radial basis functions for efficient spline approximation.

    Parameters
    ----------
    layers_hidden : List[int]
        Architecture e.g. [13, 32, 16, 2] for Cleveland EHR binary classification
        or [187, 64, 32, 5] for MIT-BIH ECG 5-class classification.
    grid_min, grid_max, num_grids : spline grid parameters
    use_base_update : bool — add a base linear branch (recommended True)
    """

    def __init__(
        self,
        layers_hidden: List[int],
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        base_activation=F.silu,
        spline_weight_init_scale: float = 0.1,
    ):
        super().__init__()
        self.layers_hidden = layers_hidden
        self.layers = nn.ModuleList([
            FastKANLayer(
                in_d, out_d,
                grid_min, grid_max, num_grids,
                use_base_update, True,
                base_activation, spline_weight_init_scale,
            )
            for in_d, out_d in zip(layers_hidden[:-1], layers_hidden[1:])
        ])
        logger.info("FastKAN initialised — layers_hidden=%s", layers_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KAN(FastKAN):
    """
    KAN wrapper that flattens input and uses config-driven architecture.

    layers_hidden must be provided via config — e.g.:
      MNIST:         [784, 24, 24, 10]
      Cleveland EHR: [13, 32, 16, 2]
      MIT-BIH ECG:   [187, 64, 32, 5]
    """

    def __init__(
        self,
        layers_hidden: List[int] = None,
        grid_min: float = -2.0,
        grid_max: float = 2.0,
        num_grids: int = 8,
        use_base_update: bool = True,
        spline_weight_init_scale: float = 0.1,
    ):
        if layers_hidden is None:
            # Sensible default — override via config.yaml
            layers_hidden = [784, 24, 24, 10]
            logger.warning(
                "KAN: layers_hidden not provided, using default %s. "
                "Set layers_hidden in config.yaml for your dataset.",
                layers_hidden,
            )
        super().__init__(
            layers_hidden=layers_hidden,
            grid_min=grid_min,
            grid_max=grid_max,
            num_grids=num_grids,
            use_base_update=use_base_update,
            spline_weight_init_scale=spline_weight_init_scale,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten all dims except batch
        x = x.view(x.size(0), -1)
        return super().forward(x)