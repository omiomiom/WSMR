from __future__ import annotations

import torch
from torch import Tensor, nn


class NormalizedModel(nn.Module):
    """Keep datasets/noise in [0, 1] and normalize inside the model."""

    def __init__(
        self,
        network: nn.Module,
        mean: tuple[float, ...],
        std: tuple[float, ...],
        input_shape: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.network = network
        self.input_shape = tuple(int(value) for value in input_shape)
        self.register_buffer(
            "input_mean",
            torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1),
        )
        self.register_buffer(
            "input_std",
            torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1),
        )

    @property
    def first_weight_module(self) -> nn.Module:
        return self.network.first_weight_module

    def forward(self, images: Tensor) -> Tensor:
        normalized = (images - self.input_mean) / self.input_std
        return self.network(normalized)

    def first_layer_input_metric(
        self,
        module: nn.Module,
        input_adjoint: Tensor,
    ) -> Tensor | None:
        if module is not self.first_weight_module:
            return None
        inverse_variance = self.input_std.reciprocal().square()
        if input_adjoint.ndim == 4:
            return inverse_variance
        if input_adjoint.ndim == 2:
            channels, height, width = self.input_shape
            metric = inverse_variance.reshape(channels, 1, 1)
            metric = metric.expand(channels, height, width).reshape(1, -1)
            if metric.size(1) != input_adjoint.size(1):
                raise ValueError("Flattened first-layer metric has wrong size.")
            return metric
        raise ValueError("Unsupported first-layer adjoint rank.")

