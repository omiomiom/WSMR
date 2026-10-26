from __future__ import annotations

from torch import Tensor, nn


class MLP(nn.Module):
    def __init__(self, layers: list[int], dropout: float = 0.0) -> None:
        super().__init__()
        if len(layers) < 2:
            raise ValueError("MLP needs at least input and output dimensions.")
        self.flatten = nn.Flatten()
        self.linears = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(layers[:-1], layers[1:])
        )
        self.dropout = nn.Dropout(float(dropout))
        self.activation = nn.ReLU(inplace=False)

    @property
    def first_weight_module(self) -> nn.Module:
        return self.linears[0]

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.flatten(inputs)
        for layer in self.linears[:-1]:
            value = self.activation(layer(value))
            value = self.dropout(value)
        return self.linears[-1](value)

