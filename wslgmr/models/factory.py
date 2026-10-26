from __future__ import annotations

from torch import nn

from .common import NormalizedModel
from .mlp import MLP
from .resnet import build_resnet
from .wrn import WideResNet


WRN_PRESETS = {
    "wrn28_2": (28, 2),
    "wrn28_10": (28, 10),
    "wrn40_10": (40, 10),
}


def build_model(
    config: dict,
    *,
    input_shape: tuple[int, int, int],
    mean: tuple[float, ...],
    std: tuple[float, ...],
    num_classes: int,
) -> nn.Module:
    name = config["model"]
    options = config["model_options"]
    dropout = float(options["dropout"])
    if name == "mlp":
        layers = [int(value) for value in options["mlp_layers"]]
        expected_inputs = input_shape[0] * input_shape[1] * input_shape[2]
        if layers[0] != expected_inputs or layers[-1] != num_classes:
            raise ValueError(
                "model_options.mlp_layers must match input and class sizes."
            )
        network = MLP(layers, dropout=dropout)
    elif name in WRN_PRESETS:
        depth, widen_factor = WRN_PRESETS[name]
        network = WideResNet(
            depth=depth,
            widen_factor=widen_factor,
            num_classes=num_classes,
            dropout=dropout,
        )
    elif name.startswith("resnet"):
        network = build_resnet(
            variant=name,
            num_classes=num_classes,
            zero_init_residual=bool(options["zero_init_residual"]),
        )
    else:
        raise ValueError(f"Unsupported model: {name}")
    return NormalizedModel(network, mean, std, input_shape)
