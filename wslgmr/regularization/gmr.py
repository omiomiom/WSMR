from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def choose_competitors(
    logits: Tensor,
    targets: Tensor,
) -> Tensor:
    """Select the highest-scoring non-target class for each sample."""

    masked = logits.detach().clone()
    masked.scatter_(1, targets.view(-1, 1), float("-inf"))
    return masked.argmax(dim=1)


def compute_margins(
    logits: Tensor,
    targets: Tensor,
    competitors: Tensor,
) -> Tensor:
    rows = torch.arange(logits.size(0), device=logits.device)
    return logits[rows, targets] - logits[rows, competitors]


def compute_gmr_quantities(
    margins: Tensor,
    input_gradients: Tensor,
    config: dict,
) -> dict[str, Tensor]:
    sigma = float(config["sigma"])
    z_value = float(config["z_value"])
    target_margin = float(config["target_margin"])
    beta = float(config["softplus_beta"])
    epsilon = float(config["variance_epsilon"])
    if sigma < 0.0 or z_value < 0.0:
        raise ValueError("sigma and z_value must be non-negative.")
    if beta <= 0.0 or epsilon <= 0.0:
        raise ValueError("softplus_beta and variance_epsilon must be positive.")

    squared_norm = input_gradients.detach().float().flatten(1).square().sum(1)
    rho = torch.sqrt(sigma * sigma * squared_norm + epsilon)
    softplus_argument = target_margin - margins.detach() + z_value * rho
    activation = torch.sigmoid(beta * softplus_argument)
    kappa = activation * z_value * sigma * sigma / rho
    regularizer = F.softplus(softplus_argument, beta=beta).mean()
    return {
        "rho": rho,
        "activation": activation,
        "kappa": kappa,
        "regularizer": regularizer,
    }


def compute_exact_gmr_loss(
    margins: Tensor,
    input_gradients: Tensor,
    config: dict,
) -> Tensor:
    """Compute the differentiable GMR loss for the exact_g0 comparison method."""

    sigma = float(config["sigma"])
    z_value = float(config["z_value"])
    target_margin = float(config["target_margin"])
    beta = float(config["softplus_beta"])
    epsilon = float(config["variance_epsilon"])
    squared_norm = input_gradients.float().flatten(1).square().sum(1)
    rho = torch.sqrt(sigma * sigma * squared_norm + epsilon)
    argument = target_margin - margins + z_value * rho
    return F.softplus(argument, beta=beta).mean()
