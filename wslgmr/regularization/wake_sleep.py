from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import grad as nn_grad

from ..models.common import NormalizedModel
from ..utils import dtype_from_name
from .local_adjoint import LocalAdjointRecord


def _sample_broadcast(values: Tensor, reference: Tensor) -> Tensor:
    return values.view(values.size(0), *([1] * (reference.ndim - 1)))


def _hidden_gradient_metric(
    record: LocalAdjointRecord,
    model: NormalizedModel,
    regularizer_config: dict,
) -> Tensor | None:
    """Return detached per-channel weights for the layer input gradients."""

    if record.input_adjoint is None or record.primal_input is None:
        raise RuntimeError("Local record is incomplete.")
    input_metric = model.first_layer_input_metric(
        record.module,
        record.input_adjoint,
    )
    if input_metric is not None:
        return input_metric.detach().float()

    mode = str(
        regularizer_config.get("hidden_gradient_metric", "euclidean")
    ).lower()
    if mode == "euclidean":
        return None
    if mode != "activation_rms":
        raise ValueError(
            "hidden_gradient_metric must be euclidean or activation_rms."
        )

    epsilon = float(
        regularizer_config.get("hidden_metric_epsilon", 1.0e-6)
    )
    inputs = record.primal_input.detach().float()
    if inputs.ndim == 4:
        reduction_dims = (0, 2, 3)
    elif inputs.ndim == 2:
        reduction_dims = (0,)
    else:
        raise ValueError(
            "activation_rms supports only Conv2d and Linear inputs."
        )
    return inputs.square().mean(
        dim=reduction_dims,
        keepdim=True,
    ).clamp_min_(epsilon)


@dataclass
class LayerGmrConditions:
    kappas: list[Tensor]
    metrics: list[Tensor | None]


@torch.no_grad()
def build_layer_gmr_conditions(
    records: list[LocalAdjointRecord],
    global_kappa: Tensor,
    model: NormalizedModel,
    regularizer_config: dict,
) -> LayerGmrConditions:
    """Reuse each sample's input-space GMR coefficient at every weight layer."""

    metrics = [
        _hidden_gradient_metric(record, model, regularizer_config)
        for record in records
    ]
    return LayerGmrConditions(
        kappas=[global_kappa.detach()] * len(records),
        metrics=metrics,
    )


@dataclass
class FactorCondition:
    module: nn.Module
    input_factor: Tensor
    output_factor: Tensor
    batch_size: int


def _contract(condition: FactorCondition, dtype: torch.dtype) -> Tensor:
    module = condition.module
    input_factor = condition.input_factor.to(dtype=dtype)
    output_factor = condition.output_factor.to(dtype=dtype)
    if isinstance(module, nn.Conv2d):
        value = nn_grad.conv2d_weight(
            input=input_factor,
            weight_size=module.weight.shape,
            grad_output=output_factor,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
    elif isinstance(module, nn.Linear):
        left = output_factor.reshape(-1, module.out_features)
        right = input_factor.reshape(-1, module.in_features)
        value = left.transpose(0, 1) @ right
    else:
        raise TypeError(f"Unsupported factor module: {type(module).__name__}")
    value.div_(condition.batch_size)
    return value


class WakeSleepAccumulator:
    def __init__(self, config: dict, epoch: int) -> None:
        self.storage_dtype = dtype_from_name(
            config["factor_storage_dtype"]
        )
        self.contraction_dtype = dtype_from_name(
            config["factor_contraction_dtype"]
        )
        self.reset()

    def reset(self) -> None:
        self.batch_count = 0
        self.batch_size: int | None = None
        self.signature: tuple[nn.Module, ...] | None = None
        self.factor_states: list[dict] = []

    @property
    def pending(self) -> bool:
        return self.batch_count > 0

    def accepts_batch_size(self, batch_size: int) -> bool:
        return not self.pending or self.batch_size == int(batch_size)

    def _uniform_scales(self) -> tuple[float, float]:
        next_count = self.batch_count + 1
        return (
            math.sqrt((next_count - 1) / next_count),
            1.0 / math.sqrt(next_count),
        )

    def _compress_batch(
        self,
        factors: list[FactorCondition],
    ) -> list[FactorCondition]:
        """Sum each layer's input and output factors over the batch.

        Divide both sums by the square root of the batch size. Each result
        keeps a singleton batch dimension, so accumulation-buffer shapes
        do not depend on the number of samples.
        """

        batch_size = factors[0].batch_size
        normalization = math.sqrt(float(batch_size))
        compressed: list[FactorCondition] = []
        for condition in factors:
            compressed.append(
                FactorCondition(
                    module=condition.module,
                    input_factor=(
                        condition.input_factor.sum(dim=0, keepdim=True)
                        .div_(normalization)
                    ),
                    output_factor=(
                        condition.output_factor.sum(dim=0, keepdim=True)
                        .div_(normalization)
                    ),
                    batch_size=1,
                )
            )
        return compressed

    def _factor_conditions(
        self,
        records: list[LocalAdjointRecord],
        kappas: list[Tensor],
        metrics: list[Tensor | None],
    ) -> list[FactorCondition]:
        if not (len(records) == len(kappas) == len(metrics)):
            raise ValueError("Layer condition count does not match records.")
        factors: list[FactorCondition] = []
        for record, kappa, metric in zip(records, kappas, metrics):
            module = record.module
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                raise TypeError(
                    f"Unsupported local module: {type(module).__name__}"
                )
            if (
                record.primal_input is None
                or record.input_adjoint is None
                or record.output_adjoint is None
            ):
                raise RuntimeError("Local factor record is incomplete.")
            input_adjoint = record.input_adjoint.detach()
            scale = _sample_broadcast(kappa.detach(), input_adjoint)
            if metric is not None:
                scale = scale * metric
            factors.append(
                FactorCondition(
                    module=module,
                    input_factor=(input_adjoint * scale).to(
                        dtype=self.storage_dtype,
                        copy=True,
                    ),
                    output_factor=record.output_adjoint.detach().to(
                        dtype=self.storage_dtype,
                        copy=True,
                    ),
                    batch_size=int(input_adjoint.size(0)),
                )
            )
            record.input_adjoint = None
            record.output_adjoint = None
            record.release_primal()
        return factors

    @torch.no_grad()
    def update_records(
        self,
        records: list[LocalAdjointRecord],
        kappas: list[Tensor],
        metrics: list[Tensor | None],
    ) -> None:
        physical_factors = self._factor_conditions(records, kappas, metrics)
        if not physical_factors:
            raise RuntimeError("No Conv/Linear local conditions were captured.")
        physical_batch_size = physical_factors[0].batch_size
        if not self.accepts_batch_size(physical_batch_size):
            raise ValueError("A sleep cycle cannot mix different batch sizes.")
        factors = self._compress_batch(physical_factors)
        signature = tuple(condition.module for condition in factors)
        if self.pending and signature != self.signature:
            raise RuntimeError("Tracked module order changed within a cycle.")

        old_scale, new_scale = self._uniform_scales()
        if not self.pending:
            self.batch_size = physical_batch_size
            self.signature = signature
            self.factor_states = []
            for condition in factors:
                self.factor_states.append(
                    {
                        "module": condition.module,
                        "input": condition.input_factor.mul_(new_scale),
                        "output": condition.output_factor.mul_(new_scale),
                    }
                )
        else:
            for state, condition in zip(self.factor_states, factors):
                state["input"].mul_(old_scale).add_(
                    condition.input_factor, alpha=new_scale
                )
                state["output"].mul_(old_scale).add_(
                    condition.output_factor, alpha=new_scale
                )
        self.batch_count += 1

    @torch.no_grad()
    def sleep(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        regularizer_weight: float,
    ) -> dict[str, float]:
        if not self.pending or self.batch_size is None:
            raise RuntimeError("Sleep requested without wake statistics.")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        multiplier = (
            -learning_rate
            * float(regularizer_weight)
            * float(self.batch_count)
        )
        trace_by_parameter: dict[nn.Parameter, Tensor] = {}
        for state in self.factor_states:
            condition = FactorCondition(
                module=state["module"],
                input_factor=state["input"],
                output_factor=state["output"],
                batch_size=1,
            )
            parameter = state["module"].weight
            trace_by_parameter[parameter] = _contract(
                condition, self.contraction_dtype
            )

        norm_squared: Tensor | None = None
        for parameter, trace in trace_by_parameter.items():
            update = trace.mul(multiplier)
            value = update.float().square().sum()
            norm_squared = value if norm_squared is None else norm_squared + value
            parameter.add_(update.to(dtype=parameter.dtype))

        update_norm = (
            math.sqrt(float(norm_squared))
            if norm_squared is not None
            else 0.0
        )
        result = {
            "sleep_batches": float(self.batch_count),
            "sleep_update_norm": update_norm,
        }
        self.reset()
        return result
