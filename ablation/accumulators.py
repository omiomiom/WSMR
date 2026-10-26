from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from wslgmr.regularization.local_adjoint import LocalAdjointRecord
from wslgmr.regularization.wake_sleep import WakeSleepAccumulator, _contract


class ExactDelayedAccumulator(WakeSleepAccumulator):
    """Sum same-sample local factor products in weight-shaped buffers.

    Apply the accumulated updates during Sleep. Each tracked weight tensor
    has one buffer of the same shape; no cross-sample factor products are used.
    """

    def reset(self) -> None:
        super().reset()
        self.traces: dict[nn.Parameter, Tensor] = {}

    @torch.no_grad()
    def update_records(
        self,
        records: list[LocalAdjointRecord],
        kappas: list[Tensor],
        metrics: list[Tensor | None],
    ) -> None:
        factors = self._factor_conditions(records, kappas, metrics)
        if not factors:
            raise RuntimeError("No Conv/Linear local conditions were captured.")
        batch_size = factors[0].batch_size
        signature = tuple(condition.module for condition in factors)
        if self.pending and signature != self.signature:
            raise RuntimeError("Tracked module order changed within a cycle.")
        if not self.pending:
            self.batch_size = batch_size
            self.signature = signature
        for condition in factors:
            parameter = condition.module.weight
            value = _contract(condition, self.contraction_dtype)
            if parameter in self.traces:
                self.traces[parameter].add_(value)
            else:
                self.traces[parameter] = value
        self.batch_count += 1

    @torch.no_grad()
    def sleep(
        self,
        *,
        optimizer: torch.optim.Optimizer,
        regularizer_weight: float,
    ) -> dict[str, float]:
        if not self.pending:
            raise RuntimeError("Sleep requested without wake statistics.")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        multiplier = (
            -learning_rate
            * float(regularizer_weight)
        )
        norm_squared: Tensor | None = None
        for parameter, trace in self.traces.items():
            update = trace.mul(multiplier)
            value = update.float().square().sum()
            norm_squared = value if norm_squared is None else norm_squared + value
            parameter.add_(update.to(dtype=parameter.dtype))
        result = {
            "sleep_batches": float(self.batch_count),
            "sleep_update_norm": (
                math.sqrt(float(norm_squared))
                if norm_squared is not None
                else 0.0
            ),
        }
        self.reset()
        return result
