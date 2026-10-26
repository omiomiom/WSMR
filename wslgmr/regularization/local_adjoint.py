from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


SYNAPTIC_MODULES = (nn.Conv2d, nn.Linear)
@dataclass
class LocalAdjointRecord:
    module: nn.Module
    primal_input: Tensor | None
    primal_output: Tensor | None
    input_adjoint: Tensor | None = None
    output_adjoint: Tensor | None = None

    def release_primal(self) -> None:
        self.primal_input = None
        self.primal_output = None


class LocalAdjointCapture:
    """Capture margin gradients at convolution and linear layer boundaries.

    Compute gradients of the summed batch margins with respect to raw images
    and tracked layer inputs and outputs in one autograd call. Differentiation
    includes residual paths and BatchNorm batch statistics during training.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.records: list[LocalAdjointRecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _hook(self, module: nn.Module, inputs, output) -> None:
        if not inputs or not isinstance(inputs[0], Tensor):
            return
        if not isinstance(output, Tensor):
            return
        self.records.append(
            LocalAdjointRecord(
                module=module,
                primal_input=inputs[0],
                primal_output=output,
            )
        )

    def __enter__(self) -> "LocalAdjointCapture":
        self.records = []
        for module in self.model.modules():
            if isinstance(module, SYNAPTIC_MODULES):
                self._handles.append(module.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def compute(
        self,
        margins: Tensor,
        images: Tensor,
    ) -> tuple[list[LocalAdjointRecord], Tensor]:
        tensors: list[Tensor] = []
        tensor_indices: dict[int, int] = {}

        def include(tensor: Tensor) -> None:
            key = id(tensor)
            if key not in tensor_indices:
                tensor_indices[key] = len(tensors)
                tensors.append(tensor)

        include(images)
        for record in self.records:
            if record.primal_input is None or record.primal_output is None:
                continue
            include(record.primal_input)
            include(record.primal_output)

        gradients = torch.autograd.grad(
            outputs=margins.sum(),
            inputs=tensors,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        gradient_by_id = {
            id(tensor): gradient
            for tensor, gradient in zip(tensors, gradients)
        }
        image_gradient = gradient_by_id[id(images)]
        if image_gradient is None:
            raise RuntimeError("Margin is disconnected from the input image.")

        usable_records: list[LocalAdjointRecord] = []
        for record in self.records:
            if record.primal_input is None or record.primal_output is None:
                continue
            record.input_adjoint = gradient_by_id[id(record.primal_input)]
            record.output_adjoint = gradient_by_id[id(record.primal_output)]
            if (
                record.input_adjoint is not None
                and record.output_adjoint is not None
            ):
                usable_records.append(record)
            else:
                record.release_primal()
        # Share the records list so records.clear() also releases captured activations.
        self.records = usable_records
        return self.records, image_gradient.detach()
