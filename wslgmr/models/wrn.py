from __future__ import annotations

import math

from torch import Tensor, nn
from torch.nn import functional as F


class WideBasicBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.dropout = float(dropout)
        self.shortcut = (
            None
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        preactivated = F.relu(self.bn1(inputs), inplace=False)
        shortcut = (
            inputs
            if self.shortcut is None
            else self.shortcut(preactivated)
        )
        value = self.conv1(preactivated)
        value = F.relu(self.bn2(value), inplace=False)
        if self.dropout > 0.0:
            value = F.dropout(value, p=self.dropout, training=self.training)
        value = self.conv2(value)
        return value + shortcut


class WideResNet(nn.Module):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        num_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if (int(depth) - 4) % 6 != 0:
            raise ValueError("WideResNet depth must satisfy (depth - 4) % 6 = 0.")
        blocks_per_group = (int(depth) - 4) // 6
        channels = [
            16,
            16 * int(widen_factor),
            32 * int(widen_factor),
            64 * int(widen_factor),
        ]
        self.conv1 = nn.Conv2d(
            3, channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        groups: list[nn.Module] = []
        in_channels = channels[0]
        for out_channels, stride in zip(channels[1:], (1, 2, 2)):
            blocks = []
            for index in range(blocks_per_group):
                block_stride = stride if index == 0 else 1
                blocks.append(
                    WideBasicBlock(
                        in_channels,
                        out_channels,
                        block_stride,
                        dropout,
                    )
                )
                in_channels = out_channels
            groups.append(nn.Sequential(*blocks))
        self.groups = nn.ModuleList(groups)
        self.final_bn = nn.BatchNorm2d(channels[-1])
        self.classifier = nn.Linear(channels[-1], int(num_classes))
        self._initialize()

    @property
    def first_weight_module(self) -> nn.Module:
        return self.conv1

    def _initialize(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                fan_out = (
                    module.kernel_size[0]
                    * module.kernel_size[1]
                    * module.out_channels
                )
                nn.init.normal_(module.weight, 0.0, math.sqrt(2.0 / fan_out))
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)

    def forward(self, inputs: Tensor) -> Tensor:
        value = self.conv1(inputs)
        for group in self.groups:
            value = group(value)
        value = F.relu(self.final_bn(value), inplace=False)
        value = F.adaptive_avg_pool2d(value, output_size=1).flatten(1)
        return self.classifier(value)
