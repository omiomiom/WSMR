from __future__ import annotations

from typing import Type

import torch
from torch import Tensor, nn


def conv3x3(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
    groups: int = 1,
    dilation: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(
    in_channels: int,
    out_channels: int,
    stride: int = 1,
) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = downsample

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        value = self.conv1(inputs)
        value = torch.relu(self.bn1(value))
        value = self.bn2(self.conv2(value))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        value = value + identity
        return torch.relu(value)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        width = channels
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.downsample = downsample

    def forward(self, inputs: Tensor) -> Tensor:
        identity = inputs
        value = torch.relu(self.bn1(self.conv1(inputs)))
        value = torch.relu(self.bn2(self.conv2(value)))
        value = self.bn3(self.conv3(value))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        value = value + identity
        return torch.relu(value)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock] | Type[Bottleneck],
        layers: list[int],
        num_classes: int,
        zero_init_residual: bool,
    ) -> None:
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, int(num_classes))
        self._initialize(bool(zero_init_residual))

    @property
    def first_weight_module(self) -> nn.Module:
        return self.conv1

    def _make_layer(
        self,
        block,
        channels: int,
        count: int,
        stride: int,
    ) -> nn.Sequential:
        downsample = None
        output_channels = channels * block.expansion
        if stride != 1 or self.in_channels != output_channels:
            downsample = nn.Sequential(
                conv1x1(self.in_channels, output_channels, stride),
                nn.BatchNorm2d(output_channels),
            )
        blocks = [
            block(self.in_channels, channels, stride, downsample)
        ]
        self.in_channels = output_channels
        for _ in range(1, count):
            blocks.append(block(self.in_channels, channels))
        return nn.Sequential(*blocks)

    def _initialize(self, zero_init_residual: bool) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        if zero_init_residual:
            for module in self.modules():
                if isinstance(module, Bottleneck):
                    nn.init.zeros_(module.bn3.weight)
                elif isinstance(module, BasicBlock):
                    nn.init.zeros_(module.bn2.weight)

    def forward(self, inputs: Tensor) -> Tensor:
        value = torch.relu(self.bn1(self.conv1(inputs)))
        value = self.maxpool(value)
        value = self.layer1(value)
        value = self.layer2(value)
        value = self.layer3(value)
        value = self.layer4(value)
        value = self.avgpool(value).flatten(1)
        return self.fc(value)


RESNET_SPECS = {
    "resnet18": (BasicBlock, [2, 2, 2, 2]),
    "resnet34": (BasicBlock, [3, 4, 6, 3]),
    "resnet50": (Bottleneck, [3, 4, 6, 3]),
    "resnet101": (Bottleneck, [3, 4, 23, 3]),
    "resnet152": (Bottleneck, [3, 8, 36, 3]),
}


def build_resnet(
    variant: str,
    num_classes: int,
    zero_init_residual: bool,
) -> ResNet:
    if variant not in RESNET_SPECS:
        raise ValueError(f"Unsupported ResNet variant: {variant}")
    block, layers = RESNET_SPECS[variant]
    return ResNet(block, layers, num_classes, zero_init_residual)

