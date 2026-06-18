"""Lightweight feature pyramid for breast multi-task model."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution block used inside LightFPN."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.dw = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False,
        )
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        """Apply depthwise convolution, pointwise projection, BN, and activation."""
        return self.act(self.bn(self.pw(self.dw(x))))


class LightFPN(nn.Module):
    """Top-down FPN with depthwise separable convolutions."""

    def __init__(
        self,
        in_channels: Sequence[int],
        out_channels: int = 256,
        num_outs: int | None = None,
    ) -> None:
        super().__init__()
        if len(in_channels) < 3:
            raise ValueError("LightFPN expects at least three input feature maps.")
        self.in_channels = list(in_channels)
        self.out_channels = out_channels
        self.num_outs = num_outs or len(in_channels)

        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(ch, out_channels, kernel_size=1, bias=False) for ch in in_channels]
        )
        self.output_convs = nn.ModuleList(
            [DepthwiseSeparableConv(out_channels, out_channels) for _ in in_channels]
        )

    def forward(self, inputs: Sequence[Tensor]) -> list[Tensor]:
        """Fuse multi-scale backbone outputs into a compact pyramid."""
        if len(inputs) != len(self.lateral_convs):
            raise ValueError("Number of inputs must match configured stages.")

        laterals = [lateral(feat) for lateral, feat in zip(self.lateral_convs, inputs)]
        for idx in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[idx], size=laterals[idx - 1].shape[-2:], mode="nearest")
            laterals[idx - 1] = laterals[idx - 1] + upsampled

        outs = [conv(feat) for conv, feat in zip(self.output_convs, laterals)]
        return outs[: self.num_outs]

