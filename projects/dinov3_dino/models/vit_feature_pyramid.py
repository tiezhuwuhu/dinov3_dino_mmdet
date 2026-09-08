from __future__ import annotations

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmdet.registry import MODELS


@MODELS.register_module()
class ViTFeaturePyramid(nn.Module):
    """Fuse same-resolution ViT features and build four scales.

    Inputs:
        Three feature maps with shape [B, 384, H/16, W/16].

    Outputs:
        P3: [B, 256, H/8,  W/8]
        P4: [B, 256, H/16, W/16]
        P5: [B, 256, H/32, W/32]
        P6: [B, 256, H/64, W/64]
    """

    def __init__(
        self,
        in_channels: tuple[int, ...] = (
            384,
            384,
            384,
        ),
        out_channels: int = 256,
    ) -> None:
        super().__init__()

        if len(in_channels) != 3:
            raise ValueError(
                "This first implementation expects exactly "
                "three ViT block features."
            )

        norm_cfg = dict(
            type="GN",
            num_groups=32,
            requires_grad=True,
        )

        self.lateral_convs = nn.ModuleList(
            [
                ConvModule(
                    in_channels=channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    norm_cfg=norm_cfg,
                    act_cfg=None,
                )
                for channels in in_channels
            ]
        )

        self.fuse_conv = ConvModule(
            in_channels=out_channels * len(in_channels),
            out_channels=out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p4_refine = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p3_refine = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p5_downsample = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p6_downsample = ConvModule(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(
        self,
        inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        if len(inputs) != len(self.lateral_convs):
            raise ValueError(
                f"Expected {len(self.lateral_convs)} inputs, "
                f"got {len(inputs)}."
            )

        reference_shape = inputs[0].shape[-2:]

        for index, feature in enumerate(inputs):
            if feature.shape[-2:] != reference_shape:
                raise ValueError(
                    "All ViT block features must have the "
                    "same spatial shape. "
                    f"Input 0={reference_shape}, "
                    f"input {index}={feature.shape[-2:]}"
                )

        lateral_features = [
            lateral_conv(feature)
            for lateral_conv, feature in zip(
                self.lateral_convs,
                inputs,
                strict=True,
            )
        ]

        fused = self.fuse_conv(
            torch.cat(lateral_features, dim=1)
        )

        p4 = self.p4_refine(fused)

        p3 = F.interpolate(
            p4,
            scale_factor=2.0,
            mode="bilinear",
            align_corners=False,
        )
        p3 = self.p3_refine(p3)

        p5 = self.p5_downsample(p4)
        p6 = self.p6_downsample(p5)

        return p3, p4, p5, p6