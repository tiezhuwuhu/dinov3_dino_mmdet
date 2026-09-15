from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from torch import Tensor, nn

from mmdet.registry import MODELS


@MODELS.register_module()
class LearnableAnchorViTFeaturePyramid(nn.Module):
    """Learnable anchor-dominant pyramid for 12 ViT block outputs.

    This module is completely standalone.

    It does NOT depend on:
        GroupedMeanViTFeaturePyramid
        ViTFeaturePyramid
        any previous grouped-mean implementation

    Expected ViT outputs:
        F0 ... F11

    Every feature:
        [B, 384, H/16, W/16]

    Fusion:

        Early:
            history = mean(F0, F1, F2)
            anchor  = F3

        Middle:
            history = mean(F4, F5, F6)
            anchor  = F7

        Deep:
            history = mean(F8, F9, F10)
            anchor  = F11

        fused =
            (anchor + alpha * history)
            / (1 + alpha)

    alpha:
        alpha_early
        alpha_middle
        alpha_deep

    are all learnable.

    Output:
        P3: stride 8
        P4: stride 16
        P5: stride 32
        P6: stride 64
    """

    def __init__(
        self,
        in_channels: int = 384,
        out_channels: int = 256,
        alpha_init: float = 0.25,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        if not 0.0 < alpha_init < 1.0:
            raise ValueError(
                f'alpha_init must be in (0, 1), '
                f'but got {alpha_init}'
            )

        # ====================================================
        # Learnable alpha
        #
        # We optimize raw logits, then use sigmoid:
        #
        #   alpha = sigmoid(alpha_logit)
        #
        # Therefore:
        #   0 < alpha < 1
        #
        # alpha_init = 0.25
        # ====================================================

        init_logit = math.log(
            alpha_init / (1.0 - alpha_init)
        )

        self.alpha_early_logit = nn.Parameter(
            torch.tensor(
                init_logit,
                dtype=torch.float32,
            )
        )

        self.alpha_middle_logit = nn.Parameter(
            torch.tensor(
                init_logit,
                dtype=torch.float32,
            )
        )

        self.alpha_deep_logit = nn.Parameter(
            torch.tensor(
                init_logit,
                dtype=torch.float32,
            )
        )

        norm_cfg = dict(
            type='GN',
            num_groups=32,
            requires_grad=True,
        )

        # ====================================================
        # Early -> P3
        # ====================================================

        self.early_proj = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p3_refine = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        # ====================================================
        # Middle -> P4
        # ====================================================

        self.middle_proj = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p4_refine = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        # ====================================================
        # Deep -> P5 -> P6
        # ====================================================

        self.deep_proj = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p5_downsample = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

        self.p6_downsample = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    # ========================================================
    # Alpha utilities
    # ========================================================

    def get_alphas(self):
        """Return actual learnable alpha values."""

        alpha_early = torch.sigmoid(
            self.alpha_early_logit
        )

        alpha_middle = torch.sigmoid(
            self.alpha_middle_logit
        )

        alpha_deep = torch.sigmoid(
            self.alpha_deep_logit
        )

        return (
            alpha_early,
            alpha_middle,
            alpha_deep,
        )

    # ========================================================
    # Internal standalone history mean
    #
    # This is implemented directly here.
    # It does NOT call any grouped-mean neck.
    # ========================================================

    @staticmethod
    def _mean_three(
        x0: Tensor,
        x1: Tensor,
        x2: Tensor,
    ) -> Tensor:

        return (
            x0
            + x1
            + x2
        ) / 3.0

    # ========================================================
    # Anchor fusion
    # ========================================================

    @staticmethod
    def _fuse(
        anchor: Tensor,
        history: Tensor,
        alpha: Tensor,
    ) -> Tensor:

        # anchor weight:
        #     1 / (1 + alpha)
        #
        # history total weight:
        #     alpha / (1 + alpha)

        return (
            anchor
            + alpha * history
        ) / (
            1.0 + alpha
        )

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        inputs: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:

        if len(inputs) != 12:
            raise ValueError(
                'LearnableAnchorViTFeaturePyramid '
                'requires exactly 12 ViT outputs, '
                f'but got {len(inputs)}'
            )

        # ----------------------------------------------------
        # Shape sanity check
        # ----------------------------------------------------

        spatial_size = inputs[0].shape[-2:]

        for i, feature in enumerate(inputs):

            if feature.ndim != 4:
                raise ValueError(
                    f'Feature F{i} must be BCHW, '
                    f'got {feature.shape}'
                )

            if feature.shape[1] != self.in_channels:
                raise ValueError(
                    f'Feature F{i} has '
                    f'{feature.shape[1]} channels, '
                    f'expected {self.in_channels}'
                )

            if feature.shape[-2:] != spatial_size:
                raise ValueError(
                    'All 12 ViT outputs must have '
                    'the same spatial resolution. '
                    f'F0={spatial_size}, '
                    f'F{i}={feature.shape[-2:]}'
                )

        # ----------------------------------------------------
        # Get learnable coefficients
        # ----------------------------------------------------

        (
            alpha_early,
            alpha_middle,
            alpha_deep,
        ) = self.get_alphas()

        # ====================================================
        # EARLY
        #
        # history:
        #     mean(F0, F1, F2)
        #
        # anchor:
        #     F3
        # ====================================================

        early_history = self._mean_three(
            inputs[0],
            inputs[1],
            inputs[2],
        )

        early = self._fuse(
            anchor=inputs[3],
            history=early_history,
            alpha=alpha_early,
        )

        # ====================================================
        # MIDDLE
        #
        # history:
        #     mean(F4, F5, F6)
        #
        # anchor:
        #     F7
        # ====================================================

        middle_history = self._mean_three(
            inputs[4],
            inputs[5],
            inputs[6],
        )

        middle = self._fuse(
            anchor=inputs[7],
            history=middle_history,
            alpha=alpha_middle,
        )

        # ====================================================
        # DEEP
        #
        # history:
        #     mean(F8, F9, F10)
        #
        # anchor:
        #     F11
        # ====================================================

        deep_history = self._mean_three(
            inputs[8],
            inputs[9],
            inputs[10],
        )

        deep = self._fuse(
            anchor=inputs[11],
            history=deep_history,
            alpha=alpha_deep,
        )

        # ====================================================
        # Early -> P3
        # ====================================================

        early = self.early_proj(
            early
        )

        p3 = F.interpolate(
            early,
            scale_factor=2.0,
            mode='bilinear',
            align_corners=False,
        )

        p3 = self.p3_refine(
            p3
        )

        # ====================================================
        # Middle -> P4
        # ====================================================

        middle = self.middle_proj(
            middle
        )

        p4 = self.p4_refine(
            middle
        )

        # ====================================================
        # Deep -> P5 -> P6
        # ====================================================

        deep = self.deep_proj(
            deep
        )

        p5 = self.p5_downsample(
            deep
        )

        p6 = self.p6_downsample(
            p5
        )

        return (
            p3,
            p4,
            p5,
            p6,
        )