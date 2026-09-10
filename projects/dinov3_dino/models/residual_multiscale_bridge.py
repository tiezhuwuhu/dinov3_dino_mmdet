from __future__ import annotations

from collections.abc import Sequence

import torch
from mmengine.model import BaseModule
from torch import Tensor, nn

from mmdet.registry import MODELS


class ResidualProjectionBlock(nn.Module):
    """Project one DINOSTAs feature and apply a zero-init residual refinement.

    Initial behavior:

        output = projection(input) + 0

    The residual branch is exactly zero at initialization because the final
    GroupNorm scale and bias are initialized to zero.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int,
    ) -> None:
        super().__init__()

        if out_channels % num_groups != 0:
            raise ValueError(
                f"out_channels={out_channels} must be divisible by "
                f"num_groups={num_groups}."
            )

        self.projection = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
        )

        self.refinement = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                num_groups=num_groups,
                num_channels=out_channels,
            ),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.projection(x)
        residual = self.refinement(shortcut)

        return shortcut + residual


@MODELS.register_module()
class ResidualMultiScaleBridge(BaseModule):
    """Bridge Lightly DINOSTAs features into MMDetection DINO features.

    Input:

        P3: [B, 224, H/8,  W/8]
        P4: [B, 224, H/16, W/16]
        P5: [B, 224, H/32, W/32]

    Output:

        P3: [B, 256, H/8,  W/8]
        P4: [B, 256, H/16, W/16]
        P5: [B, 256, H/32, W/32]
        P6: [B, 256, approximately H/64, W/64]

    Each input level uses:

        1x1 projection -> GroupNorm
                    +
        GELU -> 3x3 refinement -> zero-init GroupNorm

    P6 is generated from the refined P5 output using a stride-2 3x3
    convolution and GroupNorm.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (224, 224, 224),
        out_channels: int = 256,
        num_outs: int = 4,
        num_groups: int = 32,
        validate_inputs: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = tuple(int(x) for x in in_channels)
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs)
        self.num_groups = int(num_groups)
        self.validate_inputs = bool(validate_inputs)

        if len(self.in_channels) == 0:
            raise ValueError("in_channels must not be empty.")

        if self.num_outs < len(self.in_channels):
            raise ValueError(
                f"num_outs={self.num_outs} cannot be smaller than "
                f"the number of inputs={len(self.in_channels)}."
            )

        if self.out_channels % self.num_groups != 0:
            raise ValueError(
                f"out_channels={self.out_channels} must be divisible by "
                f"num_groups={self.num_groups}."
            )

        self.input_blocks = nn.ModuleList(
            [
                ResidualProjectionBlock(
                    in_channels=input_channels,
                    out_channels=self.out_channels,
                    num_groups=self.num_groups,
                )
                for input_channels in self.in_channels
            ]
        )

        extra_level_count = self.num_outs - len(self.in_channels)

        self.extra_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=self.out_channels,
                        out_channels=self.out_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(
                        num_groups=self.num_groups,
                        num_channels=self.out_channels,
                    ),
                )
                for _ in range(extra_level_count)
            ]
        )

    def init_weights(self) -> None:
        """Initialize bridge while keeping residual branches exactly zero."""

        if self._is_init:
            return

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Zero-initialize the final normalization of every refinement branch.
        # Therefore each block initially behaves exactly as its 1x1 projection.
        for block in self.input_blocks:
            final_norm = block.refinement[-1]

            if not isinstance(final_norm, nn.GroupNorm):
                raise TypeError(
                    "The final refinement layer must be GroupNorm."
                )

            nn.init.zeros_(final_norm.weight)
            nn.init.zeros_(final_norm.bias)

        self._is_init = True

    def _check_inputs(
        self,
        inputs: tuple[Tensor, ...],
    ) -> None:
        if len(inputs) != len(self.in_channels):
            raise RuntimeError(
                f"Expected {len(self.in_channels)} input features, "
                f"but received {len(inputs)}."
            )

        previous_height = None
        previous_width = None

        for level, (feature, expected_channels) in enumerate(
            zip(inputs, self.in_channels)
        ):
            if not torch.is_tensor(feature):
                raise TypeError(
                    f"Input level {level} is not a Tensor."
                )

            if feature.ndim != 4:
                raise RuntimeError(
                    f"Input level {level} must be BCHW, but its shape is "
                    f"{tuple(feature.shape)}."
                )

            if feature.shape[1] != expected_channels:
                raise RuntimeError(
                    f"Input level {level} expected {expected_channels} "
                    f"channels, but received {feature.shape[1]}."
                )

            if not torch.isfinite(feature).all():
                raise FloatingPointError(
                    f"Input level {level} contains NaN or Inf."
                )

            current_height, current_width = feature.shape[-2:]

            if previous_height is not None:
                expected_height = (previous_height + 1) // 2
                expected_width = (previous_width + 1) // 2

                if (current_height, current_width) != (
                    expected_height,
                    expected_width,
                ):
                    raise RuntimeError(
                        f"Input level {level} has shape "
                        f"{(current_height, current_width)}, but approximately "
                        f"half of the previous level should be "
                        f"{(expected_height, expected_width)}."
                    )

            previous_height = current_height
            previous_width = current_width

    def forward(
        self,
        inputs: Sequence[Tensor],
    ) -> tuple[Tensor, ...]:
        input_tuple = tuple(inputs)

        if self.validate_inputs:
            self._check_inputs(input_tuple)

        outputs = [
            block(feature)
            for block, feature in zip(
                self.input_blocks,
                input_tuple,
            )
        ]

        for extra_block in self.extra_blocks:
            outputs.append(
                extra_block(outputs[-1])
            )

        return tuple(outputs)