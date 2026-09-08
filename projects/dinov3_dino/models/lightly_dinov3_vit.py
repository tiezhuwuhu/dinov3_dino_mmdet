from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from mmdet.registry import MODELS


@MODELS.register_module()
class LightlyDINOv3ViTS16(nn.Module):
    """DINOv3 ViT-S/16 built from Lightly's exact implementation.

    This module only creates the model architecture. Its COCO-finetuned
    weights are loaded later through the merged MMEngine checkpoint.

    Args:
        out_indices:
            ViT block indices returned as spatial feature maps.
            Lightly ViT-S LTDETR uses blocks 5, 8 and 11.
        frozen:
            Freeze the complete ViT during alignment training.
    """

    def __init__(
        self,
        out_indices: Sequence[int] = (5, 8, 11),
        frozen: bool = False,
    ) -> None:
        super().__init__()

        from lightly_train._models.dinov3.dinov3_package import (
            DINOV3_PACKAGE,
        )

        self.out_indices = tuple(int(i) for i in out_indices)
        self.frozen = bool(frozen)

        if not self.out_indices:
            raise ValueError("out_indices cannot be empty.")

        for index in self.out_indices:
            if index < 0 or index >= 12:
                raise ValueError(
                    f"Invalid ViT-S block index: {index}"
                )

        # Build the exact architecture without loading Meta weights.
        # The extracted Lightly COCO weights are loaded by MMEngine.
        self.vit = DINOV3_PACKAGE.get_model(
            model_name="vits16",
            load_weights=False,
        )

        if int(self.vit.embed_dim) != 384:
            raise RuntimeError(
                f"Expected embed_dim=384, got {self.vit.embed_dim}"
            )

        if int(self.vit.patch_size) != 16:
            raise RuntimeError(
                f"Expected patch_size=16, got {self.vit.patch_size}"
            )

        if self.frozen:
            self._freeze()

    def _freeze(self) -> None:
        self.vit.eval()

        for parameter in self.vit.parameters():
            parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.frozen:
            self._freeze()

        return self

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, ...]:
        height, width = x.shape[-2:]

        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                "Input height and width must be divisible by 16. "
                f"Got {(height, width)}."
            )

        outputs = self.vit.get_intermediate_layers(
            x,
            n=list(self.out_indices),
            reshape=True,
            return_class_token=True,
        )

        spatial_features: list[Tensor] = []

        for output in outputs:
            # With return_class_token=True:
            # output = (patch_feature_map, cls_token)
            if not isinstance(output, (tuple, list)):
                raise TypeError(
                    "Expected (feature, cls_token), "
                    f"but got {type(output)}"
                )

            feature = output[0]

            if feature.ndim != 4:
                raise RuntimeError(
                    "Expected a BCHW feature map, "
                    f"got shape {tuple(feature.shape)}"
                )

            spatial_features.append(feature)

        return tuple(spatial_features)