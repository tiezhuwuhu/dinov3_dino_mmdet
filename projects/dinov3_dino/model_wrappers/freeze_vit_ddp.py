from __future__ import annotations

from torch import nn

from mmengine.logging import print_log
from mmengine.model import MMDistributedDataParallel
from mmengine.registry import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class FreezeViTMMDistributedDataParallel(
    MMDistributedDataParallel
):
    """Freeze only the DINOv3 ViT inside Lightly DINOSTAs.

    The detector remains the original MMDetection DINO and DINOHead.

    Frozen:

        backbone.dinostas._model_wrapper._model.*

    Trainable:

        backbone.dinostas.sta.*
        backbone.dinostas.convs.*
        backbone.dinostas.norms.*
        neck.*
        all MMDetection DINO detector parameters

    SyncBatchNorm running statistics inside DINOSTAs remain frozen, while
    their learnable weight and bias parameters remain trainable.
    """

    def __init__(
        self,
        module,
        freeze_adapter_running_stats: bool = True,
        **kwargs,
    ) -> None:
        if type(module).__name__ != "DINO":
            raise TypeError(
                "Stage 6 must use the original MMDetection DINO, "
                f"but received {type(module)!r}."
            )

        if not hasattr(module, "bbox_head"):
            raise AttributeError(
                "The detector has no bbox_head."
            )

        if type(module.bbox_head).__name__ != "DINOHead":
            raise TypeError(
                "Stage 6 must use the original DINOHead, "
                f"but received {type(module.bbox_head)!r}."
            )

        if not hasattr(module, "backbone"):
            raise AttributeError(
                "The detector has no backbone."
            )

        if not hasattr(module.backbone, "dinostas"):
            raise AttributeError(
                "The backbone wrapper has no dinostas module."
            )

        dinostas = module.backbone.dinostas

        if not hasattr(dinostas, "_model_wrapper"):
            raise AttributeError(
                "DINOSTAs has no _model_wrapper."
            )

        if not hasattr(dinostas._model_wrapper, "_model"):
            raise AttributeError(
                "DINOSTAs._model_wrapper has no _model."
            )

        if not hasattr(dinostas, "sta"):
            raise AttributeError(
                "DINOSTAs has no SpatialPriorModulev2 module."
            )

        if not hasattr(dinostas, "convs"):
            raise AttributeError(
                "DINOSTAs has no fusion convs."
            )

        if not hasattr(dinostas, "norms"):
            raise AttributeError(
                "DINOSTAs has no fusion norms."
            )

        if not hasattr(module, "neck"):
            raise AttributeError(
                "The detector has no neck."
            )

        if type(module.neck).__name__ != "ResidualMultiScaleBridge":
            raise TypeError(
                "Stage 6 expects ResidualMultiScaleBridge, "
                f"but received {type(module.neck)!r}."
            )

        self.freeze_adapter_running_stats = bool(
            freeze_adapter_running_stats
        )

        # Reset every parameter to trainable first. The stage-5 checkpoint
        # does not store requires_grad flags, but this makes the policy
        # independent of any previous configuration.
        for parameter in module.parameters():
            parameter.requires_grad_(True)

        # Freeze only the pretrained DINOv3 ViT-S/16.
        vit = dinostas._model_wrapper._model

        for parameter in vit.parameters():
            parameter.requires_grad_(False)

        self._verify_parameter_policy(module)

        # DDP must be constructed after requires_grad is finalized.
        super().__init__(
            module=module,
            **kwargs,
        )

        self._apply_module_modes()

        total_parameters = sum(
            parameter.numel()
            for parameter in self.module.parameters()
        )

        frozen_parameters = sum(
            parameter.numel()
            for parameter in self.module.parameters()
            if not parameter.requires_grad
        )

        trainable_parameters = total_parameters - frozen_parameters

        print_log(
            "Stage-6 freeze policy passed: "
            f"trainable={trainable_parameters:,}, "
            f"frozen_ViT={frozen_parameters:,}. "
            "Original DINO and DINOHead are unchanged.",
            logger="current",
        )

    @staticmethod
    def _verify_parameter_policy(module) -> None:
        frozen_names = [
            name
            for name, parameter in module.named_parameters()
            if not parameter.requires_grad
        ]

        trainable_names = [
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]

        expected_vit_prefix = (
            "backbone.dinostas._model_wrapper._model."
        )

        if not frozen_names:
            raise RuntimeError(
                "No frozen ViT parameters were found."
            )

        invalid_frozen_names = [
            name
            for name in frozen_names
            if not name.startswith(expected_vit_prefix)
        ]

        if invalid_frozen_names:
            raise RuntimeError(
                "Parameters outside the ViT were frozen unexpectedly:\n"
                + "\n".join(invalid_frozen_names)
            )

        accidentally_trainable_vit = [
            name
            for name in trainable_names
            if name.startswith(expected_vit_prefix)
        ]

        if accidentally_trainable_vit:
            raise RuntimeError(
                "Some ViT parameters remain trainable:\n"
                + "\n".join(accidentally_trainable_vit)
            )

        required_trainable_prefixes = (
            "backbone.dinostas.sta.",
            "backbone.dinostas.convs.",
            "backbone.dinostas.norms.",
            "neck.",
            "encoder.",
            "decoder.",
            "bbox_head.",
        )

        for prefix in required_trainable_prefixes:
            matching_names = [
                name
                for name in trainable_names
                if name.startswith(prefix)
            ]

            if not matching_names:
                raise RuntimeError(
                    f"No trainable parameters were found under {prefix}"
                )

        # Any trainable parameter inside the backbone must belong to the
        # Lightly feature adapter, never to the ViT.
        allowed_trainable_backbone_prefixes = (
            "backbone.dinostas.sta.",
            "backbone.dinostas.convs.",
            "backbone.dinostas.norms.",
        )

        invalid_trainable_backbone = [
            name
            for name in trainable_names
            if name.startswith("backbone.")
            and not name.startswith(
                allowed_trainable_backbone_prefixes
            )
        ]

        if invalid_trainable_backbone:
            raise RuntimeError(
                "Unexpected trainable backbone parameters:\n"
                + "\n".join(invalid_trainable_backbone)
            )

    def _apply_module_modes(self) -> None:
        """Keep ViT and adapter BN statistics in evaluation mode."""

        dinostas = self.module.backbone.dinostas
        vit = dinostas._model_wrapper._model

        # The frozen ViT must remain in eval mode.
        vit.eval()

        if self.freeze_adapter_running_stats:
            # SyncBatchNorm affine parameters remain trainable because
            # eval() changes only module behavior, not requires_grad.
            for child in dinostas.modules():
                if isinstance(child, nn.SyncBatchNorm):
                    child.eval()

    def train(
        self,
        mode: bool = True,
    ):
        super().train(mode)

        if mode:
            self._apply_module_modes()

        return self