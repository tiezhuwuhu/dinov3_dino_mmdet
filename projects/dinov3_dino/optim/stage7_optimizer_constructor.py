from __future__ import annotations

import re
from typing import List, Optional, Union

from torch import nn

from mmengine.logging import print_log
from mmengine.optim import DefaultOptimWrapperConstructor
from mmengine.registry import OPTIM_WRAPPER_CONSTRUCTORS


@OPTIM_WRAPPER_CONSTRUCTORS.register_module()
class Stage7OptimWrapperConstructor(
    DefaultOptimWrapperConstructor
):
    """Optimizer constructor for stage-7 end-to-end DINO fine-tuning.

    Learning-rate policy, assuming base LR = 2e-4:

    - ViT blocks 0-3:        2e-6
    - ViT blocks 4-7:        5e-6
    - ViT blocks 8-11:       1e-5
    - ViT final norm:        5e-6
    - Other ViT parameters:  2e-6
    - Lightly adapter:       5e-5
    - Sampling/reference:    2e-5
    - Bridge and detector:   2e-4

    Weight decay is disabled for:

    - normalization parameters;
    - bias parameters;
    - one-dimensional parameters;
    - class, register, mask and position tokens.
    """

    vit_prefix = (
        "backbone.dinostas._model_wrapper._model."
    )

    adapter_prefix = "backbone.dinostas."

    geometry_patterns = (
        "sampling_offsets",
        "reference_points",
    )

    no_decay_name_patterns = (
        "cls_token",
        "register_tokens",
        "mask_token",
        "pos_embed",
    )

    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.GroupNorm,
        nn.LayerNorm,
    )

    def _get_lr_mult(
        self,
        full_name: str,
    ) -> float:
        cfg = self.paramwise_cfg

        geometry_lr_mult = float(
            cfg.get("geometry_lr_mult", 0.1)
        )

        adapter_lr_mult = float(
            cfg.get("adapter_lr_mult", 0.25)
        )

        vit_early_lr_mult = float(
            cfg.get("vit_early_lr_mult", 0.01)
        )

        vit_middle_lr_mult = float(
            cfg.get("vit_middle_lr_mult", 0.025)
        )

        vit_late_lr_mult = float(
            cfg.get("vit_late_lr_mult", 0.05)
        )

        vit_final_norm_lr_mult = float(
            cfg.get("vit_final_norm_lr_mult", 0.025)
        )

        vit_other_lr_mult = float(
            cfg.get("vit_other_lr_mult", 0.01)
        )

        # Deformable attention geometry uses base LR x 0.1.
        if any(
            pattern in full_name
            for pattern in self.geometry_patterns
        ):
            return geometry_lr_mult

        # DINOv3 ViT parameters.
        if full_name.startswith(self.vit_prefix):
            block_match = re.search(
                r"\.blocks\.(\d+)\.",
                full_name,
            )

            if block_match is not None:
                block_index = int(
                    block_match.group(1)
                )

                if 0 <= block_index <= 3:
                    return vit_early_lr_mult

                if 4 <= block_index <= 7:
                    return vit_middle_lr_mult

                if 8 <= block_index <= 11:
                    return vit_late_lr_mult

                raise RuntimeError(
                    "Unexpected DINOv3 ViT block index "
                    f"in parameter: {full_name}"
                )

            final_norm_prefix = (
                self.vit_prefix + "norm."
            )

            if full_name.startswith(
                final_norm_prefix
            ):
                return vit_final_norm_lr_mult

            return vit_other_lr_mult

        # Every other trainable DINOSTAs parameter belongs to the
        # Lightly feature adapter.
        if full_name.startswith(
            self.adapter_prefix
        ):
            return adapter_lr_mult

        # Residual bridge and MMDetection DINO detector.
        return 1.0

    def _should_disable_weight_decay(
        self,
        module: nn.Module,
        parameter_name: str,
        full_name: str,
        parameter,
    ) -> bool:
        if isinstance(module, self.norm_types):
            return True

        if parameter_name == "bias":
            return True

        if parameter.ndim <= 1:
            return True

        if any(
            pattern in full_name
            for pattern in self.no_decay_name_patterns
        ):
            return True

        return False

    def add_params(
        self,
        params: List[dict],
        module: nn.Module,
        prefix: str = "",
        is_dcn_module: Optional[
            Union[int, float]
        ] = None,
    ) -> None:
        if self.base_lr is None:
            raise ValueError(
                "Stage7OptimWrapperConstructor requires "
                "an explicit optimizer base learning rate."
            )

        if self.base_wd is None:
            raise ValueError(
                "Stage7OptimWrapperConstructor requires "
                "an explicit optimizer weight decay."
            )

        for parameter_name, parameter in (
            module.named_parameters(
                recurse=False
            )
        ):
            full_name = (
                f"{prefix}.{parameter_name}"
                if prefix
                else parameter_name
            )

            if not parameter.requires_grad:
                print_log(
                    f"{full_name} is skipped because "
                    "requires_grad=False.",
                    logger="current",
                )
                continue

            candidate_group = {
                "params": [parameter],
            }

            if self._is_in(
                candidate_group,
                params,
            ):
                print_log(
                    f"{full_name} is a duplicate "
                    "parameter and was skipped.",
                    logger="current",
                )
                continue

            lr_mult = self._get_lr_mult(
                full_name
            )

            disable_weight_decay = (
                self._should_disable_weight_decay(
                    module=module,
                    parameter_name=parameter_name,
                    full_name=full_name,
                    parameter=parameter,
                )
            )

            weight_decay = (
                0.0
                if disable_weight_decay
                else self.base_wd
            )

            parameter_group = {
                "params": [parameter],
                "lr": self.base_lr * lr_mult,
                "weight_decay": weight_decay,
            }

            params.append(parameter_group)

        for child_name, child_module in (
            module.named_children()
        ):
            child_prefix = (
                f"{prefix}.{child_name}"
                if prefix
                else child_name
            )

            self.add_params(
                params=params,
                module=child_module,
                prefix=child_prefix,
                is_dcn_module=None,
            )