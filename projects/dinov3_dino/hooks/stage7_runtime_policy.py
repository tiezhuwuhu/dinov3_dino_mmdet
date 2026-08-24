from __future__ import annotations

import torch
from torch import nn

from mmengine.hooks import Hook
from mmengine.logging import print_log

from mmdet.registry import HOOKS


@HOOKS.register_module()
class Stage7RuntimePolicyHook(Hook):
    """Runtime policy for stage-7 end-to-end FP32 training."""

    priority = "VERY_HIGH"

    def __init__(
        self,
        freeze_dinostas_syncbn_stats: bool = True,
        disable_tf32: bool = True,
    ) -> None:
        self.freeze_dinostas_syncbn_stats = bool(
            freeze_dinostas_syncbn_stats
        )

        self.disable_tf32 = bool(
            disable_tf32
        )

        self._syncbn_count = None

    @staticmethod
    def _unwrap_model(model):
        while hasattr(model, "module"):
            model = model.module

        return model

    @staticmethod
    def _validate_model(model) -> None:
        if type(model).__name__ != "DINO":
            raise TypeError(
                "Stage 7 must use the original "
                "MMDetection DINO detector, but got "
                f"{type(model)!r}."
            )

        if not hasattr(model, "bbox_head"):
            raise AttributeError(
                "Stage-7 model has no bbox_head."
            )

        if (
            type(model.bbox_head).__name__
            != "DINOHead"
        ):
            raise TypeError(
                "Stage 7 must use the original "
                "DINOHead, but got "
                f"{type(model.bbox_head)!r}."
            )

        if not hasattr(model, "backbone"):
            raise AttributeError(
                "Stage-7 model has no backbone."
            )

        if not hasattr(
            model.backbone,
            "dinostas",
        ):
            raise AttributeError(
                "Stage-7 backbone has no dinostas."
            )

        if not hasattr(model, "neck"):
            raise AttributeError(
                "Stage-7 model has no neck."
            )

        if (
            type(model.neck).__name__
            != "ResidualMultiScaleBridge"
        ):
            raise TypeError(
                "Stage 7 requires "
                "ResidualMultiScaleBridge, but got "
                f"{type(model.neck)!r}."
            )

    @staticmethod
    def _verify_all_parameters_trainable(
        model,
    ) -> None:
        frozen_names = [
            name
            for name, parameter
            in model.named_parameters()
            if not parameter.requires_grad
        ]

        if frozen_names:
            raise RuntimeError(
                "Stage 7 contains frozen parameters:\n"
                + "\n".join(frozen_names)
            )

    def _set_syncbn_eval(
        self,
        model,
    ) -> int:
        if not self.freeze_dinostas_syncbn_stats:
            return 0

        dinostas = model.backbone.dinostas

        syncbn_count = 0

        for module in dinostas.modules():
            if isinstance(
                module,
                nn.SyncBatchNorm,
            ):
                module.eval()
                syncbn_count += 1

                # eval() must not freeze affine parameters.
                for parameter in module.parameters(
                    recurse=False
                ):
                    if not parameter.requires_grad:
                        raise RuntimeError(
                            "A DINOSTAs SyncBatchNorm "
                            "affine parameter is frozen."
                        )

        if syncbn_count == 0:
            raise RuntimeError(
                "No SyncBatchNorm modules were found "
                "inside DINOSTAs."
            )

        return syncbn_count

    def before_run(
        self,
        runner,
    ) -> None:
        if self.disable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = (
                False
            )

            torch.backends.cudnn.allow_tf32 = (
                False
            )

            torch.set_float32_matmul_precision(
                "highest"
            )

        print_log(
            "Stage-7 precision policy: "
            "FP32 OptimWrapper required; "
            "CUDA matmul TF32 disabled; "
            "cuDNN TF32 disabled.",
            logger="current",
        )

    def before_train(
        self,
        runner,
    ) -> None:
        model = self._unwrap_model(
            runner.model
        )

        self._validate_model(model)

        self._verify_all_parameters_trainable(
            model
        )

        self._syncbn_count = (
            self._set_syncbn_eval(model)
        )

        total_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        trainable_parameters = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )

        if total_parameters != trainable_parameters:
            raise RuntimeError(
                "Stage-7 total/trainable parameter "
                "counts are inconsistent."
            )

        print_log(
            "Stage-7 runtime verification passed: "
            f"{trainable_parameters:,} trainable "
            "parameters; "
            f"{self._syncbn_count} DINOSTAs "
            "SyncBatchNorm modules keep fixed "
            "running statistics.",
            logger="current",
        )

    def before_train_iter(
        self,
        runner,
        batch_idx: int,
        data_batch=None,
    ) -> None:
        # MMEngine calls model.train() at the start of every epoch.
        # Therefore SyncBN must be restored to eval mode after that
        # call and before the next forward pass.
        model = self._unwrap_model(
            runner.model
        )

        self._set_syncbn_eval(model)