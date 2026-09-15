from __future__ import annotations

from mmengine.logging import print_log
from mmengine.model import MMDistributedDataParallel
from mmengine.registry import MODEL_WRAPPERS


@MODEL_WRAPPERS.register_module()
class NeckOnlyMMDistributedDataParallel(
    MMDistributedDataParallel
):


    def __init__(
        self,
        module,
        trainable_prefix: str = "neck.",
        expected_trainable_params: int | None = 2_534_912,
        **kwargs,
    ) -> None:
        if type(module).__name__ != "DINO":
            raise TypeError(
                "Stage A must use the original MMDetection DINO detector, "
                f"but received {type(module)!r}."
            )

        if not hasattr(module, "bbox_head"):
            raise AttributeError(
                "The DINO detector does not contain bbox_head."
            )

        if type(module.bbox_head).__name__ != "DINOHead":
            raise TypeError(
                "Stage A must use the original DINOHead, "
                f"but received {type(module.bbox_head)!r}."
            )

        if not hasattr(module, "neck"):
            raise AttributeError(
                "The DINO detector does not contain a neck."
            )

        if type(module.neck).__name__ != "ResidualMultiScaleBridge":
            raise TypeError(
                "Stage A expects ResidualMultiScaleBridge, "
                f"but received {type(module.neck)!r}."
            )

        # Freeze every parameter in the original model.
        for parameter in module.parameters():
            parameter.requires_grad_(False)

        # Unfreeze only the new multiscale bridge.
        for name, parameter in module.named_parameters():
            if name.startswith(trainable_prefix):
                parameter.requires_grad_(True)

        trainable_parameters = [
            (name, parameter)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        ]

        if not trainable_parameters:
            raise RuntimeError(
                "No trainable neck parameters were found."
            )

        invalid_names = [
            name
            for name, _ in trainable_parameters
            if not name.startswith(trainable_prefix)
        ]

        if invalid_names:
            raise RuntimeError(
                "Non-neck parameters were accidentally made trainable:\n"
                + "\n".join(invalid_names)
            )

        trainable_parameter_count = sum(
            parameter.numel()
            for _, parameter in trainable_parameters
        )

        if (
            expected_trainable_params is not None
            and trainable_parameter_count
            != expected_trainable_params
        ):
            raise RuntimeError(
                "Unexpected trainable parameter count. "
                f"Expected {expected_trainable_params:,}, "
                f"but found {trainable_parameter_count:,}."
            )

        # DDP is constructed only after requires_grad has been set correctly.
        super().__init__(
            module=module,
            **kwargs,
        )

        self.trainable_prefix = trainable_prefix
        self.expected_trainable_params = (
            expected_trainable_params
        )

        print_log(
            "Neck-only DDP setup passed: "
            f"{trainable_parameter_count:,} trainable parameters, "
            "all under neck.*",
            logger="current",
        )