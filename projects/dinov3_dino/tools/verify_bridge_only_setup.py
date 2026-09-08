from __future__ import annotations

import os
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.dist import get_rank, init_dist
from mmengine.optim import build_optim_wrapper
from mmengine.registry import (
    MODEL_WRAPPERS,
    init_default_scope,
)
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS


EXPECTED_BRIDGE_PARAMETERS = 2_534_912


def extract_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        return checkpoint["state_dict"]

    if "model" in checkpoint:
        return checkpoint["model"]

    return checkpoint


def main() -> None:
    init_dist(
        launcher="pytorch",
        backend="nccl",
    )

    rank = get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    config_path = Path(
        "projects/dinov3_dino/configs/"
        "dino4_vits16_sta_align_640_1e.py"
    ).resolve()

    cfg = Config.fromfile(config_path)

    import_modules_from_strings(
        **cfg.custom_imports
    )

    init_default_scope(
        cfg.get("default_scope", "mmdet")
    )

    model = MODELS.build(cfg.model)
    model.init_weights()

    if type(model).__name__ != "DINO":
        raise TypeError(
            f"Expected original DINO, got {type(model)!r}."
        )

    if type(model.bbox_head).__name__ != "DINOHead":
        raise TypeError(
            f"Expected original DINOHead, got "
            f"{type(model.bbox_head)!r}."
        )

    checkpoint_path = Path(
        cfg.load_from
    ).expanduser().resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(checkpoint)

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            f"Missing keys: {incompatible.missing_keys}"
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys: {incompatible.unexpected_keys}"
        )

    model = model.cuda(local_rank)

    wrapper_cfg = dict(
        cfg.model_wrapper_cfg
    )

    wrapper_type = wrapper_cfg.pop("type")

    wrapped_model = MODEL_WRAPPERS.build(
        dict(
            type=wrapper_type,
            module=model,
            device_ids=[local_rank],
            **wrapper_cfg,
        )
    )

    base_model = wrapped_model.module

    trainable = {
        name: parameter
        for name, parameter in base_model.named_parameters()
        if parameter.requires_grad
    }

    frozen = {
        name: parameter
        for name, parameter in base_model.named_parameters()
        if not parameter.requires_grad
    }

    invalid_trainable = [
        name
        for name in trainable
        if not name.startswith("neck.")
    ]

    if invalid_trainable:
        raise RuntimeError(
            "Non-neck parameters are trainable:\n"
            + "\n".join(invalid_trainable)
        )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable.values()
    )

    if trainable_count != EXPECTED_BRIDGE_PARAMETERS:
        raise RuntimeError(
            f"Expected {EXPECTED_BRIDGE_PARAMETERS:,} trainable "
            f"parameters, got {trainable_count:,}."
        )

    optim_wrapper = build_optim_wrapper(
        wrapped_model,
        cfg.optim_wrapper,
    )

    optimizer_parameter_ids = {
        id(parameter)
        for group in optim_wrapper.optimizer.param_groups
        for parameter in group["params"]
    }

    trainable_parameter_ids = {
        id(parameter)
        for parameter in trainable.values()
    }

    if optimizer_parameter_ids != trainable_parameter_ids:
        missing_from_optimizer = (
            trainable_parameter_ids
            - optimizer_parameter_ids
        )

        unexpected_in_optimizer = (
            optimizer_parameter_ids
            - trainable_parameter_ids
        )

        raise RuntimeError(
            "Optimizer parameter mismatch:\n"
            f"missing={len(missing_from_optimizer)}, "
            f"unexpected={len(unexpected_in_optimizer)}"
        )

    required_frozen_prefixes = (
        "backbone.",
        "encoder.",
        "decoder.",
        "bbox_head.",
    )

    for prefix in required_frozen_prefixes:
        bad_names = [
            name
            for name in trainable
            if name.startswith(prefix)
        ]

        if bad_names:
            raise RuntimeError(
                f"Parameters under {prefix} are trainable:\n"
                + "\n".join(bad_names)
            )

    if rank == 0:
        total_count = sum(
            parameter.numel()
            for parameter in base_model.parameters()
        )

        frozen_count = sum(
            parameter.numel()
            for parameter in frozen.values()
        )

        print("=" * 72)
        print("Original detector:", type(base_model))
        print("Original head:", type(base_model.bbox_head))
        print("Backbone:", type(base_model.backbone))
        print("Bridge:", type(base_model.neck))
        print()
        print("Total parameters:", f"{total_count:,}")
        print("Trainable parameters:", f"{trainable_count:,}")
        print("Frozen parameters:", f"{frozen_count:,}")
        print(
            "Optimizer parameter count:",
            f"{sum(p.numel() for group in optim_wrapper.optimizer.param_groups for p in group['params']):,}",
        )
        print()
        print("Trainable names:")

        for name in trainable:
            print(" ", name)

        print()
        print("Verified:")
        print("  model is original DINO")
        print("  bbox_head is original DINOHead")
        print("  strict=True checkpoint loading passed")
        print("  only neck.* requires gradients")
        print("  optimizer contains exactly neck.*")
        print("=" * 72)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()