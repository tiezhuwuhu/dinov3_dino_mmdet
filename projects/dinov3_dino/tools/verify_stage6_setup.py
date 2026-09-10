from __future__ import annotations

import os
from collections import defaultdict
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


VIT_PREFIX = (
    "backbone.dinostas._model_wrapper._model."
)

ADAPTER_PREFIXES = (
    "backbone.dinostas.sta.",
    "backbone.dinostas.convs.",
    "backbone.dinostas.norms.",
)


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
        "dino4_vits16_sta_adapter_detector_640_2e.py"
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

    invalid_frozen = [
        name
        for name in frozen
        if not name.startswith(VIT_PREFIX)
    ]

    if invalid_frozen:
        raise RuntimeError(
            "Non-ViT parameters were frozen:\n"
            + "\n".join(invalid_frozen)
        )

    trainable_vit = [
        name
        for name in trainable
        if name.startswith(VIT_PREFIX)
    ]

    if trainable_vit:
        raise RuntimeError(
            "ViT parameters are trainable:\n"
            + "\n".join(trainable_vit)
        )

    required_trainable_prefixes = (
        *ADAPTER_PREFIXES,
        "neck.",
        "encoder.",
        "decoder.",
        "bbox_head.",
    )

    for prefix in required_trainable_prefixes:
        matches = [
            name
            for name in trainable
            if name.startswith(prefix)
        ]

        if not matches:
            raise RuntimeError(
                f"No trainable parameters under {prefix}"
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
        missing = (
            trainable_parameter_ids
            - optimizer_parameter_ids
        )

        unexpected = (
            optimizer_parameter_ids
            - trainable_parameter_ids
        )

        raise RuntimeError(
            "Optimizer parameter mismatch: "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    parameter_name_by_id = {
        id(parameter): name
        for name, parameter in base_model.named_parameters()
    }

    lr_summary = defaultdict(
        lambda: {
            "parameters": 0,
            "tensors": 0,
            "examples": [],
        }
    )

    for group in optim_wrapper.optimizer.param_groups:
        lr = float(group["lr"])

        for parameter in group["params"]:
            name = parameter_name_by_id[id(parameter)]

            lr_summary[lr]["parameters"] += parameter.numel()
            lr_summary[lr]["tensors"] += 1

            if len(lr_summary[lr]["examples"]) < 8:
                lr_summary[lr]["examples"].append(name)

    expected_lrs = {
        1.0e-5,
        5.0e-5,
        1.0e-4,
        2.0e-4,
    }

    actual_lrs = {
        round(lr, 10)
        for lr in lr_summary
    }

    if actual_lrs != expected_lrs:
        raise RuntimeError(
            f"Unexpected optimizer LRs: "
            f"actual={sorted(actual_lrs)}, "
            f"expected={sorted(expected_lrs)}"
        )

    # Verify training/eval modes.
    wrapped_model.train()

    vit = (
        base_model.backbone.dinostas
        ._model_wrapper._model
    )

    if vit.training:
        raise RuntimeError(
            "Frozen ViT is unexpectedly in train mode."
        )

    sync_bn_modules = [
        module
        for module in base_model.backbone.dinostas.modules()
        if isinstance(module, torch.nn.SyncBatchNorm)
    ]

    if not sync_bn_modules:
        raise RuntimeError(
            "No DINOSTAs SyncBatchNorm modules were found."
        )

    if any(module.training for module in sync_bn_modules):
        raise RuntimeError(
            "A DINOSTAs SyncBatchNorm is unexpectedly in train mode."
        )

    if rank == 0:
        total_count = sum(
            parameter.numel()
            for parameter in base_model.parameters()
        )

        trainable_count = sum(
            parameter.numel()
            for parameter in trainable.values()
        )

        frozen_count = sum(
            parameter.numel()
            for parameter in frozen.values()
        )

        print("=" * 80)
        print("Detector:", type(base_model))
        print("Head:", type(base_model.bbox_head))
        print("Backbone:", type(base_model.backbone))
        print("Bridge:", type(base_model.neck))
        print()
        print("Total parameters:", f"{total_count:,}")
        print("Trainable parameters:", f"{trainable_count:,}")
        print("Frozen ViT parameters:", f"{frozen_count:,}")
        print()
        print("Optimizer LR groups:")

        for lr in sorted(lr_summary):
            information = lr_summary[lr]

            print(
                f"\n  lr={lr:.8f}: "
                f"{information['parameters']:,} parameters, "
                f"{information['tensors']} tensors"
            )

            for name in information["examples"]:
                print("   ", name)

        print()
        print("Verified:")
        print("  original DINO detector")
        print("  original DINOHead")
        print("  stage-5 checkpoint strict=True")
        print("  only DINOv3 ViT is frozen")
        print("  Lightly STA/convs/norms are trainable")
        print("  residual bridge is trainable")
        print("  complete DINO detector is trainable")
        print("  ViT remains in eval mode")
        print("  adapter SyncBN running statistics are frozen")
        print("  optimizer contains every trainable parameter")
        print("=" * 80)

    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()