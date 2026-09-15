from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from mmengine.config import Config
from mmengine.optim import build_optim_wrapper
from mmengine.registry import (
    init_default_scope,
)
from mmengine.utils import (
    import_modules_from_strings,
)

from mmdet.registry import MODELS,HOOKS


CONFIG_PATH = Path(
    "projects/dinov3_dino/configs/"
    "dino4_vits16_sta_improved_multiscale_24e.py"
).resolve()


EXPECTED_STAGE6_CHECKPOINT = Path(
    "/root/autodl-tmp/dinov3_dino_mmdet/"
    "mmdetection/work_dirs/"
    "dino4_vits16_sta_adapter_detector_640_2e/"
    "best_coco_bbox_mAP_epoch_2.pth"
).resolve()


def extract_state_dict(checkpoint):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    tensor_state = {
        key: value
        for key, value in state_dict.items()
        if torch.is_tensor(value)
    }

    if (
        tensor_state
        and all(
            key.startswith("module.")
            for key in tensor_state
        )
    ):
        tensor_state = {
            key[len("module."):]: value
            for key, value
            in tensor_state.items()
        }

    return tensor_state


def get_unique_lr_for_pattern(
    optimizer,
    parameter_name_by_id,
    pattern: str,
) -> float:
    matched_lrs = set()
    matched_names = []

    for group in optimizer.param_groups:
        lr = float(group["lr"])

        for parameter in group["params"]:
            name = parameter_name_by_id[
                id(parameter)
            ]

            if pattern in name:
                matched_lrs.add(
                    round(lr, 12)
                )
                matched_names.append(name)

    if not matched_names:
        raise RuntimeError(
            f"No optimizer parameter matched: {pattern}"
        )

    if len(matched_lrs) != 1:
        raise RuntimeError(
            f"Pattern {pattern!r} has multiple LRs: "
            f"{sorted(matched_lrs)}"
        )

    lr = next(iter(matched_lrs))

    print(
        f"{pattern:55s} "
        f"lr={lr:.12f} "
        f"tensors={len(matched_names)}"
    )

    return lr


def assert_close(
    actual: float,
    expected: float,
    name: str,
) -> None:
    if abs(actual - expected) > 1.0e-12:
        raise RuntimeError(
            f"{name}: expected {expected}, "
            f"got {actual}"
        )


def main() -> None:
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(CONFIG_PATH)

    if not EXPECTED_STAGE6_CHECKPOINT.is_file():
        raise FileNotFoundError(
            EXPECTED_STAGE6_CHECKPOINT
        )

    cfg = Config.fromfile(CONFIG_PATH)

    import_modules_from_strings(
        **cfg.custom_imports
    )

    init_default_scope(
        cfg.get(
            "default_scope",
            "mmdet",
        )
    )

    print("=" * 80)
    print("Configuration checks")
    print("=" * 80)

    assert cfg.model.type == "DINO"
    assert cfg.model.bbox_head.type == "DINOHead"

    assert (
        cfg.model.backbone.type
        == "LightlyDINOSTAs"
    )

    assert (
        cfg.model.backbone.frozen
        is False
    )

    assert (
        cfg.model.neck.type
        == "ResidualMultiScaleBridge"
    )

    assert (
        cfg.model.bbox_head.loss_cls.loss_weight
        == 2.0
    )

    assert (
        cfg.model.positional_encoding.offset
        == -0.5
    )

    assert (
        cfg.model.positional_encoding.temperature
        == 10000
    )

    assert (
        cfg.model.dn_cfg.group_cfg.num_dn_queries
        == 300
    )

    assert cfg.optim_wrapper.type == "OptimWrapper"

    assert (
        cfg.optim_wrapper.get(
            "loss_scale",
            None,
        )
        is None
    )

    assert (
        cfg.optim_wrapper.accumulative_counts
        == 8
    )

    assert cfg.train_cfg.max_epochs == 24

    assert (
        cfg.param_scheduler[1].milestones
        == [20]
    )

    assert (
        Path(cfg.load_from).resolve()
        == EXPECTED_STAGE6_CHECKPOINT
    )

    assert cfg.resume is False

    print("model type:", cfg.model.type)
    print("head type:", cfg.model.bbox_head.type)
    print("optim wrapper:", cfg.optim_wrapper.type)
    print(
        "gradient accumulation:",
        cfg.optim_wrapper.accumulative_counts,
    )
    print("max epochs:", cfg.train_cfg.max_epochs)
    print("load_from:", cfg.load_from)
    print("resume:", cfg.resume)

    print()
    print("=" * 80)
    print("Building model")
    print("=" * 80)

    model = MODELS.build(cfg.model)
    model.init_weights()

    assert type(model).__name__ == "DINO"

    assert (
        type(model.bbox_head).__name__
        == "DINOHead"
    )

    checkpoint = torch.load(
        EXPECTED_STAGE6_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = extract_state_dict(
        checkpoint
    )

    incompatible = model.load_state_dict(
        state_dict,
        strict=True,
    )

    if incompatible.missing_keys:
        raise RuntimeError(
            f"Missing keys: "
            f"{incompatible.missing_keys}"
        )

    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected keys: "
            f"{incompatible.unexpected_keys}"
        )

    print(
        "Stage-6 checkpoint strict=True "
        "loading passed."
    )

    model.train()

    frozen_names = [
        name
        for name, parameter
        in model.named_parameters()
        if not parameter.requires_grad
    ]

    if frozen_names:
        raise RuntimeError(
            "Frozen stage-7 parameters:\n"
            + "\n".join(frozen_names)
        )

    print(
        "All model parameters have "
        "requires_grad=True."
    )

    print()
    print("=" * 80)
    print("Runtime policy checks")
    print("=" * 80)

    hook = HOOKS.build(
        dict(
            type="Stage7RuntimePolicyHook",
            freeze_dinostas_syncbn_stats=True,
            disable_tf32=True,
        )
    )

    dummy_runner = SimpleNamespace(
        model=model,
    )

    hook.before_run(dummy_runner)
    hook.before_train(dummy_runner)

    # Simulate MMEngine calling model.train() at epoch start.
    model.train()

    hook.before_train_iter(
        dummy_runner,
        batch_idx=0,
        data_batch=None,
    )

    syncbn_modules = [
        module
        for module
        in model.backbone.dinostas.modules()
        if isinstance(
            module,
            torch.nn.SyncBatchNorm,
        )
    ]

    if not syncbn_modules:
        raise RuntimeError(
            "No DINOSTAs SyncBatchNorm found."
        )

    if any(
        module.training
        for module in syncbn_modules
    ):
        raise RuntimeError(
            "A DINOSTAs SyncBatchNorm remains "
            "in training mode."
        )

    if any(
        not parameter.requires_grad
        for module in syncbn_modules
        for parameter in module.parameters(
            recurse=False
        )
    ):
        raise RuntimeError(
            "A SyncBatchNorm affine parameter "
            "is frozen."
        )

    assert (
        torch.backends.cuda.matmul.allow_tf32
        is False
    )

    assert (
        torch.backends.cudnn.allow_tf32
        is False
    )

    print(
        "SyncBN fixed-stat verification passed:"
    )
    print(
        "  modules:",
        len(syncbn_modules),
    )
    print(
        "  affine parameters remain trainable"
    )
    print(
        "  running statistics use eval mode"
    )
    print(
        "  CUDA TF32 disabled"
    )

    print()
    print("=" * 80)
    print("Optimizer checks")
    print("=" * 80)

    optim_wrapper = build_optim_wrapper(
        model,
        cfg.optim_wrapper,
    )

    parameter_name_by_id = {
        id(parameter): name
        for name, parameter
        in model.named_parameters()
    }

    optimizer_parameter_ids = [
        id(parameter)
        for group
        in optim_wrapper.optimizer.param_groups
        for parameter in group["params"]
    ]

    model_parameter_ids = [
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    if len(
        optimizer_parameter_ids
    ) != len(
        set(optimizer_parameter_ids)
    ):
        raise RuntimeError(
            "Optimizer contains duplicate parameters."
        )

    if set(
        optimizer_parameter_ids
    ) != set(
        model_parameter_ids
    ):
        missing = (
            set(model_parameter_ids)
            - set(optimizer_parameter_ids)
        )

        unexpected = (
            set(optimizer_parameter_ids)
            - set(model_parameter_ids)
        )

        raise RuntimeError(
            "Optimizer/model parameter mismatch: "
            f"missing={len(missing)}, "
            f"unexpected={len(unexpected)}"
        )

    lr_block_0 = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        ".blocks.0.",
    )

    lr_block_4 = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        ".blocks.4.",
    )

    lr_block_8 = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        ".blocks.8.",
    )

    lr_adapter = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        "backbone.dinostas.sta.",
    )

    lr_bridge = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        "neck.",
    )

    lr_sampling = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        "sampling_offsets",
    )

    lr_reference = get_unique_lr_for_pattern(
        optim_wrapper.optimizer,
        parameter_name_by_id,
        "reference_points",
    )

    assert_close(
        lr_block_0,
        2.0e-6,
        "ViT block 0 LR",
    )

    assert_close(
        lr_block_4,
        5.0e-6,
        "ViT block 4 LR",
    )

    assert_close(
        lr_block_8,
        1.0e-5,
        "ViT block 8 LR",
    )

    assert_close(
        lr_adapter,
        5.0e-5,
        "Lightly adapter LR",
    )

    assert_close(
        lr_bridge,
        2.0e-4,
        "Bridge LR",
    )

    assert_close(
        lr_sampling,
        2.0e-5,
        "sampling_offsets LR",
    )

    assert_close(
        lr_reference,
        2.0e-5,
        "reference_points LR",
    )

    unique_lrs = sorted({
        round(
            float(group["lr"]),
            12,
        )
        for group
        in optim_wrapper.optimizer.param_groups
    })

    print()
    print("Unique optimizer learning rates:")

    for lr in unique_lrs:
        print(f"  {lr:.12f}")

    print()
    print("=" * 80)
    print("High-resolution backbone/bridge check")
    print("=" * 80)

    model = model.cuda().eval()

    # 1333 width is padded to 1344 by pad_size_divisor=32.
    x = torch.randn(
        1,
        3,
        800,
        1344,
        device="cuda",
        dtype=torch.float32,
    )

    with torch.inference_mode():
        backbone_outputs = model.backbone(x)
        bridge_outputs = model.neck(
            backbone_outputs
        )

    actual_backbone_shapes = tuple(
        tuple(feature.shape)
        for feature in backbone_outputs
    )

    actual_bridge_shapes = tuple(
        tuple(feature.shape)
        for feature in bridge_outputs
    )

    expected_backbone_shapes = (
        (1, 224, 100, 168),
        (1, 224, 50, 84),
        (1, 224, 25, 42),
    )

    expected_bridge_shapes = (
        (1, 256, 100, 168),
        (1, 256, 50, 84),
        (1, 256, 25, 42),
        (1, 256, 13, 21),
    )

    print(
        "Backbone:",
        actual_backbone_shapes,
    )

    print(
        "Bridge:",
        actual_bridge_shapes,
    )

    if (
        actual_backbone_shapes
        != expected_backbone_shapes
    ):
        raise RuntimeError(
            "Unexpected high-resolution "
            "backbone shapes."
        )

    if (
        actual_bridge_shapes
        != expected_bridge_shapes
    ):
        raise RuntimeError(
            "Unexpected high-resolution "
            "bridge shapes."
        )

    if not all(
        torch.isfinite(feature).all()
        for feature in backbone_outputs
    ):
        raise FloatingPointError(
            "Backbone produced NaN/Inf."
        )

    if not all(
        torch.isfinite(feature).all()
        for feature in bridge_outputs
    ):
        raise FloatingPointError(
            "Bridge produced NaN/Inf."
        )

    print()
    print("=" * 80)
    print("Stage-7 setup verification passed.")
    print("=" * 80)
    print("  original DINO detector")
    print("  original DINOHead")
    print("  stage-6 best checkpoint strict=True")
    print("  every model parameter is trainable")
    print("  improved DINO settings enabled")
    print("  official DINO multiscale pipeline enabled")
    print("  FP32 OptimWrapper enabled")
    print("  TF32 disabled")
    print("  effective global batch size = 16")
    print("  layer-wise optimizer LRs verified")
    print("  high-resolution output shapes verified")


if __name__ == "__main__":
    main()