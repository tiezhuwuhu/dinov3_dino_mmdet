from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path("/root/autodl-tmp/dinov3_dino_mmdet/")

CACHE_DIR = ROOT / "checkpoints" / "lightly" / "cache"

OUTPUT_PATH = (
    ROOT
    / "checkpoints"
    / "lightly"
    / "dinov3_vits16_ltdetr_coco_backbone.pth"
)

MODEL_NAME = "/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/lightly/cache/dinov3_vits16_ltdetr_coco_251218_4812416b.pt"


def find_vits16(root_model: nn.Module) -> tuple[str, nn.Module]:
    """Find the pure DINOv3 ViT-S/16 inside the loaded LTDETR model."""

    from lightly_train._models.dinov3.dinov3_src.models.vision_transformer import (
        DinoVisionTransformer,
    )

    candidates: list[tuple[str, nn.Module]] = []

    for name, module in root_model.named_modules():
        if not isinstance(module, DinoVisionTransformer):
            continue

        embed_dim = int(getattr(module, "embed_dim", -1))
        patch_size = int(getattr(module, "patch_size", -1))
        n_blocks = int(
            getattr(
                module,
                "n_blocks",
                len(getattr(module, "blocks", [])),
            )
        )

        print(
            "Found DINOv3 candidate:",
            f"name={name!r}",
            f"embed_dim={embed_dim}",
            f"patch_size={patch_size}",
            f"n_blocks={n_blocks}",
        )

        if embed_dim == 384 and patch_size == 16 and n_blocks == 12:
            candidates.append((name, module))

    if len(candidates) != 1:
        names = [name for name, _ in candidates]
        raise RuntimeError(
            "Expected exactly one ViT-S/16 candidate, "
            f"but found {len(candidates)}: {names}"
        )

    return candidates[0]


def extract_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state_dict: dict[str, torch.Tensor] = {}

    for key, value in model.state_dict().items():
        state_dict[key] = value.detach().cpu().clone()

    return state_dict


def verify_strict_load(
    state_dict: dict[str, torch.Tensor],
    original_model: nn.Module,
) -> None:
    """Build a clean ViT-S/16 and verify exact weight loading/output."""

    from lightly_train._models.dinov3.dinov3_package import (
        DINOV3_PACKAGE,
    )

    clean_model = DINOV3_PACKAGE.get_model(
        model_name="vits16",
        load_weights=False,
    )

    load_result = clean_model.load_state_dict(
        state_dict,
        strict=True,
    )

    print("Strict load result:", load_result)

    original_model.eval()
    clean_model.eval()

    torch.manual_seed(0)
    x = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        original_outputs: Any = (
            original_model.get_intermediate_layers(
                x,
                n=[5, 8, 11],
                reshape=True,
                return_class_token=True,
            )
        )

        clean_outputs: Any = (
            clean_model.get_intermediate_layers(
                x,
                n=[5, 8, 11],
                reshape=True,
                return_class_token=True,
            )
        )

    max_abs_error = 0.0

    for original_item, clean_item in zip(
        original_outputs,
        clean_outputs,
        strict=True,
    ):
        original_feature, original_cls = original_item
        clean_feature, clean_cls = clean_item

        feature_error = (
            original_feature - clean_feature
        ).abs().max().item()

        cls_error = (
            original_cls - clean_cls
        ).abs().max().item()

        max_abs_error = max(
            max_abs_error,
            feature_error,
            cls_error,
        )

    print(f"Maximum output error: {max_abs_error:.12g}")

    if max_abs_error > 1e-7:
        raise RuntimeError(
            "Extracted backbone output does not exactly match "
            f"the original backbone. Error={max_abs_error}"
        )


def main() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Must be set before importing lightly_train.
    os.environ["LIGHTLY_TRAIN_MODEL_CACHE_DIR"] = str(
        CACHE_DIR
    )

    import lightly_train

    print(f"Loading Lightly model: {MODEL_NAME}")
    print(f"Model cache: {CACHE_DIR}")

    loaded_model = lightly_train.load_model(
        MODEL_NAME,
        device="cpu",
    )

    if not isinstance(loaded_model, nn.Module):
        raise TypeError(
            "lightly_train.load_model did not return nn.Module: "
            f"{type(loaded_model)}"
        )

    loaded_model.eval()

    # Preferred path for the current LTDETR implementation.
    try:
        vit = (
            loaded_model
            .backbone
            ._model_wrapper
            .get_model()
        )
        module_name = (
            "backbone._model_wrapper.get_model()"
        )

        embed_dim = int(getattr(vit, "embed_dim", -1))
        patch_size = int(getattr(vit, "patch_size", -1))
        n_blocks = int(getattr(vit, "n_blocks", -1))

        if (
            embed_dim != 384
            or patch_size != 16
            or n_blocks != 12
        ):
            raise RuntimeError(
                "Preferred module is not ViT-S/16."
            )

    except (AttributeError, RuntimeError):
        module_name, vit = find_vits16(loaded_model)

    state_dict = extract_state_dict(vit)

    num_parameters = sum(
        parameter.numel()
        for parameter in vit.parameters()
    )

    print(f"Selected module: {module_name}")
    print(f"Parameter count: {num_parameters / 1e6:.3f} M")
    print(f"Number of state tensors: {len(state_dict)}")

    verify_strict_load(
        state_dict=state_dict,
        original_model=vit,
    )

    checkpoint = {
        "state_dict": state_dict,
        "meta": {
            "source_model": MODEL_NAME,
            "selected_module": module_name,
            "architecture": "DINOv3 ViT-S/16",
            "embed_dim": 384,
            "patch_size": 16,
            "num_blocks": 12,
            "intermediate_blocks": [5, 8, 11],
            "parameter_count": num_parameters,
            "lightly_train_version": lightly_train.__version__,
        },
    }

    torch.save(checkpoint, OUTPUT_PATH)

    print(f"Saved backbone to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()