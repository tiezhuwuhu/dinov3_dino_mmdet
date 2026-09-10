from __future__ import annotations

from pathlib import Path

import torch


ROOT = Path("/root/autodl-tmp/dinov3_dino_mmdet/")

BACKBONE_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "lightly"
    / "dinov3_vits16_ltdetr_coco_backbone.pth"
)

DINO_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "mmdet"
    / "dino_r50_4scale_coco.pth"
)

OUTPUT_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "merged"
    / "dinov3_vits16_lightly_coco_dino4_init.pth"
)


def unwrap_state_dict(
    checkpoint: object,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Checkpoint must be dict, got {type(checkpoint)}"
        )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    if not isinstance(state_dict, dict):
        raise TypeError("state_dict is not a dictionary.")

    cleaned: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            continue

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        cleaned[new_key] = value.detach().cpu()

    return cleaned


def main() -> None:
    if not BACKBONE_CHECKPOINT.exists():
        raise FileNotFoundError(BACKBONE_CHECKPOINT)

    if not DINO_CHECKPOINT.exists():
        raise FileNotFoundError(DINO_CHECKPOINT)

    backbone_raw = torch.load(
        BACKBONE_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    dino_raw = torch.load(
        DINO_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    backbone_state = unwrap_state_dict(backbone_raw)
    dino_state = unwrap_state_dict(dino_raw)

    merged_state: dict[str, torch.Tensor] = {}

    # Our custom MMDet backbone has:
    # model.backbone.vit.<original DINOv3 key>
    for key, value in backbone_state.items():
        target_key = f"backbone.vit.{key}"

        if target_key in merged_state:
            raise KeyError(f"Duplicate key: {target_key}")

        merged_state[target_key] = value

    kept_dino_keys = 0
    removed_dino_keys = 0

    for key, value in dino_state.items():
        # The original ResNet and ChannelMapper do not match
        # the new ViT and ViTFeaturePyramid.
        if key.startswith("backbone."):
            removed_dino_keys += 1
            continue

        if key.startswith("neck."):
            removed_dino_keys += 1
            continue

        if key in merged_state:
            raise KeyError(
                f"DINO key collides with backbone key: {key}"
            )

        merged_state[key] = value
        kept_dino_keys += 1

    OUTPUT_CHECKPOINT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = {
        "state_dict": merged_state,
        "meta": {
            "description": (
                "Lightly COCO-finetuned DINOv3 ViT-S/16 "
                "+ MMDetection DINO-4scale R50 detection "
                "transformer/head. New ViTFeaturePyramid "
                "is randomly initialized."
            ),
            "backbone_source": str(BACKBONE_CHECKPOINT),
            "dino_source": str(DINO_CHECKPOINT),
            "backbone_prefix": "backbone.vit.",
            "dino_removed_prefixes": [
                "backbone.",
                "neck.",
            ],
            "new_random_modules": [
                "neck",
            ],
        },
    }

    torch.save(output, OUTPUT_CHECKPOINT)

    print(f"Backbone tensors: {len(backbone_state)}")
    print(f"Kept DINO tensors: {kept_dino_keys}")
    print(f"Removed DINO tensors: {removed_dino_keys}")
    print(f"Total merged tensors: {len(merged_state)}")
    print(f"Saved to: {OUTPUT_CHECKPOINT}")


if __name__ == "__main__":
    main()