from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
from mmengine.config import Config

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules

import projects.dinov3_dino  # noqa: F401


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    "/root/autodl-tmp/dinov3_dino_mmdet/mmdetection"
)

CONFIG = (
    ROOT
    / "projects/dinov3_dino/configs/"
      "dino4_vits16_learnable_anchor_improved_multiscale_24e.py"
)

VIT_SOURCE = Path(
    "/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/"
    "lightly/dinov3_vits16_ltdetr_coco_backbone.pth"
)

DINO_SOURCE = Path(
    "/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/mmdet/dino_r50_4scale_coco.pth"
)

OUTPUT = Path(
    "/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/"
    "merged/"
    "dinov3_vits16_learnable_anchor_official_dino4_init.pth"
)


# ============================================================
# Helper
# ============================================================

def load_state_dict(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key[len("module."):]

        cleaned[key] = value

    return cleaned


def print_section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# Main
# ============================================================

def main():

    register_all_modules(
        init_default_scope=True
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Build target model
    #
    # IMPORTANT:
    # deepcopy avoids DINO mutating the config object.
    # ========================================================

    print_section("BUILD TARGET MODEL")

    cfg = Config.fromfile(
        str(CONFIG)
    )

    model = MODELS.build(
        deepcopy(cfg.model)
    )

    model.init_weights()

    target_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    target_keys = set(
        target_state.keys()
    )

    vit_target_keys = {
        key
        for key in target_keys
        if key.startswith("backbone.vit.")
    }

    neck_target_keys = {
        key
        for key in target_keys
        if key.startswith("neck.")
    }

    detector_target_keys = (
        target_keys
        - {
            key
            for key in target_keys
            if key.startswith("backbone.")
        }
        - neck_target_keys
    )

    print(
        "Target total tensors:",
        len(target_keys),
    )

    print(
        "Target ViT tensors:",
        len(vit_target_keys),
    )

    print(
        "New neck tensors:",
        len(neck_target_keys),
    )

    print(
        "DINO detector tensors:",
        len(detector_target_keys),
    )

    # ========================================================
    # Check alpha initialization
    # ========================================================

    print_section("CHECK ALPHA INITIALIZATION")

    (
        alpha_early,
        alpha_middle,
        alpha_deep,
    ) = model.neck.get_alphas()

    alpha_values = [
        alpha_early.detach().cpu().item(),
        alpha_middle.detach().cpu().item(),
        alpha_deep.detach().cpu().item(),
    ]

    print(
        "alpha early :",
        alpha_values[0],
    )

    print(
        "alpha middle:",
        alpha_values[1],
    )

    print(
        "alpha deep  :",
        alpha_values[2],
    )

    for alpha in alpha_values:

        if abs(alpha - 0.25) > 1e-6:
            raise RuntimeError(
                "Unexpected alpha initialization: "
                f"{alpha}"
            )

    print("ALPHA INITIALIZATION: OK")

    # ========================================================
    # Preserve new neck initialization
    # ========================================================

    neck_initial = {
        key: target_state[key].clone()
        for key in neck_target_keys
    }

    # ========================================================
    # Load source checkpoints
    # ========================================================

    print_section("LOAD SOURCE CHECKPOINTS")

    vit_source = load_state_dict(
        VIT_SOURCE
    )

    dino_source = load_state_dict(
        DINO_SOURCE
    )

    print(
        "Lightly ViT source tensors:",
        len(vit_source),
    )

    print(
        "Official DINO source tensors:",
        len(dino_source),
    )

    # ========================================================
    # 1. Lightly ViT
    # ========================================================

    print_section("1. LOAD LIGHTLY VIT")

    loaded_vit = set()

    for source_key, source_tensor in vit_source.items():

        target_key = (
            "backbone.vit."
            + source_key
        )

        if target_key not in target_state:
            raise KeyError(
                "ViT target key not found:\n"
                f"source: {source_key}\n"
                f"target: {target_key}"
            )

        target_tensor = target_state[
            target_key
        ]

        if (
            target_tensor.shape
            != source_tensor.shape
        ):
            raise RuntimeError(
                "ViT shape mismatch:\n"
                f"{target_key}\n"
                f"target={target_tensor.shape}\n"
                f"source={source_tensor.shape}"
            )

        target_state[target_key] = (
            source_tensor
            .detach()
            .cpu()
            .clone()
        )

        loaded_vit.add(
            target_key
        )

    print(
        "Loaded Lightly ViT tensors:",
        len(loaded_vit),
    )

    missing_vit = (
        vit_target_keys
        - loaded_vit
    )

    extra_vit = (
        loaded_vit
        - vit_target_keys
    )

    if missing_vit:
        raise RuntimeError(
            "Missing ViT target keys:\n"
            + "\n".join(
                sorted(missing_vit)
            )
        )

    if extra_vit:
        raise RuntimeError(
            "Unexpected ViT keys:\n"
            + "\n".join(
                sorted(extra_vit)
            )
        )

    print("VIT KEY COVERAGE: OK")

    # ========================================================
    # 2. Official DINO detector
    # ========================================================

    print_section("2. LOAD OFFICIAL DINO DETECTOR")

    loaded_dino = set()

    for target_key in sorted(
        detector_target_keys
    ):

        if target_key not in dino_source:
            raise KeyError(
                "DINO source missing target key:\n"
                f"{target_key}"
            )

        source_tensor = dino_source[
            target_key
        ]

        target_tensor = target_state[
            target_key
        ]

        if (
            target_tensor.shape
            != source_tensor.shape
        ):
            raise RuntimeError(
                "DINO shape mismatch:\n"
                f"{target_key}\n"
                f"target={target_tensor.shape}\n"
                f"source={source_tensor.shape}"
            )

        target_state[target_key] = (
            source_tensor
            .detach()
            .cpu()
            .clone()
        )

        loaded_dino.add(
            target_key
        )

    if loaded_dino != detector_target_keys:

        missing = (
            detector_target_keys
            - loaded_dino
        )

        raise RuntimeError(
            "Incomplete DINO coverage:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    print(
        "Loaded official DINO tensors:",
        len(loaded_dino),
    )

    print("DINO KEY COVERAGE: OK")

    # ========================================================
    # 3. Verify new neck untouched
    # ========================================================

    print_section(
        "3. CHECK NEW LEARNABLE-ANCHOR NECK"
    )

    for key in neck_target_keys:

        if not torch.equal(
            target_state[key],
            neck_initial[key],
        ):
            raise RuntimeError(
                "New neck was unexpectedly "
                f"overwritten: {key}"
            )

    print(
        "NEW LEARNABLE-ANCHOR NECK UNTOUCHED: OK"
    )

    # ========================================================
    # Save
    # ========================================================

    print_section("SAVE")

    torch.save(
        {
            "state_dict": target_state,

            "meta": {
                "description": (
                    "Lightly DINOv3 ViT-S/16 "
                    "+ learnable-anchor pyramid "
                    "+ official MMDetection DINO"
                ),

                "vit_source": str(
                    VIT_SOURCE
                ),

                "dino_source": str(
                    DINO_SOURCE
                ),

                "config": str(
                    CONFIG
                ),

                "vit_tensors": len(
                    loaded_vit
                ),

                "dino_tensors": len(
                    loaded_dino
                ),

                "neck_tensors": len(
                    neck_target_keys
                ),

                "alpha_init": 0.25,
            },
        },
        OUTPUT,
    )

    print(
        "Saved:",
        OUTPUT,
    )

    # ========================================================
    # Strict load verification
    #
    # IMPORTANT:
    # Reload a fresh Config.
    #
    # Do NOT reuse cfg.model because DINO build mutates
    # nested bbox_head configuration.
    # ========================================================

    print_section(
        "STRICT LOAD VERIFICATION"
    )

    verify_cfg = Config.fromfile(
        str(CONFIG)
    )

    verify_model = MODELS.build(
        deepcopy(
            verify_cfg.model
        )
    )

    saved_checkpoint = torch.load(
        OUTPUT,
        map_location="cpu",
        weights_only=False,
    )

    saved_state = saved_checkpoint[
        "state_dict"
    ]

    verify_model.load_state_dict(
        saved_state,
        strict=True,
    )

    print("STRICT LOAD: OK")

    # ========================================================
    # Exact ViT verification
    # ========================================================

    for source_key, source_tensor in vit_source.items():

        target_key = (
            "backbone.vit."
            + source_key
        )

        if not torch.equal(
            saved_state[target_key],
            source_tensor.cpu(),
        ):
            raise RuntimeError(
                "ViT exact check failed: "
                f"{target_key}"
            )

    print(
        "BACKBONE EXACT MATCH: OK "
        f"({len(vit_source)} tensors)"
    )

    # ========================================================
    # Exact DINO verification
    # ========================================================

    for target_key in detector_target_keys:

        if not torch.equal(
            saved_state[target_key],
            dino_source[target_key].cpu(),
        ):
            raise RuntimeError(
                "DINO exact check failed: "
                f"{target_key}"
            )

    print(
        "DINO DETECTOR EXACT MATCH: OK "
        f"({len(detector_target_keys)} tensors)"
    )

    # ========================================================
    # Exact new-neck verification
    # ========================================================

    for key in neck_target_keys:

        if not torch.equal(
            saved_state[key],
            neck_initial[key],
        ):
            raise RuntimeError(
                "Neck initialization changed: "
                f"{key}"
            )

    print(
        "NEW NECK INITIALIZATION PRESERVED: OK "
        f"({len(neck_target_keys)} tensors)"
    )

    # ========================================================
    # Alpha after save/load
    # ========================================================

    verify_model.load_state_dict(
        saved_state,
        strict=True,
    )

    (
        ae,
        am,
        ad,
    ) = verify_model.neck.get_alphas()

    print()
    print(
        "Saved alpha early :",
        ae.item(),
    )

    print(
        "Saved alpha middle:",
        am.item(),
    )

    print(
        "Saved alpha deep  :",
        ad.item(),
    )

    for alpha in (
        ae.item(),
        am.item(),
        ad.item(),
    ):

        if abs(alpha - 0.25) > 1e-6:
            raise RuntimeError(
                "Saved alpha check failed."
            )

    print("ALPHA SAVE/LOAD CHECK: OK")

    print_section("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()