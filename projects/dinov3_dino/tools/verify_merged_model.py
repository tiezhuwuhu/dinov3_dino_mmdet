from __future__ import annotations

from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
MMDET_ROOT = SCRIPT_PATH.parents[3]
PROJECTS_ROOT = MMDET_ROOT / "projects"

for search_path in (
    str(MMDET_ROOT),
    str(PROJECTS_ROOT),
):
    if search_path not in sys.path:
        sys.path.insert(0, search_path)


import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


ROOT = Path("/root/autodl-tmp/dinov3_dino_mmdet")

CONFIG_PATH = (
    ROOT
    / "mmdetection"
    / "projects"
    / "dinov3_dino"
    / "configs"
    / "dino4_vits16_coco_align_640_1e.py"
)

BACKBONE_CHECKPOINT = (
    ROOT
    / "checkpoints"
    / "lightly"
    / "dinov3_vits16_ltdetr_coco_backbone.pth"
)


def main() -> None:
    register_all_modules()

    # Register the custom project.
    import projects.dinov3_dino  # noqa: F401

    cfg = Config.fromfile(CONFIG_PATH)

    print("Building model...")
    model = MODELS.build(cfg.model)

    print("Initializing model...")
    model.init_weights()

    print(f"Loading merged checkpoint: {cfg.load_from}")
    load_checkpoint(
        model,
        cfg.load_from,
        map_location="cpu",
        strict=False,
    )

    # Verify that the loaded ViT tensors exactly match the
    # standalone extracted backbone checkpoint.
    backbone_raw = torch.load(
        BACKBONE_CHECKPOINT,
        map_location="cpu",
    )
    backbone_state = backbone_raw["state_dict"]

    full_state = model.state_dict()

    checked = 0

    for key, expected_value in backbone_state.items():
        full_key = f"backbone.vit.{key}"

        if full_key not in full_state:
            raise KeyError(
                f"Missing merged backbone key: {full_key}"
            )

        actual_value = full_state[full_key].cpu()

        if not torch.equal(
            actual_value,
            expected_value.cpu(),
        ):
            raise RuntimeError(
                f"Backbone tensor mismatch: {full_key}"
            )

        checked += 1

    print(
        f"Verified {checked} backbone tensors: exact match."
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = model.to(device)
    model.eval()

    x = torch.randn(
        1,
        3,
        640,
        640,
        device=device,
    )

    with torch.no_grad():
        features = model.extract_feat(x)

    expected_shapes = [
        (1, 256, 80, 80),
        (1, 256, 40, 40),
        (1, 256, 20, 20),
        (1, 256, 10, 10),
    ]

    print("Extracted feature shapes:")

    for index, (feature, expected) in enumerate(
        zip(features, expected_shapes, strict=True)
    ):
        actual = tuple(feature.shape)
        print(f"  level {index}: {actual}")

        if actual != expected:
            raise RuntimeError(
                f"Level {index}: expected {expected}, "
                f"got {actual}"
            )

        if not torch.isfinite(feature).all():
            raise RuntimeError(
                f"Level {index} contains NaN or Inf."
            )

    print("Merged model verification passed.")


if __name__ == "__main__":
    main()