from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()

    cfg = Config.fromfile(config_path)

    import_modules_from_strings(
        **cfg.custom_imports
    )

    init_default_scope(
        cfg.get("default_scope", "mmdet")
    )

    model = MODELS.build(cfg.model)
    model.init_weights()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state = checkpoint["state_dict"]

    model.load_state_dict(
        state,
        strict=True,
    )

    print("Checkpoint loaded with strict=True.")
    print()
    print("Checkpoint meta:")

    for key, value in checkpoint.get("meta", {}).items():
        print(f"  {key}: {value}")

    model = model.cuda().eval()

    for height, width in (
        (640, 640),
        (640, 800),
        (768, 768),
    ):
        x = torch.randn(
            1,
            3,
            height,
            width,
            device="cuda",
        )

        with torch.inference_mode():
            backbone_outputs = model.backbone(x)
            neck_outputs = model.neck(backbone_outputs)

        print()
        print("=" * 72)
        print("input:", tuple(x.shape))

        print("backbone:")
        for feature in backbone_outputs:
            print(" ", tuple(feature.shape))

        print("neck:")
        for feature in neck_outputs:
            print(" ", tuple(feature.shape))

        if len(backbone_outputs) != 3:
            raise RuntimeError(
                "Expected 3 DINOSTAs features."
            )

        if len(neck_outputs) != 4:
            raise RuntimeError(
                "Expected 4 ChannelMapper features."
            )

        if not all(
            torch.isfinite(feature).all()
            for feature in neck_outputs
        ):
            raise FloatingPointError(
                "Neck output contains NaN or Inf."
            )

    print()
    print("Strict initialization verification passed.")


if __name__ == "__main__":
    main()