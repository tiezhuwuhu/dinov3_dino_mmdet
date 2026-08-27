from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS


SOURCE_EXCLUDED_PREFIXES = (
    "backbone.",
    "neck.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict combined initialization checkpoint:\n"
            "1. Lightly DINOSTAs supplies backbone and multiscale adapter.\n"
            "2. Official DINO R50 supplies every detector tensor.\n"
            "3. New ChannelMapper remains randomly initialized."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--official-dino",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    return parser.parse_args()


def normalize_key(key: str) -> str:
    prefixes = (
        "module.",
        "model.",
        "_orig_mod.",
    )

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True

    return key


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        for candidate in (
            "state_dict",
            "model",
            "model_state_dict",
        ):
            if candidate in checkpoint:
                state = checkpoint[candidate]
                break
        else:
            state = checkpoint
    else:
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)!r}"
        )

    return {
        normalize_key(key): tensor
        for key, tensor in state.items()
        if torch.is_tensor(tensor)
    }


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    official_path = Path(args.official_dino).resolve()
    output_path = Path(args.output).resolve()

    if not config_path.is_file():
        raise FileNotFoundError(config_path)

    if not official_path.is_file():
        raise FileNotFoundError(official_path)

    cfg = Config.fromfile(config_path)

    if cfg.get("custom_imports") is not None:
        import_modules_from_strings(
            **cfg.custom_imports
        )

    init_default_scope(
        cfg.get("default_scope", "mmdet")
    )

    model = MODELS.build(cfg.model)

    # Initialize only newly introduced MMDetection modules, principally
    # ChannelMapper. LightlyDINOSTAs.init_weights() is intentionally a no-op.
    model.init_weights()

    target_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }

    official_state = load_state_dict(
        official_path
    )

    official_detector_state = {
        key: tensor
        for key, tensor in official_state.items()
        if not key.startswith(SOURCE_EXCLUDED_PREFIXES)
    }

    target_detector_keys = {
        key
        for key in target_state
        if not key.startswith(SOURCE_EXCLUDED_PREFIXES)
    }

    official_detector_keys = set(
        official_detector_state
    )

    missing_in_official = sorted(
        target_detector_keys - official_detector_keys
    )

    unexpected_in_official = sorted(
        official_detector_keys - target_detector_keys
    )

    shape_mismatches = []

    for key in sorted(
        target_detector_keys & official_detector_keys
    ):
        official_shape = tuple(
            official_detector_state[key].shape
        )
        target_shape = tuple(
            target_state[key].shape
        )

        if official_shape != target_shape:
            shape_mismatches.append(
                (
                    key,
                    official_shape,
                    target_shape,
                )
            )

    if missing_in_official:
        print("\nTarget detector keys absent from official checkpoint:")

        for key in missing_in_official:
            print(" ", key)

    if unexpected_in_official:
        print("\nOfficial detector keys absent from target model:")

        for key in unexpected_in_official:
            print(" ", key)

    if shape_mismatches:
        print("\nDetector shape mismatches:")

        for key, source_shape, target_shape in shape_mismatches:
            print(
                f"  {key}: official={source_shape}, "
                f"target={target_shape}"
            )

    if (
        missing_in_official
        or unexpected_in_official
        or shape_mismatches
    ):
        raise RuntimeError(
            "Strict official detector compatibility check failed."
        )

    merged_state = dict(target_state)

    for key, tensor in official_detector_state.items():
        merged_state[key] = tensor.detach().cpu().clone()

    # Verify every detector tensor is an exact official copy.
    for key, official_tensor in official_detector_state.items():
        merged_tensor = merged_state[key]

        if not torch.equal(
            merged_tensor,
            official_tensor.cpu(),
        ):
            raise RuntimeError(
                f"Detector tensor failed exact verification: {key}"
            )

    # Verify the exported Lightly backbone is present exactly.
    lightly_checkpoint_path = Path(
        cfg.lightly_backbone_checkpoint
    ).resolve()

    lightly_checkpoint = torch.load(
        lightly_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    lightly_state = lightly_checkpoint["state_dict"]

    for key, lightly_tensor in lightly_state.items():
        target_key = f"backbone.dinostas.{key}"

        if target_key not in merged_state:
            raise KeyError(
                f"Lightly backbone key absent from target model: "
                f"{target_key}"
            )

        if not torch.equal(
            merged_state[target_key],
            lightly_tensor.cpu(),
        ):
            raise RuntimeError(
                f"Lightly tensor failed exact verification: {key}"
            )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "state_dict": merged_state,
            "meta": {
                "lightly_backbone_source": str(
                    lightly_checkpoint_path
                ),
                "official_dino_source": str(
                    official_path
                ),
                "detector_tensor_count": len(
                    official_detector_state
                ),
                "policy": (
                    "backbone.dinostas.* = exported Lightly DINOSTAs; "
                    "neck.* = newly initialized ChannelMapper; "
                    "all remaining tensors = official DINO R50."
                ),
            },
        },
        output_path,
    )

    print()
    print("Strict initialization build passed.")
    print("Lightly source:", lightly_checkpoint_path)
    print("Official DINO source:", official_path)
    print("Detector tensors:", len(official_detector_state))
    print("Output:", output_path)


if __name__ == "__main__":
    main()