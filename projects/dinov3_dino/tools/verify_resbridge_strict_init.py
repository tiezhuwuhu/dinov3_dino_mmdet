from __future__ import annotations

import argparse
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings

from mmdet.registry import MODELS


EXCLUDED_OFFICIAL_PREFIXES = (
    "backbone.",
    "neck.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--merged",
        required=True,
    )

    parser.add_argument(
        "--lightly-backbone",
        required=True,
    )

    parser.add_argument(
        "--official-dino",
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


def extract_state_dict(
    checkpoint,
) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint type: {type(checkpoint)!r}"
        )

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

    return {
        normalize_key(key): tensor
        for key, tensor in state.items()
        if torch.is_tensor(tensor)
    }


def load_checkpoint_state(
    path: Path,
) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    return extract_state_dict(checkpoint)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    merged_path = Path(args.merged).resolve()
    lightly_path = Path(args.lightly_backbone).resolve()
    official_path = Path(args.official_dino).resolve()

    for path in (
        config_path,
        merged_path,
        lightly_path,
        official_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    cfg = Config.fromfile(config_path)

    import_modules_from_strings(
        **cfg.custom_imports
    )

    init_default_scope(
        cfg.get("default_scope", "mmdet")
    )

    model = MODELS.build(cfg.model)
    model.init_weights()

    merged_checkpoint = torch.load(
        merged_path,
        map_location="cpu",
        weights_only=False,
    )

    merged_state = extract_state_dict(
        merged_checkpoint
    )

    incompatible = model.load_state_dict(
        merged_state,
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

    print("1. strict=True model loading passed.")

    lightly_checkpoint = torch.load(
        lightly_path,
        map_location="cpu",
        weights_only=False,
    )

    lightly_state = extract_state_dict(
        lightly_checkpoint
    )

    verified_lightly_tensors = 0

    for key, source_tensor in lightly_state.items():
        target_key = f"backbone.dinostas.{key}"

        if target_key not in merged_state:
            raise KeyError(
                f"Missing Lightly tensor in merged state: {target_key}"
            )

        if not torch.equal(
            merged_state[target_key].cpu(),
            source_tensor.cpu(),
        ):
            raise RuntimeError(
                f"Lightly tensor mismatch: {target_key}"
            )

        verified_lightly_tensors += 1

    print(
        "2. Exact Lightly DINOSTAs verification passed: "
        f"{verified_lightly_tensors} tensors."
    )

    official_state = load_checkpoint_state(
        official_path
    )

    verified_detector_tensors = 0

    for key, source_tensor in official_state.items():
        if key.startswith(EXCLUDED_OFFICIAL_PREFIXES):
            continue

        if key not in merged_state:
            raise KeyError(
                f"Missing official detector tensor: {key}"
            )

        if tuple(merged_state[key].shape) != tuple(source_tensor.shape):
            raise RuntimeError(
                f"Official detector shape mismatch for {key}: "
                f"merged={tuple(merged_state[key].shape)}, "
                f"official={tuple(source_tensor.shape)}"
            )

        if not torch.equal(
            merged_state[key].cpu(),
            source_tensor.cpu(),
        ):
            raise RuntimeError(
                f"Official detector tensor mismatch: {key}"
            )

        verified_detector_tensors += 1

    print(
        "3. Exact official DINO detector verification passed: "
        f"{verified_detector_tensors} tensors."
    )

    for level, block in enumerate(model.neck.input_blocks):
        final_norm = block.refinement[-1]

        weight_nonzero = torch.count_nonzero(
            final_norm.weight.detach()
        ).item()

        bias_nonzero = torch.count_nonzero(
            final_norm.bias.detach()
        ).item()

        if weight_nonzero != 0:
            raise RuntimeError(
                f"Bridge level {level} final norm weight is not zero."
            )

        if bias_nonzero != 0:
            raise RuntimeError(
                f"Bridge level {level} final norm bias is not zero."
            )

    print(
        "4. Residual bridge zero initialization verification passed."
    )

    model = model.cuda().eval()

    test_cases = (
        (
            640,
            640,
            (
                (1, 224, 80, 80),
                (1, 224, 40, 40),
                (1, 224, 20, 20),
            ),
            (
                (1, 256, 80, 80),
                (1, 256, 40, 40),
                (1, 256, 20, 20),
                (1, 256, 10, 10),
            ),
        ),
        (
            640,
            800,
            (
                (1, 224, 80, 100),
                (1, 224, 40, 50),
                (1, 224, 20, 25),
            ),
            (
                (1, 256, 80, 100),
                (1, 256, 40, 50),
                (1, 256, 20, 25),
                (1, 256, 10, 13),
            ),
        ),
        (
            768,
            768,
            (
                (1, 224, 96, 96),
                (1, 224, 48, 48),
                (1, 224, 24, 24),
            ),
            (
                (1, 256, 96, 96),
                (1, 256, 48, 48),
                (1, 256, 24, 24),
                (1, 256, 12, 12),
            ),
        ),
    )

    for (
        height,
        width,
        expected_backbone,
        expected_neck,
    ) in test_cases:
        x = torch.randn(
            1,
            3,
            height,
            width,
            device="cuda",
            dtype=torch.float32,
        )

        with torch.inference_mode():
            backbone_outputs = model.backbone(x)
            neck_outputs = model.neck(backbone_outputs)

        actual_backbone = tuple(
            tuple(feature.shape)
            for feature in backbone_outputs
        )

        actual_neck = tuple(
            tuple(feature.shape)
            for feature in neck_outputs
        )

        print()
        print(f"Input: {(1, 3, height, width)}")
        print("Backbone:", actual_backbone)
        print("Bridge:", actual_neck)

        if actual_backbone != expected_backbone:
            raise RuntimeError(
                f"Backbone output mismatch for {(height, width)}: "
                f"actual={actual_backbone}, "
                f"expected={expected_backbone}"
            )

        if actual_neck != expected_neck:
            raise RuntimeError(
                f"Bridge output mismatch for {(height, width)}: "
                f"actual={actual_neck}, "
                f"expected={expected_neck}"
            )

        if not all(
            torch.isfinite(feature).all()
            for feature in backbone_outputs
        ):
            raise FloatingPointError(
                f"Backbone produced NaN/Inf for {(height, width)}."
            )

        if not all(
            torch.isfinite(feature).all()
            for feature in neck_outputs
        ):
            raise FloatingPointError(
                f"Bridge produced NaN/Inf for {(height, width)}."
            )

    print()
    print("5. Output shape and finite-value verification passed.")

    print()
    print("=" * 72)
    print("All strict residual-bridge initialization checks passed.")
    print("=" * 72)


if __name__ == "__main__":
    main()