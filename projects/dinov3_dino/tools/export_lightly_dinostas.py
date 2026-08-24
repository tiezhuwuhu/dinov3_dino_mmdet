from __future__ import annotations

import argparse
from pathlib import Path

import torch
import lightly_train


EXPECTED_PARAMETER_COUNT = 22_173_408

REQUIRED_PREFIXES = (
    "_model_wrapper._model.",
    "sta.",
    "convs.",
    "norms.",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the complete Lightly DINOSTAs backbone, including "
            "DINOv3 ViT-S/16, SpatialPriorModulev2 and multiscale fusion."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Complete Lightly LTDETR .pt checkpoint.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output DINOSTAs state-dict checkpoint.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Lightly checkpoint does not exist: {input_path}"
        )

    print("Loading complete Lightly model:")
    print(input_path)

    full_model = lightly_train.load_model(
        input_path,
        device="cpu",
    )

    if not hasattr(full_model, "backbone"):
        raise AttributeError(
            "The loaded Lightly model has no top-level backbone."
        )

    backbone = full_model.backbone

    if type(backbone).__name__ != "DINOSTAs":
        raise TypeError(
            "Expected Lightly backbone type DINOSTAs, but received "
            f"{type(backbone)!r}."
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in backbone.parameters()
    )

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_PARAMETER_COUNT:,} parameters, "
            f"but found {parameter_count:,}."
        )

    backbone_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in backbone.state_dict().items()
    }

    print()
    print("Extracted tensor groups:")

    for prefix in REQUIRED_PREFIXES:
        matching_keys = [
            key
            for key in backbone_state
            if key.startswith(prefix)
        ]

        if not matching_keys:
            raise RuntimeError(
                f"No tensors were found under prefix '{prefix}'."
            )

        tensor_count = len(matching_keys)
        element_count = sum(
            backbone_state[key].numel()
            for key in matching_keys
        )

        print(
            f"{prefix:35s} "
            f"tensors={tensor_count:4d}, "
            f"elements={element_count:,}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "state_dict": backbone_state,
        "meta": {
            "source_checkpoint": str(input_path),
            "source_model_class": type(full_model).__name__,
            "backbone_class": type(backbone).__name__,
            "parameter_count": parameter_count,
            "output_channels": [224, 224, 224],
            "output_strides": [8, 16, 32],
            "components": [
                "DINOv3 ViT-S/16",
                "SpatialPriorModulev2",
                "multiscale fusion convolutions",
                "multiscale fusion SyncBatchNorm",
            ],
        },
    }

    torch.save(
        checkpoint,
        output_path,
    )

    # Reload the written file and verify exact equality.
    written_checkpoint = torch.load(
        output_path,
        map_location="cpu",
        weights_only=False,
    )

    written_state = written_checkpoint["state_dict"]

    if written_state.keys() != backbone_state.keys():
        raise RuntimeError(
            "Written checkpoint keys do not match source backbone keys."
        )

    for key, source_tensor in backbone_state.items():
        written_tensor = written_state[key]

        if not torch.equal(source_tensor, written_tensor):
            raise RuntimeError(
                f"Written tensor differs from source: {key}"
            )

    print()
    print("DINOSTAs export passed.")
    print("Output:", output_path)
    print("Parameters:", f"{parameter_count:,}")
    print("State-dict tensors:", len(backbone_state))


if __name__ == "__main__":
    main()