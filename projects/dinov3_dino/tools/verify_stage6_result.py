from __future__ import annotations

import argparse
from pathlib import Path

import torch


VIT_PREFIX = (
    "backbone.dinostas._model_wrapper._model."
)

ADAPTER_PREFIXES = (
    "backbone.dinostas.sta.",
    "backbone.dinostas.convs.",
    "backbone.dinostas.norms.",
)

SYNC_BN_BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--initial",
        required=True,
    )

    parser.add_argument(
        "--trained",
        required=True,
    )

    return parser.parse_args()


def load_state(
    path: Path,
) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state = checkpoint["model"]
    else:
        state = checkpoint

    return {
        key.removeprefix("module."): tensor
        for key, tensor in state.items()
        if torch.is_tensor(tensor)
    }


def count_changed(
    initial_state,
    trained_state,
    predicate,
):
    changed = []
    unchanged = []

    for key in initial_state:
        if not predicate(key):
            continue

        if torch.equal(
            initial_state[key],
            trained_state[key],
        ):
            unchanged.append(key)
        else:
            changed.append(key)

    return changed, unchanged


def main() -> None:
    args = parse_args()

    initial_path = Path(
        args.initial
    ).expanduser().resolve()

    trained_path = Path(
        args.trained
    ).expanduser().resolve()

    if not initial_path.is_file():
        raise FileNotFoundError(initial_path)

    if not trained_path.is_file():
        raise FileNotFoundError(trained_path)

    initial_state = load_state(initial_path)
    trained_state = load_state(trained_path)

    if initial_state.keys() != trained_state.keys():
        missing = sorted(
            initial_state.keys() - trained_state.keys()
        )

        unexpected = sorted(
            trained_state.keys() - initial_state.keys()
        )

        raise RuntimeError(
            "State-dict key mismatch:\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    changed_vit, unchanged_vit = count_changed(
        initial_state,
        trained_state,
        lambda key: key.startswith(VIT_PREFIX),
    )

    if changed_vit:
        raise RuntimeError(
            "Frozen ViT tensors changed:\n"
            + "\n".join(changed_vit)
        )

    changed_adapter, unchanged_adapter = count_changed(
        initial_state,
        trained_state,
        lambda key: key.startswith(ADAPTER_PREFIXES)
        and not key.endswith(SYNC_BN_BUFFER_SUFFIXES),
    )

    changed_bridge, unchanged_bridge = count_changed(
        initial_state,
        trained_state,
        lambda key: key.startswith("neck."),
    )

    changed_detector, unchanged_detector = count_changed(
        initial_state,
        trained_state,
        lambda key: (
            not key.startswith("backbone.")
            and not key.startswith("neck.")
        ),
    )

    changed_bn_buffers, unchanged_bn_buffers = count_changed(
        initial_state,
        trained_state,
        lambda key: (
            key.startswith(ADAPTER_PREFIXES)
            and key.endswith(SYNC_BN_BUFFER_SUFFIXES)
        ),
    )

    if not changed_adapter:
        raise RuntimeError(
            "No Lightly adapter tensor changed."
        )

    if not changed_bridge:
        raise RuntimeError(
            "No bridge tensor changed."
        )

    if not changed_detector:
        raise RuntimeError(
            "No DINO detector tensor changed."
        )

    if changed_bn_buffers:
        raise RuntimeError(
            "Frozen adapter SyncBN running statistics changed:\n"
            + "\n".join(changed_bn_buffers)
        )

    print("=" * 80)
    print("Initial:", initial_path)
    print("Trained:", trained_path)
    print()
    print(
        "ViT:",
        f"changed={len(changed_vit)}, "
        f"unchanged={len(unchanged_vit)}",
    )
    print(
        "Lightly adapter:",
        f"changed={len(changed_adapter)}, "
        f"unchanged={len(unchanged_adapter)}",
    )
    print(
        "Residual bridge:",
        f"changed={len(changed_bridge)}, "
        f"unchanged={len(unchanged_bridge)}",
    )
    print(
        "DINO detector:",
        f"changed={len(changed_detector)}, "
        f"unchanged={len(unchanged_detector)}",
    )
    print(
        "Adapter SyncBN buffers:",
        f"changed={len(changed_bn_buffers)}, "
        f"unchanged={len(unchanged_bn_buffers)}",
    )
    print()
    print("Verification passed:")
    print("  frozen DINOv3 ViT remained exactly unchanged")
    print("  Lightly feature adapter was updated")
    print("  residual multiscale bridge was updated")
    print("  MMDetection DINO detector was updated")
    print("  adapter SyncBN running statistics remained unchanged")
    print("=" * 80)


if __name__ == "__main__":
    main()