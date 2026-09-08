from __future__ import annotations

import argparse
from pathlib import Path

import torch


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


def load_state(path: Path) -> dict[str, torch.Tensor]:
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
            f"State-dict key mismatch:\n"
            f"missing={missing}\n"
            f"unexpected={unexpected}"
        )

    changed_neck = []
    unchanged_neck = []
    changed_non_neck = []

    for key in initial_state:
        is_equal = torch.equal(
            initial_state[key],
            trained_state[key],
        )

        if key.startswith("neck."):
            if is_equal:
                unchanged_neck.append(key)
            else:
                changed_neck.append(key)
        elif not is_equal:
            changed_non_neck.append(key)

    if changed_non_neck:
        raise RuntimeError(
            "Non-neck tensors changed during bridge-only training:\n"
            + "\n".join(changed_non_neck)
        )

    if not changed_neck:
        raise RuntimeError(
            "No neck tensors changed. Bridge training did not update "
            "the bridge."
        )

    print("=" * 72)
    print("Initial:", initial_path)
    print("Trained:", trained_path)
    print()
    print("Changed neck tensors:", len(changed_neck))
    print("Unchanged neck tensors:", len(unchanged_neck))
    print("Changed non-neck tensors:", len(changed_non_neck))
    print()
    print("Examples of changed neck tensors:")

    for key in changed_neck[:20]:
        print(" ", key)

    print()
    print("Verification passed:")
    print("  at least one neck tensor changed")
    print("  every non-neck tensor remained exactly unchanged")
    print("=" * 72)


if __name__ == "__main__":
    main()