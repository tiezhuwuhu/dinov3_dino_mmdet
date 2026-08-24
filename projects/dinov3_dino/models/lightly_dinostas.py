from __future__ import annotations

from pathlib import Path

import torch
from mmengine.model import BaseModule
from torch import Tensor

from mmdet.registry import MODELS


@MODELS.register_module(force=True)
class LightlyDINOSTAs(BaseModule):
    """Strict MMDetection wrapper for Lightly DINOSTAs.

    Architecture is reconstructed from the original Lightly export.

    Actual backbone weights are then loaded strictly from the separately
    exported DINOSTAs state-dict checkpoint.

    Outputs:

    - P3: [B, 224, H/8, W/8]
    - P4: [B, 224, H/16, W/16]
    - P5: [B, 224, H/32, W/32]
    """

    def __init__(
        self,
        architecture_checkpoint: str,
        backbone_checkpoint: str,
        frozen: bool = False,
        validate_outputs: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        architecture_path = (
            Path(architecture_checkpoint)
            .expanduser()
            .resolve()
        )

        backbone_path = (
            Path(backbone_checkpoint)
            .expanduser()
            .resolve()
        )

        if not architecture_path.is_file():
            raise FileNotFoundError(
                "Lightly architecture checkpoint not found: "
                f"{architecture_path}"
            )

        if not backbone_path.is_file():
            raise FileNotFoundError(
                "Exported DINOSTAs checkpoint not found: "
                f"{backbone_path}"
            )

        try:
            import lightly_train
        except ImportError as exc:
            raise ImportError(
                "lightly_train is required to reconstruct DINOSTAs."
            ) from exc

        # Reconstruct the exact LightlyTrain 0.17.0 architecture stored
        # in the complete LTDETR export.
        full_model = lightly_train.load_model(
            architecture_path,
            device="cpu",
        )

        if not hasattr(full_model, "backbone"):
            raise AttributeError(
                "The Lightly model has no top-level backbone."
            )

        if type(full_model.backbone).__name__ != "DINOSTAs":
            raise TypeError(
                "Expected DINOSTAs, but got "
                f"{type(full_model.backbone)!r}."
            )

        self.dinostas = full_model.backbone

        # Drop Lightly HybridEncoder, RT-DETR decoder, preprocessing and
        # postprocessing. Only DINOSTAs remains registered in this wrapper.
        del full_model

        exported_checkpoint = torch.load(
            backbone_path,
            map_location="cpu",
            weights_only=False,
        )

        if "state_dict" not in exported_checkpoint:
            raise KeyError(
                "Exported DINOSTAs checkpoint has no 'state_dict'."
            )

        exported_state = exported_checkpoint["state_dict"]

        # Strict loading rejects every missing, unexpected or mismatched key.
        self.dinostas.load_state_dict(
            exported_state,
            strict=True,
        )

        # Verify exact equality after loading.
        loaded_state = self.dinostas.state_dict()

        if loaded_state.keys() != exported_state.keys():
            raise RuntimeError(
                "Loaded DINOSTAs keys differ from exported keys."
            )

        for key, source_tensor in exported_state.items():
            loaded_tensor = loaded_state[key]

            if not torch.equal(
                loaded_tensor.cpu(),
                source_tensor.cpu(),
            ):
                raise RuntimeError(
                    f"DINOSTAs tensor failed exact verification: {key}"
                )

        self.architecture_checkpoint = str(architecture_path)
        self.backbone_checkpoint = str(backbone_path)
        self.frozen = bool(frozen)
        self.validate_outputs = bool(validate_outputs)

        if self.frozen:
            self._freeze_dinostas()

    def init_weights(self) -> None:
        """Do not reinitialize the already loaded Lightly weights."""

        self._is_init = True

    def _freeze_dinostas(self) -> None:
        for parameter in self.dinostas.parameters():
            parameter.requires_grad_(False)

        # DINOSTAs contains SyncBatchNorm running statistics.
        self.dinostas.eval()

    def train(self, mode: bool = True):
        super().train(mode)

        if self.frozen:
            self.dinostas.eval()

        return self

    def _validate_outputs(
        self,
        x: Tensor,
        outputs: tuple[Tensor, Tensor, Tensor],
    ) -> None:
        expected_channels = (224, 224, 224)
        expected_strides = (8, 16, 32)

        input_height = x.shape[-2]
        input_width = x.shape[-1]

        for level, (
            feature,
            expected_channel,
            expected_stride,
        ) in enumerate(
            zip(
                outputs,
                expected_channels,
                expected_strides,
            )
        ):
            if not torch.is_tensor(feature):
                raise TypeError(
                    f"Feature {level} is not a Tensor."
                )

            if feature.ndim != 4:
                raise RuntimeError(
                    f"Feature {level} is not BCHW: "
                    f"{tuple(feature.shape)}"
                )

            if feature.shape[0] != x.shape[0]:
                raise RuntimeError(
                    f"Feature {level} batch size mismatch."
                )

            if feature.shape[1] != expected_channel:
                raise RuntimeError(
                    f"Feature {level} expected "
                    f"{expected_channel} channels, but got "
                    f"{feature.shape[1]}."
                )

            expected_height = input_height // expected_stride
            expected_width = input_width // expected_stride

            if feature.shape[-2:] != (
                expected_height,
                expected_width,
            ):
                raise RuntimeError(
                    f"Feature {level} expected spatial shape "
                    f"{(expected_height, expected_width)}, but got "
                    f"{tuple(feature.shape[-2:])}."
                )

            if not torch.isfinite(feature).all():
                raise FloatingPointError(
                    f"Feature {level} contains NaN or Inf."
                )

    def forward(
        self,
        x: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        outputs = self.dinostas(x)

        if not isinstance(outputs, (tuple, list)):
            raise TypeError(
                "DINOSTAs must return tuple/list, but returned "
                f"{type(outputs)!r}."
            )

        if len(outputs) != 3:
            raise RuntimeError(
                f"DINOSTAs must return 3 features, got {len(outputs)}."
            )

        output_tuple = tuple(outputs)

        if self.validate_outputs:
            self._validate_outputs(
                x,
                output_tuple,
            )

        return output_tuple