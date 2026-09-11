from pathlib import Path

import cv2
import torch

from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint

from mmdet.registry import MODELS
from mmdet.structures import DetDataSample


# ============================================================
# Fixed model
# ============================================================

CONFIG_PATH = (
    "/root/autodl-tmp/dinov3_dino_mmdet/mmdetection/"
    "configs/dino/"
    "point_dino_r50_own12_stage4_lam0_native_100e.py"
)

CHECKPOINT_PATH = (
    "/root/autodl-tmp/dinov3_dino_mmdet/"
    "work_dirs/"
    "point_dino_own12_stage4_lam0_native_100e/"
    "best_point_f1@10px_epoch_75.pth"
)

SCORE_THRESHOLD = 0.288


# ============================================================
# Predictor
# ============================================================

class PointDINOPredictor:

    def __init__(
            self,
            config_path,
            checkpoint_path,
            score_threshold=0.5,
            device="cuda:0"):

        self.score_threshold = float(
            score_threshold
        )

        self.device = torch.device(
            device
        )

        init_default_scope("mmdet")

        cfg = Config.fromfile(
            config_path
        )

        self.model = MODELS.build(
            cfg.model
        )

        self.model.to(
            self.device
        )

        load_checkpoint(
            self.model,
            checkpoint_path,
            map_location="cpu",
            strict=False,
        )

        self.model.eval()

    @torch.no_grad()
    def predict(self, image):
        """
        Args:
            image:
                OpenCV BGR image,
                shape = [H, W, 3]

        Returns:
            dict:
                {
                    "points": [[x, y], ...],
                    "scores": [score, ...]
                }

        Coordinates are in the input image's
        original pixel coordinate system.
        """

        if image is None:
            raise ValueError(
                "input image is None"
            )

        if image.ndim != 3:
            raise ValueError(
                f"Expected HxWxC image, "
                f"got shape={image.shape}"
            )

        height, width = (
            image.shape[:2]
        )

        # ----------------------------------------------------
        # HWC BGR uint8 -> CHW tensor
        #
        # Normalization / BGR->RGB is handled by
        # the model data_preprocessor.
        # ----------------------------------------------------

        input_tensor = (
            torch.from_numpy(image)
            .permute(2, 0, 1)
            .contiguous()
        )

        # ----------------------------------------------------
        # Minimal MMDetection data sample
        # ----------------------------------------------------

        data_sample = (
            DetDataSample()
        )

        data_sample.set_metainfo(
            dict(
                img_id=0,

                ori_shape=(
                    height,
                    width,
                ),

                img_shape=(
                    height,
                    width,
                ),

                scale_factor=(
                    1.0,
                    1.0,
                ),
            )
        )

        raw_data = dict(
            inputs=[
                input_tensor
            ],
            data_samples=[
                data_sample
            ],
        )

        # ----------------------------------------------------
        # Preprocess
        # ----------------------------------------------------

        data = (
            self.model
            .data_preprocessor(
                raw_data,
                training=False,
            )
        )

        inputs = data[
            "inputs"
        ]

        data_samples = data[
            "data_samples"
        ]

        # ----------------------------------------------------
        # Point-DINO prediction
        # ----------------------------------------------------

        outputs = self.model.predict(
            inputs,
            data_samples,
            rescale=False,
        )

        pred_instances = (
            outputs[0]
            .pred_instances
        )

        points = (
            pred_instances
            .points
            .detach()
            .cpu()
        )

        scores = (
            pred_instances
            .scores
            .detach()
            .cpu()
        )

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        keep = (
            scores
            >= self.score_threshold
        )

        points = points[keep]
        scores = scores[keep]

        # ----------------------------------------------------
        # Return ordinary Python objects
        # ----------------------------------------------------

        return {
            "points": (
                points.numpy().tolist()
            ),

            "scores": (
                scores.numpy().tolist()
            ),
        }


# ============================================================
# Load model ONCE
# ============================================================

predictor = PointDINOPredictor(
    config_path=CONFIG_PATH,
    checkpoint_path=CHECKPOINT_PATH,
    score_threshold=SCORE_THRESHOLD,
)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    IMAGE_PATH = (
        "/root/autodl-tmp/dinov3_dino_mmdet/"
        "data/export/export/6.png"
    )

    image = cv2.imread(IMAGE_PATH)
    result = predictor.predict(image)
    print(result)