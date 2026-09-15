_base_ = [
    './dino4_vits16_sta_improved_multiscale_24e.py',
]


# ============================================================
# Learnable-Anchor ViT-S/16 + MMDetection DINO
#
# Backbone:
#   Lightly DINOv3 ViT-S/16
#   output all 12 transformer blocks
#
# Neck:
#   LearnableAnchorViTFeaturePyramid
#
#   Early:
#       mean(F0,F1,F2) + anchor F3
#
#   Middle:
#       mean(F4,F5,F6) + anchor F7
#
#   Deep:
#       mean(F8,F9,F10) + anchor F11
#
# alpha_init = 0.25
#
# Detector:
#   inherited improved MMDetection DINO recipe
# ============================================================


model = dict(

    backbone=dict(
        _delete_=True,
        type='LightlyDINOv3ViTS16',
        out_indices=tuple(range(12)),
        frozen=False,
    ),

    neck=dict(
        _delete_=True,
        type='LearnableAnchorViTFeaturePyramid',
        in_channels=384,
        out_channels=256,
        alpha_init=0.25,
    ),
)


# Initialization checkpoint will be supplied later.
load_from = None

resume = False


work_dir = (
    'work_dirs/'
    'dino4_vits16_learnable_anchor_improved_multiscale_24e'
)