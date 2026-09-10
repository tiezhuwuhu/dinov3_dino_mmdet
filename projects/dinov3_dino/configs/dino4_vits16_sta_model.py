_base_ = (
    "../../../configs/dino/"
    "dino-4scale_r50_8xb2-12e_coco.py"
)


custom_imports = dict(
    imports=[
        "projects.dinov3_dino",
    ],
    allow_failed_imports=False,
)


lightly_architecture_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/lightly/cache/"
    "dinov3_vits16_ltdetr_coco_251218_4812416b.pt"
)

lightly_backbone_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/lightly/"
    "dinov3_vits16_lightly_coco_dinostas.pth"
)


model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        std=[
            58.395,
            57.12,
            57.375,
        ],
        bgr_to_rgb=True,

        # DINOv3 uses a patch size of 16 and the adapter outputs through
        # stride 32. Padding to 32 keeps all levels spatially consistent.
        pad_size_divisor=32,
    ),

    backbone=dict(
        _delete_=True,
        type="LightlyDINOSTAs",
        architecture_checkpoint=lightly_architecture_checkpoint,
        backbone_checkpoint=lightly_backbone_checkpoint,
        frozen=False,
        validate_outputs=False,
    ),

    neck=dict(
        _delete_=True,
        type="ResidualMultiScaleBridge",
    
        in_channels=(
            224,
            224,
            224,
        ),
    
        out_channels=256,
        num_outs=4,
        num_groups=32,
    
        validate_inputs=False,
    ),


)


# This file describes the model only.
# The strict combined initialization checkpoint is built separately.
load_from = None
resume = False