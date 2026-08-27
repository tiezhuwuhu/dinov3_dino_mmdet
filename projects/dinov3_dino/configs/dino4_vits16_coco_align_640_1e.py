_base_ = (
    "../../../configs/dino/"
    "dino-4scale_r50_8xb2-12e_coco.py"
)

custom_imports = dict(
    imports=["projects.dinov3_dino"],
    allow_failed_imports=False,
)

data_root = (
    "/root/autodl-tmp/dinov3_dino_mmdet/data/coco/"
)

merged_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/"
    "checkpoints/merged/"
    "dinov3_vits16_lightly_coco_dino4_init.pth"
)

model = dict(
    data_preprocessor=dict(
        type="DetDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=16,
    ),

    backbone=dict(
        _delete_=True,
        type="LightlyDINOv3ViTS16",
        out_indices=(5, 8, 11),

        # Stage 1: preserve the COCO-finetuned backbone.
        frozen=True,
    ),

    neck=dict(
        _delete_=True,
        type="ViTFeaturePyramid",
        in_channels=(384, 384, 384),
        out_channels=256,
    ),
)

# Letterbox-style fixed 640.
train_pipeline = [
    dict(
        type="LoadImageFromFile",
        backend_args=None,
    ),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
    ),
    dict(
        type="RandomFlip",
        prob=0.5,
    ),
    dict(
        type="Resize",
        scale=(640, 640),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114)),
    ),
    dict(
        type="PackDetInputs",
    ),
]

test_pipeline = [
    dict(
        type="LoadImageFromFile",
        backend_args=None,
    ),
    dict(
        type="Resize",
        scale=(640, 640),
        keep_ratio=True,
    ),
    dict(
        type="Pad",
        size=(640, 640),
        pad_val=dict(img=(114, 114, 114)),
    ),
    dict(
        type="LoadAnnotations",
        with_bbox=True,
    ),
    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=(
            "annotations/instances_train2017.json"
        ),
        data_prefix=dict(
            img="train2017/",
        ),
        filter_cfg=dict(
            filter_empty_gt=False,
            min_size=32,
        ),
        pipeline=train_pipeline,
        backend_args=None,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),
    dataset=dict(
        type="CocoDataset",
        data_root=data_root,
        ann_file=(
            "annotations/instances_val2017.json"
        ),
        data_prefix=dict(
            img="val2017/",
        ),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=None,
    ),
)

test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=(
        data_root
        + "annotations/instances_val2017.json"
    ),
    metric="bbox",
    format_only=False,
    backend_args=None,
)

test_evaluator = val_evaluator

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=1e-4,
    ),
    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,
    ),
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(
                lr_mult=0.1,
            ),
        },
    ),
)

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=1,
    val_interval=1,
)

val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# One-epoch structural alignment; no LR decay.
param_scheduler = []

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=2,
    ),
)

load_from = merged_checkpoint
resume = False

randomness = dict(
    seed=0,
    deterministic=False,
)

auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)