_base_ = [
    './dino4_vits16_learnable_anchor_improved_multiscale_24e.py',
]


# ============================================================
# Stage A2
#
# ViT        : frozen
# Anchor Neck: train
# DINO       : train
#
# Fixed 640
#
# batch/GPU = 1
# GPUs = 4
# accumulation = 4
# effective global batch = 16
# ============================================================


stage_a1_checkpoint = (
    '/root/autodl-tmp/dinov3_dino_mmdet/mmdetection/'
    'work_dirs/'
    'dino4_vits16_learnable_anchor_neck_only_640_1e/'
    'epoch_1.pth'
)


# ============================================================
# Model
# ============================================================

model = dict(
    type='FreezeableDINO',

    # Do not use neck-only mode anymore.
    train_neck_only=False,

    # Freeze ViT only.
    backbone=dict(
        frozen=True,
    ),
)


# ============================================================
# Fixed 640 pipeline
# ============================================================

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=True,
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
    ),

    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True,
    ),

    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114),
        ),
    ),

    dict(
        type='PackDetInputs',
    ),
]


val_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=None,
    ),

    dict(
        type='Resize',
        scale=(640, 640),
        keep_ratio=True,
    ),

    dict(
        type='Pad',
        size=(640, 640),
        pad_val=dict(
            img=(114, 114, 114),
        ),
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=True,
    ),

    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
    ),
]


# ============================================================
# Dataloader
# ============================================================

train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,

    dataset=dict(
        pipeline=train_pipeline,
    ),
)


val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,

    dataset=dict(
        pipeline=val_pipeline,
    ),
)


test_dataloader = val_dataloader


# ============================================================
# 2 epochs
# ============================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=2,
    val_interval=1,
)


# Short adaptation stage.
# Keep LR constant.
param_scheduler = []


# ============================================================
# Optimizer
#
# Inherit:
# AdamW
# lr = 2e-4
# weight_decay
# reference_points 0.1x
# sampling_offsets 0.1x
# clip_grad = 0.1
#
# Just change accumulation.
# ============================================================

optim_wrapper = dict(
    accumulative_counts=4,
)


# ============================================================
# Load A1
# ============================================================

load_from = stage_a1_checkpoint

resume = False


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=3,
    ),
)


work_dir = (
    'work_dirs/'
    'dino4_vits16_learnable_anchor_neck_dino_640_2e'
)