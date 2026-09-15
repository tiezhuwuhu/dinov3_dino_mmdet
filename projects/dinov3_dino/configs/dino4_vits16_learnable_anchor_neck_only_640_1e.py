_base_ = [
    './dino4_vits16_learnable_anchor_improved_multiscale_24e.py',
]


# ============================================================
# Stage A1
#
# ViT       : frozen
# Anchor Neck: train
# DINO      : frozen
#
# Fixed input = 640x640
#
# 4 GPUs
# batch/GPU = 1
# accumulation = 4
# effective global batch = 16
# ============================================================


init_checkpoint = (
    '/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/'
    'merged/'
    'dinov3_vits16_learnable_anchor_official_dino4_init.pth'
)


# ============================================================
# Model
# ============================================================

model = dict(
    type='FreezeableDINO',

    # Freeze all except neck.
    train_neck_only=True,

    backbone=dict(
        frozen=True,
    ),
)


# ============================================================
# Fixed 640 training pipeline
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
# Training schedule
# ============================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=1,
    val_interval=1,
)


# One epoch alignment stage:
# no LR decay.
param_scheduler = []


# ============================================================
# Optimizer
#
# Inherit AdamW / LR / clipping from improved config.
#
# Only modify accumulation.
# ============================================================

optim_wrapper = dict(
    accumulative_counts=4,
)


# ============================================================
# Checkpoint
# ============================================================

load_from = init_checkpoint

resume = False


default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        save_best='coco/bbox_mAP',
        rule='greater',
        max_keep_ckpts=2,
    ),
)


work_dir = (
    'work_dirs/'
    'dino4_vits16_learnable_anchor_neck_only_640_1e'
)