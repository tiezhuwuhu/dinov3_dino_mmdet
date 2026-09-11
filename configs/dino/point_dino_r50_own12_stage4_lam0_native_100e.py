_base_ = (
    './point_dino_r50_bcdata_'
    'stage4_local_fidt_lam0_512_12e.py'
)


# ============================================================
# Custom grinding-point dataset
# ============================================================

data_root = (
    '/root/autodl-tmp/'
    'dinov3_dino_mmdet/'
    'data/export/'
)

metainfo = dict(
    classes=('root', ),
)


# ============================================================
# Model
# ============================================================

model = dict(

    num_queries=900,

    bbox_head=dict(
        num_classes=1,
        point_euclidean_weight=0.20,
    ),

    dn_cfg=dict(
        point_noise_scale=0.01,
    ),

    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0,
                ),
                dict(
                    type='PointL1Cost',
                    weight=20.0,
                ),
            ],
        ),
    ),

    test_cfg=dict(
        max_per_img=900,

        # No resize is used.
        # Prediction stays directly in native image coordinates.
        rescale_points=False,
    ),

    # Stage-4 code path, but no FIDT supervision.
    point_fidt_head=dict(
        fidt_loss_weight=0.0,
        local_radius_px=16.0,
        debug=False,
    ),
)


# ============================================================
# Native-resolution pipeline
#
# Original:
# width  = 2448
# height = 2048
#
# IMPORTANT:
# No Resize transform.
# ============================================================

train_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_point=True,
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
    ),

    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
        ),
    ),
]


val_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),

    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_point=True,
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
# Train
#
# 11 images
# ============================================================

train_dataloader = dict(
    _delete_=True,

    batch_size=1,

    num_workers=2,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

    dataset=dict(
        type='BaseDetDataset',

        data_root=data_root,

        ann_file='train_point.json',

        data_prefix=dict(
            img_path='export/',
        ),

        metainfo=metainfo,

        pipeline=train_pipeline,
    ),
)


# ============================================================
# Validation
#
# 6.png
# ============================================================

val_dataloader = dict(
    _delete_=True,

    batch_size=1,

    num_workers=1,
    persistent_workers=True,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type='BaseDetDataset',

        data_root=data_root,

        ann_file='val_point.json',

        data_prefix=dict(
            img_path='export/',
        ),

        metainfo=metainfo,

        test_mode=True,

        pipeline=val_pipeline,
    ),
)


# No independent test split yet.
test_dataloader = val_dataloader


# ============================================================
# Evaluation
#
# Native 2448x2048 coordinates.
#
# First exploratory run:
# report 5 / 10 / 20 pixels.
#
# 5px  = very strict
# 10px = medium
# 20px = looser reference
# ============================================================

val_evaluator = dict(
    _delete_=True,

    type='PointMetric',

    distance_thresholds=[
        5.0,
        10.0,
    ],

    # Do NOT inherit BCData's tuned 0.325.
    # Use a fixed neutral operating point while training.
    score_threshold=0.5,
)

test_evaluator = val_evaluator


# ============================================================
# Training
#
# 11 iterations / epoch
# 100 epochs = 1100 iterations
# ============================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',

    max_epochs=100,

    val_interval=5,
)


param_scheduler = [
    dict(
        type='MultiStepLR',

        begin=0,
        end=100,

        by_epoch=True,

        milestones=[
            70,
            90,
        ],

        gamma=0.1,
    ),
]


# ============================================================
# Best checkpoint
#
# Use native 10-pixel F1 as the first checkpoint criterion.
# ============================================================

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',

        by_epoch=True,
        interval=5,

        save_best='point/f1@10px',
        rule='greater',

        save_last=True,
        max_keep_ckpts=5,
    ),
)


# ============================================================
# Initialization
#
# Common Point-DINO initialization.
# No ShanghaiTech / BCData task-specific checkpoint.
# ============================================================

load_from = (
    '/root/autodl-tmp/'
    'dinov3_dino_mmdet/'
    'checkpoints/mmdet/'
    'point_dino_stage2_step2_init.pth'
)


work_dir = (
    '/root/autodl-tmp/'
    'dinov3_dino_mmdet/'
    'work_dirs/'
    'point_dino_own12_stage4_lam0_native_100e'
)