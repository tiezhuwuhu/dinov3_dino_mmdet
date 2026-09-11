_base_ = './point_dino_r50_shanghaitech_stage4_local_fidt_native_12e.py'


# ============================================================
# BCData benchmark protocol
#
# Original images: 640 x 640
# Paper input:     512 x 512
#
# 512 / 640 = 0.8
#
# Published metrics:
#   F1@5px  @ 512
#   F1@10px @ 512
#
# Current PointMetric evaluates rescaled predictions in original
# 640x640 coordinates, therefore:
#
#   5 / 0.8  = 6.25 px
#   10 / 0.8 = 12.5 px
# ============================================================

data_root = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'data/BCData/'
)

metainfo = dict(
    classes=('cell', )
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
            ]
        )
    ),

    test_cfg=dict(
        max_per_img=900,
        rescale_points=False,
    ),

    point_fidt_head=dict(
        enabled=True,
        local_radius_px=16.0,
        fidt_loss_weight=2.0,
        debug=False,
    ),
)


# ============================================================
# Pipeline
#
# BCData native: 640 x 640
# Benchmark input: 512 x 512
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
        type='Resize',
        scale=(512, 512),
        keep_ratio=False,
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


test_pipeline = [
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
        type='Resize',
        scale=(512, 512),
        keep_ratio=False,
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
# ============================================================

train_dataloader = dict(
    _delete_=True,

    batch_size=2,
    num_workers=2,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

    batch_sampler=dict(
        type='AspectRatioBatchSampler',
    ),

    dataset=dict(
        type='BaseDetDataset',

        data_root=data_root,
        ann_file='train_point.json',

        data_prefix=dict(
            img_path='images/train/',
        ),

        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)


# ============================================================
# Validation
#
# Validation is used for best-checkpoint selection.
# ============================================================

val_dataloader = dict(
    _delete_=True,

    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type='BaseDetDataset',

        data_root=data_root,
        ann_file='validation_point.json',

        data_prefix=dict(
            img_path='images/validation/',
        ),

        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)


# ============================================================
# Test
#
# Test is used only for final reported result.
# ============================================================

test_dataloader = dict(
    _delete_=True,

    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type='BaseDetDataset',

        data_root=data_root,
        ann_file='test_point.json',

        data_prefix=dict(
            img_path='images/test/',
        ),

        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)


# ============================================================
# Evaluation
#
# MMDetection rescale=True:
# model input 512 -> prediction output back to original 640.
#
# Therefore:
#
# 6.25 px @640 == 5 px  @512
# 12.5 px @640 == 10 px @512
# ============================================================

val_evaluator = dict(
    _delete_=True,

    type='PointMetric',

    distance_thresholds=[
        5,
        10,
    ],

    score_threshold=0.5,
)


test_evaluator = dict(
    _delete_=True,

    type='PointMetric',

    distance_thresholds=[
        5,
        10,
    ],

    score_threshold=0.5,
)


# ============================================================
# Checkpoint selection
#
# Equivalent to benchmark F1@10 @512
# ============================================================

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,

        save_best='point/f1@10px',
        rule='greater',

        save_last=True,
        max_keep_ckpts=3,
    ),
)


# ============================================================
# Initialization
#
# Fair cross-dataset experiment:
# start from the common Stage 2 Point-DINO initialization.
#
# Do NOT initialize from a ShanghaiTech-trained Stage3/Stage4 model.
# ============================================================

load_from = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'checkpoints/mmdet/'
    'point_dino_stage2_step2_init.pth'
)


work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/'
    'point_dino_r50_bcdata_stage4_local_fidt_512_12e'
)