_base_ = [
    './ocrdino_r50_totaltext_rec1e4_dn04_100e.py'
]


# ============================================================
# OCRDINO / ESTDINO Stage-1 pretraining
#
# Model architecture:
#   KEEP CURRENT OCRDINO 4-SCALE ARCHITECTURE UNCHANGED.
#
# Training datasets:
#   MLT2017
#   CurvedSynText150k Part 1
#   CurvedSynText150k Part 2
#
# Hardware:
#   2 GPUs
#   batch_size = 4 / GPU
#   accumulation = 2
#
# Effective batch:
#   2 * 4 * 2 = 16
# ============================================================


metainfo = dict(
    classes=('text', ),
)


# ============================================================
# MODEL
#
# IMPORTANT:
# Do NOT override:
#   backbone.out_indices
#   neck.num_outs
#   encoder num_levels
#   decoder num_levels
#   text_decoder num_feature_levels
#
# The base OCRDINO model is already the validated 4-scale model.
#
# We only override the official pretraining DN settings.
# ============================================================

model = dict(
    dn_cfg=dict(
        label_noise_scale=0.5,

        # ESTextSpotter Pretrain.sh overrides the config value
        # and actually launches pretraining with 1.0.
        box_noise_scale=1.0,

        group_cfg=dict(
            dynamic=True,
            num_groups=None,
            num_dn_queries=100,
        ),
    ),
)


# ============================================================
# OFFICIAL ESTextSpotter PRETRAIN RESIZE
#
# main.py runtime defaults:
#
# short side:
# 640, 672, 704, 736, 768,
# 800, 832, 864, 896
#
# max long side:
# 1600
#
# This part is safe for bbox + OCR text alignment.
# ============================================================

train_scales = [
    (1600, 640),
    (1600, 672),
    (1600, 704),
    (1600, 736),
    (1600, 768),
    (1600, 800),
    (1600, 832),
    (1600, 864),
    (1600, 896),
]


train_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),

    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
    ),

    dict(
        type='RandomChoiceResize',
        scales=train_scales,
        keep_ratio=True,
    ),

    dict(
        type='TokenizeOCRText',
        text_types=(
            'text',
            'latex',
        ),
        max_length=25,
        ignore_index=-100,
    ),

    dict(
        type='PackOCRDinoInputs',
    ),
]


# ============================================================
# DATASETS
# ============================================================

mlt2017_dataset = dict(
    type='OCRDinoDataset',

    data_root=(
        '/root/autodl-tmp/dataset/'
        'ocrdino_pretrain/mlt2017/'
    ),

    ann_file='ocrdino_train.json',

    data_prefix=dict(
        img_path='',
    ),

    metainfo=metainfo,
    filter_cfg=None,
    serialize_data=False,
    pipeline=train_pipeline,
)


syntext1_dataset = dict(
    type='OCRDinoDataset',

    data_root=(
        '/root/autodl-tmp/dataset/'
        'ocrdino_pretrain/syntext1/'
    ),

    ann_file='ocrdino_train.json',

    data_prefix=dict(
        img_path='',
    ),

    metainfo=metainfo,
    filter_cfg=None,
    serialize_data=False,
    pipeline=train_pipeline,
)


syntext2_dataset = dict(
    type='OCRDinoDataset',

    data_root=(
        '/root/autodl-tmp/dataset/'
        'ocrdino_pretrain/syntext2/'
    ),

    ann_file='ocrdino_train.json',

    data_prefix=dict(
        img_path='',
    ),

    metainfo=metainfo,
    filter_cfg=None,
    serialize_data=False,
    pipeline=train_pipeline,
)


train_dataloader = dict(
    batch_size=4,

    num_workers=2,
    persistent_workers=True,

    drop_last=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

    batch_sampler=None,

    dataset=dict(
        _delete_=True,

        type='ConcatDataset',

        datasets=[
            mlt2017_dataset,
            syntext1_dataset,
            syntext2_dataset,
        ],
    ),
)

# ============================================================
# VALIDATION
#
# Official ESTextSpotter pretrain validates on Total-Text.
#
# short side = 1000
# max long side = 1824
# ============================================================

val_pipeline = [
    dict(
        type='LoadImageFromFile',
    ),

    dict(
        type='Resize',
        scale=(1824, 1000),
        keep_ratio=True,
    ),

    # IMPORTANT:
    # Load GT after Resize so evaluation GT remains
    # in original-image coordinates.
    dict(
        type='LoadOCRAnnotations',
        with_bbox=True,
        with_label=True,
    ),

    dict(
        type='TokenizeOCRText',
        text_types=(
            'text',
            'latex',
        ),
        max_length=25,
        ignore_index=-100,
    ),

    dict(
        type='PackOCRDinoInputs',
    ),
]

val_dataloader = dict(
    batch_size=1,

    num_workers=2,
    persistent_workers=True,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type='OCRDinoDataset',

        data_root=(
            '/root/autodl-tmp/'
            'dataset/totaltext/'
        ),

        ann_file='ocrdino_test.json',

        data_prefix=dict(
            img_path='',
        ),

        metainfo=metainfo,
        filter_cfg=None,
        serialize_data=False,
        test_mode=True,
        pipeline=val_pipeline,
    ),
)


test_dataloader = val_dataloader


# ============================================================
# OPTIMIZER
#
# Official ESTextSpotter:
#
# AdamW
# lr                  = 1e-4
# backbone lr         = 1e-5
# reference_points    = 1e-5
# sampling_offsets    = 1e-5
# weight_decay        = 1e-4
# grad clip           = 0.1
#
# IMPORTANT:
# Remove our old Total-Text-specific decoder lr_mult=10.
# ============================================================

optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    accumulative_counts=2,

    optimizer=dict(
        type='AdamW',
        lr=1.0e-4,
        weight_decay=1.0e-4,
    ),

    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,
    ),

    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(
                lr_mult=0.1,
            ),

            'reference_points': dict(
                lr_mult=0.1,
            ),

            'sampling_offsets': dict(
                lr_mult=0.1,
            ),
        },
    ),
)


# ============================================================
# LR SCHEDULE
#
# Official:
#
# epochs = 25
# MultiStepLR
# drops = [18, 21]
#
# No warmup.
# ============================================================

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=25,
        by_epoch=True,

        milestones=[
            18,
            21,
        ],

        gamma=0.1,
    ),
]


# ============================================================
# TRAIN LOOP
# ============================================================

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=25,
    val_interval=1,
)

val_cfg = dict(
    type='ValLoop',
)

test_cfg = dict(
    type='TestLoop',
)


# ============================================================
# INITIALIZATION
#
# Official Stage-1 starts from pretrained ResNet-50 backbone,
# not from our Total-Text epoch100 model.
#
# The inherited backbone init_cfg handles ResNet-50 init.
# ============================================================

load_from = None
resume = False


# ============================================================
# EFFECTIVE BATCH SIZE
#
# 2 GPU * 4 images * accumulation 2 = 16
# ============================================================

auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)


# ============================================================
# RANDOMNESS
# ============================================================

randomness = dict(
    seed=42,
    deterministic=False,
)


# ============================================================
# CHECKPOINT / LOG
# ============================================================

default_hooks = dict(
    checkpoint=dict(
        _delete_=True,
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=25,
        save_last=True,
        save_optimizer=True,
        save_param_scheduler=True,
    ),

    logger=dict(
        type='LoggerHook',
        interval=50,
    ),
)


work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_estextspotter_4scale_'
    'pretrain_3datasets_25e_2gpu_bs4_acc2'
)