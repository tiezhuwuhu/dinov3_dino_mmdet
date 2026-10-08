_base_ = [
    './ocrdino_r50_estextspotter_4scale_pretrain_3datasets_25e_2gpu_bs4_acc2.py'
]


# ============================================================
# Stage-2: Total-Text fine-tuning
#
# Initialization:
#   4-scale OCRDINO Stage-1 pretrained model
#
# GPUs:
#   2
#
# local batch:
#   4 / GPU
#
# accumulation:
#   2
#
# effective batch:
#   2 * 4 * 2 = 16
# ============================================================


stage1_checkpoint = (
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_estextspotter_4scale_stage1_3datasets_25e.pth'
)


# ============================================================
# MODEL
#
# Keep the existing 4-scale OCRDINO architecture unchanged.
#
# Total-Text finetune uses DN box noise = 0.4.
# ============================================================

model = dict(
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=0.4,

        group_cfg=dict(
            dynamic=True,
            num_groups=None,
            num_dn_queries=100,
        ),
    ),
)


# ============================================================
# TRAIN PIPELINE
#
# Keep the Stage-1 augmentation/resize pipeline.
#
# IMPORTANT:
# Training must remain:
#
#   Load annotations
#       -> Resize
#
# because training GT boxes must follow the resized image.
# ============================================================

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

        scales=[
            (1600, 640),
            (1600, 672),
            (1600, 704),
            (1600, 736),
            (1600, 768),
            (1600, 800),
            (1600, 832),
            (1600, 864),
            (1600, 896),
        ],

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
# TOTAL-TEXT TRAIN
# ============================================================

train_dataloader = dict(
    batch_size=4,

    num_workers=4,
    persistent_workers=True,

    drop_last=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

    batch_sampler=None,

    dataset=dict(
        _delete_=True,

        type='OCRDinoDataset',

        data_root=(
            '/root/autodl-tmp/dataset/'
            'totaltext/'
        ),

        ann_file='ocrdino_train.json',

        data_prefix=dict(
            img_path='',
        ),

        metainfo=dict(
            classes=('text', ),
        ),

        filter_cfg=None,

        serialize_data=False,

        pipeline=train_pipeline,
    ),
)


# ============================================================
# VALIDATION / TEST
#
# IMPORTANT:
#
# This is the coordinate-system fix we have already verified.
#
# Validation MUST be:
#
#   LoadImage
#   -> Resize
#   -> LoadAnnotations
#
# Therefore:
#
# GT bbox   = original-image coordinates
# Pred bbox = rescale=True -> original-image coordinates
#
# Metric then compares the same coordinate system.
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

    num_workers=4,
    persistent_workers=True,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        _delete_=True,

        type='OCRDinoDataset',

        data_root=(
            '/root/autodl-tmp/dataset/'
            'totaltext/'
        ),

        ann_file='ocrdino_test.json',

        data_prefix=dict(
            img_path='',
        ),

        metainfo=dict(
            classes=('text', ),
        ),

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
# Total-Text official fine-tuning settings:
#
# model LR               = 1e-4
# backbone               = 1e-5
# reference_points       = 1e-5
# sampling_offsets       = 1e-5
# weight decay           = 1e-4
# grad clip              = 0.1
#
# No old decoder lr_mult=10.
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
# OFFICIAL TOTAL-TEXT FINETUNE:
#
# epochs = 5
# lr_drop = 11
#
# Therefore no LR drop occurs during the 5 epochs.
# ============================================================

param_scheduler = []


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=5,
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
# load_from = initialize model weights only.
#
# resume=False means:
#
# DO NOT restore:
# optimizer
# scheduler
# epoch
# iteration
#
# Fine-tuning begins from epoch 1 with a new optimizer.
# ============================================================

load_from = stage1_checkpoint
resume = False


auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)


randomness = dict(
    seed=42,
    deterministic=False,
)


default_hooks = dict(
    checkpoint=dict(
        _delete_=True,

        type='CheckpointHook',

        by_epoch=True,
        interval=1,

        max_keep_ckpts=5,

        save_last=True,

        save_optimizer=True,
        save_param_scheduler=True,

        save_best='ocrdino/e2e_f1',
        rule='greater',
    ),

    logger=dict(
        type='LoggerHook',
        interval=20,
    ),
)


work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_totaltext_'
    'stage1pretrain_finetune_5e_'
    '2gpu_bs4_acc2'
)