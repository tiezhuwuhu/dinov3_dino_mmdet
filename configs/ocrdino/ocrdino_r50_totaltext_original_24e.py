_base_ = [
    '../dino/dino-4scale_r50_8xb2-12e_coco.py',
]

custom_imports = dict(
    imports=[
        'mmdet.models.detectors.est_dino_totaltext',
        'mmdet.evaluation.metrics.ocrdino_totaltext_metric',
    ],
    allow_failed_imports=False,
)

work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_totaltext_original_24e_val'
)

data_root = (
    '/root/autodl-tmp/dataset/totaltext/'
)

metainfo = dict(
    classes=('text',),
)


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


val_pipeline = [
    dict(
        type='LoadImageFromFile',
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


model = dict(
    type='ESTDINOTotalText',

    data_preprocessor=dict(
        type='DetDataPreprocessor',
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
        pad_size_divisor=1,
        batch_augments=None,
    ),

    num_queries=100,

    bbox_head=dict(
        type='ESTDINOHead',
        num_classes=1,
        rec_loss_weight=1.0,
        dn_rec_loss_weight=1.0,
        rec_ignore_index=-100,
    ),

    recognition_query_cfg=dict(
        num_rec_queries=25,
        roi_height=8,
        featmap_strides=(
            8,
            16,
            32,
        ),
        sampling_ratio=1,
        aligned=True,
        canonical_box_size=224,
        canonical_level=4,
        box_format='cxcywh',
        normalized_boxes=True,
        detach_boxes=True,
    ),

    text_query_group_cfg=dict(
        num_rec_queries=25,
    ),

    text_decoder_cfg=dict(
        num_layers=6,
        num_rec_classes=97,
        num_heads=8,
        num_feature_levels=4,
        num_points=4,
        ffn_dims=2048,
        dropout=0.0,
        activation='relu',
        im2col_step=64,
    ),

    test_cfg=dict(
        max_per_img=100,
    ),
)


train_dataloader = dict(
    _delete_=True,

    batch_size=1,

    num_workers=4,

    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

    batch_sampler=None,

    dataset=dict(
        type='OCRDinoDataset',

        data_root=data_root,

        ann_file='ocrdino_train.json',

        data_prefix=dict(
            img_path='',
        ),

        metainfo=metainfo,

        filter_cfg=None,

        serialize_data=False,

        pipeline=train_pipeline,
    ),
)


val_dataloader = dict(
    _delete_=True,

    batch_size=1,

    num_workers=4,

    persistent_workers=True,

    drop_last=False,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),

    dataset=dict(
        type='OCRDinoDataset',

        data_root=data_root,

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


val_evaluator = dict(
    _delete_=True,

    type='OCRDinoTotalTextMetric',

    iou_thr=0.5,

    score_thr=0.3,

    ap_iou_thresholds=(
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ),
)


test_evaluator = dict(
    _delete_=True,

    type='OCRDinoTotalTextMetric',

    iou_thr=0.5,

    score_thr=0.3,

    ap_iou_thresholds=(
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
    ),
)


optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=1.0e-5,
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
        },
    ),

    accumulative_counts=1,
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500,
    ),
    dict(
        type='MultiStepLR',
        by_epoch=True,
        begin=0,
        end=24,
        milestones=[
            18,
            22,
        ],
        gamma=0.1,
    ),
]


train_cfg = dict(
    _delete_=True,

    type='EpochBasedTrainLoop',

    max_epochs=24,

    val_interval=1,
)


val_cfg = dict(
    type='ValLoop',
)


test_cfg = dict(
    type='TestLoop',
)


default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        type='CheckpointHook',

        interval=1,

        by_epoch=True,

        max_keep_ckpts=8,

        save_last=True,

        save_best='ocrdino/e2e_f1',

        rule='greater',
    ),
)


log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True,
)


randomness = dict(
    seed=42,
    deterministic=False,
)


env_cfg = dict(
    cudnn_benchmark=False,

    mp_cfg=dict(
        mp_start_method='fork',
        opencv_num_threads=0,
    ),

    dist_cfg=dict(
        backend='nccl',
    ),
)


load_from = (
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_totaltext_init.pth'
)

resume = False


auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)