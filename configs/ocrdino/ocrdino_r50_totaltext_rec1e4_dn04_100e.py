_base_ = '../dino/dino-4scale_r50_8xb2-12e_coco.py'


custom_imports = dict(
    imports=[
        'mmdet.models.detectors.est_dino_totaltext',
        'mmdet.evaluation.metrics.ocrdino_totaltext_metric',
    ],
    allow_failed_imports=False,
)


data_root = '/root/autodl-tmp/dataset/totaltext/'
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
        text_types=('text', 'latex'),
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
        text_types=('text', 'latex'),
        max_length=25,
        ignore_index=-100,
    ),
    dict(
        type='PackOCRDinoInputs',
    ),
]


test_pipeline = val_pipeline


model = dict(
    _delete_=True,

    type='ESTDINOTotalText',

    num_queries=100,

    with_box_refine=True,
    as_two_stage=True,

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

    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(
            type='BN',
            requires_grad=False,
        ),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet50',
        ),
    ),

    neck=dict(
        type='ChannelMapper',
        in_channels=[
            512,
            1024,
            2048,
        ],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(
            type='GN',
            num_groups=32,
        ),
        num_outs=4,
    ),

    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,
                dropout=0.0,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
        ),
    ),

    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        post_norm_cfg=None,
        layer_cfg=dict(
            self_attn_cfg=dict(
                embed_dims=256,
                num_heads=8,
                dropout=0.0,
            ),
            cross_attn_cfg=dict(
                embed_dims=256,
                num_levels=4,
                dropout=0.0,
            ),
            ffn_cfg=dict(
                embed_dims=256,
                feedforward_channels=2048,
                ffn_drop=0.0,
            ),
        ),
    ),

    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20,
    ),

    bbox_head=dict(
        type='ESTDINOHead',

        num_classes=1,

        sync_cls_avg_factor=True,

        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0,
        ),

        loss_bbox=dict(
            type='L1Loss',
            loss_weight=5.0,
        ),

        loss_iou=dict(
            type='GIoULoss',
            loss_weight=2.0,
        ),

        rec_loss_weight=1.0,
        dn_rec_loss_weight=1.0,
        rec_ignore_index=-100,
    ),

    dn_cfg=dict(
        label_noise_scale=0.5,

        # Changed from 1.0 to the ESTextSpotter value.
        box_noise_scale=0.4,

        group_cfg=dict(
            dynamic=True,
            num_groups=None,
            num_dn_queries=100,
        ),
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

        box_format='cxcywh',

        normalized_boxes=True,

        detach_boxes=True,

        canonical_box_size=224,

        canonical_level=4,
    ),

    text_query_group_cfg=dict(
        num_rec_queries=25,
    ),

    text_decoder_cfg=dict(
        num_layers=6,

        embed_dims=256,

        num_feature_levels=4,

        num_heads=8,

        num_points=4,

        ffn_dims=2048,

        dropout=0.0,

        activation='relu',

        im2col_step=64,

        num_rec_classes=97,
    ),

    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0,
                ),
                dict(
                    type='BBoxL1Cost',
                    weight=5.0,
                    box_format='xywh',
                ),
                dict(
                    type='IoUCost',
                    iou_mode='giou',
                    weight=2.0,
                ),
            ],
        ),
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

    batch_sampler=None,

    sampler=dict(
        type='DefaultSampler',
        shuffle=True,
    ),

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


test_evaluator = val_evaluator


optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',

        # Normal detector learning rate.
        lr=1.0e-5,

        weight_decay=1.0e-4,
    ),

    paramwise_cfg=dict(
        custom_keys=dict(

            # Pretrained ResNet backbone:
            # 1e-5 * 0.1 = 1e-6.
            backbone=dict(
                lr_mult=0.1,
            ),

            # Randomly initialized ESTextSpotter task-aware decoder:
            # 1e-5 * 10 = 1e-4.
            decoder=dict(
                lr_mult=10.0,
            ),
        ),
    ),

    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,
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

        end=100,

        milestones=[
            80,
            90,
        ],

        gamma=0.1,
    ),
]


train_cfg = dict(
    _delete_=True,

    type='EpochBasedTrainLoop',

    max_epochs=100,

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


randomness = dict(
    seed=42,
    deterministic=False,
)


load_from = (
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_totaltext_init.pth'
)

resume = False


work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_totaltext_rec1e4_dn04_100e'
)