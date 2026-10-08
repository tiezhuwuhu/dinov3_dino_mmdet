_base_ = '../dino/dino-4scale_r50_8xb2-12e_coco.py'


data_root = '/root/autodl-tmp/dataset/OmniDocBench/'

vocab_file = (
    '/root/autodl-tmp/dataset/OmniDocBench/'
    'ocrdino_train_vocab.json'
)

num_classes = 28
ocr_vocab_size = 4292


model = dict(
    type='OCRDINO',

    backbone=dict(
        init_cfg=None,
    ),

    bbox_head=dict(
        type='OCRDINOHead',
        num_classes=num_classes,

        ocr_vocab_size=ocr_vocab_size,
        ocr_vocab_file=vocab_file,

        ocr_num_visual_queries=64,
        ocr_max_seq_len=1024,

        ocr_visual_num_layers=2,
        ocr_text_num_layers=2,

        ocr_num_heads=8,
        ocr_ffn_channels=1024,

        ocr_dropout=0.1,
        ocr_loss_weight=1.0,

        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,

        ocr_class_ids=(
            0,
            1,
            2,
            3,
            5,
            7,
            8,
            10,
            12,
            13,
            14,
            15,
            17,
            19,
            23,
            24,
            26,
            27,
        ),

        ocr_score_thr=0.3,
        ocr_decode_max_len=1024,
        ocr_chunk_size=8,
    ),

    test_cfg=dict(
        max_per_img=300,
        score_thr=0.3,
    ),
)


load_from = (
    '/root/autodl-tmp/checkpoint/dino/'
    'dino-4scale_r50_ocrdino_pretrain.pth'
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
        type='Resize',
        scale=(800, 480),
        keep_ratio=True,
    ),
    dict(
        type='TokenizeOCRText',
        vocab_file=vocab_file,
        text_types=(
            'text',
            'latex',
        ),
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
        type='Resize',
        scale=(800, 480),
        keep_ratio=True,
    ),
    dict(
        type='PackOCRDinoInputs',
    ),
]


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
        type='OCRDinoDataset',

        data_root=data_root,
        ann_file='ocrdino_train.json',

        data_prefix=dict(
            img_path='',
        ),

        pipeline=train_pipeline,
        serialize_data=False,
    ),
)


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
        type='OCRDinoDataset',

        data_root=data_root,
        ann_file='ocrdino_val.json',

        data_prefix=dict(
            img_path='',
        ),

        test_mode=True,
        pipeline=val_pipeline,
        serialize_data=False,
    ),
)


test_dataloader = val_dataloader


val_evaluator = dict(
    _delete_=True,

    type='OCRDinoMetric',

    iou_thr=0.5,

    text_types=(
        'text',
        'latex',
    ),
)


test_evaluator = val_evaluator


optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    accumulative_counts=2,

    optimizer=dict(
        type='AdamW',
        lr=5e-5,
        weight_decay=1e-4,
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

            'bbox_head.block_visual_extractor': dict(
                lr_mult=4.0,
            ),

            'bbox_head.block_text_decoder': dict(
                lr_mult=4.0,
            ),

            'bbox_head.cls_branches': dict(
                lr_mult=2.0,
            ),

            'dn_query_generator.label_embedding': dict(
                lr_mult=2.0,
            ),
        }
    ),
)


max_epochs = 12


train_cfg = dict(
    _delete_=True,

    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1,
)


val_cfg = dict(
    _delete_=True,
    type='ValLoop',
)


test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)


param_scheduler = [
    dict(
        type='LinearLR',

        start_factor=0.1,

        by_epoch=False,

        begin=0,
        end=500,
    ),

    dict(
        type='MultiStepLR',

        begin=0,
        end=max_epochs,

        by_epoch=True,

        milestones=[
            8,
            11,
        ],

        gamma=0.1,
    ),
]


default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=20,
    ),

    checkpoint=dict(
        type='CheckpointHook',

        interval=1,

        max_keep_ckpts=3,

        save_best=(
            'ocrdino/e2e_exact_recall'
        ),

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
)


auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)