_base_ = './ocrdino_r50_totaltext_rec1e4_dn04_100e.py'


train_scales = [
    (480, 1333),
    (512, 1333),
    (544, 1333),
    (576, 1333),
    (608, 1333),
    (640, 1333),
    (672, 1333),
    (704, 1333),
    (736, 1333),
    (768, 1333),
    (800, 1333),
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
        type='Resize',
        scale=(1000, 1824),
        keep_ratio=True,
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


train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline,
    ),
    batch_size=4,
    num_workers=8,
)


val_dataloader = dict(
    dataset=dict(
        pipeline=val_pipeline,
    ),
    batch_size=4,
    num_workers=8,
)


test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    ),
    batch_size=4,
    num_workers=8,
)


load_from = (
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_estextspotter_pretrain_init.pth'
)


resume = False


work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_totaltext_estpretrain_resize_100e'
)