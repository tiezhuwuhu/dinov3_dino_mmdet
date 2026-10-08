_base_ = './ocr-dino-4scale_r50_overfit8.py'

data_root = '/root/autodl-tmp/dataset/OmniDocBench/'

test_pipeline = [
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

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        type='OCRDinoDataset',
        data_root=data_root,
        ann_file='ocrdino_v1_overfit8.json',
        data_prefix=dict(
            img_path='',
        ),
        test_mode=True,
        pipeline=test_pipeline,
        serialize_data=False,
    ),
)

test_evaluator = dict(
    _delete_=True,
    type='OCRDinoMetric',
    iou_thr=0.5,
    text_types=(
        'text',
        'latex',
    ),
)

test_cfg = dict(
    _delete_=True,
    type='TestLoop',
)