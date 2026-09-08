_base_ = './point_dino_r50_shanghaitech_stage2_dn001_12e.py'


# =========================================================
# Native ShanghaiTech Part-B input: 1024 x 768
# =========================================================

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_point=True),
    dict(
        type='Resize',
        scale=(1024, 768),
        keep_ratio=False),
    dict(
        type='RandomFlip',
        prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations',
        with_bbox=False,
        with_label=False,
        with_point=True),
    dict(
        type='Resize',
        scale=(1024, 768),
        keep_ratio=False),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor'))
]


train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline))

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline))


# =========================================================
# Point-DINO
# DN noise = 0.01 inherited from dn001 config
# Hungarian: Focal 2 + PointL1 5
# Euclidean loss lambda = 0.1
# =========================================================

model = dict(
    bbox_head=dict(
        point_euclidean_weight=0.1),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0),
                dict(
                    type='PointL1Cost',
                    weight=5.0)
            ]
        )
    )
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/'
    'point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e'
)