_base_ = './point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py'

model = dict(
    bbox_head=dict(
        point_euclidean_weight=0.5),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0),
                dict(
                    type='PointL1Cost',
                    weight=25.0)
            ]
        )
    )
)

work_dir = '/root/autodl-tmp/dinov3_dino_mmdet/work_dirs/stage3_grid_ext/point_dino_stage3_lam050_cost25'
