_base_ = './point_dino_r50_shanghaitech_stage2_dn001_12e.py'

# Ablation:
# Keep Point-DN noise_scale = 0.01.
# Only increase Hungarian point matching cost:
# PointL1Cost 5.0 -> 10.0.
model = dict(
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0),
                dict(
                    type='PointL1Cost',
                    weight=10.0)
            ]
        )
    )
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/'
    'point_dino_r50_shanghaitech_stage2_dn001_match10_12e'
)