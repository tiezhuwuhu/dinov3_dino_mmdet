_base_ = './point_dino_r50_shanghaitech_stage2_dn001_euc01_native_12e.py'

# ============================================================
# Stage 3 grid ablation
#
# Euclidean lambda = 0.2
# PointL1Cost      = 20
# FocalLossCost    = 2.0
# Point DN         = 0.01
# Input            = native 1024x768
# ============================================================

model = dict(
    bbox_head=dict(
        point_euclidean_weight=0.2),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(
                    type='FocalLossCost',
                    weight=2.0),
                dict(
                    type='PointL1Cost',
                    weight=20.0)
            ]
        )
    )
)

work_dir = '/root/autodl-tmp/dinov3_dino_mmdet/work_dirs/stage3_grid/point_dino_stage3_lam020_cost20'
