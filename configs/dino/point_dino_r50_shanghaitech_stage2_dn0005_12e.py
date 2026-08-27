_base_ = './point_dino_r50_shanghaitech_stage2_12e.py'

# Point-DN ablation:
# only change point noise scale to 0.005.
model = dict(
    use_dn=True,
    dn_cfg=dict(
        point_noise_scale=0.005
    )
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/point_dino_r50_shanghaitech_stage2_dn0005_12e'
)

# Keep F1@8px as the best-checkpoint criterion.
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_last=True,
        max_keep_ckpts=3,
        save_best='point/f1@8px',
        rule='greater'
    )
)