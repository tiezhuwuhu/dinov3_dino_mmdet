_base_ = './point_dino_r50_shanghaitech_stage2_12e.py'

# Point-DN ablation:
# keep all Stage-2 settings unchanged,
# only change point noise scale from 0.05 to 0.01.
model = dict(
    use_dn=True,
    dn_cfg=dict(
        point_noise_scale=0.01
    )
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/point_dino_r50_shanghaitech_stage2_dn001_12e'
)