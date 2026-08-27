_base_ = './point_dino_r50_shanghaitech_stage2_12e.py'

model = dict(
    use_dn=False
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/point_dino_r50_shanghaitech_stage2_nodn_12e'
)