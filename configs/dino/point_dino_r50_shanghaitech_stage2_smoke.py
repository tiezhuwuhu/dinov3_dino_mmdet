_base_ = './point_dino_r50_shanghaitech_smoke.py'

load_from = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'checkpoints/mmdet/'
    'point_dino_stage2_step2_init.pth'
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/point_dino_stage2_smoke'
)