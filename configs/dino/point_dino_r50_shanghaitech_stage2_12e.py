_base_ = './point_dino_r50_shanghaitech_12e.py'

# Stage-2 initialization:
# Stage-1 trained weights converted to:
# - 2D ref_point_head
# - 2D regression branches
load_from = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'checkpoints/mmdet/'
    'point_dino_stage2_step2_init.pth'
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/point_dino_r50_shanghaitech_stage2_12e'
)

# The evaluator now reports:
# point/f1@4px
# point/f1@8px
#
# Use F1@8px as the main checkpoint selection metric.
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