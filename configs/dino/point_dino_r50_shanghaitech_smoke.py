_base_ = './dino-4scale_r50_8xb2-12e_coco.py'

# --------------------------------------------------
# Smoke test only:
# run 10 iterations and do not run validation.
# --------------------------------------------------
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=10,
    val_interval=100)

# Iter-based loop should use an infinite sampler.
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    sampler=dict(
        _delete_=True,
        type='InfiniteSampler',
        shuffle=True))

# We only want to verify that optimization can run.
# Disable the original epoch-based 12e LR scheduler for this 10-iter test.
param_scheduler = []

# Log every iteration and save once at iteration 10.
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=1),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=10,
        max_keep_ckpts=1))

log_processor = dict(
    type='LogProcessor',
    window_size=1,
    by_epoch=False)

# Load the Point-DINO-compatible COCO initialization.
load_from = (
    '/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/mmdet/'
    'dino_r50_4scale_coco_point1cls.pth'
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/work_dirs/'
    'point_dino_shanghaitech_smoke'
)