_base_ = './point_dino_r50_shanghaitech_stage2_dn001_12e.py'

# Euclidean-loss smoke test.
# The Euclidean loss is implemented directly in DINOHead,
# so no extra model option is needed for this first lambda=1.0 version.

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/point_dino_stage2_dn001_euc_smoke'
)

# Only verify forward/backward and loss behavior.
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=10,
    val_interval=100
)

# Do not run validation during this smoke test.
val_dataloader = None
val_cfg = None
val_evaluator = None

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=1
    ),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        by_epoch=False
    )
)