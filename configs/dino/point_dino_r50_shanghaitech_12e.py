_base_ = './dino-4scale_r50_8xb2-12e_coco.py'


# ============================================================
# Point-DINO Stage-1 formal baseline
# ShanghaiTech Part B
# ============================================================

# ------------------------------------------------------------
# Initialization
#
# COCO DINO checkpoint with the 15 incompatible single-class
# parameters removed.
# ------------------------------------------------------------
load_from = (
    '/root/autodl-tmp/dinov3_dino_mmdet/checkpoints/mmdet/'
    'dino_r50_4scale_coco_point1cls.pth'
)

resume = False


# ------------------------------------------------------------
# Training dataloader
#
# Keep DINO's original per-GPU batch size = 2.
# Dataset / pipeline are inherited from our modified base config.
# ------------------------------------------------------------
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=True))


# ------------------------------------------------------------
# Validation / test
# ------------------------------------------------------------
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False))


# ------------------------------------------------------------
# Optimizer
#
# Keep official DINO R50 settings.
# ------------------------------------------------------------
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=1e-4,
        weight_decay=1e-4),
    clip_grad=dict(
        max_norm=0.1,
        norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1)
        }))


# ------------------------------------------------------------
# Training schedule
#
# First formal baseline: keep official DINO 12e schedule.
# ------------------------------------------------------------
max_epochs = 12

train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


# ------------------------------------------------------------
# LR scheduler
#
# Official DINO 12e config reduces LR at epoch 11.
# ------------------------------------------------------------
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]


# ------------------------------------------------------------
# Automatic LR scaling
#
# Disabled by default to reproduce the original DINO optimizer
# setting directly.
#
# base_batch_size=16 corresponds to official 8 GPUs x 2 images.
# If later needed, tools/train.py --auto-scale-lr can enable it.
# ------------------------------------------------------------
auto_scale_lr = dict(
    enable=False,
    base_batch_size=16)


# ------------------------------------------------------------
# Checkpoint / logging
# ------------------------------------------------------------
default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=20),

    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        by_epoch=True,
        save_last=True,
        max_keep_ckpts=3,
        save_best='point/f1',
        rule='greater'))


log_processor = dict(
    type='LogProcessor',
    window_size=20,
    by_epoch=True)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
randomness = dict(
    seed=0,
    deterministic=False)


# ------------------------------------------------------------
# Work directory
# ------------------------------------------------------------
work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/work_dirs/'
    'point_dino_r50_shanghaitech_12e'
)