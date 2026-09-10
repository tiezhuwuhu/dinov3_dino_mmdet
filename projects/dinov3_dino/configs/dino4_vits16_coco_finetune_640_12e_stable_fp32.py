_base_ = "./dino4_vits16_coco_align_640_1e.py"


alignment_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/"
    "mmdetection/work_dirs/"
    "dino4_vits16_align_640/"
    "epoch_1.pth"
)


# ============================================================
# Model
# ============================================================

model = dict(
    backbone=dict(
        frozen=False,
    ),
)


# ============================================================
# Data
# ============================================================

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
)


# ============================================================
# Optimizer
# ============================================================
#
# Important changes:
#
# 1. Explicit BF16 autocast instead of default CUDA FP16.
# 2. Static scale 1.0: BF16 does not need FP16-style loss scaling.
# 3. Main LR 8e-5; backbone LR 8e-6.
# 4. Abort immediately if the gradient norm is NaN or Inf.
#
# _delete_=True prevents fields from the inherited optimizer
# configuration from being silently retained.
# ============================================================

optim_wrapper = dict(
    _delete_=True,

    type="OptimWrapper",

    # Wide exponent range, much safer than FP16 for this model.
    #dtype="bfloat16",

    # Static scaling. MMEngine requires this value to be float.
    #loss_scale=1.0,

    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
    ),

    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,

        # Fail fast rather than continuing for several epochs
        # after a non-finite gradient appears.
        error_if_nonfinite=True,
    ),

    paramwise_cfg=dict(
        custom_keys={
            # Backbone LR = 8e-5 * 0.1 = 8e-6.
            "backbone": dict(
                lr_mult=0.1,
            ),
        },
    ),
)


# ============================================================
# Schedule
# ============================================================

max_epochs = 12

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)

param_scheduler = [
    # This warmup is only for the fresh optimizer and newly
    # unfrozen backbone at the beginning. It is not being used
    # as an explanation or fix for the previous late collapse.
    dict(
        type="LinearLR",
        start_factor=0.2,
        begin=0,
        end=500,
        by_epoch=False,
    ),

    # The previous run reached its best result around epochs
    # 7-9 and became unstable near the end of epoch 10.
    #
    # MMEngine applies milestone=8 from epoch 9 onward:
    # epochs 1-8: 8e-5
    # epochs 9-12: 8e-6
    dict(
        type="MultiStepLR",
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[11],
        gamma=0.1,
    ),
]


# ============================================================
# Fail-fast checks
# ============================================================

custom_hooks = [
    # Stops immediately if total loss becomes NaN or Inf.
    dict(
        type="CheckInvalidLossHook",
        interval=1,
    ),
]


# ============================================================
# Checkpoints
# ============================================================

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,

        # Keep every complete checkpoint so that the optimizer,
        # scheduler and AMP state of the best epoch are available.
        max_keep_ckpts=12,

        save_optimizer=True,
        save_param_scheduler=True,
        save_last=True,

        save_best="coco/bbox_mAP",
        rule="greater",
    ),
)


# ============================================================
# Initialization
# ============================================================

# Start again from the clean Stage-1 alignment result.
# Do not load any epoch from the collapsed 12-epoch run.
load_from = alignment_checkpoint

# Create a fresh optimizer for the unfrozen model.
resume = False


# The learning rates above are explicit.
auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)


# Fixed 640x640 input can benefit from cuDNN benchmarking.
env_cfg = dict(
    cudnn_benchmark=True,
)
