_base_ = "./dino4_vits16_coco_align_640_1e.py"

alignment_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/"
    "mmdetection/work_dirs/"
    "dino4_vits16_align_640/"
    "epoch_1.pth"
)

model = dict(
    backbone=dict(
        # Keep the same architecture and weights,
        # but allow gradients through the ViT.
        frozen=False,
    ),
)

optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(
        type="AdamW",
        lr=1e-4,
        weight_decay=1e-4,
    ),
    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,
    ),
    paramwise_cfg=dict(
        custom_keys={
            # ViT LR = 1e-5 when the main LR is 1e-4.
            "backbone": dict(
                lr_mult=0.1,
            ),
        },
    ),
)

max_epochs = 12

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)

param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1,
    ),
]

default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=3,
        save_best="coco/bbox_mAP",
        rule="greater",
    ),
)

# Load model tensors only and create a fresh optimizer.
# Do not set resume=True because Stage 1 optimizer did not
# contain trainable backbone parameters.
load_from = alignment_checkpoint
resume = False

auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)