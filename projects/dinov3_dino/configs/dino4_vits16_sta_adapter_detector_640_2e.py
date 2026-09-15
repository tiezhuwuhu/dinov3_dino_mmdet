_base_ = "./dino4_vits16_sta_align_640_1e.py"


stage5_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/mmdetection/"
    "work_dirs/dino4_vits16_sta_resbridge_only_640_1e/"
    "epoch_1.pth"
)


# The detector remains the original MMDetection DINO.
model = dict(
    # frozen=False allows the Lightly adapter to train.
    # The custom DDP wrapper freezes only the internal ViT.
    backbone=dict(
        frozen=False,
        validate_outputs=False,
    ),

    neck=dict(
        validate_inputs=False,
    ),
)


model_wrapper_cfg = dict(
    _delete_=True,

    type="FreezeViTMMDistributedDataParallel",

    # Keep pretrained adapter SyncBN running statistics unchanged.
    # Their affine weight and bias remain trainable.
    freeze_adapter_running_stats=True,

    find_unused_parameters=False,
)


# Load model weights from stage 5, but create a new optimizer and scheduler.
load_from = stage5_checkpoint
resume = False


optim_wrapper = dict(
    _delete_=True,

    type="OptimWrapper",

    optimizer=dict(
        type="AdamW",

        # Base LR applies to MMDetection DINO detector parameters.
        lr=1.0e-4,

        betas=(
            0.9,
            0.999,
        ),

        weight_decay=1.0e-4,
    ),

    paramwise_cfg=dict(
        custom_keys={
            # New residual bridge: 2e-4.
            "neck": dict(
                lr_mult=2.0,
            ),

            # Lightly feature adapter: 5e-5.
            "backbone.dinostas.sta": dict(
                lr_mult=0.5,
            ),
            "backbone.dinostas.convs": dict(
                lr_mult=0.5,
            ),
            "backbone.dinostas.norms": dict(
                lr_mult=0.5,
                decay_mult=0.0,
            ),

            # Stabilize deformable attention geometry.
            "sampling_offsets": dict(
                lr_mult=0.1,
            ),
            "reference_points": dict(
                lr_mult=0.1,
            ),
        },

        # No weight decay on bias and normalization affine parameters.
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
    ),

    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,
        error_if_nonfinite=True,
    ),
)


# Stage 6 is short fixed-resolution adaptation.
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.2,
        begin=0,
        end=500,
        by_epoch=False,
    ),
]


train_cfg = dict(
    _delete_=True,
    type="EpochBasedTrainLoop",
    max_epochs=2,
    val_interval=1,
)


default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=2,
        save_last=True,
        save_optimizer=True,
        save_param_scheduler=True,
        save_best="coco/bbox_mAP",
        rule="greater",
    ),
)


work_dir = (
    "work_dirs/"
    "dino4_vits16_sta_adapter_detector_640_2e"
)