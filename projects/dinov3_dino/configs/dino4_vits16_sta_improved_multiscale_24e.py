_base_ = (
    "./dino4_vits16_sta_adapter_detector_640_2e.py"
)


# -------------------------------------------------------------------------
# Stage-6 initialization
# -------------------------------------------------------------------------

stage6_best_checkpoint = (
    "/root/autodl-tmp/dinov3_dino_mmdet/mmdetection/"
    "work_dirs/"
    "dino4_vits16_sta_adapter_detector_640_2e/"
    "best_coco_bbox_mAP_epoch_2.pth"
)

load_from = stage6_best_checkpoint

# Only load model weights. Build a new stage-7 optimizer and scheduler.
resume = False


# -------------------------------------------------------------------------
# Original MMDetection DINO + improved DINO hyperparameters
# -------------------------------------------------------------------------

model = dict(
    # Inherited type remains the original MMDetection DINO.
    type="DINO",

    data_preprocessor=dict(
        # Necessary adaptation for the ViT/STA stride hierarchy.
        # Padding is performed by DetDataPreprocessor, not by the pipeline.
        pad_size_divisor=32,
    ),

    backbone=dict(
        # Remove the explicit stage-6 ViT freeze.
        # Parameters naturally marked requires_grad=False by Lightly
        # remain unchanged.
        frozen=False,
        validate_outputs=False,
    ),

    neck=dict(
        validate_inputs=False,
    ),

    # Official improved DINO.
    bbox_head=dict(
        loss_cls=dict(
            loss_weight=2.0,
        ),
    ),

    positional_encoding=dict(
        offset=-0.5,
        temperature=10000,
    ),

    dn_cfg=dict(
        group_cfg=dict(
            num_dn_queries=300,
        ),
    ),
)


# Remove the stage-6 FreezeViT DDP wrapper.
# MMEngine will use its standard distributed wrapper.
model_wrapper_cfg = None


# -------------------------------------------------------------------------
# Official DINO multiscale augmentation
# -------------------------------------------------------------------------

backend_args = None


dino_train_scales = [
    (480, 1333),
    (512, 1333),
    (544, 1333),
    (576, 1333),
    (608, 1333),
    (640, 1333),
    (672, 1333),
    (704, 1333),
    (736, 1333),
    (768, 1333),
    (800, 1333),
]


train_pipeline = [
    dict(
        type="LoadImageFromFile",
        backend_args=backend_args,
    ),

    dict(
        type="LoadAnnotations",
        with_bbox=True,
    ),

    dict(
        type="RandomFlip",
        prob=0.5,
    ),

    dict(
        type="RandomChoice",
        transforms=[
            # Official direct multiscale branch.
            [
                dict(
                    type="RandomChoiceResize",
                    scales=dino_train_scales,
                    keep_ratio=True,
                ),
            ],

            # Official resize -> crop -> resize branch.
            [
                dict(
                    type="RandomChoiceResize",
                    scales=[
                        (400, 4200),
                        (500, 4200),
                        (600, 4200),
                    ],
                    keep_ratio=True,
                ),

                dict(
                    type="RandomCrop",
                    crop_type="absolute_range",
                    crop_size=(384, 600),
                    allow_negative_crop=True,
                ),

                dict(
                    type="RandomChoiceResize",
                    scales=dino_train_scales,
                    keep_ratio=True,
                ),
            ],
        ],
    ),

    dict(
        type="PackDetInputs",
    ),
]


# Official COCO single-scale validation.
test_pipeline = [
    dict(
        type="LoadImageFromFile",
        backend_args=backend_args,
    ),

    dict(
        type="Resize",
        scale=(1333, 800),
        keep_ratio=True,
    ),

    dict(
        type="LoadAnnotations",
        with_bbox=True,
    ),

    dict(
        type="PackDetInputs",
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
    ),
]


# Two GPUs × one image per GPU × accumulation 8 = effective batch 16.
#
# This preserves the official DINO optimizer-step batch size:
# 8 GPUs × 2 images = 16.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type="DefaultSampler",
        shuffle=True,
    ),

    batch_sampler=dict(
        type="AspectRatioBatchSampler",
    ),

    dataset=dict(
        filter_cfg=dict(
            filter_empty_gt=False,
        ),

        pipeline=train_pipeline,
    ),
)


val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,

    sampler=dict(
        type="DefaultSampler",
        shuffle=False,
    ),

    dataset=dict(
        pipeline=test_pipeline,
    ),
)


test_dataloader = val_dataloader


# -------------------------------------------------------------------------
# Official improved DINO optimizer
# -------------------------------------------------------------------------

optim_wrapper = dict(
    _delete_=True,

    # Standard FP32 training.
    type="OptimWrapper",

    # Effective global batch:
    accumulative_counts=2,

    optimizer=dict(
        type="AdamW",
        lr=2.0e-4,
        weight_decay=1.0e-4,
    ),

    clip_grad=dict(
        max_norm=0.1,
        norm_type=2,
    ),

    # Match official improved DINO:
    # backbone and deformable-attention geometry use base LR × 0.1.
    paramwise_cfg=dict(
        custom_keys={
            "backbone": dict(
                lr_mult=0.1,
            ),

            "sampling_offsets": dict(
                lr_mult=0.1,
            ),

            "reference_points": dict(
                lr_mult=0.1,
            ),
        },
    ),
)


# Effective batch size is already matched manually.
auto_scale_lr = dict(
    enable=False,
    base_batch_size=16,
)


# -------------------------------------------------------------------------
# Official 24-epoch schedule
# -------------------------------------------------------------------------

max_epochs = 24


train_cfg = dict(
    _delete_=True,
    type="EpochBasedTrainLoop",
    max_epochs=max_epochs,
    val_interval=1,
)


val_cfg = dict(
    _delete_=True,
    type="ValLoop",
)


test_cfg = dict(
    _delete_=True,
    type="TestLoop",
)


# Official 24e schedule: decay once at epoch 20.
# No additional warmup.
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[
            20,
        ],
        gamma=0.1,
    ),
]


# -------------------------------------------------------------------------
# Runtime
# -------------------------------------------------------------------------

# Remove stage-specific custom hooks.
custom_hooks = []


# Official multiscale training uses many changing shapes.
env_cfg = dict(
    cudnn_benchmark=False,
)


default_hooks = dict(
    logger=dict(
        type="LoggerHook",
        interval=50,
    ),

    checkpoint=dict(
        type="CheckpointHook",
        interval=1,
        max_keep_ckpts=5,
        save_last=True,
        save_optimizer=True,
        save_param_scheduler=True,
        save_best="coco/bbox_mAP",
        rule="greater",
        filename_tmpl="epoch_{}.pth",
    ),
)


work_dir = (
    "work_dirs/"
    "dino4_vits16_sta_improved_multiscale_24e_batch_2"
)