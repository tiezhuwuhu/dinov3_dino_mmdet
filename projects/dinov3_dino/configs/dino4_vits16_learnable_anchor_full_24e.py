_base_ = [
    './dino4_vits16_learnable_anchor_improved_multiscale_24e.py',
]


# ============================================================
# Stage A3
#
# ViT        : train
# Anchor Neck: train
# DINO       : train
#
# Improved DINO multiscale 24e
#
# batch/GPU = 1
# GPUs = 4
# accumulation = 4
# effective global batch = 16
# ============================================================


# ============================================================
# Model
# ============================================================

model = dict(
    type='FreezeableDINO',

    train_neck_only=False,

    backbone=dict(
        frozen=False,
    ),
)


# ============================================================
# Dataloader
#
# Do NOT redefine pipeline here.
#
# Therefore it keeps the inherited improved multiscale:
# short edge 480 ... 800
# max long edge 1333
# resize/crop augmentation branch
# ============================================================

train_dataloader = dict(
    batch_size=1,
)


val_dataloader = dict(
    batch_size=1,
)


test_dataloader = dict(
    batch_size=1,
)


# ============================================================
# Optimizer
#
# Keep all inherited settings:
#
# base lr              = 2e-4
# backbone lr_mult     = 0.1
# backbone actual lr   = 2e-5
#
# reference_points     = 0.1x
# sampling_offsets     = 0.1x
#
# max_norm             = 0.1
#
# Only change accumulation:
# ============================================================

optim_wrapper = dict(
    accumulative_counts=4,
)


# ============================================================
# Checkpoint supplied from command line.
# ============================================================

load_from = None

resume = False


work_dir = (
    'work_dirs/'
    'dino4_vits16_learnable_anchor_full_24e_batch1_acc4'
)