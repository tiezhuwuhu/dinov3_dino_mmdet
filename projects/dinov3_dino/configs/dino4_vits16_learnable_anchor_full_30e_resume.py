_base_ = [
    './dino4_vits16_learnable_anchor_full_24e.py',
]


# ============================================================
# Continue A3:
# epoch 24 -> epoch 30
#
# Additional epochs:
# 25, 26, 27, 28, 29, 30
#
# Keep low LR = 2e-5 after milestone 20
# ============================================================


train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=30,
    val_interval=1,
)


# Keep exactly the same LR schedule.
#
# Original:
#   epoch 1-20 : base lr = 2e-4
#   epoch 21+  : base lr = 2e-5
#
# We ONLY extend scheduler end from 24 -> 30.
# Do NOT add another LR drop.
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=30,
        by_epoch=True,
        milestones=[20],
        gamma=0.1,
    ),
]


# Keep using the SAME work directory,
# so new epoch_25...epoch_30 checkpoints stay together.
work_dir = (
    'work_dirs/'
    'dino4_vits16_learnable_anchor_full_24e_batch1_acc4'
)


# Resume is supplied explicitly from command line.
load_from = None
resume = False