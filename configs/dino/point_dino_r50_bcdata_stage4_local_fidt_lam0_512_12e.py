_base_ = './point_dino_r50_bcdata_stage4_local_fidt_512_12e.py'


# ============================================================
# FIDT ablation
#
# Same Stage-4 code path, but remove the FIDT loss contribution.
# Everything else is inherited unchanged.
# ============================================================

model = dict(
    point_fidt_head=dict(
        fidt_loss_weight=0.0,
    )
)


work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/'
    'point_dino_r50_bcdata_stage4_local_fidt_lam0_512_12e'
)