_base_ = './point_dino_stage3_lam020_cost20.py'

# Stage 4: Local FIDT Auxiliary Supervision.
# Keep the selected Stage 3 baseline and native 1024 x 768 pipeline.
# FIDT supervises the highest-resolution shared neck feature during loss()
# only. Query decoding, point prediction, matching, DN and evaluation stay
# on the existing Point-DINO path.
model = dict(
    use_dn=True,
    bbox_head=dict(point_euclidean_weight=0.20),
    train_cfg=dict(
        assigner=dict(
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                # This is a matching cost, not the Point L1 training weight.
                dict(type='PointL1Cost', weight=20.0)
            ])),
    dn_cfg=dict(point_noise_scale=0.01),
    point_fidt_head=dict(
        enabled=True,
        in_channels=256,
        hidden_channels=64,
        num_groups=8,
        # Neck level 0: stride 8, [B, 256, 96, 128]. Only the FIDT
        # branch upsamples, producing [B, 1, 192, 256] at stride 4.
        upsample_factor=2,
        gamma=0.02,
        phi=0.75,
        xi=1.0,
        # Temporary functional-smoke value only. Inspect fidt_mse and
        # loss_fidt scales before selecting any experiment weight.
        # Set this to 0, or enabled=False, to disable the branch entirely.
        fidt_loss_weight=1.0,
        chunk_size=4096,
        point_chunk_size=256,
        # Detached fidt_mse statistic is logged without adding to total loss.
        debug=True))

# Keep the inherited dataset and initialization settings. Their existing
# absolute /root/autodl-tmp paths must be overridden for another machine.
# In particular, load_from still names the inherited Stage 2 initialization;
# set it explicitly at runtime to use an actual selected Stage 3 checkpoint.
work_dir = './work_dirs/point_dino_r50_shanghaitech_stage4_fidt_native_12e'
