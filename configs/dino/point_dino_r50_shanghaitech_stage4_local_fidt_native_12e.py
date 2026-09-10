_base_ = './point_dino_r50_shanghaitech_stage4_fidt_native_12e.py'

# Local FIDT changes only the cells selected for the auxiliary MSE.
# Keep the original full-map config available for the corresponding ablation.
# Inherited baseline: Euclidean weight 0.20, Hungarian FocalLossCost 2.0 /
# PointL1Cost 20.0, Point L1 training weight 5.0, Point DN noise 0.01,
# native 1024 x 768 input, and 12 epochs.
# Keep the same inherited point_dino_stage2_step2_init.pth initialization
# for a fair comparison; do not replace load_from with a Stage 3 best model.
model = dict(
    point_fidt_head=dict(
        # Current training-image pixels, measured from each map cell center
        # to its nearest GT point. This initial radius is not claimed optimal.
        local_radius_px=16.0,
        # Smoke settings for observing the changed raw MSE scale only.
        # Select experiment weights from real-data measurements separately;
        # use debug=False for formal training. No sweep is defined here.
        fidt_loss_weight=1.0,
        debug=True))

work_dir = './work_dirs/point_dino_r50_shanghaitech_stage4_local_fidt_native_12e'
