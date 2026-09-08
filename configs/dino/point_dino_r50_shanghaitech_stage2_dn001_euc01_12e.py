_base_ = './point_dino_r50_shanghaitech_stage2_dn001_12e.py'

# Experiment 3:
# Pixel-space Euclidean point loss.
#
# L_reg =
#     L_point_L1
#     + lambda_euc * (pixel_euclidean_distance / 8)
#
# lambda_euc = 0.1
model = dict(
    bbox_head=dict(
        point_euclidean_weight=0.1
    )
)

work_dir = (
    '/root/autodl-tmp/dinov3_dino_mmdet/'
    'work_dirs/'
    'point_dino_r50_shanghaitech_stage2_dn001_euc01_12e'
)