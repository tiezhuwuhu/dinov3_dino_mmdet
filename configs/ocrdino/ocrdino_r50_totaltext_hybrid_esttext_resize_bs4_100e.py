_base_ = './ocrdino_r50_totaltext_estpretrain_resize_100e.py'


train_dataloader = dict(
    batch_size=4,
    num_workers=8,
)


val_dataloader = dict(
    batch_size=4,
    num_workers=8,
)


test_dataloader = dict(
    batch_size=4,
    num_workers=8,
)


load_from = (
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_dino_det_est_text_pretrain_init.pth'
)


resume = False


work_dir = (
    '/root/autodl-tmp/work_dirs/'
    'ocrdino_r50_totaltext_hybrid_esttext_resize_bs4_100e'
)