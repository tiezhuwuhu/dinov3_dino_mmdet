import math
from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.optim import OptimWrapper

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


# ============================================================
# Settings
# ============================================================

DATA_ROOT = (
    '/root/autodl-tmp/dataset/totaltext/'
)

CONFIG = (
    'configs/dino/'
    'dino-4scale_r50_8xb2-12e_coco.py'
)

CHECKPOINT = (
    '/root/autodl-tmp/checkpoint/dino/'
    'dino-4scale_r50_8xb2-12e_coco_'
    '20221202_182705-55b2bba2.pth'
)

DEVICE = 'cuda'

# Use two REAL images so batching/padding/collation is tested too.
SAMPLE_INDICES = [
    0,
    1,
]


# ============================================================
# Init
# ============================================================

register_all_modules()

torch.manual_seed(42)

device = torch.device(
    DEVICE
)

print(
    'device:',
    device,
)


# ============================================================
# 1. Dataset
# ============================================================

dataset_cfg = dict(
    type='OCRDinoDataset',

    data_root=DATA_ROOT,

    ann_file='ocrdino_train.json',

    data_prefix=dict(
        img_path='',
    ),

    serialize_data=False,

    pipeline=[
        dict(
            type='LoadImageFromFile',
        ),

        dict(
            type='LoadOCRAnnotations',
            with_bbox=True,
            with_label=True,
        ),

        dict(
            type='Resize',
            scale=(
                800,
                480,
            ),
            keep_ratio=True,
        ),

        dict(
            type='TokenizeOCRText',
            text_types=(
                'text',
                'latex',
            ),
            max_length=25,
            ignore_index=-100,
        ),

        dict(
            type='PackOCRDinoInputs',
        ),
    ],
)


dataset = DATASETS.build(
    dataset_cfg
)


print()
print(
    '========================================'
)
print(
    'REAL DATASET'
)
print(
    '========================================'
)

print(
    'dataset length:',
    len(dataset),
)

assert len(dataset) == 1255


# ============================================================
# 2. Fetch two real samples
# ============================================================

items = [
    dataset[index]
    for index
    in SAMPLE_INDICES
]


for index, item in zip(
    SAMPLE_INDICES,
    items,
):

    sample = (
        item[
            'data_samples'
        ]
    )

    gt = (
        sample.gt_instances
    )

    print()
    print(
        'sample:',
        index,
    )

    print(
        'img path:',
        sample.img_path,
    )

    print(
        'input:',
        tuple(
            item[
                'inputs'
            ].shape
        ),
    )

    print(
        'GT count:',
        len(gt),
    )

    print(
        'rec shape:',
        tuple(
            gt.rec.shape
        ),
    )

    print(
        'texts:',
        gt.ocr_texts,
    )

    assert tuple(
        gt.rec.shape
    ) == (
        len(gt),
        25,
    )

    assert (
        gt.rec.dtype
        == torch.long
    )

    assert (
        gt.labels.min().item()
        >= 0
    )

    assert (
        gt.labels.max().item()
        == 0
    )


# ============================================================
# 3. MMEngine-style batch collation
# ============================================================

batch = pseudo_collate(
    items
)


print()
print(
    '========================================'
)
print(
    'COLLATED BATCH'
)
print(
    '========================================'
)

print(
    'batch keys:',
    batch.keys(),
)

print(
    'number of inputs:',
    len(
        batch[
            'inputs'
        ]
    ),
)

print(
    'number of samples:',
    len(
        batch[
            'data_samples'
        ]
    ),
)


assert (
    len(
        batch[
            'inputs'
        ]
    )
    == 2
)

assert (
    len(
        batch[
            'data_samples'
        ]
    )
    == 2
)


# ============================================================
# 4. Build ESTDINO
# ============================================================

cfg = Config.fromfile(
    CONFIG
)


cfg.model.type = (
    'ESTDINO'
)

cfg.model.num_queries = (
    100
)


# ------------------------------------------------------------
# Detection head
# ------------------------------------------------------------

cfg.model.bbox_head.type = (
    'ESTDINOHead'
)

cfg.model.bbox_head.num_classes = (
    1
)

cfg.model.bbox_head.rec_loss_weight = (
    1.0
)

cfg.model.bbox_head.dn_rec_loss_weight = (
    1.0
)

cfg.model.bbox_head.rec_ignore_index = (
    -100
)


# ------------------------------------------------------------
# Recognition query initializer
# ------------------------------------------------------------

cfg.model.recognition_query_cfg = dict(
    num_rec_queries=25,

    roi_height=8,

    featmap_strides=(
        8,
        16,
        32,
    ),

    sampling_ratio=1,

    aligned=True,

    canonical_box_size=224,

    canonical_level=4,

    box_format='cxcywh',

    normalized_boxes=True,

    detach_boxes=True,
)


# ------------------------------------------------------------
# Task-aware decoder
# ------------------------------------------------------------

cfg.model.text_decoder_cfg = dict(
    num_layers=6,

    num_rec_classes=97,

    num_heads=8,

    num_feature_levels=4,

    num_points=4,

    ffn_dims=2048,

    dropout=0.0,

    activation='relu',

    im2col_step=64,
)


print()
print(
    '========================================'
)
print(
    'BUILD MODEL'
)
print(
    '========================================'
)


model = MODELS.build(
    cfg.model
)

model.init_weights()


print(
    'model:',
    model.__class__.__name__,
)

print(
    'bbox head:',
    model.bbox_head.__class__.__name__,
)

print(
    'detection classes:',
    model.bbox_head.num_classes,
)

print(
    'matching queries:',
    model.num_queries,
)

print(
    'recognition queries:',
    model.num_rec_queries,
)

print(
    'recognition classes:',
    model.decoder
    .recognition_semantics
    .rec_classifier
    .out_features,
)


assert (
    model.__class__.__name__
    == 'ESTDINO'
)

assert (
    model.bbox_head.num_classes
    == 1
)

assert (
    model.num_queries
    == 100
)

assert (
    model.num_rec_queries
    == 25
)

assert (
    model.decoder
    .recognition_semantics
    .rec_classifier
    .out_features
    == 97
)


# ============================================================
# 5. Load pretrained DINO
# ============================================================

checkpoint_path = Path(
    CHECKPOINT
)

assert checkpoint_path.exists()


checkpoint = torch.load(
    checkpoint_path,
    map_location='cpu',
    weights_only=False,
)

state_dict = checkpoint.get(
    'state_dict',
    checkpoint,
)

model_state = (
    model.state_dict()
)

clean_state_dict = {}

adapted = []

shape_skipped = []

decoder_skipped = 0


for original_key, value in (
    state_dict.items()
):

    key = original_key

    if key.startswith(
        'module.'
    ):
        key = key[
            len('module.') :
        ]

    # New OCRDINO decoder is intentionally initialized separately.
    if key.startswith(
        'decoder.'
    ):

        decoder_skipped += 1

        continue

    # Adapt 900 matching queries -> 100.
    if (
        key
        == 'query_embedding.weight'
    ):

        target = (
            model_state[
                key
            ]
        )

        assert (
            value.ndim
            == 2
        )

        assert (
            value.shape[1]
            == target.shape[1]
        )

        assert (
            value.shape[0]
            >= target.shape[0]
        )

        new_value = (
            value[
                :target.shape[0]
            ].clone()
        )

        clean_state_dict[
            key
        ] = new_value

        adapted.append(
            (
                key,
                tuple(
                    value.shape
                ),
                tuple(
                    new_value.shape
                ),
            )
        )

        continue

    if key not in model_state:
        continue

    if (
        value.shape
        != model_state[
            key
        ].shape
    ):

        shape_skipped.append(
            (
                key,
                tuple(
                    value.shape
                ),
                tuple(
                    model_state[
                        key
                    ].shape
                ),
            )
        )

        continue

    clean_state_dict[
        key
    ] = value


load_result = (
    model.load_state_dict(
        clean_state_dict,
        strict=False,
    )
)


print()
print(
    '========================================'
)
print(
    'CHECKPOINT'
)
print(
    '========================================'
)

print(
    'loaded tensors:',
    len(
        clean_state_dict
    ),
)

print(
    'decoder skipped:',
    decoder_skipped,
)

print(
    'shape skipped:',
    len(
        shape_skipped
    ),
)

print(
    'adapted:',
    adapted,
)

print(
    'missing keys:',
    len(
        load_result.missing_keys
    ),
)

print(
    'unexpected keys:',
    len(
        load_result.unexpected_keys
    ),
)


# The known Total-Text mismatch count should be exactly 15:
#
#   14 = seven classification branches, weight+bias
#    1 = DN label embedding
#
assert (
    len(
        shape_skipped
    )
    == 15
)


# ============================================================
# 6. GPU + training mode
# ============================================================

model = model.to(
    device
)

model.train()


# ============================================================
# 7. Build a real optimizer + OptimWrapper
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),

    lr=1e-5,

    weight_decay=1e-4,
)


optim_wrapper = OptimWrapper(
    optimizer=optimizer
)


print()
print(
    '========================================'
)
print(
    'OPTIMIZER'
)
print(
    '========================================'
)

print(
    'optimizer:',
    optimizer.__class__.__name__,
)

print(
    'lr:',
    optimizer.param_groups[
        0
    ][
        'lr'
    ],
)

print(
    'weight decay:',
    optimizer.param_groups[
        0
    ][
        'weight_decay'
    ],
)


# ============================================================
# 8. Save parameter snapshots BEFORE train_step
# ============================================================

rec_classifier = (
    model.decoder
    .recognition_semantics
    .rec_classifier
)


rec_weight_before = (
    rec_classifier
    .weight
    .detach()
    .clone()
)


bbox_parameter_name = None

bbox_parameter_before = None


for name, parameter in (
    model.bbox_head
    .named_parameters()
):

    if (
        parameter.requires_grad
        and parameter.numel() > 0
    ):

        bbox_parameter_name = (
            name
        )

        bbox_parameter_before = (
            parameter
            .detach()
            .clone()
        )

        break


assert (
    bbox_parameter_before
    is not None
)


backbone_parameter_name = None

backbone_parameter_before = None


for name, parameter in (
    model.backbone
    .named_parameters()
):

    if (
        parameter.requires_grad
        and parameter.numel() > 0
    ):

        backbone_parameter_name = (
            name
        )

        backbone_parameter_before = (
            parameter
            .detach()
            .clone()
        )

        break


assert (
    backbone_parameter_before
    is not None
)


# ============================================================
# 9. REAL train_step()
#
# BaseDetector/BaseModel train_step will perform:
#
#   data_preprocessor
#   forward(mode='loss')
#   parse_losses
#   optim_wrapper.update_params
#       backward
#       optimizer.step
#       zero_grad
# ============================================================

print()
print(
    '========================================'
)
print(
    'RUN REAL train_step()'
)
print(
    '========================================'
)


log_vars = model.train_step(
    batch,
    optim_wrapper,
)


print()
print(
    'train_step() completed.'
)

print()
print(
    'log vars:'
)


for key in sorted(
    log_vars.keys()
):

    value = (
        log_vars[
            key
        ]
    )

    print(
        key,
        '=',
        value,
    )


# ============================================================
# 10. Validate log losses
# ============================================================

required_keys = [
    'loss',

    'loss_cls',
    'loss_bbox',
    'loss_iou',
    'loss_rec',

    'dn_loss_cls',
    'dn_loss_bbox',
    'dn_loss_iou',
    'dn_loss_rec',

    'enc_loss_cls',
    'enc_loss_bbox',
    'enc_loss_iou',

    'd0.loss_rec',
    'd1.loss_rec',
    'd2.loss_rec',
    'd3.loss_rec',
    'd4.loss_rec',

    'd0.dn_loss_rec',
    'd1.dn_loss_rec',
    'd2.dn_loss_rec',
    'd3.dn_loss_rec',
    'd4.dn_loss_rec',
]


missing_keys = [
    key
    for key
    in required_keys
    if key not in log_vars
]


print()
print(
    'missing required log keys:',
    missing_keys,
)


assert not (
    missing_keys
)


for key, value in (
    log_vars.items()
):

    if torch.is_tensor(
        value
    ):

        scalar = float(
            value.detach().cpu()
        )

    else:

        scalar = float(
            value
        )

    if not math.isfinite(
        scalar
    ):

        raise RuntimeError(
            'Non-finite train_step log: '
            f'{key}={scalar}'
        )


print(
    'all train_step logs finite: PASS'
)


# ============================================================
# 11. Verify the optimizer actually changed parameters
# ============================================================

rec_weight_after = (
    rec_classifier
    .weight
    .detach()
)


rec_update_abs_sum = (
    rec_weight_after
    - rec_weight_before
).abs().sum().item()


print()
print(
    'recognition classifier '
    'parameter update abs sum:',
    rec_update_abs_sum,
)


assert (
    rec_update_abs_sum
    > 0.0
)


# ------------------------------------------------------------
# BBox head
# ------------------------------------------------------------

bbox_parameter_after = dict(
    model.bbox_head
    .named_parameters()
)[
    bbox_parameter_name
].detach()


bbox_update_abs_sum = (
    bbox_parameter_after
    - bbox_parameter_before
).abs().sum().item()


print(
    'bbox parameter:',
    bbox_parameter_name,
)

print(
    'bbox parameter update abs sum:',
    bbox_update_abs_sum,
)


assert (
    bbox_update_abs_sum
    > 0.0
)


# ------------------------------------------------------------
# Backbone
# ------------------------------------------------------------

backbone_parameter_after = dict(
    model.backbone
    .named_parameters()
)[
    backbone_parameter_name
].detach()


backbone_update_abs_sum = (
    backbone_parameter_after
    - backbone_parameter_before
).abs().sum().item()


print(
    'backbone parameter:',
    backbone_parameter_name,
)

print(
    'backbone parameter update abs sum:',
    backbone_update_abs_sum,
)


assert (
    backbone_update_abs_sum
    > 0.0
)


# ============================================================
# 12. Verify all trainable parameters remain finite
# ============================================================

nonfinite_parameter_names = []


for name, parameter in (
    model.named_parameters()
):

    if not parameter.requires_grad:
        continue

    if not torch.isfinite(
        parameter
    ).all().item():

        nonfinite_parameter_names.append(
            name
        )


print()
print(
    'non-finite parameter tensors '
    'after optimizer step:',
    nonfinite_parameter_names,
)


assert not (
    nonfinite_parameter_names
)


# ============================================================
# 13. Optimizer state validation
# ============================================================

nonfinite_optimizer_states = []


for parameter, state in (
    optimizer.state.items()
):

    for key, value in (
        state.items()
    ):

        if not torch.is_tensor(
            value
        ):
            continue

        if not torch.isfinite(
            value
        ).all().item():

            nonfinite_optimizer_states.append(
                key
            )


print(
    'non-finite optimizer states:',
    nonfinite_optimizer_states,
)


assert not (
    nonfinite_optimizer_states
)


# ============================================================
# 14. Verify both samples survived the real batch
# ============================================================

# train_step itself preprocesses internally, so the original batch is
# intentionally still Python-list style here.

assert (
    len(
        batch[
            'data_samples'
        ]
    )
    == 2
)

total_real_gt = sum(
    len(
        sample.gt_instances
    )
    for sample
    in batch[
        'data_samples'
    ]
)


print()
print(
    'real batch size:',
    len(
        batch[
            'data_samples'
        ]
    ),
)

print(
    'real GT instances in batch:',
    total_real_gt,
)


assert (
    total_real_gt
    == 26
)


# ============================================================
# Final
# ============================================================

print()
print(
    '=============================================='
)

print(
    'OCRDINO REAL TRAIN_STEP PASSED'
)

print(
    '=============================================='
)

print()

print(
    'two-image real batch: PASS'
)

print(
    'pseudo_collate: PASS'
)

print(
    'data_preprocessor inside train_step: PASS'
)

print(
    'model.loss inside train_step: PASS'
)

print(
    'matching recognition loss: PASS'
)

print(
    'DN recognition loss: PASS'
)

print(
    'auxiliary recognition losses: PASS'
)

print(
    'loss parsing: PASS'
)

print(
    'backward via OptimWrapper: PASS'
)

print(
    'optimizer.step: PASS'
)

print(
    'recognition classifier updated: PASS'
)

print(
    'bbox head updated: PASS'
)

print(
    'backbone updated: PASS'
)

print(
    'all parameters finite after update: PASS'
)

print(
    'optimizer state finite: PASS'
)

print()

print(
    'OCRDINO is ready for an actual '
    'Total-Text training configuration.'
)