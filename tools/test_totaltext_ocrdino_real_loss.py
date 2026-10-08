import math
from pathlib import Path

import torch
from mmengine.config import Config

from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules


# ============================================================
# Settings
# ============================================================

DATA_ROOT = (
    '/root/autodl-tmp/dataset/totaltext/'
)

CHECKPOINT = (
    '/root/autodl-tmp/checkpoint/dino/'
    'dino-4scale_r50_8xb2-12e_coco_'
    '20221202_182705-55b2bba2.pth'
)

CONFIG = (
    'configs/dino/'
    'dino-4scale_r50_8xb2-12e_coco.py'
)

DEVICE = 'cuda'

SAMPLE_INDEX = 0


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
# 1. Build real Total-Text dataset
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


packed = dataset[
    SAMPLE_INDEX
]

raw_inputs = packed[
    'inputs'
]

raw_sample = packed[
    'data_samples'
]


print(
    'sample index:',
    SAMPLE_INDEX,
)

print(
    'img path:',
    raw_sample.img_path,
)

print(
    'raw image tensor:',
    tuple(
        raw_inputs.shape
    ),
    raw_inputs.dtype,
)

print(
    'ori shape:',
    raw_sample.ori_shape,
)

print(
    'img shape:',
    raw_sample.img_shape,
)

print(
    'GT count:',
    len(
        raw_sample.gt_instances
    ),
)

print(
    'GT bbox:',
    tuple(
        raw_sample.gt_instances
        .bboxes.shape
    ),
)

print(
    'GT labels:',
    raw_sample.gt_instances
    .labels.tolist(),
)

print(
    'GT rec:',
    tuple(
        raw_sample.gt_instances
        .rec.shape
    ),
)

print(
    'texts:',
    raw_sample.gt_instances
    .ocr_texts,
)


# ============================================================
# 2. Verify real targets before model
# ============================================================

gt = raw_sample.gt_instances

num_gt = len(
    gt
)

assert num_gt > 0

assert tuple(
    gt.bboxes.shape
) == (
    num_gt,
    4,
)

assert tuple(
    gt.labels.shape
) == (
    num_gt,
)

assert tuple(
    gt.rec.shape
) == (
    num_gt,
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

# Total-Text is one-class.
assert (
    gt.labels.max().item()
    == 0
)

assert (
    gt.rec.min().item()
    >= 0
)

assert (
    gt.rec.max().item()
    <= 96
)


print()
print(
    'real target validation: PASS'
)


# ============================================================
# 3. Build ESTDINO
#
# Keep exactly the model structure already validated by the
# synthetic end-to-end test.
#
# Only detection classes become 1 for Total-Text.
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
# Total-Text:
#
# bbox_label = 0
# classes = ["text"]
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
# Task-Aware Recognition Query Initialization
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
# ESTextSpotter-style decoder
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
    'num detection classes:',
    model.bbox_head.num_classes,
)

print(
    'num recognition queries:',
    model.num_rec_queries,
)

print(
    'recognition classifier classes:',
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
    model.bbox_head
    .__class__.__name__
    == 'ESTDINOHead'
)

assert (
    model.bbox_head.num_classes
    == 1
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
# 4. Load COCO DINO pretrained weights
#
# Important:
#
# - stock DINO decoder is NOT loaded
# - query embedding 900 -> 100
# - COCO 80-class classifier is skipped because Total-Text
#   uses one detection class
# - DN label embedding mismatch is also allowed to remain
#   randomly initialized
# ============================================================

checkpoint_path = Path(
    CHECKPOINT
)

assert checkpoint_path.exists(), (
    checkpoint_path
)


print()
print(
    '========================================'
)
print(
    'LOAD DINO CHECKPOINT'
)
print(
    '========================================'
)


checkpoint = torch.load(
    checkpoint_path,
    map_location='cpu',
    weights_only=False,
)

state_dict = checkpoint.get(
    'state_dict',
    checkpoint,
)

model_state_dict = (
    model.state_dict()
)

clean_state_dict = {}

adapted_keys = []

shape_skipped_keys = []

missing_in_model_keys = []

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

    # --------------------------------------------------------
    # Do not load stock DINO decoder because OCRDINO replaced it
    # with the task-aware decoder.
    # --------------------------------------------------------

    if key.startswith(
        'decoder.'
    ):

        decoder_skipped += 1

        continue

    # --------------------------------------------------------
    # Adapt matching query count:
    #
    # pretrained = 900
    # OCRDINO test model = 100
    # --------------------------------------------------------

    if (
        key
        == 'query_embedding.weight'
    ):

        if key not in model_state_dict:

            raise RuntimeError(
                'query_embedding.weight '
                'not found in model'
            )

        target = (
            model_state_dict[
                key
            ]
        )

        if not (
            value.ndim == 2
            and target.ndim == 2
            and value.shape[1]
            == target.shape[1]
            and value.shape[0]
            >= target.shape[0]
        ):

            raise RuntimeError(
                'Cannot adapt '
                'query_embedding.weight: '
                f'checkpoint='
                f'{tuple(value.shape)}, '
                f'model='
                f'{tuple(target.shape)}'
            )

        adapted = (
            value[
                :target.shape[0]
            ]
            .clone()
        )

        clean_state_dict[
            key
        ] = adapted

        adapted_keys.append(
            (
                key,
                tuple(
                    value.shape
                ),
                tuple(
                    adapted.shape
                ),
            )
        )

        continue

    # --------------------------------------------------------
    # Ignore keys that no longer exist.
    # --------------------------------------------------------

    if key not in model_state_dict:

        missing_in_model_keys.append(
            key
        )

        continue

    # --------------------------------------------------------
    # Skip incompatible shapes:
    #
    # e.g.
    #
    # COCO cls branch: 80 classes
    # Total-Text:       1 class
    # --------------------------------------------------------

    if (
        value.shape
        != model_state_dict[
            key
        ].shape
    ):

        shape_skipped_keys.append(
            (
                key,
                tuple(
                    value.shape
                ),
                tuple(
                    model_state_dict[
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


print(
    'loaded tensors:',
    len(
        clean_state_dict
    ),
)

print(
    'decoder tensors skipped:',
    decoder_skipped,
)

print(
    'shape mismatches skipped:',
    len(
        shape_skipped_keys
    ),
)

print(
    'missing-in-model skipped:',
    len(
        missing_in_model_keys
    ),
)

print(
    'adapted keys:',
    adapted_keys,
)

print(
    'missing keys after load:',
    len(
        load_result.missing_keys
    ),
)

print(
    'unexpected keys after load:',
    len(
        load_result.unexpected_keys
    ),
)


print()
print(
    'first shape mismatches:'
)

for item in (
    shape_skipped_keys[
        :20
    ]
):

    print(
        item
    )


# ============================================================
# 5. Move model to GPU
# ============================================================

model = model.to(
    device
)

model.train()


# ============================================================
# 6. Run the REAL MMDetection data preprocessor
#
# This performs:
#
# uint8
#   ->
# float
#   ->
# BGR/RGB conversion
#   ->
# normalization
#   ->
# batch padding
#
# and also moves DetDataSample / gt_instances.rec to CUDA.
# ============================================================

data = dict(
    inputs=[
        raw_inputs
    ],

    data_samples=[
        raw_sample
    ],
)


processed = (
    model.data_preprocessor(
        data,
        training=True,
    )
)


batch_inputs = (
    processed[
        'inputs'
    ]
)

batch_data_samples = (
    processed[
        'data_samples'
    ]
)


print()
print(
    '========================================'
)
print(
    'AFTER DATA PREPROCESSOR'
)
print(
    '========================================'
)

print(
    'batch inputs:',
    tuple(
        batch_inputs.shape
    ),
)

print(
    'batch dtype:',
    batch_inputs.dtype,
)

print(
    'batch device:',
    batch_inputs.device,
)


assert (
    batch_inputs.ndim
    == 4
)

assert (
    batch_inputs.shape[0]
    == 1
)

assert (
    batch_inputs.device.type
    == 'cuda'
)


sample = (
    batch_data_samples[
        0
    ]
)

gt = (
    sample.gt_instances
)


print(
    'img shape:',
    sample.img_shape,
)

print(
    'batch input shape:',
    sample.batch_input_shape,
)

print(
    'GT bboxes:',
    tuple(
        gt.bboxes.shape
    ),
    gt.bboxes.device,
)

print(
    'GT labels:',
    tuple(
        gt.labels.shape
    ),
    gt.labels.device,
)

print(
    'GT rec:',
    tuple(
        gt.rec.shape
    ),
    gt.rec.dtype,
    gt.rec.device,
)


assert tuple(
    gt.rec.shape
) == (
    num_gt,
    25,
)

assert (
    gt.rec.dtype
    == torch.long
)

assert (
    gt.rec.device.type
    == 'cuda'
)

assert (
    gt.bboxes.device.type
    == 'cuda'
)

assert (
    gt.labels.device.type
    == 'cuda'
)


print(
    'data preprocessor: PASS'
)


# ============================================================
# 7. Clear previous gradients
# ============================================================

model.zero_grad(
    set_to_none=True
)


# ============================================================
# 8. REAL model.loss()
# ============================================================

print()
print(
    '========================================'
)
print(
    'RUN REAL model.loss()'
)
print(
    '========================================'
)


losses = model.loss(
    batch_inputs,
    batch_data_samples,
)


print()
print(
    'model.loss() completed.'
)

print()

print(
    'complete OCRDINO loss dictionary:'
)


for key in sorted(
    losses.keys()
):

    value = (
        losses[
            key
        ]
    )

    if torch.is_tensor(
        value
    ):

        print(
            key,
            '=',
            value.detach().item(),
        )

    else:

        print(
            key,
            '=',
            value,
        )


# ============================================================
# 9. Required recognition loss keys
# ============================================================

expected_recognition_keys = [
    'loss_rec',
    'dn_loss_rec',

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


missing_recognition_keys = [
    key
    for key
    in expected_recognition_keys
    if key not in losses
]


print()
print(
    'missing recognition loss keys:',
    missing_recognition_keys,
)


assert not (
    missing_recognition_keys
)


# ============================================================
# 10. Required detection losses
# ============================================================

expected_detection_keys = [
    'loss_cls',
    'loss_bbox',
    'loss_iou',

    'enc_loss_cls',
    'enc_loss_bbox',
    'enc_loss_iou',

    'dn_loss_cls',
    'dn_loss_bbox',
    'dn_loss_iou',
]


missing_detection_keys = [
    key
    for key
    in expected_detection_keys
    if key not in losses
]


print(
    'missing detection loss keys:',
    missing_detection_keys,
)


assert not (
    missing_detection_keys
)


# ============================================================
# 11. Every loss must be finite
# ============================================================

loss_tensors = []

for key, value in (
    losses.items()
):

    if (
        'loss'
        not in key
    ):
        continue

    if not torch.is_tensor(
        value
    ):
        continue

    if value.numel() != 1:

        raise RuntimeError(
            f'Loss {key} is not scalar: '
            f'{tuple(value.shape)}'
        )

    if not torch.isfinite(
        value
    ).item():

        raise RuntimeError(
            f'Non-finite loss: '
            f'{key}={value}'
        )

    loss_tensors.append(
        value
    )


print()
print(
    'finite loss tensors:',
    len(
        loss_tensors
    ),
)


assert len(
    loss_tensors
) > 0


# ============================================================
# 12. Joint loss
# ============================================================

total_loss = sum(
    loss_tensors
)


recognition_loss = sum(
    value
    for key, value
    in losses.items()
    if (
        'loss_rec'
        in key
        and torch.is_tensor(
            value
        )
    )
)


print()
print(
    'total joint loss:',
    total_loss.detach().item(),
)

print(
    'summed recognition loss:',
    recognition_loss.detach().item(),
)


assert torch.isfinite(
    total_loss
).item()

assert torch.isfinite(
    recognition_loss
).item()

assert (
    recognition_loss.detach().item()
    > 0.0
)


# ============================================================
# 13. Backward
# ============================================================

print()
print(
    'running backward() ...'
)


total_loss.backward()


print(
    'backward completed.'
)


# ============================================================
# 14. Gradient helpers
# ============================================================

def gradient_stats(
    module,
):

    parameter_tensors = 0
    parameter_tensors_with_grad = 0

    num_trainable = 0
    num_with_grad = 0

    grad_sum = 0.0

    finite = True

    for parameter in (
        module.parameters()
    ):

        if not parameter.requires_grad:
            continue

        parameter_tensors += 1

        num_trainable += (
            parameter.numel()
        )

        if parameter.grad is None:
            continue

        parameter_tensors_with_grad += 1

        num_with_grad += (
            parameter.numel()
        )

        gradient = (
            parameter.grad
        )

        if not torch.isfinite(
            gradient
        ).all().item():

            finite = False

        grad_sum += (
            gradient
            .detach()
            .abs()
            .sum()
            .item()
        )

    return dict(
        parameter_tensors=(
            parameter_tensors
        ),

        parameter_tensors_with_grad=(
            parameter_tensors_with_grad
        ),

        num_trainable=(
            num_trainable
        ),

        num_with_grad=(
            num_with_grad
        ),

        grad_sum=(
            grad_sum
        ),

        finite=(
            finite
        ),
    )


# ============================================================
# 15. Backbone gradient
# ============================================================

backbone_stats = (
    gradient_stats(
        model.backbone
    )
)


print()
print(
    'backbone gradient stats:',
    backbone_stats,
)


assert (
    backbone_stats[
        'parameter_tensors_with_grad'
    ]
    > 0
)

assert (
    backbone_stats[
        'grad_sum'
    ]
    > 0.0
)

assert (
    backbone_stats[
        'finite'
    ]
)


# ============================================================
# 16. Task-aware decoder gradient
# ============================================================

decoder_stats = (
    gradient_stats(
        model.decoder
    )
)


print()
print(
    'task-aware decoder gradient stats:',
    decoder_stats,
)


assert (
    decoder_stats[
        'parameter_tensors_with_grad'
    ]
    > 0
)

assert (
    decoder_stats[
        'grad_sum'
    ]
    > 0.0
)

assert (
    decoder_stats[
        'finite'
    ]
)


# ============================================================
# 17. BBox head gradient
# ============================================================

bbox_head_stats = (
    gradient_stats(
        model.bbox_head
    )
)


print()
print(
    'bbox head gradient stats:',
    bbox_head_stats,
)


assert (
    bbox_head_stats[
        'parameter_tensors_with_grad'
    ]
    > 0
)

assert (
    bbox_head_stats[
        'grad_sum'
    ]
    > 0.0
)

assert (
    bbox_head_stats[
        'finite'
    ]
)


# ============================================================
# 18. Recognition classifier gradient
# ============================================================

rec_classifier = (
    model.decoder
    .recognition_semantics
    .rec_classifier
)


rec_weight_grad = (
    rec_classifier
    .weight
    .grad
)

rec_bias_grad = (
    rec_classifier
    .bias
    .grad
)


print()
print(
    'recognition classifier weight '
    'grad is None:',
    rec_weight_grad is None,
)

print(
    'recognition classifier bias '
    'grad is None:',
    rec_bias_grad is None,
)


assert (
    rec_weight_grad
    is not None
)

assert (
    rec_bias_grad
    is not None
)


weight_grad_sum = (
    rec_weight_grad
    .detach()
    .abs()
    .sum()
    .item()
)

bias_grad_sum = (
    rec_bias_grad
    .detach()
    .abs()
    .sum()
    .item()
)


print(
    'recognition classifier weight '
    'grad abs sum:',
    weight_grad_sum,
)

print(
    'recognition classifier bias '
    'grad abs sum:',
    bias_grad_sum,
)


assert (
    weight_grad_sum
    > 0.0
)

assert (
    bias_grad_sum
    > 0.0
)

assert torch.isfinite(
    rec_weight_grad
).all().item()

assert torch.isfinite(
    rec_bias_grad
).all().item()


# ============================================================
# 19. Global gradient validation
# ============================================================

trainable_parameter_tensors = 0

parameter_tensors_with_grad = 0

nonfinite_gradient_names = []


for name, parameter in (
    model.named_parameters()
):

    if not parameter.requires_grad:
        continue

    trainable_parameter_tensors += 1

    if parameter.grad is None:
        continue

    parameter_tensors_with_grad += 1

    if not torch.isfinite(
        parameter.grad
    ).all().item():

        nonfinite_gradient_names.append(
            name
        )


print()
print(
    'global gradient inspection:'
)

print(
    'trainable parameter tensors:',
    trainable_parameter_tensors,
)

print(
    'parameter tensors with gradient:',
    parameter_tensors_with_grad,
)

print(
    'non-finite gradient tensors:',
    nonfinite_gradient_names,
)


assert (
    parameter_tensors_with_grad
    > 0
)

assert not (
    nonfinite_gradient_names
)


# ============================================================
# 20. Final
# ============================================================

print()
print(
    '============================================'
)

print(
    'OCRDINO REAL TOTAL-TEXT TRAINING PATH PASSED'
)

print(
    '============================================'
)

print()

print(
    'real image loading: PASS'
)

print(
    'real bbox targets: PASS'
)

print(
    'real recognition targets [N,25]: PASS'
)

print(
    'data_preprocessor: PASS'
)

print(
    'model.loss(): PASS'
)

print(
    'DINO detection losses: PASS'
)

print(
    'matching recognition losses: PASS'
)

print(
    'DN recognition losses: PASS'
)

print(
    'six-layer auxiliary recognition losses: PASS'
)

print(
    'joint total loss finite: PASS'
)

print(
    'backward(): PASS'
)

print(
    'backbone gradient: PASS'
)

print(
    'task-aware decoder gradient: PASS'
)

print(
    'bbox head gradient: PASS'
)

print(
    'recognition classifier gradient: PASS'
)

print(
    'all observed gradients finite: PASS'
)