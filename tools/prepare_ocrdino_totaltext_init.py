from pathlib import Path

import torch
from mmengine.config import Config

from mmdet.registry import MODELS
from mmdet.utils import register_all_modules


CONFIG = (
    'configs/ocrdino/'
    'ocrdino_r50_totaltext_original_24e.py'
)

SOURCE_CHECKPOINT = (
    '/root/autodl-tmp/checkpoint/dino/'
    'dino-4scale_r50_8xb2-12e_coco_'
    '20221202_182705-55b2bba2.pth'
)

OUTPUT_CHECKPOINT = (
    '/root/autodl-tmp/checkpoint/ocrdino/'
    'ocrdino_r50_totaltext_init.pth'
)


register_all_modules()

torch.manual_seed(42)


# ============================================================
# Build the exact training model
# ============================================================

cfg = Config.fromfile(
    CONFIG
)

model = MODELS.build(
    cfg.model
)

model.init_weights()


print()
print(
    '========================================'
)

print(
    'OCRDINO TOTAL-TEXT INITIALIZATION'
)

print(
    '========================================'
)

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
# Load source DINO checkpoint
# ============================================================

source_path = Path(
    SOURCE_CHECKPOINT
)

if not source_path.is_file():

    raise FileNotFoundError(
        source_path
    )


checkpoint = torch.load(
    source_path,

    map_location='cpu',

    weights_only=False,
)


source_state = checkpoint.get(
    'state_dict',

    checkpoint,
)


model_state = (
    model.state_dict()
)


# ============================================================
# Adapt source weights
# ============================================================

clean_state = {}

adapted = []

shape_skipped = []

missing_skipped = []

decoder_skipped = []


for original_key, value in (
    source_state.items()
):

    key = original_key

    if key.startswith(
        'module.'
    ):

        key = key[
            len('module.') :
        ]

    # Do not load the stock DINO decoder.
    if key.startswith(
        'decoder.'
    ):

        decoder_skipped.append(
            key
        )

        continue

    # Adapt 900 matching queries to 100 matching queries.
    if (
        key
        == 'query_embedding.weight'
    ):

        target = (
            model_state[
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
                'Cannot adapt query_embedding.weight. '
                f'checkpoint={tuple(value.shape)}, '
                f'model={tuple(target.shape)}'
            )

        adapted_value = (
            value[
                :target.shape[0]
            ].clone()
        )

        clean_state[
            key
        ] = adapted_value

        adapted.append(
            (
                key,

                tuple(
                    value.shape
                ),

                tuple(
                    adapted_value.shape
                ),
            )
        )

        continue

    if key not in model_state:

        missing_skipped.append(
            key
        )

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

    clean_state[
        key
    ] = value


# ============================================================
# Load adapted weights
# ============================================================

load_result = (
    model.load_state_dict(
        clean_state,

        strict=False,
    )
)


print()
print(
    'loaded tensors:',
    len(
        clean_state
    ),
)

print(
    'decoder tensors skipped:',
    len(
        decoder_skipped
    ),
)

print(
    'shape mismatches:',
    len(
        shape_skipped
    ),
)

print(
    'missing source keys:',
    len(
        missing_skipped
    ),
)

print(
    'adapted:',
    adapted,
)

print(
    'missing keys after load:',
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


print()
print(
    'shape mismatches:'
)


for item in shape_skipped:

    print(
        item
    )


# ============================================================
# Expected adaptation checks
# ============================================================

assert (
    len(
        decoder_skipped
    )
    == 138
), len(decoder_skipped)


assert (
    len(
        shape_skipped
    )
    == 15
), shape_skipped


assert (
    adapted
    == [
        (
            'query_embedding.weight',

            (
                900,
                256,
            ),

            (
                100,
                256,
            ),
        )
    ]
)


assert not (
    load_result.unexpected_keys
)


# ============================================================
# Save
# ============================================================

output_path = Path(
    OUTPUT_CHECKPOINT
)


output_path.parent.mkdir(
    parents=True,

    exist_ok=True,
)


save_data = dict(
    state_dict=(
        model.state_dict()
    ),

    meta=dict(
        source_checkpoint=(
            str(
                source_path
            )
        ),

        model='ESTDINO',

        dataset='Total-Text',

        detection_classes=1,

        matching_queries=100,

        recognition_queries=25,

        recognition_classes=97,

        recognition_length=25,

        initialization=(
            'Selective COCO DINO initialization'
        ),
    ),
)


torch.save(
    save_data,

    output_path,
)


print()
print(
    '========================================'
)

print(
    'OCRDINO INIT CHECKPOINT CREATED'
)

print(
    '========================================'
)

print(
    output_path
)