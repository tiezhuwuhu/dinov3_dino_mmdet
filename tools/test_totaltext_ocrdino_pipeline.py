import json
from pathlib import Path

import torch
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules


DATA_ROOT = (
    '/root/autodl-tmp/dataset/totaltext/'
)

ANN_FILE = (
    'ocrdino_train.json'
)


# ============================================================
# Recognition reference encoder
# ============================================================

def reference_encode(text):

    result = []

    for char in text:

        code = ord(char)

        if 32 <= code <= 126:

            result.append(
                code - 32
            )

        else:

            result.append(
                95
            )

        if len(result) == 25:
            break

    if len(result) < 25:

        result.extend(
            [96]
            * (
                25
                - len(result)
            )
        )

    return result


def decode(rec):

    chars = []

    for token in rec:

        token = int(token)

        if token == 96:
            break

        if 0 <= token <= 94:

            chars.append(
                chr(
                    token + 32
                )
            )

        elif token == 95:

            chars.append(
                '\ufffd'
            )

        elif token == -100:

            continue

        else:

            raise RuntimeError(
                f'Invalid token {token}'
            )

    return ''.join(chars)


# ============================================================
# Registry
# ============================================================

register_all_modules(
    init_default_scope=True
)

init_default_scope(
    'mmdet'
)


# ============================================================
# Build REAL Total-Text dataset
# ============================================================

dataset_cfg = dict(
    type='OCRDinoDataset',

    data_root=(
        DATA_ROOT
    ),

    ann_file=(
        ANN_FILE
    ),

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
    'DATASET BUILT'
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
# Load raw annotation JSON for reference
# ============================================================

raw_path = Path(
    DATA_ROOT
) / ANN_FILE

with raw_path.open(
    'r',
    encoding='utf-8',
) as f:

    raw_data = json.load(f)


assert (
    len(
        raw_data[
            'data_list'
        ]
    )
    == len(dataset)
)


# ============================================================
# Test several images instead of only one
# ============================================================

test_indices = [
    0,
    1,
    10,
    37,
    100,
]


total_instances = 0


for dataset_index in test_indices:

    print()
    print(
        '========================================'
    )
    print(
        'SAMPLE',
        dataset_index,
    )
    print(
        '========================================'
    )

    packed = dataset[
        dataset_index
    ]

    assert 'inputs' in packed
    assert 'data_samples' in packed

    inputs = packed[
        'inputs'
    ]

    sample = packed[
        'data_samples'
    ]

    gt = (
        sample.gt_instances
    )

    raw_item = (
        raw_data[
            'data_list'
        ][
            dataset_index
        ]
    )

    raw_instances = (
        raw_item[
            'instances'
        ]
    )

    print(
        'image tensor:',
        tuple(
            inputs.shape
        ),
        inputs.dtype,
    )

    print(
        'img_path:',
        sample.img_path,
    )

    print(
        'ori_shape:',
        sample.ori_shape,
    )

    print(
        'img_shape:',
        sample.img_shape,
    )

    print(
        'scale_factor:',
        sample.scale_factor,
    )

    print(
        'GT count:',
        len(gt),
    )

    print(
        'bboxes shape:',
        tuple(
            gt.bboxes.shape
        ),
    )

    print(
        'labels shape:',
        tuple(
            gt.labels.shape
        ),
    )

    print(
        'rec shape:',
        tuple(
            gt.rec.shape
        ),
    )

    print(
        'rec dtype:',
        gt.rec.dtype,
    )

    print(
        'ocr texts:',
        gt.ocr_texts,
    )

    print(
        'ocr text types:',
        gt.ocr_text_types,
    )

    print(
        'ocr trainable:',
        gt.ocr_trainable,
    )

    # --------------------------------------------------------
    # Shapes
    # --------------------------------------------------------

    num_gt = len(
        raw_instances
    )

    assert len(gt) == num_gt

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
        len(
            gt.ocr_texts
        )
        == num_gt
    )

    assert (
        len(
            gt.ocr_text_types
        )
        == num_gt
    )

    assert (
        len(
            gt.ocr_trainable
        )
        == num_gt
    )

    # --------------------------------------------------------
    # Text alignment
    # --------------------------------------------------------

    for instance_index in range(
        num_gt
    ):

        raw_instance = (
            raw_instances[
                instance_index
            ]
        )

        expected_text = (
            raw_instance[
                'text'
            ]
        )

        expected_type = (
            raw_instance[
                'text_type'
            ]
        )

        packed_text = (
            gt.ocr_texts[
                instance_index
            ]
        )

        packed_type = (
            gt.ocr_text_types[
                instance_index
            ]
        )

        assert (
            packed_text
            == expected_text
        ), (
            dataset_index,
            instance_index,
            packed_text,
            expected_text,
        )

        assert (
            packed_type
            == expected_type
        )

        # ----------------------------------------------------
        # Recognition target exact check
        # ----------------------------------------------------

        expected_rec = torch.tensor(
            reference_encode(
                expected_text
            ),
            dtype=torch.long,
        )

        actual_rec = (
            gt.rec[
                instance_index
            ]
            .cpu()
        )

        assert torch.equal(
            actual_rec,
            expected_rec,
        ), (
            dataset_index,
            instance_index,
            expected_text,
            actual_rec.tolist(),
            expected_rec.tolist(),
        )

        decoded = decode(
            actual_rec
        )

        expected_decoded = (
            expected_text[
                :25
            ]
        )

        # Total-Text has printable ASCII only.
        assert (
            decoded
            == expected_decoded
        ), (
            dataset_index,
            instance_index,
            decoded,
            expected_decoded,
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    assert (
        sample.metainfo[
            'ocr_num_classes'
        ]
        == 97
    )

    assert (
        sample.metainfo[
            'ocr_vocab_size'
        ]
        == 97
    )

    assert (
        sample.metainfo[
            'ocr_max_length'
        ]
        == 25
    )

    assert (
        sample.metainfo[
            'ocr_unk_token_id'
        ]
        == 95
    )

    assert (
        sample.metainfo[
            'ocr_eos_token_id'
        ]
        == 96
    )

    assert (
        sample.metainfo[
            'ocr_ignore_index'
        ]
        == -100
    )

    total_instances += num_gt

    if num_gt > 0:

        print()
        print(
            'first text:',
            repr(
                gt.ocr_texts[
                    0
                ]
            ),
        )

        print(
            'first rec:',
            gt.rec[
                0
            ].tolist(),
        )

        print(
            'first decoded:',
            repr(
                decode(
                    gt.rec[
                        0
                    ]
                )
            ),
        )

    print()
    print(
        'sample alignment: PASS'
    )


# ============================================================
# Final
# ============================================================

print()
print(
    '============================================'
)
print(
    'REAL TOTAL-TEXT PIPELINE PASSED'
)
print(
    '============================================'
)

print(
    'tested images:',
    len(
        test_indices
    ),
)

print(
    'tested instances:',
    total_instances,
)

print(
    'bbox/text/rec alignment: PASS'
)

print(
    'rec shape [N,25]: PASS'
)

print(
    'rec dtype torch.long: PASS'
)

print(
    '97-class encoding: PASS'
)

print(
    'EOS=96: PASS'
)

print(
    'Total-Text real data pipeline is ready.'
)