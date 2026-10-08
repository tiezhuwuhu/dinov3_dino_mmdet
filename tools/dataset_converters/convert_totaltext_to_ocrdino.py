import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


# ============================================================
# OCRDINO v1
# ============================================================

FORMAT_VERSION = 'ocrdino_v1'

SOURCE_NAME = 'totaltext'

DEFAULT_LANGUAGE = 'en'

TEXT_TYPE = 'text'

DEFAULT_TEXT_BBOX_LABEL = 0

DEFAULT_CLASSES = [
    'text',
]


# ============================================================
# ESTextSpotter / SPTS recognition encoding
#
# printable ASCII:
#     token 0..94
#     ASCII 32..126
#
# token 95:
#     UNK / reserved
#
# token 96:
#     EOS
#
# sequence length:
#     25
# ============================================================

ASCII_FIRST = 32

ASCII_CLASS_MIN = 0
ASCII_CLASS_MAX = 94

UNK_ID = 95
EOS_ID = 96

REC_LENGTH = 25

UNK_CHAR = '\ufffd'


# ============================================================
# Recognition
# ============================================================

def decode_rec(rec):
    """Decode Total-Text ESTextSpotter/SPTS recognition target."""

    if not isinstance(rec, list):
        raise TypeError(
            f'rec must be list, got {type(rec)}'
        )

    if len(rec) != REC_LENGTH:
        raise ValueError(
            f'rec length must be {REC_LENGTH}, '
            f'got {len(rec)}'
        )

    chars = []

    eos_seen = False

    for position, token in enumerate(rec):

        if not isinstance(token, int):
            raise TypeError(
                f'rec[{position}] must be int, '
                f'got {type(token)}'
            )

        if token < 0 or token > EOS_ID:
            raise ValueError(
                f'invalid rec token={token} '
                f'at position={position}'
            )

        if eos_seen:

            if token != EOS_ID:
                raise ValueError(
                    'Non-EOS token appears after EOS: '
                    f'position={position}, '
                    f'token={token}'
                )

            continue

        if token == EOS_ID:

            eos_seen = True
            continue

        if ASCII_CLASS_MIN <= token <= ASCII_CLASS_MAX:

            chars.append(
                chr(
                    token
                    + ASCII_FIRST
                )
            )

        elif token == UNK_ID:

            chars.append(
                UNK_CHAR
            )

        else:

            raise ValueError(
                f'Unexpected rec token: {token}'
            )

    return ''.join(chars), eos_seen


# ============================================================
# Image
# ============================================================

def get_real_image_size(image_path):
    """Read actual JPEG/PNG size from disk."""

    with Image.open(image_path) as image:

        width, height = image.size

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            f'Invalid real image size: '
            f'{width}x{height}'
        )

    return width, height


# ============================================================
# Source bbox
#
# Total-Text converted annotation uses:
#
#     [x, y, w, h]
#
# Examination of the provided train/test annotations shows
# that width/height correspond to inclusive integer extent:
#
#     xmax = x + w - 1
#     ymax = y + h - 1
#
# OCRDINO stores:
#
#     [x1, y1, x2, y2]
# ============================================================

def source_bbox_to_xyxy(source_bbox):

    if not isinstance(
        source_bbox,
        list,
    ):
        raise TypeError(
            'bbox must be list'
        )

    if len(source_bbox) != 4:
        raise ValueError(
            f'bbox must contain 4 values, '
            f'got {len(source_bbox)}'
        )

    values = []

    for value in source_bbox:

        if not isinstance(
            value,
            (int, float),
        ):
            raise TypeError(
                'bbox coordinates must be numeric'
            )

        value = float(value)

        if not math.isfinite(value):
            raise ValueError(
                'bbox contains non-finite value'
            )

        values.append(value)

    x, y, w, h = values

    if w <= 0 or h <= 0:
        raise ValueError(
            f'Invalid source bbox size: '
            f'{source_bbox}'
        )

    x1 = x
    y1 = y

    x2 = (
        x
        + w
        - 1.0
    )

    y2 = (
        y
        + h
        - 1.0
    )

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2),
    ]


# ============================================================
# Clip bbox to actual image boundary.
#
# MMDetection XYXY uses continuous coordinates.
#
# Therefore:
#
#     0 <= x <= width
#     0 <= y <= height
#
# is permitted.
# ============================================================

def clip_bbox_to_image(
    bbox,
    width,
    height,
):

    x1, y1, x2, y2 = bbox

    original = [
        x1,
        y1,
        x2,
        y2,
    ]

    clipped = [
        min(
            max(
                x1,
                0.0,
            ),
            float(width),
        ),

        min(
            max(
                y1,
                0.0,
            ),
            float(height),
        ),

        min(
            max(
                x2,
                0.0,
            ),
            float(width),
        ),

        min(
            max(
                y2,
                0.0,
            ),
            float(height),
        ),
    ]

    changed = (
        clipped
        != original
    )

    left_overflow = max(
        0.0,
        -x1,
    )

    top_overflow = max(
        0.0,
        -y1,
    )

    right_overflow = max(
        0.0,
        x2 - width,
    )

    bottom_overflow = max(
        0.0,
        y2 - height,
    )

    overflow = dict(
        left=float(
            left_overflow
        ),
        top=float(
            top_overflow
        ),
        right=float(
            right_overflow
        ),
        bottom=float(
            bottom_overflow
        ),
    )

    return (
        clipped,
        changed,
        overflow,
    )


def bbox_has_positive_area(bbox):

    x1, y1, x2, y2 = bbox

    return (
        x2 > x1
        and
        y2 > y1
    )


# ============================================================
# Convert one split
# ============================================================

def convert_split(
    data_root,
    input_json_name,
    image_dir_name,
    output_json_name,
    split_name,
    text_bbox_label,
):

    data_root = Path(
        data_root
    )

    input_json = (
        data_root
        / input_json_name
    )

    image_dir = (
        data_root
        / image_dir_name
    )

    output_json = (
        data_root
        / output_json_name
    )

    # --------------------------------------------------------
    # Load source JSON
    # --------------------------------------------------------

    with input_json.open(
        'r',
        encoding='utf-8',
    ) as f:

        source = json.load(f)

    for key in (
        'images',
        'annotations',
        'categories',
    ):

        if key not in source:

            raise KeyError(
                f'Missing top-level key: {key}'
            )

    images = (
        source[
            'images'
        ]
    )

    annotations = (
        source[
            'annotations'
        ]
    )

    categories = (
        source[
            'categories'
        ]
    )

    print()
    print(
        f'===== converting {split_name} ====='
    )

    print(
        'source:',
        input_json,
    )

    print(
        'images:',
        len(images),
    )

    print(
        'annotations:',
        len(annotations),
    )

    print(
        'categories:',
        len(categories),
    )

    # ========================================================
    # Build image table
    # ========================================================

    image_map = {}

    num_image_size_mismatches = 0

    num_swapped_image_sizes = 0

    image_size_mismatch_examples = []

    for image in images:

        for key in (
            'id',
            'file_name',
            'width',
            'height',
        ):

            if key not in image:

                raise KeyError(
                    f'Image record missing '
                    f'{key}: {image}'
                )

        image_id = (
            image[
                'id'
            ]
        )

        if image_id in image_map:

            raise ValueError(
                f'Duplicate image_id: '
                f'{image_id}'
            )

        file_name = str(
            image[
                'file_name'
            ]
        )

        image_path = (
            image_dir
            / file_name
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f'Missing image: '
                f'{image_path}'
            )

        json_width = int(
            image[
                'width'
            ]
        )

        json_height = int(
            image[
                'height'
            ]
        )

        (
            real_width,
            real_height,
        ) = get_real_image_size(
            image_path
        )

        size_mismatch = (
            json_width
            != real_width
            or
            json_height
            != real_height
        )

        if size_mismatch:

            num_image_size_mismatches += 1

            swapped = (
                json_width
                == real_height
                and
                json_height
                == real_width
            )

            if swapped:
                num_swapped_image_sizes += 1

            if (
                len(
                    image_size_mismatch_examples
                )
                < 20
            ):

                image_size_mismatch_examples.append(
                    {
                        'image_id': (
                            image_id
                        ),

                        'file_name': (
                            file_name
                        ),

                        'json_size': [
                            json_width,
                            json_height,
                        ],

                        'real_size': [
                            real_width,
                            real_height,
                        ],

                        'swapped': (
                            swapped
                        ),
                    }
                )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # OCRDINO always uses ACTUAL image dimensions.
        # ----------------------------------------------------

        image_map[
            image_id
        ] = {
            'file_name': (
                file_name
            ),

            'width': (
                real_width
            ),

            'height': (
                real_height
            ),

            'json_width': (
                json_width
            ),

            'json_height': (
                json_height
            ),
        }

    # ========================================================
    # Group annotations
    # ========================================================

    annotations_by_image = (
        defaultdict(list)
    )

    annotation_ids = set()

    for annotation in annotations:

        for key in (
            'id',
            'image_id',
            'bbox',
            'rec',
            'category_id',
        ):

            if key not in annotation:

                raise KeyError(
                    f'Annotation missing '
                    f'{key}: '
                    f'{annotation}'
                )

        annotation_id = (
            annotation[
                'id'
            ]
        )

        if (
            annotation_id
            in annotation_ids
        ):

            raise ValueError(
                'Duplicate annotation_id: '
                f'{annotation_id}'
            )

        annotation_ids.add(
            annotation_id
        )

        image_id = (
            annotation[
                'image_id'
            ]
        )

        if image_id not in image_map:

            raise KeyError(
                f'annotation {annotation_id} '
                f'references missing '
                f'image_id={image_id}'
            )

        annotations_by_image[
            image_id
        ].append(
            annotation
        )

    # ========================================================
    # Statistics
    # ========================================================

    token_counter = Counter()

    char_counter = Counter()

    text_length_counter = Counter()

    num_instances = 0

    num_ignored = 0

    num_unknown_tokens = 0

    num_sequences_with_eos = 0

    num_full_length_without_eos = 0

    num_clipped_bboxes = 0

    clipped_image_ids = set()

    max_overflow = {
        'left': 0.0,
        'top': 0.0,
        'right': 0.0,
        'bottom': 0.0,
    }

    clipping_examples = []

    output_data_list = []

    # ========================================================
    # Convert images
    # ========================================================

    for image in images:

        image_id = (
            image[
                'id'
            ]
        )

        image_info = (
            image_map[
                image_id
            ]
        )

        file_name = (
            image_info[
                'file_name'
            ]
        )

        width = (
            image_info[
                'width'
            ]
        )

        height = (
            image_info[
                'height'
            ]
        )

        image_annotations = sorted(
            annotations_by_image[
                image_id
            ],
            key=lambda item: item['id'],
        )

        output_instances = []

        # ====================================================
        # Convert instances
        # ====================================================

        for order, annotation in enumerate(
            image_annotations
        ):

            annotation_id = (
                annotation[
                    'id'
                ]
            )

            # ------------------------------------------------
            # Recognition
            # ------------------------------------------------

            rec = (
                annotation[
                    'rec'
                ]
            )

            (
                text,
                eos_seen,
            ) = decode_rec(
                rec
            )

            if len(text) == 0:

                raise RuntimeError(
                    f'annotation '
                    f'{annotation_id}: '
                    'decoded text is empty'
                )

            for token in rec:

                token_counter[
                    token
                ] += 1

                if token == UNK_ID:

                    num_unknown_tokens += 1

            for char in text:

                char_counter[
                    char
                ] += 1

            text_length_counter[
                len(text)
            ] += 1

            if eos_seen:

                num_sequences_with_eos += 1

            else:

                if (
                    len(text)
                    == REC_LENGTH
                ):

                    num_full_length_without_eos += 1

                else:

                    raise RuntimeError(
                        f'annotation '
                        f'{annotation_id}: '
                        'no EOS but '
                        f'text length={len(text)}'
                    )

            # ------------------------------------------------
            # BBox
            #
            # Source:
            #     [x, y, w, h]
            #
            # OCRDINO:
            #     [x1, y1, x2, y2]
            # ------------------------------------------------

            raw_bbox = (
                source_bbox_to_xyxy(
                    annotation[
                        'bbox'
                    ]
                )
            )

            (
                bbox,
                bbox_was_clipped,
                overflow,
            ) = clip_bbox_to_image(
                bbox=raw_bbox,
                width=width,
                height=height,
            )

            if bbox_was_clipped:

                num_clipped_bboxes += 1

                clipped_image_ids.add(
                    image_id
                )

                for direction in (
                    'left',
                    'top',
                    'right',
                    'bottom',
                ):

                    max_overflow[
                        direction
                    ] = max(
                        max_overflow[
                            direction
                        ],
                        overflow[
                            direction
                        ],
                    )

                if (
                    len(
                        clipping_examples
                    )
                    < 20
                ):

                    clipping_examples.append(
                        {
                            'annotation_id': (
                                annotation_id
                            ),

                            'image_id': (
                                image_id
                            ),

                            'image_size': [
                                width,
                                height,
                            ],

                            'raw_bbox': (
                                raw_bbox
                            ),

                            'clipped_bbox': (
                                bbox
                            ),

                            'overflow': (
                                overflow
                            ),
                        }
                    )

            # ------------------------------------------------
            # Must still have positive area.
            # ------------------------------------------------

            if not (
                bbox_has_positive_area(
                    bbox
                )
            ):

                raise RuntimeError(
                    f'annotation '
                    f'{annotation_id}: '
                    'bbox became degenerate '
                    'after clipping. '
                    f'raw={raw_bbox}, '
                    f'clipped={bbox}, '
                    f'image={width}x{height}'
                )

            x1, y1, x2, y2 = bbox

            # ------------------------------------------------
            # Final geometry validation
            # ------------------------------------------------

            if not (
                0.0 <= x1 <= width
                and
                0.0 <= x2 <= width
                and
                0.0 <= y1 <= height
                and
                0.0 <= y2 <= height
            ):

                raise RuntimeError(
                    f'annotation '
                    f'{annotation_id}: '
                    'bbox still outside image '
                    f'after clipping: '
                    f'{bbox}'
                )

            # ------------------------------------------------
            # Ignore
            # ------------------------------------------------

            iscrowd = int(
                annotation.get(
                    'iscrowd',
                    0,
                )
            )

            ignore_flag = int(
                iscrowd != 0
            )

            if ignore_flag:

                num_ignored += 1

            # ------------------------------------------------
            # OCRDINO v1 instance
            #
            # IMPORTANT:
            #
            # NO polygon
            # NO bezier
            # NO original rec
            # ------------------------------------------------

            output_instance = {
                'bbox': [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2),
                ],

                'bbox_label': int(
                    text_bbox_label
                ),

                'ignore_flag': (
                    ignore_flag
                ),

                'text': (
                    text
                ),

                'text_type': (
                    TEXT_TYPE
                ),

                'order': int(
                    order
                ),
            }

            output_instances.append(
                output_instance
            )

            num_instances += 1

        # ====================================================
        # OCRDINO image record
        # ====================================================

        output_data_list.append(
            {
                'img_path': (
                    f'{image_dir_name}/'
                    f'{file_name}'
                ),

                'height': int(
                    height
                ),

                'width': int(
                    width
                ),

                'language': (
                    DEFAULT_LANGUAGE
                ),

                'source': (
                    SOURCE_NAME
                ),

                'instances': (
                    output_instances
                ),
            }
        )

    # ========================================================
    # OCRDINO v1
    # ========================================================

    output = {
        'metainfo': {
            'dataset_name': (
                f'{SOURCE_NAME}_{split_name}'
            ),

            'format_version': (
                FORMAT_VERSION
            ),

            'classes': list(
                DEFAULT_CLASSES
            ),

            'source': (
                SOURCE_NAME
            ),

            'split': (
                split_name
            ),
        },

        'data_list': (
            output_data_list
        ),
    }

    # ========================================================
    # Write
    # ========================================================

    with output_json.open(
        'w',
        encoding='utf-8',
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Final verification
    # ========================================================

    if (
        len(
            output_data_list
        )
        != len(images)
    ):

        raise RuntimeError(
            'Image count mismatch'
        )

    if (
        num_instances
        != len(annotations)
    ):

        raise RuntimeError(
            'Annotation count mismatch'
        )

    # --------------------------------------------------------
    # Make sure polygon really disappeared.
    # --------------------------------------------------------

    for record in output_data_list:

        for instance in (
            record[
                'instances'
            ]
        ):

            forbidden_fields = (
                'poly',
                'segmentation',
                'bezier_pts',
                'rec',
            )

            for field in forbidden_fields:

                if field in instance:

                    raise RuntimeError(
                        'Forbidden field found '
                        f'in OCRDINO output: '
                        f'{field}'
                    )

    # ========================================================
    # Report
    # ========================================================

    print()
    print(
        'output:',
        output_json,
    )

    print()
    print(
        'basic statistics:'
    )

    print(
        'converted images:',
        len(
            output_data_list
        ),
    )

    print(
        'converted instances:',
        num_instances,
    )

    print(
        'ignored instances:',
        num_ignored,
    )

    print()
    print(
        'image size validation:'
    )

    print(
        'JSON/real size mismatches:',
        num_image_size_mismatches,
    )

    print(
        'width/height swapped:',
        num_swapped_image_sizes,
    )

    if image_size_mismatch_examples:

        print()
        print(
            'image size mismatch examples:'
        )

        for example in (
            image_size_mismatch_examples
        ):

            print(
                example
            )

    print()
    print(
        'bbox correction:'
    )

    print(
        'clipped bboxes:',
        num_clipped_bboxes,
    )

    print(
        'affected images:',
        len(
            clipped_image_ids
        ),
    )

    print(
        'max left overflow:',
        max_overflow[
            'left'
        ],
    )

    print(
        'max top overflow:',
        max_overflow[
            'top'
        ],
    )

    print(
        'max right overflow:',
        max_overflow[
            'right'
        ],
    )

    print(
        'max bottom overflow:',
        max_overflow[
            'bottom'
        ],
    )

    if clipping_examples:

        print()
        print(
            'first clipping examples:'
        )

        for example in (
            clipping_examples
        ):

            print(
                example
            )

    print()
    print(
        'recognition statistics:'
    )

    print(
        'sequences with EOS:',
        num_sequences_with_eos,
    )

    print(
        'full 25-char sequences '
        'without EOS:',
        num_full_length_without_eos,
    )

    print(
        'UNK token count:',
        num_unknown_tokens,
    )

    print(
        'recognition token IDs:',
        sorted(
            token_counter.keys()
        ),
    )

    print(
        'unique decoded characters:',
        len(
            char_counter
        ),
    )

    if text_length_counter:

        print(
            'minimum decoded length:',
            min(
                text_length_counter
            ),
        )

        print(
            'maximum decoded length:',
            max(
                text_length_counter
            ),
        )

    print()
    print(
        f'{split_name} conversion PASS'
    )

    return {
        'images': (
            len(
                output_data_list
            )
        ),

        'instances': (
            num_instances
        ),

        'image_size_mismatches': (
            num_image_size_mismatches
        ),

        'swapped_image_sizes': (
            num_swapped_image_sizes
        ),

        'clipped_bboxes': (
            num_clipped_bboxes
        ),

        'clipped_images': (
            len(
                clipped_image_ids
            )
        ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            'Convert Total-Text '
            'ESTextSpotter/SPTS JSON '
            'to OCRDINO v1.'
        )
    )

    parser.add_argument(
        '--data-root',
        default=(
            '/root/autodl-tmp/'
            'dataset/totaltext'
        ),
    )

    parser.add_argument(
        '--text-bbox-label',
        type=int,
        default=(
            DEFAULT_TEXT_BBOX_LABEL
        ),
    )

    return parser.parse_args()


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    data_root = Path(
        args.data_root
    )

    required_paths = [
        data_root / 'train.json',
        data_root / 'test.json',
        data_root / 'train_images',
        data_root / 'test_images',
    ]

    for path in required_paths:

        if not path.exists():

            raise FileNotFoundError(
                f'Missing required path: '
                f'{path}'
            )

    print(
        'OCRDINO Total-Text converter'
    )

    print(
        'data root:',
        data_root,
    )

    print(
        'polygon output: DISABLED'
    )

    print(
        'text bbox label:',
        args.text_bbox_label,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_stats = convert_split(
        data_root=data_root,

        input_json_name=(
            'train.json'
        ),

        image_dir_name=(
            'train_images'
        ),

        output_json_name=(
            'ocrdino_train.json'
        ),

        split_name=(
            'train'
        ),

        text_bbox_label=(
            args.text_bbox_label
        ),
    )

    # --------------------------------------------------------
    # Test
    # --------------------------------------------------------

    test_stats = convert_split(
        data_root=data_root,

        input_json_name=(
            'test.json'
        ),

        image_dir_name=(
            'test_images'
        ),

        output_json_name=(
            'ocrdino_test.json'
        ),

        split_name=(
            'test'
        ),

        text_bbox_label=(
            args.text_bbox_label
        ),
    )

    print()
    print(
        '=============================================='
    )

    print(
        'TOTAL-TEXT -> OCRDINO V1 CONVERSION PASSED'
    )

    print(
        '=============================================='
    )

    print()

    print(
        'train:',
        train_stats,
    )

    print(
        'test:',
        test_stats,
    )

    print()

    print(
        'train output:',
        data_root
        / 'ocrdino_train.json',
    )

    print(
        'test output:',
        data_root
        / 'ocrdino_test.json',
    )

    print()

    print(
        'POLYGON FIELDS: NOT STORED'
    )

    print(
        'BEZIER FIELDS: NOT STORED'
    )

    print(
        'SOURCE REC: decoded to text only'
    )


if __name__ == '__main__':

    main()