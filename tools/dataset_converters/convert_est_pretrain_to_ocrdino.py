from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path


ROOT = Path(
    '/root/autodl-tmp/dataset/ocrdino_pretrain'
)

UNK_TOKEN_ID = 95
EOS_TOKEN_ID = 96
UNK_SENTINEL = '\ufffd'

MAX_REC_LENGTH = 25
MAX_ALLOWED_OVERFLOW = 1.0


DATASETS = {
    'mlt2017': {
        'ann_file': (
            ROOT
            / 'mlt2017'
            / 'train.json'
        ),
        'image_root': (
            ROOT
            / 'mlt2017'
            / 'MLT_train_images'
        ),
        'relative_image_root': (
            'MLT_train_images'
        ),
        'output': (
            ROOT
            / 'mlt2017'
            / 'ocrdino_train.json'
        ),
        'stats': (
            ROOT
            / 'mlt2017'
            / 'ocrdino_train_stats.json'
        ),
        'language': 'multilingual',
        'expected_images': 9999,
        'expected_annotations': 89418,
    },

    'syntext1': {
        'ann_file': (
            ROOT
            / 'syntext1'
            / 'train.json'
        ),
        'image_root': (
            ROOT
            / 'syntext1'
            / 'images'
            / 'syntext_word_eng'
        ),
        'relative_image_root': (
            'images/syntext_word_eng'
        ),
        'output': (
            ROOT
            / 'syntext1'
            / 'ocrdino_train.json'
        ),
        'stats': (
            ROOT
            / 'syntext1'
            / 'ocrdino_train_stats.json'
        ),
        'language': 'en',
        'expected_images': 94723,
        'expected_annotations': 1318214,
    },

    'syntext2': {
        'ann_file': (
            ROOT
            / 'syntext2'
            / 'train.json'
        ),
        'image_root': (
            ROOT
            / 'syntext2'
            / 'emcs_imgs'
        ),
        'relative_image_root': (
            'emcs_imgs'
        ),
        'output': (
            ROOT
            / 'syntext2'
            / 'ocrdino_train.json'
        ),
        'stats': (
            ROOT
            / 'syntext2'
            / 'ocrdino_train_stats.json'
        ),
        'language': 'en',
        'expected_images': 54327,
        'expected_annotations': 479834,
    },
}


def log(*args):
    print(
        *args,
        flush=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        choices=[
            'all',
            'mlt2017',
            'syntext1',
            'syntext2',
        ],
        required=True,
    )

    parser.add_argument(
        '--overwrite',
        action='store_true',
    )

    return parser.parse_args()


def load_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(
            f'File not found: {path}'
        )

    size_mb = (
        path.stat().st_size
        / 1024
        / 1024
    )

    log(
        f'Loading JSON: {path}'
    )

    log(
        f'JSON size: {size_mb:.2f} MB'
    )

    with path.open(
        'r',
        encoding='utf-8',
    ) as f:
        data = json.load(f)

    log(
        'JSON loaded.'
    )

    return data


def validate_source(
    name: str,
    data: dict,
    cfg: dict,
):
    log(
        f'{name}: validating source structure...'
    )

    if not isinstance(
        data,
        dict,
    ):
        raise TypeError(
            'Top-level JSON must be dict.'
        )

    for key in (
        'images',
        'annotations',
        'categories',
    ):
        if key not in data:
            raise KeyError(
                f'Missing top-level key: {key}'
            )

    images = data['images']
    annotations = data['annotations']

    if (
        len(images)
        != cfg['expected_images']
    ):
        raise RuntimeError(
            f'{name}: image count mismatch: '
            f'{len(images)} != '
            f'{cfg["expected_images"]}'
        )

    if (
        len(annotations)
        != cfg[
            'expected_annotations'
        ]
    ):
        raise RuntimeError(
            f'{name}: annotation count mismatch: '
            f'{len(annotations)} != '
            f'{cfg["expected_annotations"]}'
        )

    image_ids = [
        image['id']
        for image in images
    ]

    if (
        len(image_ids)
        != len(set(image_ids))
    ):
        raise RuntimeError(
            f'{name}: duplicate image IDs'
        )

    annotation_ids = [
        ann['id']
        for ann in annotations
    ]

    if (
        len(annotation_ids)
        != len(set(annotation_ids))
    ):
        raise RuntimeError(
            f'{name}: duplicate annotation IDs'
        )

    log(
        f'{name}: source validation PASS'
    )


def decode_rec(
    name: str,
    ann_id: int,
    rec: list,
):
    if not isinstance(
        rec,
        list,
    ):
        raise TypeError(
            f'{name}: ann {ann_id}: '
            'rec is not list'
        )

    if len(rec) != MAX_REC_LENGTH:
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            f'rec length={len(rec)}'
        )

    for token in rec:
        if not isinstance(
            token,
            int,
        ):
            raise TypeError(
                f'{name}: ann {ann_id}: '
                'non-integer rec token'
            )

        if token < 0 or token > 96:
            raise RuntimeError(
                f'{name}: ann {ann_id}: '
                f'invalid token {token}'
            )

    first_eos = None

    for i, token in enumerate(rec):
        if token == EOS_TOKEN_ID:
            first_eos = i
            break

    if first_eos is None:
        active = rec
        no_eos = True

    else:
        active = rec[
            :first_eos
        ]

        no_eos = False

        for token in rec[
            first_eos:
        ]:
            if token != EOS_TOKEN_ID:
                raise RuntimeError(
                    f'{name}: ann {ann_id}: '
                    'non-EOS token exists '
                    'after EOS'
                )

    chars = []
    unk_count = 0

    for token in active:
        if 0 <= token <= 94:
            chars.append(
                chr(token + 32)
            )

        elif token == UNK_TOKEN_ID:
            chars.append(
                UNK_SENTINEL
            )

            unk_count += 1

        else:
            raise RuntimeError(
                f'{name}: ann {ann_id}: '
                f'unexpected active token '
                f'{token}'
            )

    text = ''.join(chars)

    return (
        text,
        unk_count,
        no_eos,
    )


def convert_bbox(
    name: str,
    ann: dict,
    image: dict,
    stats: dict,
):
    ann_id = ann[
        'id'
    ]

    bbox = ann.get(
        'bbox'
    )

    if not (
        isinstance(bbox, list)
        and len(bbox) == 4
    ):
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            f'invalid bbox {bbox}'
        )

    x, y, w, h = [
        float(v)
        for v in bbox
    ]

    if w <= 0 or h <= 0:
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            f'invalid bbox size {bbox}'
        )

    width = float(
        image['width']
    )

    height = float(
        image['height']
    )

    area = float(
        ann['area']
    )

    expected_area = (
        w * h
    )

    if abs(
        area - expected_area
    ) > max(
        1e-4,
        expected_area * 1e-6,
    ):
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            f'area mismatch: '
            f'{area} vs {expected_area}'
        )

    raw_x1 = x
    raw_y1 = y

    raw_x2 = (
        x + w - 1.0
    )

    raw_y2 = (
        y + h - 1.0
    )

    max_x = (
        width - 1.0
    )

    max_y = (
        height - 1.0
    )

    over_left = max(
        0.0,
        -raw_x1,
    )

    over_top = max(
        0.0,
        -raw_y1,
    )

    over_right = max(
        0.0,
        raw_x2 - max_x,
    )

    over_bottom = max(
        0.0,
        raw_y2 - max_y,
    )

    max_overflow = max(
        over_left,
        over_top,
        over_right,
        over_bottom,
    )

    if (
        max_overflow
        > MAX_ALLOWED_OVERFLOW
    ):
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            'bbox overflow too large: '
            f'{bbox}, '
            f'image=({width},{height}), '
            f'overflow={max_overflow}'
        )

    if (
        raw_x2 < 0
        or raw_y2 < 0
        or raw_x1 > max_x
        or raw_y1 > max_y
    ):
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            'bbox completely outside image'
        )

    clipped = (
        max_overflow > 0
    )

    x1 = min(
        max(raw_x1, 0.0),
        max_x,
    )

    y1 = min(
        max(raw_y1, 0.0),
        max_y,
    )

    x2 = min(
        max(raw_x2, 0.0),
        max_x,
    )

    y2 = min(
        max(raw_y2, 0.0),
        max_y,
    )

    if (
        x2 < x1
        or y2 < y1
    ):
        raise RuntimeError(
            f'{name}: ann {ann_id}: '
            'invalid bbox after clip'
        )

    if clipped:
        stats[
            'clipped_bboxes'
        ] += 1

    return [
        x1,
        y1,
        x2,
        y2,
    ]


def build_indices(
    name: str,
    data: dict,
):
    log(
        f'{name}: building image index...'
    )

    image_by_id = {}

    total_images = len(
        data['images']
    )

    for i, image in enumerate(
        data['images'],
        start=1,
    ):
        image_id = image[
            'id'
        ]

        image_by_id[
            image_id
        ] = image

        if (
            i % 10000 == 0
            or i == total_images
        ):
            log(
                f'  images: '
                f'{i}/{total_images}'
            )

    log(
        f'{name}: building '
        'annotation index...'
    )

    annotations_by_image = (
        defaultdict(list)
    )

    total_annotations = len(
        data['annotations']
    )

    for i, ann in enumerate(
        data['annotations'],
        start=1,
    ):
        image_id = ann[
            'image_id'
        ]

        if (
            image_id
            not in image_by_id
        ):
            raise RuntimeError(
                f'{name}: ann '
                f'{ann["id"]}: '
                f'unknown image_id '
                f'{image_id}'
            )

        if (
            ann.get(
                'category_id'
            )
            != 1
        ):
            raise RuntimeError(
                f'{name}: ann '
                f'{ann["id"]}: '
                'category_id != 1'
            )

        if (
            ann.get(
                'iscrowd'
            )
            != 0
        ):
            raise RuntimeError(
                f'{name}: ann '
                f'{ann["id"]}: '
                'iscrowd != 0'
            )

        annotations_by_image[
            image_id
        ].append(
            ann
        )

        if (
            i % 100000 == 0
            or i == total_annotations
        ):
            log(
                f'  annotations: '
                f'{i}/'
                f'{total_annotations}'
            )

    log(
        f'{name}: index PASS'
    )

    return (
        image_by_id,
        annotations_by_image,
    )


def make_instance(
    name: str,
    ann: dict,
    image: dict,
    stats: dict,
):
    bbox = convert_bbox(
        name=name,
        ann=ann,
        image=image,
        stats=stats,
    )

    (
        text,
        unk_count,
        no_eos,
    ) = decode_rec(
        name=name,
        ann_id=ann['id'],
        rec=ann['rec'],
    )

    stats[
        'instances'
    ] += 1

    stats[
        'characters'
    ] += len(text)

    stats[
        'unk_tokens'
    ] += unk_count

    if unk_count > 0:
        stats[
            'instances_with_unk'
        ] += 1

    if (
        len(text) > 0
        and unk_count == len(text)
    ):
        stats[
            'instances_all_unk'
        ] += 1

    if no_eos:
        stats[
            'instances_without_eos'
        ] += 1

    stats[
        'max_text_length'
    ] = max(
        stats[
            'max_text_length'
        ],
        len(text),
    )

    if text:
        text_type = 'text'
    else:
        text_type = 'none'

    return {
        'bbox': bbox,
        'bbox_label': 0,
        'ignore_flag': 0,
        'text': text,
        'text_type': text_type,
        'order': 0,
    }


def make_record(
    name: str,
    cfg: dict,
    image: dict,
    annotations: list,
    stats: dict,
):
    for key in (
        'id',
        'file_name',
        'width',
        'height',
    ):
        if key not in image:
            raise RuntimeError(
                f'{name}: image missing '
                f'field {key}'
            )

    file_name = image[
        'file_name'
    ]

    real_path = (
        cfg[
            'image_root'
        ]
        / file_name
    )

    if not real_path.is_file():
        raise FileNotFoundError(
            f'{name}: missing image: '
            f'{real_path}'
        )

    relative_path = str(
        Path(
            cfg[
                'relative_image_root'
            ]
        )
        / file_name
    ).replace(
        os.sep,
        '/',
    )

    instances = []

    for ann in annotations:
        instances.append(
            make_instance(
                name=name,
                ann=ann,
                image=image,
                stats=stats,
            )
        )

    stats[
        'images'
    ] += 1

    if not instances:
        stats[
            'images_without_instances'
        ] += 1

    return {
        'img_id': image[
            'id'
        ],
        'img_path': relative_path,
        'width': int(
            image[
                'width'
            ]
        ),
        'height': int(
            image[
                'height'
            ]
        ),
        'language': cfg[
            'language'
        ],
        'source': name,
        'instances': instances,
    }


def write_output(
    name: str,
    cfg: dict,
    data: dict,
    annotations_by_image: dict,
    stats: dict,
):
    output_path = cfg[
        'output'
    ]

    tmp_path = Path(
        str(output_path)
        + '.tmp'
    )

    if tmp_path.exists():
        tmp_path.unlink()

    metainfo = {
        'format_version': (
            'ocrdino_v1'
        ),
        'classes': [
            'text'
        ],
        'dataset_name': name,
        'source_bbox_format': (
            'xywh'
        ),
        'target_bbox_format': (
            'xyxy'
        ),
        'source_rec_length': 25,
        'source_unk_token_id': 95,
        'source_eos_token_id': 96,
        'unk_text_sentinel': (
            UNK_SENTINEL
        ),
    }

    total_images = len(
        data['images']
    )

    log(
        f'{name}: writing output...'
    )

    with tmp_path.open(
        'w',
        encoding='utf-8',
    ) as f:

        f.write(
            '{"metainfo":'
        )

        json.dump(
            metainfo,
            f,
            ensure_ascii=True,
            separators=(
                ',',
                ':',
            ),
        )

        f.write(
            ',"data_list":['
        )

        first = True

        for i, image in enumerate(
            data['images'],
            start=1,
        ):
            record = make_record(
                name=name,
                cfg=cfg,
                image=image,
                annotations=(
                    annotations_by_image.get(
                        image['id'],
                        [],
                    )
                ),
                stats=stats,
            )

            if not first:
                f.write(',')

            json.dump(
                record,
                f,
                ensure_ascii=True,
                separators=(
                    ',',
                    ':',
                ),
            )

            first = False

            if (
                i % 5000 == 0
                or i == total_images
            ):
                log(
                    f'  written images: '
                    f'{i}/{total_images}'
                )

        f.write(
            ']}'
        )

    log(
        f'{name}: temporary JSON '
        'written.'
    )

    return tmp_path


def verify_output(
    name: str,
    cfg: dict,
    tmp_path: Path,
):
    log(
        f'{name}: verifying '
        'generated JSON...'
    )

    with tmp_path.open(
        'r',
        encoding='utf-8',
    ) as f:
        generated = json.load(f)

    if (
        generated[
            'metainfo'
        ][
            'format_version'
        ]
        != 'ocrdino_v1'
    ):
        raise RuntimeError(
            'Bad format_version'
        )

    data_list = generated[
        'data_list'
    ]

    if (
        len(data_list)
        != cfg[
            'expected_images'
        ]
    ):
        raise RuntimeError(
            f'Output image count '
            f'mismatch: '
            f'{len(data_list)}'
        )

    instance_count = sum(
        len(
            item[
                'instances'
            ]
        )
        for item in data_list
    )

    if (
        instance_count
        != cfg[
            'expected_annotations'
        ]
    ):
        raise RuntimeError(
            f'Output instance count '
            f'mismatch: '
            f'{instance_count}'
        )

    log(
        f'{name}: generated JSON '
        'validation PASS'
    )


def convert_dataset(
    name: str,
    overwrite: bool,
):
    cfg = DATASETS[
        name
    ]

    log()
    log(
        '=' * 80
    )

    log(
        f'START DATASET: {name}'
    )

    log(
        '=' * 80
    )

    log(
        'Source:',
        cfg[
            'ann_file'
        ],
    )

    log(
        'Image root:',
        cfg[
            'image_root'
        ],
    )

    log(
        'Output:',
        cfg[
            'output'
        ],
    )

    if (
        cfg['output'].exists()
        and not overwrite
    ):
        raise FileExistsError(
            f'Output exists: '
            f'{cfg["output"]}\n'
            'Use --overwrite.'
        )

    data = load_json(
        cfg[
            'ann_file'
        ]
    )

    validate_source(
        name=name,
        data=data,
        cfg=cfg,
    )

    (
        image_by_id,
        annotations_by_image,
    ) = build_indices(
        name=name,
        data=data,
    )

    stats = {
        'dataset': name,
        'images': 0,
        'instances': 0,
        'images_without_instances': 0,
        'characters': 0,
        'unk_tokens': 0,
        'instances_with_unk': 0,
        'instances_all_unk': 0,
        'instances_without_eos': 0,
        'max_text_length': 0,
        'clipped_bboxes': 0,
    }

    tmp_path = write_output(
        name=name,
        cfg=cfg,
        data=data,
        annotations_by_image=(
            annotations_by_image
        ),
        stats=stats,
    )

    verify_output(
        name=name,
        cfg=cfg,
        tmp_path=tmp_path,
    )

    output_path = cfg[
        'output'
    ]

    if output_path.exists():
        output_path.unlink()

    tmp_path.replace(
        output_path
    )

    with cfg[
        'stats'
    ].open(
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=True,
        )

    log()
    log(
        'CONVERSION PASS'
    )

    log(
        'images:',
        stats[
            'images'
        ],
    )

    log(
        'instances:',
        stats[
            'instances'
        ],
    )

    log(
        'clipped_bboxes:',
        stats[
            'clipped_bboxes'
        ],
    )

    log(
        'instances_with_unk:',
        stats[
            'instances_with_unk'
        ],
    )

    log(
        'instances_all_unk:',
        stats[
            'instances_all_unk'
        ],
    )

    log(
        'unk_tokens:',
        stats[
            'unk_tokens'
        ],
    )

    log(
        'instances_without_eos:',
        stats[
            'instances_without_eos'
        ],
    )

    log(
        'max_text_length:',
        stats[
            'max_text_length'
        ],
    )

    log(
        'Final output:',
        output_path,
    )


def main():
    log(
        'convert_est_pretrain_to_ocrdino.py START'
    )

    log(
        'Python:',
        sys.version,
    )

    args = parse_args()

    log(
        'Requested dataset:',
        args.dataset,
    )

    if args.dataset == 'all':
        datasets = [
            'mlt2017',
            'syntext1',
            'syntext2',
        ]
    else:
        datasets = [
            args.dataset
        ]

    for name in datasets:
        convert_dataset(
            name=name,
            overwrite=args.overwrite,
        )

    log()
    log(
        '=' * 80
    )

    log(
        'ALL CONVERSIONS PASSED'
    )

    log(
        '=' * 80
    )


if __name__ == '__main__':
    try:
        main()

    except Exception:
        log()
        log(
            '=' * 80
        )

        log(
            'CONVERSION FAILED'
        )

        log(
            '=' * 80
        )

        traceback.print_exc()

        sys.exit(1)