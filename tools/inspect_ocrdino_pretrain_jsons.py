import gc
import json
import os
from collections import Counter
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None


ROOT = Path(
    '/root/autodl-tmp/dataset/ocrdino_pretrain'
)

DATASETS = [
    {
        'name': 'mlt2017',
        'json': ROOT / 'mlt2017' / 'train.json',
        'image_root': ROOT / 'mlt2017' / 'MLT_train_images',
    },
    {
        'name': 'syntext1',
        'json': ROOT / 'syntext1' / 'train.json',
        'image_root': ROOT / 'syntext1' / 'images',
    },
    {
        'name': 'syntext2',
        'json': ROOT / 'syntext2' / 'train.json',
        'image_root': ROOT / 'syntext2' / 'emcs_imgs',
    },
    {
        'name': 'syntext2_extra',
        'json': (
            ROOT
            / 'syntext2'
            / 'annotations'
            / 'ecms_v1_maxlen25.json'
        ),
        'image_root': ROOT / 'syntext2' / 'emcs_imgs',
    },
]


MAX_SAMPLES_TO_PRINT = 5
MAX_IMAGE_CHECK = 50
MAX_ANNOTATIONS_FOR_TOKEN_STATS = 200000


TEXT_FIELD_CANDIDATES = (
    'text',
    'transcription',
    'utf8_string',
    'word',
    'label',
)

REC_FIELD_CANDIDATES = (
    'rec',
    'rec_label',
    'rec_labels',
    'tokens',
    'token_ids',
)

POLYGON_FIELD_CANDIDATES = (
    'poly',
    'polygon',
    'segmentation',
    'bezier_pts',
    'bezier',
)

IGNORE_FIELD_CANDIDATES = (
    'ignore',
    'ignore_flag',
    'iscrowd',
    'illegibility',
)


def print_header(title):
    print()
    print('=' * 100)
    print(title)
    print('=' * 100)


def short_value(value, max_items=20):
    if isinstance(value, dict):
        keys = list(value.keys())

        if len(keys) > max_items:
            keys = keys[:max_items] + ['...']

        return {
            'type': 'dict',
            'keys': keys,
        }

    if isinstance(value, list):
        preview = value[:max_items]

        return {
            'type': 'list',
            'length': len(value),
            'preview': preview,
        }

    if isinstance(value, str):
        if len(value) > 200:
            return value[:200] + '...'

        return value

    return value


def print_object(obj, prefix=''):
    if isinstance(obj, dict):
        for key, value in obj.items():
            print(
                prefix
                + str(key)
                + ': '
                + str(
                    short_value(value)
                )
            )
    else:
        print(
            prefix
            + str(
                short_value(obj)
            )
        )


def decode_rec_guess(rec):
    if not isinstance(rec, list):
        return None

    if not rec:
        return ''

    if not all(
        isinstance(x, int)
        for x in rec
    ):
        return None

    chars = []

    for token in rec:
        if token == 96:
            break

        if 0 <= token <= 94:
            chars.append(
                chr(token + 32)
            )
        elif token == 95:
            chars.append('?')
        else:
            chars.append(
                f'<{token}>'
            )

    return ''.join(chars)


def find_existing_field(
    obj,
    candidates,
):
    if not isinstance(obj, dict):
        return None

    for key in candidates:
        if key in obj:
            return key

    return None


def inspect_generic_structure(data):
    print()
    print('TOP LEVEL TYPE:')
    print(type(data).__name__)

    if isinstance(data, dict):
        print()
        print('TOP LEVEL KEYS:')

        for key, value in data.items():
            if isinstance(value, list):
                print(
                    f'  {key}: list, length={len(value)}'
                )
            elif isinstance(value, dict):
                print(
                    f'  {key}: dict, keys={list(value.keys())[:20]}'
                )
            else:
                print(
                    f'  {key}: '
                    f'{type(value).__name__} '
                    f'{short_value(value)}'
                )

    elif isinstance(data, list):
        print(
            'TOP LEVEL LIST LENGTH:',
            len(data),
        )

        if data:
            print()
            print('FIRST ITEM:')
            print_object(data[0], '  ')


def inspect_categories(categories):
    print()
    print('CATEGORIES')
    print('-' * 100)

    if categories is None:
        print('categories: MISSING')
        return

    print(
        'count:',
        len(categories),
    )

    for category in categories[:20]:
        print(
            '  ',
            category,
        )


def inspect_images(
    images,
    image_root,
):
    print()
    print('IMAGES')
    print('-' * 100)

    print(
        'count:',
        len(images),
    )

    if not images:
        return {}

    key_counter = Counter()

    ids = []
    file_names = []

    for image in images:
        if not isinstance(image, dict):
            continue

        key_counter.update(
            image.keys()
        )

        if 'id' in image:
            ids.append(
                image['id']
            )

        if 'file_name' in image:
            file_names.append(
                image['file_name']
            )

    print()
    print('IMAGE FIELD FREQUENCY:')

    for key, count in (
        key_counter.most_common()
    ):
        print(
            f'  {key:30s} '
            f'{count}/{len(images)}'
        )

    print()
    print('IMAGE ID CHECK:')

    print(
        'ids:',
        len(ids),
    )

    print(
        'unique ids:',
        len(set(ids)),
    )

    print(
        'duplicate ids:',
        len(ids) - len(set(ids)),
    )

    print()
    print('FIRST IMAGE RECORDS:')

    for index, image in enumerate(
        images[:MAX_SAMPLES_TO_PRINT]
    ):
        print()
        print(
            f'IMAGE SAMPLE {index}'
        )

        print_object(
            image,
            '  ',
        )

    image_by_id = {}

    for image in images:
        if (
            isinstance(image, dict)
            and 'id' in image
        ):
            image_by_id[
                image['id']
            ] = image

    print()
    print('IMAGE FILE CHECK')
    print(
        'image root:',
        image_root,
    )

    if not image_root.exists():
        print(
            'WARNING: image root does not exist'
        )

        return image_by_id

    checked = 0
    exists_count = 0
    size_match = 0
    size_mismatch = 0

    mismatch_examples = []

    for image in images:
        if checked >= MAX_IMAGE_CHECK:
            break

        file_name = image.get(
            'file_name',
            None,
        )

        if not file_name:
            continue

        image_path = (
            image_root
            / file_name
        )

        checked += 1

        if not image_path.is_file():
            print(
                'MISSING:',
                image_path,
            )

            continue

        exists_count += 1

        if (
            Image is not None
            and 'width' in image
            and 'height' in image
        ):
            try:
                with Image.open(
                    image_path
                ) as pil_image:
                    real_width = (
                        pil_image.width
                    )

                    real_height = (
                        pil_image.height
                    )

                json_width = int(
                    image['width']
                )

                json_height = int(
                    image['height']
                )

                if (
                    real_width == json_width
                    and real_height
                    == json_height
                ):
                    size_match += 1
                else:
                    size_mismatch += 1

                    if (
                        len(
                            mismatch_examples
                        )
                        < 10
                    ):
                        mismatch_examples.append(
                            {
                                'file_name':
                                    file_name,
                                'json':
                                    (
                                        json_width,
                                        json_height,
                                    ),
                                'real':
                                    (
                                        real_width,
                                        real_height,
                                    ),
                            }
                        )

            except Exception as exc:
                print(
                    'IMAGE READ ERROR:',
                    image_path,
                    repr(exc),
                )

    print()
    print(
        'checked files:',
        checked,
    )

    print(
        'existing files:',
        exists_count,
    )

    if Image is not None:
        print(
            'size matches:',
            size_match,
        )

        print(
            'size mismatches:',
            size_mismatch,
        )

        if mismatch_examples:
            print(
                'size mismatch examples:'
            )

            for example in mismatch_examples:
                print(
                    '  ',
                    example,
                )

    return image_by_id


def check_bbox_formats(
    annotations,
    image_by_id,
):
    xywh_valid = 0
    xyxy_valid = 0
    both_valid = 0
    neither_valid = 0

    area_matches_xywh = 0
    area_matches_xyxy = 0
    area_checked = 0

    bbox_count = 0

    negative_value_count = 0
    zero_size_xywh = 0
    zero_size_xyxy = 0

    examples = []

    for ann in annotations:
        bbox = ann.get(
            'bbox',
            None,
        )

        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
        ):
            continue

        if not all(
            isinstance(x, (int, float))
            for x in bbox
        ):
            continue

        bbox_count += 1

        if len(examples) < 10:
            examples.append(
                {
                    'annotation_id':
                        ann.get('id'),
                    'image_id':
                        ann.get('image_id'),
                    'bbox':
                        bbox,
                    'area':
                        ann.get('area'),
                }
            )

        x0, y0, x2_or_w, y2_or_h = (
            map(
                float,
                bbox,
            )
        )

        if min(
            x0,
            y0,
            x2_or_w,
            y2_or_h,
        ) < 0:
            negative_value_count += 1

        image = image_by_id.get(
            ann.get('image_id')
        )

        if image is None:
            continue

        width = float(
            image.get(
                'width',
                0,
            )
        )

        height = float(
            image.get(
                'height',
                0,
            )
        )

        tol = 2.0

        xywh_ok = (
            x0 >= -tol
            and y0 >= -tol
            and x2_or_w > 0
            and y2_or_h > 0
            and x0 + x2_or_w
            <= width + tol
            and y0 + y2_or_h
            <= height + tol
        )

        xyxy_ok = (
            x0 >= -tol
            and y0 >= -tol
            and x2_or_w > x0
            and y2_or_h > y0
            and x2_or_w
            <= width + tol
            and y2_or_h
            <= height + tol
        )

        if xywh_ok:
            xywh_valid += 1

        if xyxy_ok:
            xyxy_valid += 1

        if xywh_ok and xyxy_ok:
            both_valid += 1
        elif not xywh_ok and not xyxy_ok:
            neither_valid += 1

        if (
            x2_or_w <= 0
            or y2_or_h <= 0
        ):
            zero_size_xywh += 1

        if (
            x2_or_w <= x0
            or y2_or_h <= y0
        ):
            zero_size_xyxy += 1

        area = ann.get(
            'area',
            None,
        )

        if isinstance(
            area,
            (int, float),
        ):
            area_checked += 1

            area = float(area)

            area_xywh = (
                x2_or_w
                * y2_or_h
            )

            area_xyxy = (
                max(
                    x2_or_w - x0,
                    0.0,
                )
                * max(
                    y2_or_h - y0,
                    0.0,
                )
            )

            tolerance = max(
                1.0,
                abs(area) * 0.01,
            )

            if (
                abs(
                    area - area_xywh
                )
                <= tolerance
            ):
                area_matches_xywh += 1

            if (
                abs(
                    area - area_xyxy
                )
                <= tolerance
            ):
                area_matches_xyxy += 1

    print()
    print('BBOX ANALYSIS')
    print('-' * 100)

    print(
        'bbox records:',
        bbox_count,
    )

    print(
        'negative bbox values:',
        negative_value_count,
    )

    print()
    print(
        'valid if interpreted as COCO xywh:',
        xywh_valid,
    )

    print(
        'valid if interpreted as xyxy:',
        xyxy_valid,
    )

    print(
        'valid under both interpretations:',
        both_valid,
    )

    print(
        'valid under neither interpretation:',
        neither_valid,
    )

    print()
    print(
        'invalid/non-positive xywh sizes:',
        zero_size_xywh,
    )

    print(
        'invalid/non-positive xyxy sizes:',
        zero_size_xyxy,
    )

    if area_checked:
        print()
        print(
            'annotations with area:',
            area_checked,
        )

        print(
            'area matches xywh interpretation:',
            area_matches_xywh,
        )

        print(
            'area matches xyxy interpretation:',
            area_matches_xyxy,
        )

    print()
    print('BBOX EXAMPLES:')

    for example in examples:
        print(
            '  ',
            example,
        )


def inspect_recognition(
    annotations,
):
    rec_field_counter = Counter()
    text_field_counter = Counter()

    rec_type_counter = Counter()
    rec_length_counter = Counter()

    token_counter = Counter()

    min_token = None
    max_token = None

    eos_96_count = 0
    unk_95_count = 0

    string_text_examples = []
    rec_examples = []

    checked = 0

    for ann in annotations:

        for field in TEXT_FIELD_CANDIDATES:
            if field in ann:
                text_field_counter[
                    field
                ] += 1

                value = ann[field]

                if (
                    isinstance(value, str)
                    and len(
                        string_text_examples
                    )
                    < 20
                ):
                    string_text_examples.append(
                        (
                            ann.get('id'),
                            field,
                            value,
                        )
                    )

        for field in REC_FIELD_CANDIDATES:
            if field not in ann:
                continue

            rec_field_counter[
                field
            ] += 1

            value = ann[field]

            rec_type_counter[
                type(value).__name__
            ] += 1

            if isinstance(value, list):
                rec_length_counter[
                    len(value)
                ] += 1

                if all(
                    isinstance(
                        token,
                        int,
                    )
                    for token in value
                ):
                    for token in value:
                        token_counter[
                            token
                        ] += 1

                        if min_token is None:
                            min_token = token
                            max_token = token
                        else:
                            min_token = min(
                                min_token,
                                token,
                            )

                            max_token = max(
                                max_token,
                                token,
                            )

                        if token == 96:
                            eos_96_count += 1

                        if token == 95:
                            unk_95_count += 1

            if len(rec_examples) < 20:
                rec_examples.append(
                    (
                        ann.get('id'),
                        field,
                        short_value(value, 30),
                        decode_rec_guess(value),
                    )
                )

        checked += 1

        if (
            checked
            >= MAX_ANNOTATIONS_FOR_TOKEN_STATS
        ):
            break

    print()
    print('RECOGNITION ANALYSIS')
    print('-' * 100)

    print(
        'annotation sample size:',
        checked,
    )

    print()
    print(
        'text field frequency:',
        dict(
            text_field_counter
        ),
    )

    print(
        'rec field frequency:',
        dict(
            rec_field_counter
        ),
    )

    print(
        'rec value types:',
        dict(
            rec_type_counter
        ),
    )

    if rec_length_counter:
        print()
        print(
            'rec sequence length distribution:'
        )

        for length, count in sorted(
            rec_length_counter.items()
        ):
            print(
                f'  length {length:4d}: '
                f'{count}'
            )

    if token_counter:
        print()
        print(
            'token min:',
            min_token,
        )

        print(
            'token max:',
            max_token,
        )

        print(
            'token 95 count:',
            unk_95_count,
        )

        print(
            'token 96 count:',
            eos_96_count,
        )

        print()
        print(
            'top 30 token ids:'
        )

        for token, count in (
            token_counter.most_common(
                30
            )
        ):
            if 0 <= token <= 94:
                symbol = repr(
                    chr(token + 32)
                )
            elif token == 95:
                symbol = 'UNK?'
            elif token == 96:
                symbol = 'EOS?'
            else:
                symbol = ''

            print(
                f'  {token:5d}: '
                f'{count:10d} '
                f'{symbol}'
            )

    print()
    print('TEXT EXAMPLES:')

    for example in (
        string_text_examples
    ):
        print(
            '  ',
            example,
        )

    print()
    print('REC EXAMPLES:')

    for ann_id, field, value, decoded in (
        rec_examples
    ):
        print(
            '  ann_id=',
            ann_id,
            'field=',
            field,
        )

        print(
            '    value=',
            value,
        )

        print(
            '    ASCII+32 decode guess=',
            repr(decoded),
        )


def inspect_polygon(
    annotations,
):
    field_counter = Counter()
    type_counter = Counter()
    length_counter = Counter()

    examples = []

    missing_all = 0

    for ann in annotations:
        found = False

        for field in (
            POLYGON_FIELD_CANDIDATES
        ):
            if field not in ann:
                continue

            found = True

            value = ann[field]

            field_counter[
                field
            ] += 1

            type_counter[
                (
                    field,
                    type(value).__name__,
                )
            ] += 1

            if isinstance(value, list):
                length_counter[
                    (
                        field,
                        len(value),
                    )
                ] += 1

            if len(examples) < 20:
                examples.append(
                    (
                        ann.get('id'),
                        field,
                        short_value(value, 30),
                    )
                )

        if not found:
            missing_all += 1

    print()
    print('POLYGON / SEGMENTATION ANALYSIS')
    print('-' * 100)

    print(
        'field frequency:',
        dict(field_counter),
    )

    print(
        'annotations without any '
        'polygon-like field:',
        missing_all,
    )

    print()
    print(
        'field types:'
    )

    for key, count in (
        type_counter.most_common()
    ):
        print(
            '  ',
            key,
            count,
        )

    print()
    print(
        'list length distribution:'
    )

    for key, count in sorted(
        length_counter.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
        ),
    ):
        print(
            '  ',
            key,
            count,
        )

    print()
    print('POLYGON EXAMPLES:')

    for example in examples:
        print(
            '  ',
            example,
        )


def inspect_ignore_fields(
    annotations,
):
    counters = {
        field: Counter()
        for field
        in IGNORE_FIELD_CANDIDATES
    }

    for ann in annotations:
        for field in (
            IGNORE_FIELD_CANDIDATES
        ):
            if field in ann:
                value = ann[field]

                try:
                    counters[field][
                        str(value)
                    ] += 1
                except Exception:
                    counters[field][
                        type(value).__name__
                    ] += 1

    print()
    print('IGNORE FIELD ANALYSIS')
    print('-' * 100)

    for field, counter in (
        counters.items()
    ):
        if not counter:
            continue

        print(
            field,
            dict(
                counter.most_common(
                    20
                )
            ),
        )


def inspect_annotations(
    annotations,
    image_by_id,
    categories,
):
    print()
    print('ANNOTATIONS')
    print('-' * 100)

    print(
        'count:',
        len(annotations),
    )

    if not annotations:
        return

    key_counter = Counter()

    annotation_ids = []
    image_ids = []
    category_ids = []

    for ann in annotations:
        if not isinstance(ann, dict):
            continue

        key_counter.update(
            ann.keys()
        )

        if 'id' in ann:
            annotation_ids.append(
                ann['id']
            )

        if 'image_id' in ann:
            image_ids.append(
                ann['image_id']
            )

        if 'category_id' in ann:
            category_ids.append(
                ann['category_id']
            )

    print()
    print(
        'ANNOTATION FIELD FREQUENCY:'
    )

    for key, count in (
        key_counter.most_common()
    ):
        print(
            f'  {key:30s} '
            f'{count}/{len(annotations)}'
        )

    print()
    print('ID CHECK:')

    print(
        'annotation ids:',
        len(annotation_ids),
    )

    print(
        'unique annotation ids:',
        len(
            set(annotation_ids)
        ),
    )

    print(
        'duplicate annotation ids:',
        len(annotation_ids)
        - len(
            set(annotation_ids)
        ),
    )

    if image_by_id:
        missing_image_refs = sum(
            image_id not in image_by_id
            for image_id in image_ids
        )

        print(
            'annotation image references:',
            len(image_ids),
        )

        print(
            'missing image references:',
            missing_image_refs,
        )

    if categories:
        category_set = {
            category.get('id')
            for category in categories
            if isinstance(
                category,
                dict,
            )
        }

        missing_category_refs = sum(
            category_id
            not in category_set
            for category_id
            in category_ids
        )

        print(
            'category ids used:',
            sorted(
                set(category_ids)
            )[:50],
        )

        print(
            'missing category references:',
            missing_category_refs,
        )

    print()
    print(
        'FIRST ANNOTATION RECORDS:'
    )

    for index, ann in enumerate(
        annotations[
            :MAX_SAMPLES_TO_PRINT
        ]
    ):
        print()
        print(
            f'ANNOTATION SAMPLE {index}'
        )

        print_object(
            ann,
            '  ',
        )

    check_bbox_formats(
        annotations,
        image_by_id,
    )

    inspect_recognition(
        annotations,
    )

    inspect_polygon(
        annotations,
    )

    inspect_ignore_fields(
        annotations,
    )


def inspect_coco_like(
    data,
    image_root,
):
    images = data.get(
        'images',
        [],
    )

    annotations = data.get(
        'annotations',
        [],
    )

    categories = data.get(
        'categories',
        [],
    )

    if not isinstance(
        images,
        list,
    ):
        raise TypeError(
            'images is not a list'
        )

    if not isinstance(
        annotations,
        list,
    ):
        raise TypeError(
            'annotations is not a list'
        )

    if not isinstance(
        categories,
        list,
    ):
        raise TypeError(
            'categories is not a list'
        )

    inspect_categories(
        categories
    )

    image_by_id = inspect_images(
        images,
        image_root,
    )

    inspect_annotations(
        annotations,
        image_by_id,
        categories,
    )


def inspect_file(
    name,
    json_path,
    image_root,
):
    print_header(
        f'DATASET: {name}'
    )

    print(
        'json:',
        json_path,
    )

    print(
        'image root:',
        image_root,
    )

    if not json_path.is_file():
        print(
            'ERROR: JSON file does not exist'
        )

        return

    size_mb = (
        json_path.stat().st_size
        / 1024
        / 1024
    )

    print(
        'json size MB:',
        f'{size_mb:.2f}',
    )

    print()
    print(
        'Loading JSON...'
    )

    with json_path.open(
        'r',
        encoding='utf-8',
    ) as handle:
        data = json.load(
            handle
        )

    print(
        'JSON loaded successfully.'
    )

    inspect_generic_structure(
        data
    )

    if (
        isinstance(data, dict)
        and isinstance(
            data.get('images'),
            list,
        )
        and isinstance(
            data.get('annotations'),
            list,
        )
    ):
        print()
        print(
            'FORMAT DETECTED: '
            'COCO-LIKE'
        )

        inspect_coco_like(
            data,
            image_root,
        )

    else:
        print()
        print(
            'FORMAT DETECTED: '
            'NON-COCO / UNKNOWN'
        )

        print()
        print(
            'Detailed first-level preview:'
        )

        print_object(
            data,
            '  ',
        )

    del data
    gc.collect()


def main():
    print_header(
        'OCRDINO PRETRAIN JSON AUDIT'
    )

    print(
        'root:',
        ROOT,
    )

    print(
        'Pillow available:',
        Image is not None,
    )

    for dataset in DATASETS:
        try:
            inspect_file(
                name=dataset['name'],
                json_path=dataset['json'],
                image_root=dataset[
                    'image_root'
                ],
            )

        except Exception as exc:
            print()
            print(
                'FAILED:',
                dataset['name'],
            )

            print(
                type(exc).__name__,
                str(exc),
            )

            print()

            raise

    print_header(
        'ALL JSON AUDITS COMPLETED'
    )


if __name__ == '__main__':
    main()