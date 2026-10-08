import argparse
import json
import os
from collections import Counter


OCRDINO_CLASSES = (
    'text_block',
    'title',
    'equation_isolated',
    'header',
    'figure',
    'page_number',
    'abandon',
    'footer',
    'figure_caption',
    'table',
    'table_caption',
    'text_mask',
    'equation_caption',
    'reference',
    'table_footnote',
    'figure_footnote',
    'equation_semantic',
    'page_footnote',
    'list_group',
    'code_txt',
    'chart_mask',
    'table_mask',
    'unknown_mask',
    'equation_explanation',
    'organic_chemical_formula_mask',
    'need_mask',
    'code_txt_caption',
    'algorithm_mask',
)


def polygon_to_bbox(poly, width, height):
    if not isinstance(poly, (list, tuple)):
        return None

    if len(poly) < 4 or len(poly) % 2 != 0:
        return None

    xs = poly[0::2]
    ys = poly[1::2]

    x1 = max(0.0, min(xs))
    y1 = max(0.0, min(ys))
    x2 = min(float(width), max(xs))
    y2 = min(float(height), max(ys))

    if x2 <= x1 or y2 <= y1:
        return None

    return [
        float(x1),
        float(y1),
        float(x2),
        float(y2),
    ]


def get_text_target(det):
    text = det.get('text')
    latex = det.get('latex')
    html = det.get('html')

    if isinstance(text, str) and text:
        return text, 'text'

    if isinstance(latex, str) and latex:
        return latex, 'latex'

    if isinstance(html, str) and html:
        return html, 'html'

    return '', 'none'


def convert(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    class_to_label = {
        name: index
        for index, name in enumerate(OCRDINO_CLASSES)
    }

    output = {
        'metainfo': {
            'dataset_name': 'OmniDocBench',
            'format_version': 'ocrdino_v1',
            'classes': list(OCRDINO_CLASSES),
        },
        'data_list': [],
    }

    category_counter = Counter()
    text_type_counter = Counter()

    total_instances = 0
    skipped_instances = 0

    for page_idx, page in enumerate(raw_data):
        page_info = page['page_info']

        width = int(page_info['width'])
        height = int(page_info['height'])

        original_image_path = page_info['image_path']
        image_name = os.path.basename(original_image_path)

        page_attribute = page_info.get(
            'page_attribute',
            {},
        )

        language = page_attribute.get(
            'language',
            'unknown',
        )

        instances = []

        for det in page.get('layout_dets', []):
            category = det.get('category_type')

            if category not in class_to_label:
                skipped_instances += 1
                continue

            poly = det.get('poly')

            bbox = polygon_to_bbox(
                poly=poly,
                width=width,
                height=height,
            )

            if bbox is None:
                skipped_instances += 1
                continue

            text, text_type = get_text_target(det)

            instance = {
                'bbox': bbox,
                'bbox_label': class_to_label[category],
                'ignore_flag': int(
                    bool(det.get('ignore', False))
                ),
                'text': text,
                'text_type': text_type,
                'poly': [
                    float(value)
                    for value in poly
                ],
                'order': det.get('order', -1),
            }

            instances.append(instance)

            total_instances += 1
            category_counter[category] += 1
            text_type_counter[text_type] += 1

        item = {
            'img_path': os.path.join(
                'images',
                image_name,
            ).replace('\\', '/'),
            'height': height,
            'width': width,
            'language': language,
            'source': 'OmniDocBench',
            'instances': instances,
        }

        output['data_list'].append(item)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print('=' * 70)
    print('OCRDINO CONVERSION COMPLETE')
    print('=' * 70)

    print('Input:')
    print(input_path)

    print('\nOutput:')
    print(output_path)

    print('\nPages:')
    print(len(output['data_list']))

    print('\nClasses:')
    print(len(OCRDINO_CLASSES))

    print('\nInstances:')
    print(total_instances)

    print('\nSkipped instances:')
    print(skipped_instances)

    print('\nText types:')
    for name, count in text_type_counter.most_common():
        print(f'{name:20s} {count}')

    print('\nCategories:')
    for name, count in category_counter.most_common():
        print(f'{name:35s} {count}')

    print('=' * 70)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        required=True,
        help='Path to OmniDocBench.json',
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output OCRDINO JSON path',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
    )


if __name__ == '__main__':
    main()