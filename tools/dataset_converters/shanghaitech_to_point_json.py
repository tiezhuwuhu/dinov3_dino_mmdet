import argparse
import json
import os
import os.path as osp
import re

import scipy.io as sio
from PIL import Image


def natural_key(name):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r'(\d+)', name)
    ]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert ShanghaiTech point annotations '
                    'to Point-DINO JSON format.')

    parser.add_argument(
        '--img-dir',
        required=True,
        help='ShanghaiTech images directory.')

    parser.add_argument(
        '--gt-dir',
        required=True,
        help='ShanghaiTech ground_truth directory.')

    parser.add_argument(
        '--out',
        required=True,
        help='Output JSON file.')

    return parser.parse_args()


def main():
    args = parse_args()

    image_files = [
        name for name in os.listdir(args.img_dir)
        if name.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    image_files.sort(key=natural_key)

    data_list = []

    for img_id, image_name in enumerate(image_files):

        img_path = osp.join(args.img_dir, image_name)

        stem = osp.splitext(image_name)[0]

        gt_path = osp.join(
            args.gt_dir,
            f'GT_{stem}.mat'
        )

        if not osp.isfile(gt_path):
            raise FileNotFoundError(
                f'Cannot find annotation: {gt_path}'
            )

        mat = sio.loadmat(gt_path)

        if 'image_info' not in mat:
            raise KeyError(
                f'image_info not found in {gt_path}'
            )

        # ShanghaiTech original point annotation: N x 2 (x, y)
        points = mat['image_info'][0, 0][0, 0][0]

        with Image.open(img_path) as img:
            width, height = img.size

        instances = []

        for point in points:

            x = float(point[0])
            y = float(point[1])

            instances.append(
                dict(
                    point=[x, y],
                    point_label=0,
                    ignore_flag=0
                )
            )

        data_list.append(
            dict(
                img_id=img_id,
                img_path=image_name,
                width=width,
                height=height,
                instances=instances
            )
        )

    output = dict(
        metainfo=dict(
            classes=['point']
        ),
        data_list=data_list
    )

    out_dir = osp.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.out, 'w') as f:
        json.dump(output, f)

    total_points = sum(
        len(item['instances'])
        for item in data_list
    )

    print('images:', len(data_list))
    print('points:', total_points)
    print('saved:', args.out)


if __name__ == '__main__':
    main()