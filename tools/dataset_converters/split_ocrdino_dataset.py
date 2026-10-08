import argparse
import copy
import json
import random


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        required=True,
    )

    parser.add_argument(
        '--train-output',
        required=True,
    )

    parser.add_argument(
        '--val-output',
        required=True,
    )

    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    with open(
        args.input,
        'r',
        encoding='utf-8',
    ) as f:
        data = json.load(f)

    if (
        data.get('metainfo', {})
        .get('format_version')
        != 'ocrdino_v1'
    ):
        raise ValueError(
            'Input must use OCRDINO v1 format.'
        )

    data_list = list(
        data['data_list']
    )

    rng = random.Random(
        args.seed
    )

    indices = list(
        range(len(data_list))
    )

    rng.shuffle(indices)

    num_val = max(
        1,
        round(
            len(indices)
            * args.val_ratio
        ),
    )

    val_indices = set(
        indices[:num_val]
    )

    train_list = []
    val_list = []

    for index, item in enumerate(
        data_list
    ):
        if index in val_indices:
            val_list.append(item)
        else:
            train_list.append(item)

    train_data = {
        'metainfo': copy.deepcopy(
            data['metainfo']
        ),
        'data_list': train_list,
    }

    val_data = {
        'metainfo': copy.deepcopy(
            data['metainfo']
        ),
        'data_list': val_list,
    }

    train_data['metainfo'][
        'split'
    ] = 'train'

    val_data['metainfo'][
        'split'
    ] = 'val'

    with open(
        args.train_output,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            train_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(
        args.val_output,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            val_data,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print('=' * 70)
    print('OCRDINO DATASET SPLIT')
    print('=' * 70)
    print('total:', len(data_list))
    print('train:', len(train_list))
    print('val:', len(val_list))
    print('seed:', args.seed)
    print('val ratio:', args.val_ratio)
    print()
    print('train output:')
    print(args.train_output)
    print()
    print('val output:')
    print(args.val_output)
    print('=' * 70)


if __name__ == '__main__':
    main()