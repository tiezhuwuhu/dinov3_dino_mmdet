import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--input',
        required=True,
        help='Original DINO checkpoint.',
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Filtered checkpoint.',
    )

    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint = torch.load(
        args.input,
        map_location='cpu',
        weights_only=False,
    )

    state_dict = checkpoint['state_dict']

    remove_prefixes = (
        'bbox_head.cls_branches.',
        'dn_query_generator.label_embedding.',
    )

    removed_keys = []

    for key in list(state_dict.keys()):
        if key.startswith(remove_prefixes):
            removed_keys.append(key)
            del state_dict[key]

    torch.save(
        checkpoint,
        args.output,
    )

    print('=' * 70)
    print('DINO CHECKPOINT FILTER')
    print('=' * 70)

    print('Removed parameters:')
    for key in removed_keys:
        print(key)

    print()
    print('Removed count:', len(removed_keys))
    print('Remaining count:', len(state_dict))
    print('Output:', args.output)

    print('=' * 70)


if __name__ == '__main__':
    main()