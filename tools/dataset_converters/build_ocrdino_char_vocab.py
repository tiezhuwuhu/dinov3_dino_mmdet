import argparse
import json
from collections import Counter


SPECIAL_TOKENS = {
    'pad_token': '<PAD>',
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'unk_token': '<UNK>',
}


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--inputs',
        nargs='+',
        required=True,
        help='OCRDINO v1 annotation files.',
    )

    parser.add_argument(
        '--output',
        required=True,
        help='Output vocabulary JSON.',
    )

    parser.add_argument(
        '--text-types',
        nargs='+',
        default=['text', 'latex'],
        help='Text types used to build the vocabulary.',
    )

    parser.add_argument(
        '--min-freq',
        type=int,
        default=1,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    counter = Counter()
    total_strings = 0
    total_chars = 0

    allowed_types = set(args.text_types)

    for path in args.inputs:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        version = data.get(
            'metainfo',
            {},
        ).get(
            'format_version'
        )

        if version != 'ocrdino_v1':
            raise ValueError(
                f'Unsupported format in {path}: {version}'
            )

        for item in data['data_list']:
            for instance in item.get('instances', []):

                text_type = instance.get(
                    'text_type',
                    'none',
                )

                if text_type not in allowed_types:
                    continue

                text = instance.get('text', '')

                if not isinstance(text, str) or not text:
                    continue

                counter.update(text)

                total_strings += 1
                total_chars += len(text)

    characters = sorted(
        char
        for char, freq in counter.items()
        if freq >= args.min_freq
    )

    token_to_id = {
        SPECIAL_TOKENS['pad_token']: 0,
        SPECIAL_TOKENS['bos_token']: 1,
        SPECIAL_TOKENS['eos_token']: 2,
        SPECIAL_TOKENS['unk_token']: 3,
    }

    for char in characters:
        if char not in token_to_id:
            token_to_id[char] = len(token_to_id)

    vocab = {
        'format_version': 'ocrdino_char_vocab_v1',
        'tokenizer_type': 'character',
        'special_tokens': {
            'pad_token': SPECIAL_TOKENS['pad_token'],
            'pad_token_id': 0,
            'bos_token': SPECIAL_TOKENS['bos_token'],
            'bos_token_id': 1,
            'eos_token': SPECIAL_TOKENS['eos_token'],
            'eos_token_id': 2,
            'unk_token': SPECIAL_TOKENS['unk_token'],
            'unk_token_id': 3,
        },
        'vocab_size': len(token_to_id),
        'token_to_id': token_to_id,
    }

    with open(
        args.output,
        'w',
        encoding='utf-8',
    ) as f:
        json.dump(
            vocab,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print('=' * 70)
    print('OCRDINO CHARACTER VOCABULARY')
    print('=' * 70)

    print('Input files:', len(args.inputs))
    print('Text types:', sorted(allowed_types))
    print('Text strings:', total_strings)
    print('Characters:', total_chars)
    print('Unique characters:', len(characters))
    print('Vocabulary size:', len(token_to_id))

    print()
    print('Special tokens:')
    for key, value in vocab['special_tokens'].items():
        print(key, '=', value)

    print()
    print('Output:')
    print(args.output)

    print('=' * 70)


if __name__ == '__main__':
    main()