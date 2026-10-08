import json
from typing import List, Optional, Sequence


class OCRDinoCharTokenizer:
    """Character tokenizer for OCR-DINO."""

    def __init__(self, vocab_file: str):
        with open(
            vocab_file,
            'r',
            encoding='utf-8',
        ) as f:
            vocab = json.load(f)

        version = vocab.get('format_version')

        if version != 'ocrdino_char_vocab_v1':
            raise ValueError(
                f'Unsupported vocabulary format: {version}'
            )

        self.token_to_id = vocab['token_to_id']

        self.id_to_token = {
            int(token_id): token
            for token, token_id
            in self.token_to_id.items()
        }

        special = vocab['special_tokens']

        self.pad_token_id = int(
            special['pad_token_id']
        )

        self.bos_token_id = int(
            special['bos_token_id']
        )

        self.eos_token_id = int(
            special['eos_token_id']
        )

        self.unk_token_id = int(
            special['unk_token_id']
        )

        self.vocab_size = len(
            self.token_to_id
        )

    def __len__(self) -> int:
        return self.vocab_size

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = False,
    ) -> List[int]:

        if not isinstance(text, str):
            raise TypeError(
                'text must be a string.'
            )

        token_ids = []

        if add_bos:
            token_ids.append(
                self.bos_token_id
            )

        token_ids.extend(
            self.token_to_id.get(
                char,
                self.unk_token_id,
            )
            for char in text
        )

        if add_eos:
            token_ids.append(
                self.eos_token_id
            )

        if (
            max_length is not None
            and len(token_ids) > max_length
        ):
            if not truncation:
                raise ValueError(
                    f'Encoded sequence length '
                    f'{len(token_ids)} exceeds '
                    f'max_length={max_length}.'
                )

            token_ids = token_ids[
                :max_length
            ]

            if add_eos:
                token_ids[-1] = (
                    self.eos_token_id
                )

        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        skip_special_tokens: bool = True,
        stop_at_eos: bool = True,
    ) -> str:

        output = []

        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }

        for token_id in token_ids:
            token_id = int(token_id)

            if (
                stop_at_eos
                and token_id
                == self.eos_token_id
            ):
                break

            if (
                skip_special_tokens
                and token_id in special_ids
            ):
                continue

            token = self.id_to_token.get(
                token_id,
                self.id_to_token[
                    self.unk_token_id
                ],
            )

            output.append(token)

        return ''.join(output)