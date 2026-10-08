import json
from pathlib import Path
from typing import Iterable, List, Mapping, Sequence, Union


CHARSET_FORMAT_VERSION = 'text_recognition_charset_v1'


class TextRecognitionCharset:
    """Generic character-level vocabulary for text recognition.

    Character IDs occupy [0, num_characters - 1].

    Two special classes are appended automatically:

        UNK = num_characters
        EOS = num_characters + 1

    The charset is intentionally independent from recognition sequence
    length. The same charset can therefore be reused with different numbers
    of recognition queries.
    """

    def __init__(
        self,
        characters: Sequence[str],
        unk_token: str = '<UNK>',
        eos_token: str = '<EOS>',
    ) -> None:
        if not characters:
            raise ValueError('characters must not be empty.')

        normalized_characters: List[str] = []
        seen = set()

        for char in characters:
            if not isinstance(char, str):
                raise TypeError(
                    'Every character entry must be a string.'
                )

            if len(char) != 1:
                raise ValueError(
                    'Every character entry must contain exactly one '
                    f'Unicode code point, but got {repr(char)}.'
                )

            if char in seen:
                raise ValueError(
                    f'Duplicate character: {repr(char)}'
                )

            seen.add(char)
            normalized_characters.append(char)

        if not isinstance(unk_token, str) or not unk_token:
            raise ValueError(
                'unk_token must be a non-empty string.'
            )

        if not isinstance(eos_token, str) or not eos_token:
            raise ValueError(
                'eos_token must be a non-empty string.'
            )

        if unk_token == eos_token:
            raise ValueError(
                'unk_token and eos_token must be different.'
            )

        if unk_token in seen:
            raise ValueError(
                'unk_token conflicts with a real character.'
            )

        if eos_token in seen:
            raise ValueError(
                'eos_token conflicts with a real character.'
            )

        self.characters = tuple(
            normalized_characters
        )

        self.unk_token = unk_token
        self.eos_token = eos_token

        self.char_to_id = {
            char: index
            for index, char in enumerate(
                self.characters
            )
        }

        self.id_to_char = {
            index: char
            for index, char in enumerate(
                self.characters
            )
        }

        self.num_characters = len(
            self.characters
        )

        self.unk_id = self.num_characters
        self.eos_id = self.num_characters + 1

        self.num_classes = self.num_characters + 2

    @classmethod
    def english_ascii(
        cls,
    ) -> 'TextRecognitionCharset':
        """Build the printable-ASCII vocabulary used by ESTextSpotter."""

        characters = [
            chr(code)
            for code in range(32, 127)
        ]

        return cls(
            characters=characters,
        )

    @classmethod
    def from_file(
        cls,
        path: Union[str, Path],
    ) -> 'TextRecognitionCharset':
        """Load a charset from a UTF-8 JSON file."""

        path = Path(path)

        with path.open(
            'r',
            encoding='utf-8',
        ) as file:
            data = json.load(file)

        format_version = data.get(
            'format_version'
        )

        if (
            format_version
            != CHARSET_FORMAT_VERSION
        ):
            raise ValueError(
                'Unsupported charset format version: '
                f'{repr(format_version)}'
            )

        characters = data.get(
            'characters'
        )

        if not isinstance(characters, list):
            raise ValueError(
                'Charset JSON must contain a list named '
                '"characters".'
            )

        return cls(
            characters=characters,
            unk_token=data.get(
                'unk_token',
                '<UNK>',
            ),
            eos_token=data.get(
                'eos_token',
                '<EOS>',
            ),
        )

    def to_dict(self) -> dict:
        """Serialize charset metadata."""

        return {
            'format_version': (
                CHARSET_FORMAT_VERSION
            ),
            'characters': list(
                self.characters
            ),
            'num_characters': (
                self.num_characters
            ),
            'unk_token': self.unk_token,
            'unk_id': self.unk_id,
            'eos_token': self.eos_token,
            'eos_id': self.eos_id,
            'num_classes': self.num_classes,
        }

    def save(
        self,
        path: Union[str, Path],
    ) -> None:
        """Save the charset as UTF-8 JSON."""

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with path.open(
            'w',
            encoding='utf-8',
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                ensure_ascii=False,
                indent=2,
            )

    def encode(
        self,
        text: str,
        max_length: int,
    ) -> List[int]:
        """Encode text into a fixed-length ordered target.

        Text shorter than max_length is filled with EOS.
        Text longer than max_length is truncated. The caller is responsible
        for deciding whether truncation should be allowed.
        """

        if not isinstance(text, str):
            raise TypeError(
                'text must be a string.'
            )

        if max_length <= 0:
            raise ValueError(
                'max_length must be positive.'
            )

        token_ids = [
            self.char_to_id.get(
                char,
                self.unk_id,
            )
            for char in text[:max_length]
        ]

        remaining = (
            max_length - len(token_ids)
        )

        if remaining > 0:
            token_ids.extend(
                [self.eos_id] * remaining
            )

        return token_ids

    def decode(
        self,
        token_ids: Union[
            Sequence[int],
            Iterable[int],
        ],
        stop_at_eos: bool = False,
        unk_replacement: str = '',
    ) -> str:
        """Decode recognition IDs into a string."""

        output: List[str] = []

        for token_id in token_ids:
            token_id = int(token_id)

            if token_id == self.eos_id:
                if stop_at_eos:
                    break

                continue

            if token_id == self.unk_id:
                output.append(
                    unk_replacement
                )
                continue

            if (
                0
                <= token_id
                < self.num_characters
            ):
                output.append(
                    self.id_to_char[
                        token_id
                    ]
                )

        return ''.join(output)

    def contains(
        self,
        char: str,
    ) -> bool:
        """Return whether a character exists in the vocabulary."""

        return char in self.char_to_id

    def unknown_characters(
        self,
        text: str,
    ) -> List[str]:
        """Return unique out-of-vocabulary characters in text."""

        unknown = []
        seen = set()

        for char in text:
            if (
                char not in self.char_to_id
                and char not in seen
            ):
                seen.add(char)
                unknown.append(char)

        return unknown

    def validate_ids(
        self,
        token_ids: Sequence[int],
        expected_length: int = None,
    ) -> None:
        """Validate recognition target IDs."""

        if expected_length is not None:
            if (
                len(token_ids)
                != expected_length
            ):
                raise ValueError(
                    'Recognition target must contain '
                    f'exactly {expected_length} IDs, '
                    f'but got {len(token_ids)}.'
                )

        for token_id in token_ids:
            token_id = int(token_id)

            if not (
                0
                <= token_id
                < self.num_classes
            ):
                raise ValueError(
                    'Recognition ID must be in '
                    f'[0, {self.num_classes - 1}], '
                    f'but got {token_id}.'
                )


def build_text_recognition_charset(
    cfg: Union[
        TextRecognitionCharset,
        Mapping,
    ],
) -> TextRecognitionCharset:
    """Build a recognition charset from configuration.

    Supported configurations:

    English ESTextSpotter-compatible vocabulary:

        dict(type='english_ascii')

    Vocabulary stored in JSON:

        dict(
            type='file',
            path='path/to/charset.json',
        )

    Inline vocabulary:

        dict(
            type='characters',
            characters=[...],
        )
    """

    if isinstance(
        cfg,
        TextRecognitionCharset,
    ):
        return cfg

    if not isinstance(cfg, Mapping):
        raise TypeError(
            'charset configuration must be a mapping or '
            'TextRecognitionCharset instance.'
        )

    cfg = dict(cfg)

    charset_type = cfg.pop(
        'type',
        None,
    )

    if charset_type == 'english_ascii':
        if cfg:
            raise ValueError(
                'english_ascii does not accept additional '
                f'arguments: {sorted(cfg.keys())}'
            )

        return (
            TextRecognitionCharset
            .english_ascii()
        )

    if charset_type == 'file':
        path = cfg.pop(
            'path',
            None,
        )

        if path is None:
            raise ValueError(
                'file charset requires "path".'
            )

        if cfg:
            raise ValueError(
                'Unexpected file charset arguments: '
                f'{sorted(cfg.keys())}'
            )

        return (
            TextRecognitionCharset
            .from_file(path)
        )

    if charset_type == 'characters':
        characters = cfg.pop(
            'characters',
            None,
        )

        if characters is None:
            raise ValueError(
                'characters charset requires '
                '"characters".'
            )

        return TextRecognitionCharset(
            characters=characters,
            **cfg,
        )

    raise ValueError(
        'Unsupported charset type: '
        f'{repr(charset_type)}'
    )