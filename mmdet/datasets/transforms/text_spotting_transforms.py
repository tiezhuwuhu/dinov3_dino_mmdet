from typing import Optional

import numpy as np
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS
from mmdet.utils.text_recognition_charset import (
    build_text_recognition_charset,
)

from .formatting import PackDetInputs
from .loading import LoadAnnotations


@TRANSFORMS.register_module()
class LoadTextSpottingAnnotations(
    LoadAnnotations
):
    """Load detection and transcription annotations.

    This transform extends MMDetection's LoadAnnotations while preserving
    one transcription string for every instance.

    Required instance fields:

        bbox
        bbox_label
        ignore_flag
        text

    Added result field:

        gt_texts: list[str | None]
    """

    def __init__(
        self,
        text_key: str = 'text',
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.text_key = text_key

    def transform(
        self,
        results: dict,
    ) -> dict:
        results = super().transform(
            results
        )

        instances = results.get(
            'instances',
            [],
        )

        gt_texts = []

        for instance in instances:
            gt_texts.append(
                instance.get(
                    self.text_key,
                    None,
                )
            )

        results['gt_texts'] = (
            gt_texts
        )

        return results


@TRANSFORMS.register_module()
class EncodeRecognitionText(
    BaseTransform
):
    """Encode ordered transcription targets.

    The transform converts every instance transcription into a fixed-length
    sequence of character IDs.

    Instance order is preserved exactly. No character-level matching is
    performed.

    Args:
        charset_cfg: Configuration used to construct the character vocabulary.
        max_length: Number of recognition queries per text instance.
        empty_text_policy: Action for empty transcription. Supported values
            are "ignore" and "error".
        overlength_policy: Action when transcription is longer than
            max_length. Supported values are "ignore", "truncate", and
            "error".
        unknown_policy: Action when transcription contains characters that
            are not present in the vocabulary. Supported values are
            "encode", "ignore", and "error".
    """

    _EMPTY_POLICIES = {
        'ignore',
        'error',
    }

    _OVERLENGTH_POLICIES = {
        'ignore',
        'truncate',
        'error',
    }

    _UNKNOWN_POLICIES = {
        'encode',
        'ignore',
        'error',
    }

    def __init__(
        self,
        charset_cfg: dict,
        max_length: int = 25,
        empty_text_policy: str = 'ignore',
        overlength_policy: str = 'ignore',
        unknown_policy: str = 'encode',
    ) -> None:
        if max_length <= 0:
            raise ValueError(
                'max_length must be positive.'
            )

        if (
            empty_text_policy
            not in self._EMPTY_POLICIES
        ):
            raise ValueError(
                'Unsupported empty_text_policy: '
                f'{empty_text_policy}'
            )

        if (
            overlength_policy
            not in self._OVERLENGTH_POLICIES
        ):
            raise ValueError(
                'Unsupported overlength_policy: '
                f'{overlength_policy}'
            )

        if (
            unknown_policy
            not in self._UNKNOWN_POLICIES
        ):
            raise ValueError(
                'Unsupported unknown_policy: '
                f'{unknown_policy}'
            )

        self.charset_cfg = dict(
            charset_cfg
        )

        self.charset = (
            build_text_recognition_charset(
                self.charset_cfg
            )
        )

        self.max_length = int(
            max_length
        )

        self.empty_text_policy = (
            empty_text_policy
        )

        self.overlength_policy = (
            overlength_policy
        )

        self.unknown_policy = (
            unknown_policy
        )

    def _mark_ignored(
        self,
        ignore_flags: np.ndarray,
        index: int,
    ) -> None:
        ignore_flags[index] = True

    def transform(
        self,
        results: dict,
    ) -> dict:
        if 'gt_texts' not in results:
            raise KeyError(
                'EncodeRecognitionText requires '
                '"gt_texts". Run '
                'LoadTextSpottingAnnotations first.'
            )

        gt_texts = results[
            'gt_texts'
        ]

        num_instances = len(
            gt_texts
        )

        if 'gt_bboxes_labels' in results:
            num_labels = len(
                results[
                    'gt_bboxes_labels'
                ]
            )

            if (
                num_labels
                != num_instances
            ):
                raise ValueError(
                    'gt_texts and '
                    'gt_bboxes_labels are '
                    'not instance-aligned: '
                    f'{num_instances} texts '
                    f'vs {num_labels} labels.'
                )

        if 'gt_ignore_flags' in results:
            ignore_flags = np.asarray(
                results[
                    'gt_ignore_flags'
                ],
                dtype=bool,
            ).copy()

            if (
                len(ignore_flags)
                != num_instances
            ):
                raise ValueError(
                    'gt_ignore_flags and '
                    'gt_texts are '
                    'not instance-aligned.'
                )
        else:
            ignore_flags = np.zeros(
                num_instances,
                dtype=bool,
            )

        encoded_targets = []
        recognition_lengths = []

        for index, text in enumerate(
            gt_texts
        ):
            if text is None:
                if (
                    self.empty_text_policy
                    == 'error'
                ):
                    raise ValueError(
                        'Missing transcription '
                        f'at instance {index}.'
                    )

                self._mark_ignored(
                    ignore_flags,
                    index,
                )

                text = ''

            if not isinstance(
                text,
                str,
            ):
                raise TypeError(
                    'Transcription must be '
                    'a string or None, but '
                    f'instance {index} has '
                    f'{type(text).__name__}.'
                )

            if len(text) == 0:
                if (
                    self.empty_text_policy
                    == 'error'
                ):
                    raise ValueError(
                        'Empty transcription '
                        f'at instance {index}.'
                    )

                self._mark_ignored(
                    ignore_flags,
                    index,
                )

            if (
                len(text)
                > self.max_length
            ):
                if (
                    self.overlength_policy
                    == 'error'
                ):
                    raise ValueError(
                        'Transcription exceeds '
                        'max_length at instance '
                        f'{index}: '
                        f'{len(text)} > '
                        f'{self.max_length}.'
                    )

                if (
                    self.overlength_policy
                    == 'ignore'
                ):
                    self._mark_ignored(
                        ignore_flags,
                        index,
                    )

            unknown_characters = (
                self.charset
                .unknown_characters(text)
            )

            if unknown_characters:
                if (
                    self.unknown_policy
                    == 'error'
                ):
                    raise ValueError(
                        'Unknown characters at '
                        f'instance {index}: '
                        f'{unknown_characters}'
                    )

                if (
                    self.unknown_policy
                    == 'ignore'
                ):
                    self._mark_ignored(
                        ignore_flags,
                        index,
                    )

            target = (
                self.charset.encode(
                    text=text,
                    max_length=(
                        self.max_length
                    ),
                )
            )

            self.charset.validate_ids(
                target,
                expected_length=(
                    self.max_length
                ),
            )

            encoded_targets.append(
                target
            )

            recognition_lengths.append(
                min(
                    len(text),
                    self.max_length,
                )
            )

        if num_instances == 0:
            gt_rec = np.empty(
                (
                    0,
                    self.max_length,
                ),
                dtype=np.int64,
            )

            gt_rec_lengths = np.empty(
                (0,),
                dtype=np.int64,
            )
        else:
            gt_rec = np.asarray(
                encoded_targets,
                dtype=np.int64,
            )

            gt_rec_lengths = np.asarray(
                recognition_lengths,
                dtype=np.int64,
            )

        expected_shape = (
            num_instances,
            self.max_length,
        )

        if (
            gt_rec.shape
            != expected_shape
        ):
            raise RuntimeError(
                'Unexpected recognition '
                'target shape: '
                f'{gt_rec.shape}, '
                f'expected '
                f'{expected_shape}.'
            )

        results[
            'gt_rec'
        ] = gt_rec

        results[
            'gt_rec_lengths'
        ] = gt_rec_lengths

        results[
            'gt_ignore_flags'
        ] = ignore_flags

        return results

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '('
            + f'charset_cfg={self.charset_cfg}, '
            + f'max_length={self.max_length}, '
            + 'empty_text_policy='
            + f'{self.empty_text_policy}, '
            + 'overlength_policy='
            + f'{self.overlength_policy}, '
            + 'unknown_policy='
            + f'{self.unknown_policy}'
            + ')'
        )


@TRANSFORMS.register_module()
class PackTextSpottingInputs(
    PackDetInputs
):
    """Pack detection and recognition targets into DetDataSample.

    Added InstanceData fields:

        rec:
            LongTensor [num_instances, max_length]

        rec_lengths:
            LongTensor [num_instances]

    Ignore filtering is inherited from PackDetInputs, so recognition targets
    remain aligned with boxes and labels.
    """

    mapping_table = dict(
        PackDetInputs.mapping_table,
        gt_rec='rec',
        gt_rec_lengths='rec_lengths',
    )