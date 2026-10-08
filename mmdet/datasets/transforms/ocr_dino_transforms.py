# Copyright (c) OpenMMLab. All rights reserved.

from typing import Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform

from mmdet.registry import TRANSFORMS

from .formatting import PackDetInputs
from .loading import LoadAnnotations


# ============================================================
# ESTextSpotter-compatible OCR recognition definition
# ============================================================

OCR_REC_LENGTH = 25

OCR_ASCII_FIRST = 32
OCR_ASCII_LAST = 126

OCR_ASCII_ID_MIN = 0
OCR_ASCII_ID_MAX = 94

OCR_UNK_ID = 95
OCR_EOS_ID = 96

OCR_NUM_CLASSES = 97

# Used for instances that participate in detection but do not
# participate in recognition supervision.
OCR_IGNORE_INDEX = -100


# ============================================================
# Recognition helpers
# ============================================================

def encode_estextspotter_text(
    text: str,
    max_length: int = OCR_REC_LENGTH,
) -> list:
    """Encode text using the OCRDINO/ESTextSpotter 97-class scheme.

    Character mapping:

        printable ASCII 32..126 -> class 0..94
        unsupported character  -> UNK 95
        EOS                     -> 96

    Sequence rule:

        len(text) < max_length:
            characters + EOS repeated to max_length

        len(text) >= max_length:
            first max_length characters
            no extra EOS

    Example:

        "HELLO"

        H -> 72 - 32 = 40
        E -> 69 - 32 = 37
        L -> 76 - 32 = 44
        L -> 44
        O -> 79 - 32 = 47

        =>
        [40, 37, 44, 44, 47, 96, ..., 96]
    """

    if not isinstance(text, str):
        raise TypeError(
            f'text must be str, got {type(text)}'
        )

    if max_length <= 0:
        raise ValueError(
            'max_length must be positive'
        )

    token_ids = []

    for char in text:

        code = ord(char)

        if (
            OCR_ASCII_FIRST
            <= code
            <= OCR_ASCII_LAST
        ):
            token_id = (
                code
                - OCR_ASCII_FIRST
            )

        else:
            token_id = (
                OCR_UNK_ID
            )

        token_ids.append(
            token_id
        )

        if (
            len(token_ids)
            >= max_length
        ):
            break

    if len(token_ids) < max_length:

        token_ids.extend(
            [
                OCR_EOS_ID
            ]
            * (
                max_length
                - len(token_ids)
            )
        )

    if len(token_ids) != max_length:
        raise RuntimeError(
            'Encoded recognition sequence '
            f'has unexpected length '
            f'{len(token_ids)}'
        )

    return token_ids


def decode_estextspotter_rec(
    rec: Sequence[int],
) -> str:
    """Decode a recognition target for debugging/tests."""

    chars = []

    for token in rec:

        token = int(token)

        if token == OCR_EOS_ID:
            break

        if (
            OCR_ASCII_ID_MIN
            <= token
            <= OCR_ASCII_ID_MAX
        ):
            chars.append(
                chr(
                    token
                    + OCR_ASCII_FIRST
                )
            )

        elif token == OCR_UNK_ID:
            chars.append(
                '\ufffd'
            )

        elif token == OCR_IGNORE_INDEX:
            continue

        else:
            raise ValueError(
                f'Invalid OCR token: {token}'
            )

    return ''.join(chars)


# ============================================================
# Load OCR annotations
# ============================================================

@TRANSFORMS.register_module()
class LoadOCRAnnotations(LoadAnnotations):
    """Load detection annotations and OCR-specific fields.

    OCR-specific fields are kept parallel to the normal MMDetection
    instance annotations:

        gt_texts
        gt_text_types
        gt_orders

    Recognition encoding is intentionally performed later by
    ``TokenizeOCRText``.
    """

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
        gt_text_types = []
        gt_orders = []

        for instance in instances:

            text = instance.get(
                'text',
                '',
            )

            text_type = instance.get(
                'text_type',
                'none',
            )

            order = instance.get(
                'order',
                -1,
            )

            if text is None:
                text = ''

            if text_type is None:
                text_type = 'none'

            if order is None:
                order = -1

            gt_texts.append(
                text
            )

            gt_text_types.append(
                text_type
            )

            gt_orders.append(
                int(order)
            )

        results[
            'gt_texts'
        ] = gt_texts

        results[
            'gt_text_types'
        ] = gt_text_types

        results[
            'gt_orders'
        ] = np.asarray(
            gt_orders,
            dtype=np.int64,
        )

        # ----------------------------------------------------
        # Alignment validation
        # ----------------------------------------------------

        num_instances = len(
            instances
        )

        if (
            len(gt_texts)
            != num_instances
        ):
            raise RuntimeError(
                'gt_texts are not aligned '
                'with instances'
            )

        if (
            len(gt_text_types)
            != num_instances
        ):
            raise RuntimeError(
                'gt_text_types are not aligned '
                'with instances'
            )

        if (
            len(gt_orders)
            != num_instances
        ):
            raise RuntimeError(
                'gt_orders are not aligned '
                'with instances'
            )

        return results


# ============================================================
# Fixed ESTextSpotter tokenizer
# ============================================================

@TRANSFORMS.register_module()
class TokenizeOCRText(BaseTransform):
    """Build fixed-length OCR recognition targets.

    This replaces the old OCRDINO dynamic vocabulary tokenizer.

    No vocabulary file is required.

    Recognition definition:

        sequence length = 25
        classes         = 97

        ASCII 32..126   -> 0..94
        UNK             -> 95
        EOS             -> 96

    Instances whose ``text_type`` is not included in ``text_types``
    remain detection targets, but their recognition target is filled
    entirely with ``ignore_index``.

    The output ``gt_rec`` always has shape:

        [N, 25]
    """

    def __init__(
        self,
        text_types: Sequence[str] = (
            'text',
            'latex',
        ),
        max_length: int = OCR_REC_LENGTH,
        ignore_index: int = OCR_IGNORE_INDEX,
    ) -> None:

        super().__init__()

        self.text_types = set(
            text_types
        )

        self.max_length = int(
            max_length
        )

        self.ignore_index = int(
            ignore_index
        )

        if (
            self.max_length
            != OCR_REC_LENGTH
        ):
            raise ValueError(
                'Current OCRDINO model requires '
                f'max_length={OCR_REC_LENGTH}, '
                f'got {self.max_length}'
            )

        if (
            self.ignore_index
            != OCR_IGNORE_INDEX
        ):
            raise ValueError(
                'Current OCRDINO recognition '
                f'loss requires ignore_index='
                f'{OCR_IGNORE_INDEX}'
            )

    def transform(
        self,
        results: dict,
    ) -> dict:

        gt_texts = results.get(
            'gt_texts',
            [],
        )

        gt_text_types = results.get(
            'gt_text_types',
            [],
        )

        if (
            len(gt_texts)
            != len(gt_text_types)
        ):
            raise ValueError(
                'gt_texts and gt_text_types '
                'must have the same length'
            )

        num_instances = len(
            gt_texts
        )

        # ----------------------------------------------------
        # [N, 25]
        #
        # Start as completely ignored. Recognition-valid
        # instances are overwritten below.
        # ----------------------------------------------------

        gt_rec = np.full(
            (
                num_instances,
                self.max_length,
            ),
            fill_value=(
                self.ignore_index
            ),
            dtype=np.int64,
        )

        gt_ocr_trainable = np.zeros(
            num_instances,
            dtype=np.bool_,
        )

        for index, (
            text,
            text_type,
        ) in enumerate(
            zip(
                gt_texts,
                gt_text_types,
            )
        ):

            is_trainable = (
                text_type
                in self.text_types
                and isinstance(
                    text,
                    str,
                )
                and len(text) > 0
            )

            if not is_trainable:
                continue

            token_ids = (
                encode_estextspotter_text(
                    text=text,
                    max_length=(
                        self.max_length
                    ),
                )
            )

            gt_rec[
                index,
                :,
            ] = np.asarray(
                token_ids,
                dtype=np.int64,
            )

            gt_ocr_trainable[
                index
            ] = True

        # ----------------------------------------------------
        # Structural validation
        # ----------------------------------------------------

        if (
            gt_rec.shape
            != (
                num_instances,
                self.max_length,
            )
        ):
            raise RuntimeError(
                'Unexpected gt_rec shape: '
                f'{gt_rec.shape}'
            )

        trainable_rec = gt_rec[
            gt_ocr_trainable
        ]

        if trainable_rec.size > 0:

            min_token = int(
                trainable_rec.min()
            )

            max_token = int(
                trainable_rec.max()
            )

            if min_token < 0:
                raise RuntimeError(
                    'Trainable recognition '
                    'targets contain negative IDs'
                )

            if (
                max_token
                >= OCR_NUM_CLASSES
            ):
                raise RuntimeError(
                    'Recognition target class '
                    'exceeds classifier size: '
                    f'{max_token}'
                )

        results[
            'gt_rec'
        ] = gt_rec

        results[
            'gt_ocr_trainable'
        ] = gt_ocr_trainable

        # Useful metadata for debugging/checkpoint/config checks.
        results[
            'ocr_vocab_size'
        ] = OCR_NUM_CLASSES

        results[
            'ocr_num_classes'
        ] = OCR_NUM_CLASSES

        results[
            'ocr_max_length'
        ] = self.max_length

        results[
            'ocr_unk_token_id'
        ] = OCR_UNK_ID

        results[
            'ocr_eos_token_id'
        ] = OCR_EOS_ID

        results[
            'ocr_ignore_index'
        ] = self.ignore_index

        return results


# ============================================================
# Pack OCR targets into DetDataSample
# ============================================================

@TRANSFORMS.register_module()
class PackOCRDinoInputs(PackDetInputs):
    """Pack detection + OCR targets into ``DetDataSample``.

    The key output required by ESTDINOHead is:

        data_sample.gt_instances.rec

    with shape:

        [N, 25]

    and dtype:

        torch.long
    """

    def __init__(
        self,
        meta_keys: Sequence[str] = (
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'flip',
            'flip_direction',
            'language',
            'source',
        ),
    ) -> None:

        super().__init__(
            meta_keys=meta_keys
        )

    def transform(
        self,
        results: dict,
    ) -> dict:

        # ----------------------------------------------------
        # Detection annotations are packed by stock
        # PackDetInputs first.
        # ----------------------------------------------------

        packed_results = (
            super().transform(
                results
            )
        )

        data_sample = (
            packed_results[
                'data_samples'
            ]
        )

        gt_texts = results.get(
            'gt_texts',
            [],
        )

        gt_text_types = results.get(
            'gt_text_types',
            [],
        )

        gt_orders = results.get(
            'gt_orders',
            np.full(
                len(gt_texts),
                -1,
                dtype=np.int64,
            ),
        )

        # ----------------------------------------------------
        # Recognition matrix
        # ----------------------------------------------------

        if 'gt_rec' in results:

            gt_rec = np.asarray(
                results[
                    'gt_rec'
                ],
                dtype=np.int64,
            )

        else:

            # This allows evaluation pipelines that do not run
            # TokenizeOCRText.
            gt_rec = np.full(
                (
                    len(gt_texts),
                    OCR_REC_LENGTH,
                ),
                OCR_IGNORE_INDEX,
                dtype=np.int64,
            )

        gt_ocr_trainable = results.get(
            'gt_ocr_trainable',
            np.zeros(
                len(gt_texts),
                dtype=np.bool_,
            ),
        )

        gt_ocr_trainable = np.asarray(
            gt_ocr_trainable,
            dtype=np.bool_,
        )

        # ----------------------------------------------------
        # Basic alignment before valid/ignored split
        # ----------------------------------------------------

        num_instances = len(
            gt_texts
        )

        if (
            len(gt_text_types)
            != num_instances
        ):
            raise RuntimeError(
                'gt_text_types are not '
                'aligned with gt_texts'
            )

        if (
            len(gt_orders)
            != num_instances
        ):
            raise RuntimeError(
                'gt_orders are not aligned '
                'with gt_texts'
            )

        if (
            gt_ocr_trainable.shape
            != (
                num_instances,
            )
        ):
            raise RuntimeError(
                'Unexpected '
                'gt_ocr_trainable shape: '
                f'{gt_ocr_trainable.shape}'
            )

        if (
            gt_rec.shape
            != (
                num_instances,
                OCR_REC_LENGTH,
            )
        ):
            raise RuntimeError(
                'Unexpected gt_rec shape '
                'before packing: '
                f'{gt_rec.shape}, expected '
                f'({num_instances}, '
                f'{OCR_REC_LENGTH})'
            )

        # ----------------------------------------------------
        # Match PackDetInputs ignore semantics.
        # ----------------------------------------------------

        if (
            'gt_ignore_flags'
            in results
        ):

            ignore_flags = np.asarray(
                results[
                    'gt_ignore_flags'
                ]
            ).astype(
                np.bool_
            )

            if (
                ignore_flags.shape
                != (
                    num_instances,
                )
            ):
                raise RuntimeError(
                    'gt_ignore_flags are not '
                    'aligned with OCR targets'
                )

            valid_idx = np.where(
                ~ignore_flags
            )[0]

            ignore_idx = np.where(
                ignore_flags
            )[0]

        else:

            valid_idx = np.arange(
                num_instances,
                dtype=np.int64,
            )

            ignore_idx = np.empty(
                0,
                dtype=np.int64,
            )

        # ----------------------------------------------------
        # Valid instances
        # ----------------------------------------------------

        valid_rec = gt_rec[
            valid_idx
        ]

        valid_texts = [
            gt_texts[index]
            for index
            in valid_idx
        ]

        valid_text_types = [
            gt_text_types[index]
            for index
            in valid_idx
        ]

        valid_orders = np.asarray(
            gt_orders,
            dtype=np.int64,
        )[
            valid_idx
        ]

        valid_trainable = (
            gt_ocr_trainable[
                valid_idx
            ]
        )

        # ----------------------------------------------------
        # Ignored instances
        # ----------------------------------------------------

        ignored_rec = gt_rec[
            ignore_idx
        ]

        ignored_texts = [
            gt_texts[index]
            for index
            in ignore_idx
        ]

        ignored_text_types = [
            gt_text_types[index]
            for index
            in ignore_idx
        ]

        ignored_orders = np.asarray(
            gt_orders,
            dtype=np.int64,
        )[
            ignore_idx
        ]

        ignored_trainable = (
            gt_ocr_trainable[
                ignore_idx
            ]
        )

        # ----------------------------------------------------
        # Final length consistency with PackDetInputs output.
        # ----------------------------------------------------

        num_valid_detection_gt = len(
            data_sample.gt_instances
        )

        if (
            num_valid_detection_gt
            != len(valid_idx)
        ):
            raise RuntimeError(
                'Detection/OCR valid-instance '
                'alignment failed: '
                f'detection={num_valid_detection_gt}, '
                f'ocr={len(valid_idx)}'
            )

        num_ignored_detection_gt = len(
            data_sample.ignored_instances
        )

        if (
            num_ignored_detection_gt
            != len(ignore_idx)
        ):
            raise RuntimeError(
                'Detection/OCR ignored-instance '
                'alignment failed: '
                f'detection='
                f'{num_ignored_detection_gt}, '
                f'ocr={len(ignore_idx)}'
            )

        # ====================================================
        # This is the field ESTDINOHead consumes.
        # ====================================================

        data_sample.gt_instances.rec = (
            torch.as_tensor(
                valid_rec,
                dtype=torch.long,
            )
        )

        data_sample.ignored_instances.rec = (
            torch.as_tensor(
                ignored_rec,
                dtype=torch.long,
            )
        )

        # ----------------------------------------------------
        # Keep human-readable OCR metadata for debugging and
        # future evaluation.
        # ----------------------------------------------------

        data_sample.gt_instances.ocr_texts = (
            valid_texts
        )

        data_sample.gt_instances.ocr_text_types = (
            valid_text_types
        )

        data_sample.gt_instances.ocr_orders = (
            valid_orders
        )

        data_sample.gt_instances.ocr_trainable = (
            valid_trainable
        )

        data_sample.ignored_instances.ocr_texts = (
            ignored_texts
        )

        data_sample.ignored_instances.ocr_text_types = (
            ignored_text_types
        )

        data_sample.ignored_instances.ocr_orders = (
            ignored_orders
        )

        data_sample.ignored_instances.ocr_trainable = (
            ignored_trainable
        )

        # ----------------------------------------------------
        # Tokenizer/model metadata
        # ----------------------------------------------------

        tokenizer_meta = {}

        for key in (
            'ocr_vocab_size',
            'ocr_num_classes',
            'ocr_max_length',
            'ocr_unk_token_id',
            'ocr_eos_token_id',
            'ocr_ignore_index',
        ):

            if key in results:

                tokenizer_meta[
                    key
                ] = results[
                    key
                ]

        if tokenizer_meta:

            data_sample.set_metainfo(
                tokenizer_meta
            )

        return packed_results