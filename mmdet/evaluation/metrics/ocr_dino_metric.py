# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Sequence

import torch
from mmengine.evaluator import BaseMetric
from torch import Tensor

from mmdet.registry import METRICS
from mmdet.structures.bbox import bbox_overlaps


@METRICS.register_module()
class OCRDinoMetric(BaseMetric):
    """Evaluation metric for OCR-DINO."""

    default_prefix = 'ocrdino'

    def __init__(
        self,
        iou_thr: float = 0.5,
        text_types: Sequence[str] = ('text', 'latex'),
        collect_device: str = 'cpu',
        prefix: str = None,
    ):
        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
        )

        self.iou_thr = float(iou_thr)
        self.text_types = set(text_types)

    @staticmethod
    def _get_field(
        container,
        key: str,
        default=None,
    ):
        if isinstance(container, dict):
            return container.get(
                key,
                default,
            )

        return getattr(
            container,
            key,
            default,
        )

    @staticmethod
    def _get_sample_field(
        data_sample,
        key: str,
        default=None,
    ):
        if isinstance(data_sample, dict):
            if key in data_sample:
                return data_sample[key]

            metainfo = data_sample.get(
                'metainfo',
                {},
            )

            if isinstance(metainfo, dict):
                return metainfo.get(
                    key,
                    default,
                )

            return default

        if hasattr(data_sample, 'metainfo'):
            return data_sample.metainfo.get(
                key,
                default,
            )

        return getattr(
            data_sample,
            key,
            default,
        )

    @staticmethod
    def _to_tensor(boxes) -> Tensor:
        if boxes is None:
            return torch.empty(
                (0, 4),
                dtype=torch.float32,
            )

        if hasattr(boxes, 'tensor'):
            return boxes.tensor

        if torch.is_tensor(boxes):
            return boxes

        return torch.as_tensor(
            boxes,
            dtype=torch.float32,
        )

    @staticmethod
    def _edit_distance(
        reference: str,
        prediction: str,
    ) -> int:
        """Compute character-level Levenshtein distance."""

        if reference == prediction:
            return 0

        if not reference:
            return len(prediction)

        if not prediction:
            return len(reference)

        previous = list(
            range(len(prediction) + 1)
        )

        for i, ref_char in enumerate(
            reference,
            start=1,
        ):
            current = [i]

            for j, pred_char in enumerate(
                prediction,
                start=1,
            ):
                insert_cost = (
                    current[j - 1] + 1
                )

                delete_cost = (
                    previous[j] + 1
                )

                replace_cost = (
                    previous[j - 1]
                    + int(
                        ref_char != pred_char
                    )
                )

                current.append(
                    min(
                        insert_cost,
                        delete_cost,
                        replace_cost,
                    )
                )

            previous = current

        return previous[-1]

    @staticmethod
    def _safe_div(
        numerator: float,
        denominator: float,
    ) -> float:
        if denominator == 0:
            return 0.0

        return (
            float(numerator)
            / float(denominator)
        )

    def _restore_gt_boxes(
        self,
        gt_boxes: Tensor,
        data_sample,
    ) -> Tensor:
        """Restore transformed GT boxes to original image coordinates."""

        if gt_boxes.numel() == 0:
            return gt_boxes

        scale_factor = (
            self._get_sample_field(
                data_sample,
                'scale_factor',
                None,
            )
        )

        if scale_factor is None:
            return gt_boxes

        scale = gt_boxes.new_tensor(
            scale_factor
        ).flatten()

        if scale.numel() == 2:
            scale = scale.repeat(2)

        if scale.numel() != 4:
            raise ValueError(
                'scale_factor must contain 2 or 4 values.'
            )

        return gt_boxes / scale

    def process(
        self,
        data_batch: dict,
        data_samples: Sequence[dict],
    ) -> None:
        """Process predictions and ground truth."""

        for data_sample in data_samples:
            gt_instances = (
                self._get_sample_field(
                    data_sample,
                    'gt_instances',
                )
            )

            pred_instances = (
                self._get_sample_field(
                    data_sample,
                    'pred_instances',
                )
            )

            if gt_instances is None:
                raise KeyError(
                    'gt_instances is missing from data_sample.'
                )

            if pred_instances is None:
                raise KeyError(
                    'pred_instances is missing from data_sample.'
                )

            gt_boxes = self._to_tensor(
                self._get_field(
                    gt_instances,
                    'bboxes',
                )
            ).detach().cpu()

            gt_boxes = self._restore_gt_boxes(
                gt_boxes,
                data_sample,
            )

            gt_labels = self._get_field(
                gt_instances,
                'labels',
            )

            if gt_labels is None:
                gt_labels = torch.empty(
                    0,
                    dtype=torch.long,
                )

            gt_labels = (
                torch.as_tensor(
                    gt_labels,
                )
                .detach()
                .cpu()
            )

            pred_boxes = self._to_tensor(
                self._get_field(
                    pred_instances,
                    'bboxes',
                )
            ).detach().cpu()

            pred_labels = self._get_field(
                pred_instances,
                'labels',
            )

            if pred_labels is None:
                pred_labels = torch.empty(
                    0,
                    dtype=torch.long,
                )

            pred_labels = (
                torch.as_tensor(
                    pred_labels,
                )
                .detach()
                .cpu()
            )

            pred_scores = self._get_field(
                pred_instances,
                'scores',
            )

            if pred_scores is None:
                pred_scores = torch.empty(
                    0,
                    dtype=torch.float32,
                )

            pred_scores = (
                torch.as_tensor(
                    pred_scores,
                )
                .detach()
                .cpu()
            )

            gt_texts = list(
                self._get_field(
                    gt_instances,
                    'ocr_texts',
                    [''] * len(gt_labels),
                )
            )

            gt_text_types = list(
                self._get_field(
                    gt_instances,
                    'ocr_text_types',
                    ['none'] * len(gt_labels),
                )
            )

            pred_texts = list(
                self._get_field(
                    pred_instances,
                    'texts',
                    [''] * len(pred_labels),
                )
            )

            if len(gt_texts) != len(gt_labels):
                raise ValueError(
                    'GT text count does not match '
                    'GT instance count.'
                )

            if (
                len(gt_text_types)
                != len(gt_labels)
            ):
                raise ValueError(
                    'GT text type count does not match '
                    'GT instance count.'
                )

            if (
                len(pred_texts)
                != len(pred_labels)
            ):
                raise ValueError(
                    'Prediction text count does not match '
                    'prediction instance count.'
                )

            ocr_gt_mask = []

            for text, text_type in zip(
                gt_texts,
                gt_text_types,
            ):
                is_ocr = (
                    text_type in self.text_types
                    and isinstance(text, str)
                    and len(text) > 0
                )

                ocr_gt_mask.append(
                    is_ocr
                )

            total_ocr_gt = sum(
                ocr_gt_mask
            )

            total_ocr_chars = sum(
                len(gt_texts[index])
                for index, is_ocr
                in enumerate(ocr_gt_mask)
                if is_ocr
            )

            if pred_scores.numel() > 0:
                pred_order = torch.argsort(
                    pred_scores,
                    descending=True,
                ).tolist()
            else:
                pred_order = []

            matched_gt = set()
            matched_ocr_gt = set()

            det_tp = 0
            det_fp = 0

            ocr_exact = 0
            ocr_edit_distance = 0
            ocr_reference_chars = 0

            for pred_index in pred_order:
                candidate_gt_indices = []

                pred_label = int(
                    pred_labels[pred_index]
                )

                for gt_index in range(
                    len(gt_labels)
                ):
                    if gt_index in matched_gt:
                        continue

                    if (
                        int(gt_labels[gt_index])
                        != pred_label
                    ):
                        continue

                    candidate_gt_indices.append(
                        gt_index
                    )

                if not candidate_gt_indices:
                    det_fp += 1
                    continue

                candidate_boxes = gt_boxes[
                    candidate_gt_indices
                ]

                pred_box = pred_boxes[
                    pred_index:
                    pred_index + 1
                ]

                ious = bbox_overlaps(
                    pred_box,
                    candidate_boxes,
                )[0]

                best_position = int(
                    torch.argmax(ious)
                )

                best_iou = float(
                    ious[best_position]
                )

                if best_iou < self.iou_thr:
                    det_fp += 1
                    continue

                gt_index = (
                    candidate_gt_indices[
                        best_position
                    ]
                )

                matched_gt.add(
                    gt_index
                )

                det_tp += 1

                if not ocr_gt_mask[
                    gt_index
                ]:
                    continue

                matched_ocr_gt.add(
                    gt_index
                )

                reference = gt_texts[
                    gt_index
                ]

                prediction = pred_texts[
                    pred_index
                ]

                if not isinstance(
                    prediction,
                    str,
                ):
                    prediction = str(
                        prediction
                    )

                distance = (
                    self._edit_distance(
                        reference,
                        prediction,
                    )
                )

                ocr_edit_distance += (
                    distance
                )

                ocr_reference_chars += len(
                    reference
                )

                if prediction == reference:
                    ocr_exact += 1

            det_fn = (
                len(gt_labels)
                - det_tp
            )

            missed_ocr_chars = 0

            for gt_index, is_ocr in enumerate(
                ocr_gt_mask
            ):
                if not is_ocr:
                    continue

                if (
                    gt_index
                    not in matched_ocr_gt
                ):
                    missed_ocr_chars += len(
                        gt_texts[gt_index]
                    )

            self.results.append(
                dict(
                    det_tp=det_tp,
                    det_fp=det_fp,
                    det_fn=det_fn,
                    total_ocr_gt=(
                        total_ocr_gt
                    ),
                    matched_ocr_gt=len(
                        matched_ocr_gt
                    ),
                    ocr_exact=ocr_exact,
                    ocr_edit_distance=(
                        ocr_edit_distance
                    ),
                    ocr_reference_chars=(
                        ocr_reference_chars
                    ),
                    total_ocr_chars=(
                        total_ocr_chars
                    ),
                    missed_ocr_chars=(
                        missed_ocr_chars
                    ),
                )
            )

    def compute_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:
        """Compute final metrics."""

        totals = {
            'det_tp': 0,
            'det_fp': 0,
            'det_fn': 0,
            'total_ocr_gt': 0,
            'matched_ocr_gt': 0,
            'ocr_exact': 0,
            'ocr_edit_distance': 0,
            'ocr_reference_chars': 0,
            'total_ocr_chars': 0,
            'missed_ocr_chars': 0,
        }

        for result in results:
            for key in totals:
                totals[key] += result[key]

        det_precision = self._safe_div(
            totals['det_tp'],
            totals['det_tp']
            + totals['det_fp'],
        )

        det_recall = self._safe_div(
            totals['det_tp'],
            totals['det_tp']
            + totals['det_fn'],
        )

        det_f1 = self._safe_div(
            2.0
            * det_precision
            * det_recall,
            det_precision
            + det_recall,
        )

        ocr_match_recall = self._safe_div(
            totals['matched_ocr_gt'],
            totals['total_ocr_gt'],
        )

        ocr_exact_match = self._safe_div(
            totals['ocr_exact'],
            totals['matched_ocr_gt'],
        )

        ocr_cer = self._safe_div(
            totals['ocr_edit_distance'],
            totals['ocr_reference_chars'],
        )

        e2e_exact_recall = self._safe_div(
            totals['ocr_exact'],
            totals['total_ocr_gt'],
        )

        e2e_edit_distance = (
            totals['ocr_edit_distance']
            + totals['missed_ocr_chars']
        )

        e2e_cer = self._safe_div(
            e2e_edit_distance,
            totals['total_ocr_chars'],
        )

        return {
            'det_precision': det_precision,
            'det_recall': det_recall,
            'det_f1': det_f1,
            'ocr_match_recall': (
                ocr_match_recall
            ),
            'ocr_exact_match': (
                ocr_exact_match
            ),
            'ocr_cer': ocr_cer,
            'e2e_exact_recall': (
                e2e_exact_recall
            ),
            'e2e_cer': e2e_cer,
            'num_gt': float(
                totals['det_tp']
                + totals['det_fn']
            ),
            'num_ocr_gt': float(
                totals['total_ocr_gt']
            ),
        }