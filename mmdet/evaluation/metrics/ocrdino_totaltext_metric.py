from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


def _to_numpy(value):
    if torch.is_tensor(value):
        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    return np.asarray(
        value
    )


def _bbox_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:

    boxes1 = np.asarray(
        boxes1,
        dtype=np.float64,
    ).reshape(-1, 4)

    boxes2 = np.asarray(
        boxes2,
        dtype=np.float64,
    ).reshape(-1, 4)

    if (
        len(boxes1) == 0
        or len(boxes2) == 0
    ):
        return np.zeros(
            (
                len(boxes1),
                len(boxes2),
            ),
            dtype=np.float64,
        )

    lt = np.maximum(
        boxes1[
            :,
            None,
            :2,
        ],
        boxes2[
            None,
            :,
            :2,
        ],
    )

    rb = np.minimum(
        boxes1[
            :,
            None,
            2:,
        ],
        boxes2[
            None,
            :,
            2:,
        ],
    )

    wh = np.maximum(
        rb - lt,
        0.0,
    )

    intersection = (
        wh[
            ...,
            0
        ]
        * wh[
            ...,
            1
        ]
    )

    area1 = (
        np.maximum(
            boxes1[
                :,
                2
            ]
            - boxes1[
                :,
                0
            ],
            0.0,
        )
        * np.maximum(
            boxes1[
                :,
                3
            ]
            - boxes1[
                :,
                1
            ],
            0.0,
        )
    )

    area2 = (
        np.maximum(
            boxes2[
                :,
                2
            ]
            - boxes2[
                :,
                0
            ],
            0.0,
        )
        * np.maximum(
            boxes2[
                :,
                3
            ]
            - boxes2[
                :,
                1
            ],
            0.0,
        )
    )

    union = (
        area1[
            :,
            None
        ]
        + area2[
            None,
            :
        ]
        - intersection
    )

    iou = np.divide(
        intersection,
        union,
        out=np.zeros_like(
            intersection
        ),
        where=union > 0,
    )

    return iou


def _levenshtein(
    source: str,
    target: str,
) -> int:

    if source == target:
        return 0

    if len(source) == 0:
        return len(target)

    if len(target) == 0:
        return len(source)

    previous = list(
        range(
            len(target) + 1
        )
    )

    for i, source_char in enumerate(
        source,
        start=1,
    ):
        current = [
            i
        ]

        for j, target_char in enumerate(
            target,
            start=1,
        ):
            insert_cost = (
                current[
                    j - 1
                ]
                + 1
            )

            delete_cost = (
                previous[
                    j
                ]
                + 1
            )

            replace_cost = (
                previous[
                    j - 1
                ]
                + (
                    source_char
                    != target_char
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

    return int(
        previous[-1]
    )


def _compute_ap(
    recall: np.ndarray,
    precision: np.ndarray,
) -> float:

    recall_points = np.linspace(
        0.0,
        1.0,
        101,
    )

    interpolated = []

    for recall_level in (
        recall_points
    ):
        valid = (
            recall
            >= recall_level
        )

        if np.any(valid):
            interpolated.append(
                float(
                    np.max(
                        precision[
                            valid
                        ]
                    )
                )
            )
        else:
            interpolated.append(
                0.0
            )

    return float(
        np.mean(
            interpolated
        )
    )


@METRICS.register_module()
class OCRDinoTotalTextMetric(BaseMetric):

    default_prefix = 'ocrdino'

    def __init__(
        self,
        iou_thr: float = 0.5,
        score_thr: float = 0.3,
        ap_iou_thresholds: Sequence[float] = (
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
        ),
        collect_device: str = 'cpu',
        prefix: str = None,
    ) -> None:

        super().__init__(
            collect_device=collect_device,
            prefix=prefix,
        )

        self.iou_thr = float(
            iou_thr
        )

        self.score_thr = float(
            score_thr
        )

        self.ap_iou_thresholds = tuple(
            float(value)
            for value
            in ap_iou_thresholds
        )

        if not (
            0.0
            <= self.iou_thr
            <= 1.0
        ):
            raise ValueError(
                'iou_thr must be in [0, 1]'
            )

        if not (
            0.0
            <= self.score_thr
            <= 1.0
        ):
            raise ValueError(
                'score_thr must be in [0, 1]'
            )

    def process(
        self,
        data_batch: dict,
        data_samples: List,
    ) -> None:

        for data_sample in (
            data_samples
        ):

            if isinstance(
                data_sample,
                dict,
            ):
                pred_instances = (
                    data_sample[
                        'pred_instances'
                    ]
                )

                gt_instances = (
                    data_sample[
                        'gt_instances'
                    ]
                )

                pred_bboxes = (
                    pred_instances[
                        'bboxes'
                    ]
                )

                pred_scores = (
                    pred_instances[
                        'scores'
                    ]
                )

                pred_texts = (
                    pred_instances[
                        'rec_texts'
                    ]
                )

                gt_bboxes = (
                    gt_instances[
                        'bboxes'
                    ]
                )

                gt_texts = (
                    gt_instances[
                        'ocr_texts'
                    ]
                )

            else:
                pred_instances = (
                    data_sample
                    .pred_instances
                )

                gt_instances = (
                    data_sample
                    .gt_instances
                )

                pred_bboxes = (
                    pred_instances
                    .bboxes
                )

                pred_scores = (
                    pred_instances
                    .scores
                )

                pred_texts = (
                    pred_instances
                    .rec_texts
                )

                gt_bboxes = (
                    gt_instances
                    .bboxes
                )

                gt_texts = (
                    gt_instances
                    .ocr_texts
                )

            pred_bboxes = (
                _to_numpy(
                    pred_bboxes
                )
                .astype(
                    np.float64
                )
                .reshape(
                    -1,
                    4,
                )
            )

            pred_scores = (
                _to_numpy(
                    pred_scores
                )
                .astype(
                    np.float64
                )
                .reshape(-1)
            )

            gt_bboxes = (
                _to_numpy(
                    gt_bboxes
                )
                .astype(
                    np.float64
                )
                .reshape(
                    -1,
                    4,
                )
            )

            pred_texts = [
                str(value)
                for value
                in pred_texts
            ]

            gt_texts = [
                str(value)
                for value
                in gt_texts
            ]

            if (
                len(pred_bboxes)
                != len(pred_scores)
                or len(pred_bboxes)
                != len(pred_texts)
            ):
                raise RuntimeError(
                    'Prediction fields are '
                    'not aligned.'
                )

            if (
                len(gt_bboxes)
                != len(gt_texts)
            ):
                raise RuntimeError(
                    'GT bbox/text fields '
                    'are not aligned.'
                )

            finite_pred = (
                np.isfinite(
                    pred_bboxes
                ).all(
                    axis=1
                )
                & np.isfinite(
                    pred_scores
                )
            )

            valid_geometry = (
                (
                    pred_bboxes[
                        :,
                        2
                    ]
                    > pred_bboxes[
                        :,
                        0
                    ]
                )
                & (
                    pred_bboxes[
                        :,
                        3
                    ]
                    > pred_bboxes[
                        :,
                        1
                    ]
                )
            )

            valid_pred = (
                finite_pred
                & valid_geometry
            )

            pred_bboxes = (
                pred_bboxes[
                    valid_pred
                ]
            )

            pred_scores = (
                pred_scores[
                    valid_pred
                ]
            )

            pred_texts = [
                pred_texts[index]
                for index
                in np.where(
                    valid_pred
                )[0]
            ]

            order = np.argsort(
                -pred_scores
            )

            pred_bboxes = (
                pred_bboxes[
                    order
                ]
            )

            pred_scores = (
                pred_scores[
                    order
                ]
            )

            pred_texts = [
                pred_texts[index]
                for index
                in order
            ]

            self.results.append(
                dict(
                    pred_bboxes=pred_bboxes,
                    pred_scores=pred_scores,
                    pred_texts=pred_texts,
                    gt_bboxes=gt_bboxes,
                    gt_texts=gt_texts,
                )
            )

    def _dataset_ap(
        self,
        results: List[dict],
        iou_thr: float,
    ) -> float:

        total_gt = sum(
            len(
                result[
                    'gt_bboxes'
                ]
            )
            for result
            in results
        )

        if total_gt == 0:
            return 0.0

        predictions = []

        for image_index, result in enumerate(
            results
        ):
            for pred_index, score in enumerate(
                result[
                    'pred_scores'
                ]
            ):
                predictions.append(
                    (
                        float(score),
                        image_index,
                        pred_index,
                    )
                )

        predictions.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        matched = [
            np.zeros(
                len(
                    result[
                        'gt_bboxes'
                    ]
                ),
                dtype=np.bool_,
            )
            for result
            in results
        ]

        tp = np.zeros(
            len(predictions),
            dtype=np.float64,
        )

        fp = np.zeros(
            len(predictions),
            dtype=np.float64,
        )

        for rank, (
            score,
            image_index,
            pred_index,
        ) in enumerate(
            predictions
        ):
            result = (
                results[
                    image_index
                ]
            )

            gt_bboxes = (
                result[
                    'gt_bboxes'
                ]
            )

            if len(gt_bboxes) == 0:
                fp[
                    rank
                ] = 1.0
                continue

            pred_box = (
                result[
                    'pred_bboxes'
                ][
                    pred_index
                    :
                    pred_index
                    + 1
                ]
            )

            ious = (
                _bbox_iou_matrix(
                    pred_box,
                    gt_bboxes,
                )[0]
            )

            available = np.where(
                ~matched[
                    image_index
                ]
            )[0]

            if len(available) == 0:
                fp[
                    rank
                ] = 1.0
                continue

            available_ious = (
                ious[
                    available
                ]
            )

            best_pos = int(
                np.argmax(
                    available_ious
                )
            )

            best_gt = int(
                available[
                    best_pos
                ]
            )

            best_iou = float(
                ious[
                    best_gt
                ]
            )

            if best_iou >= iou_thr:
                tp[
                    rank
                ] = 1.0

                matched[
                    image_index
                ][
                    best_gt
                ] = True
            else:
                fp[
                    rank
                ] = 1.0

        tp_cumulative = (
            np.cumsum(
                tp
            )
        )

        fp_cumulative = (
            np.cumsum(
                fp
            )
        )

        recall = (
            tp_cumulative
            / float(
                total_gt
            )
        )

        precision = (
            tp_cumulative
            / np.maximum(
                tp_cumulative
                + fp_cumulative,
                1e-12,
            )
        )

        return _compute_ap(
            recall,
            precision,
        )

    def _match_for_recognition(
        self,
        result: dict,
    ) -> List[
        Tuple[int, int]
    ]:

        scores = (
            result[
                'pred_scores'
            ]
        )

        keep = np.where(
            scores
            >= self.score_thr
        )[0]

        if len(keep) == 0:
            return []

        pred_bboxes = (
            result[
                'pred_bboxes'
            ][
                keep
            ]
        )

        gt_bboxes = (
            result[
                'gt_bboxes'
            ]
        )

        if len(gt_bboxes) == 0:
            return []

        iou_matrix = (
            _bbox_iou_matrix(
                pred_bboxes,
                gt_bboxes,
            )
        )

        gt_used = np.zeros(
            len(gt_bboxes),
            dtype=np.bool_,
        )

        matches = []

        for local_pred_index in range(
            len(keep)
        ):
            available = np.where(
                ~gt_used
            )[0]

            if len(available) == 0:
                break

            available_ious = (
                iou_matrix[
                    local_pred_index,
                    available,
                ]
            )

            best_pos = int(
                np.argmax(
                    available_ious
                )
            )

            gt_index = int(
                available[
                    best_pos
                ]
            )

            best_iou = float(
                iou_matrix[
                    local_pred_index,
                    gt_index,
                ]
            )

            if (
                best_iou
                < self.iou_thr
            ):
                continue

            gt_used[
                gt_index
            ] = True

            original_pred_index = int(
                keep[
                    local_pred_index
                ]
            )

            matches.append(
                (
                    original_pred_index,
                    gt_index,
                )
            )

        return matches

    def _compute_recognition_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:

        matched_count = 0

        exact_count = 0

        edit_distance_sum = 0

        gt_char_count = 0

        for result in results:
            matches = (
                self._match_for_recognition(
                    result
                )
            )

            pred_texts = (
                result[
                    'pred_texts'
                ]
            )

            gt_texts = (
                result[
                    'gt_texts'
                ]
            )

            for (
                pred_index,
                gt_index,
            ) in matches:
                pred_text = (
                    pred_texts[
                        pred_index
                    ]
                )

                gt_text = (
                    gt_texts[
                        gt_index
                    ]
                )

                matched_count += 1

                if (
                    pred_text
                    == gt_text
                ):
                    exact_count += 1

                edit_distance_sum += (
                    _levenshtein(
                        pred_text,
                        gt_text,
                    )
                )

                gt_char_count += len(
                    gt_text
                )

        exact_accuracy = (
            exact_count
            / matched_count
            if matched_count > 0
            else 0.0
        )

        if gt_char_count > 0:
            cer = (
                edit_distance_sum
                / gt_char_count
            )
        else:
            cer = 0.0

        char_accuracy = max(
            0.0,
            1.0 - cer,
        )

        return dict(
            rec_exact_accuracy=float(
                exact_accuracy
            ),
            rec_char_accuracy=float(
                char_accuracy
            ),
            rec_matched=float(
                matched_count
            ),
        )

    def _compute_e2e_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:

        tp = 0
        fp = 0
        fn = 0

        for result in results:
            scores = (
                result[
                    'pred_scores'
                ]
            )

            keep = np.where(
                scores
                >= self.score_thr
            )[0]

            pred_bboxes = (
                result[
                    'pred_bboxes'
                ]
            )

            pred_texts = (
                result[
                    'pred_texts'
                ]
            )

            gt_bboxes = (
                result[
                    'gt_bboxes'
                ]
            )

            gt_texts = (
                result[
                    'gt_texts'
                ]
            )

            gt_used = np.zeros(
                len(gt_bboxes),
                dtype=np.bool_,
            )

            for pred_index in keep:
                pred_box = (
                    pred_bboxes[
                        pred_index
                        :
                        pred_index
                        + 1
                    ]
                )

                pred_text = (
                    pred_texts[
                        pred_index
                    ]
                )

                candidate_indices = [
                    gt_index
                    for gt_index, gt_text
                    in enumerate(
                        gt_texts
                    )
                    if (
                        not gt_used[
                            gt_index
                        ]
                        and pred_text
                        == gt_text
                    )
                ]

                if not candidate_indices:
                    fp += 1
                    continue

                candidate_boxes = (
                    gt_bboxes[
                        candidate_indices
                    ]
                )

                ious = (
                    _bbox_iou_matrix(
                        pred_box,
                        candidate_boxes,
                    )[0]
                )

                best_local = int(
                    np.argmax(
                        ious
                    )
                )

                best_iou = float(
                    ious[
                        best_local
                    ]
                )

                if (
                    best_iou
                    >= self.iou_thr
                ):
                    gt_index = int(
                        candidate_indices[
                            best_local
                        ]
                    )

                    gt_used[
                        gt_index
                    ] = True

                    tp += 1
                else:
                    fp += 1

            fn += int(
                (
                    ~gt_used
                ).sum()
            )

        precision = (
            tp
            / (
                tp
                + fp
            )
            if (
                tp
                + fp
            ) > 0
            else 0.0
        )

        recall = (
            tp
            / (
                tp
                + fn
            )
            if (
                tp
                + fn
            ) > 0
            else 0.0
        )

        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
            if (
                precision
                + recall
            ) > 0
            else 0.0
        )

        return dict(
            e2e_precision=float(
                precision
            ),
            e2e_recall=float(
                recall
            ),
            e2e_f1=float(
                f1
            ),
        )

    def compute_metrics(
        self,
        results: List[dict],
    ) -> Dict[str, float]:

        ap_values = []

        ap50 = 0.0

        for iou_thr in (
            self.ap_iou_thresholds
        ):
            ap = (
                self._dataset_ap(
                    results,
                    iou_thr,
                )
            )

            ap_values.append(
                ap
            )

            if (
                abs(
                    iou_thr
                    - 0.50
                )
                < 1e-8
            ):
                ap50 = ap

        det_map = (
            float(
                np.mean(
                    ap_values
                )
            )
            if ap_values
            else 0.0
        )

        recognition_metrics = (
            self._compute_recognition_metrics(
                results
            )
        )

        e2e_metrics = (
            self._compute_e2e_metrics(
                results
            )
        )

        total_gt = sum(
            len(
                result[
                    'gt_bboxes'
                ]
            )
            for result
            in results
        )

        metrics = dict(
            det_mAP=float(
                det_map
            ),

            det_AP50=float(
                ap50
            ),

            rec_exact_accuracy=(
                recognition_metrics[
                    'rec_exact_accuracy'
                ]
            ),

            rec_char_accuracy=(
                recognition_metrics[
                    'rec_char_accuracy'
                ]
            ),

            e2e_precision=(
                e2e_metrics[
                    'e2e_precision'
                ]
            ),

            e2e_recall=(
                e2e_metrics[
                    'e2e_recall'
                ]
            ),

            e2e_f1=(
                e2e_metrics[
                    'e2e_f1'
                ]
            ),

            num_gt=float(
                total_gt
            ),

            rec_matched=(
                recognition_metrics[
                    'rec_matched'
                ]
            ),
        )

        return metrics