# Copyright (c) OpenMMLab. All rights reserved.

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from mmengine.evaluator import BaseMetric

from mmdet.registry import METRICS


@METRICS.register_module()
class PointMetric(BaseMetric):
    """Evaluation metric for single-class point detection.

    A prediction is considered a true positive when it can be matched
    one-to-one with a ground-truth point within ``distance_threshold``.

    Predictions whose confidence is lower than ``score_threshold`` are
    ignored.

    Args:
        distance_threshold (float):
            Maximum Euclidean distance in pixels for a valid match.
        score_threshold (float):
            Minimum prediction confidence used for evaluation.
        collect_device (str):
            Device used to collect results.
        prefix (str, optional):
            Metric prefix.
    """

    default_prefix = 'point'

    def __init__(
            self,
            distance_threshold: float,
            score_threshold: float,
            collect_device: str = 'cpu',
            prefix: Optional[str] = None) -> None:

        super().__init__(
            collect_device=collect_device,
            prefix=prefix)

        if distance_threshold <= 0:
            raise ValueError(
                'distance_threshold must be > 0.')

        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                'score_threshold must be in [0, 1].')

        self.distance_threshold = float(
            distance_threshold)

        self.score_threshold = float(
            score_threshold)

    def _match_points(
            self,
            pred_points: np.ndarray,
            gt_points: np.ndarray):

        num_pred = len(pred_points)
        num_gt = len(gt_points)

        if num_pred == 0 or num_gt == 0:
            return 0, 0.0

        # Pairwise Euclidean distance:
        # shape = [num_pred, num_gt]
        distance_matrix = np.linalg.norm(
            pred_points[:, None, :] -
            gt_points[None, :, :],
            axis=-1)

        # We want:
        # 1. maximum number of valid matches;
        # 2. minimum total distance among those matches.
        #
        # Invalid pairs are therefore assigned a sufficiently
        # large penalty before Hungarian matching.
        num_pairs = min(num_pred, num_gt)

        invalid_cost = (
            (num_pairs + 1) *
            max(self.distance_threshold, 1.0)
            + 1.0
        )

        cost_matrix = np.where(
            distance_matrix <= self.distance_threshold,
            distance_matrix,
            invalid_cost)

        pred_indices, gt_indices = \
            linear_sum_assignment(cost_matrix)

        matched_distances = distance_matrix[
            pred_indices,
            gt_indices
        ]

        valid_matches = (
            matched_distances <=
            self.distance_threshold
        )

        tp = int(valid_matches.sum())

        distance_sum = float(
            matched_distances[
                valid_matches
            ].sum())

        return tp, distance_sum

    def process(
            self,
            data_batch: dict,
            data_samples: List) -> None:
        """Process one validation/test batch."""
    
        for data_sample in data_samples:
    
            # MMEngine Evaluator normally passes dict-form data samples.
            # Keep the object-form branch for direct/manual unit tests.
            if isinstance(data_sample, dict):
                pred_instances = data_sample['pred_instances']
                gt_instances = data_sample['gt_instances']
    
                pred_points = pred_instances['points']
                pred_scores = pred_instances['scores']
                gt_points = gt_instances['points']
    
            else:
                pred_instances = data_sample.pred_instances
                gt_instances = data_sample.gt_instances
    
                pred_points = pred_instances.points
                pred_scores = pred_instances.scores
                gt_points = gt_instances.points
    
            pred_points = (
                pred_points.detach().cpu().numpy()
            )
    
            pred_scores = (
                pred_scores.detach().cpu().numpy()
            )
    
            gt_points = (
                gt_points.detach().cpu().numpy()
            )
    
            # Confidence filtering.
            keep = (
                pred_scores >=
                self.score_threshold
            )
    
            pred_points = pred_points[keep]
    
            tp, distance_sum = self._match_points(
                pred_points,
                gt_points)
    
            fp = len(pred_points) - tp
            fn = len(gt_points) - tp
    
            self.results.append(
                dict(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    distance_sum=distance_sum
                )
            )

    def compute_metrics(
            self,
            results: List[dict]) -> Dict[str, float]:
        """Compute dataset-level point metrics."""

        tp = sum(
            result['tp']
            for result in results)

        fp = sum(
            result['fp']
            for result in results)

        fn = sum(
            result['fn']
            for result in results)

        distance_sum = sum(
            result['distance_sum']
            for result in results)

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        mean_localization_error = (
            distance_sum / tp
            if tp > 0
            else 0.0
        )

        return dict(
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            mean_localization_error=float(
                mean_localization_error),
            tp=int(tp),
            fp=int(fp),
            fn=int(fn)
        )