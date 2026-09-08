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
            distance_thresholds=(4.0, 8.0),
            score_threshold: float = 0.5,
            collect_device: str = 'cpu',
            prefix: Optional[str] = None) -> None:
    
        super().__init__(
            collect_device=collect_device,
            prefix=prefix)
    
        self.distance_thresholds = [
            float(x) for x in distance_thresholds
        ]
    
        for threshold in self.distance_thresholds:
            if threshold <= 0:
                raise ValueError(
                    'distance thresholds must be > 0.')
    
        if not 0.0 <= score_threshold <= 1.0:
            raise ValueError(
                'score_threshold must be in [0, 1].')
    
        self.score_threshold = float(score_threshold)

    def _match_points(
            self,
            pred_points: np.ndarray,
            gt_points: np.ndarray,
            distance_threshold: float):
    
        num_pred = len(pred_points)
        num_gt = len(gt_points)
    
        if num_pred == 0 or num_gt == 0:
            return 0, 0.0
    
        distance_matrix = np.linalg.norm(
            pred_points[:, None, :] -
            gt_points[None, :, :],
            axis=-1)
    
        num_pairs = min(num_pred, num_gt)
    
        invalid_cost = (
            (num_pairs + 1) *
            max(distance_threshold, 1.0)
            + 1.0
        )
    
        cost_matrix = np.where(
            distance_matrix <= distance_threshold,
            distance_matrix,
            invalid_cost)
    
        pred_indices, gt_indices = \
            linear_sum_assignment(cost_matrix)
    
        matched_distances = distance_matrix[
            pred_indices,
            gt_indices
        ]
    
        valid_matches = (
            matched_distances <= distance_threshold
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
    
            pred_points = pred_points.detach().cpu().numpy()
            pred_scores = pred_scores.detach().cpu().numpy()
            gt_points = gt_points.detach().cpu().numpy()
    
            # Confidence filtering
            keep = pred_scores >= self.score_threshold
            pred_points = pred_points[keep]
    
            image_result = {}
    
            # Evaluate the same predictions at 4 px and 8 px.
            for threshold in self.distance_thresholds:
    
                tp, distance_sum = self._match_points(
                    pred_points,
                    gt_points,
                    threshold)
    
                fp = len(pred_points) - tp
                fn = len(gt_points) - tp
    
                key = str(int(threshold))
    
                image_result[key] = dict(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    distance_sum=distance_sum)
    
            self.results.append(image_result)

    def compute_metrics(
            self,
            results: List[dict]) -> Dict[str, float]:
        """Compute point metrics at multiple distance thresholds."""
    
        metrics = {}
    
        for threshold in self.distance_thresholds:
    
            key = str(int(threshold))
    
            tp = sum(
                result[key]['tp']
                for result in results)
    
            fp = sum(
                result[key]['fp']
                for result in results)
    
            fn = sum(
                result[key]['fn']
                for result in results)
    
            distance_sum = sum(
                result[key]['distance_sum']
                for result in results)
    
            precision = (
                tp / (tp + fp)
                if (tp + fp) > 0
                else 0.0)
    
            recall = (
                tp / (tp + fn)
                if (tp + fn) > 0
                else 0.0)
    
            f1 = (
                2.0 * precision * recall /
                (precision + recall)
                if (precision + recall) > 0
                else 0.0)
    
            mean_localization_error = (
                distance_sum / tp
                if tp > 0
                else 0.0)
    
            metrics[f'precision@{key}px'] = float(precision)
            metrics[f'recall@{key}px'] = float(recall)
            metrics[f'f1@{key}px'] = float(f1)
    
            metrics[f'mean_localization_error@{key}px'] = \
                float(mean_localization_error)
    
            metrics[f'tp@{key}px'] = int(tp)
            metrics[f'fp@{key}px'] = int(fp)
            metrics[f'fn@{key}px'] = int(fn)
    
        return metrics