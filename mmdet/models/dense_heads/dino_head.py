# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.structures.bbox import (bbox_cxcywh_to_xyxy, bbox_overlaps,
                                   bbox_xyxy_to_cxcywh)
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from ..losses import QualityFocalLoss
from ..utils import multi_apply
from .deformable_detr_head import DeformableDETRHead

import copy

import torch.nn as nn
from mmcv.cnn import Linear



@MODELS.register_module()
class DINOHead(DeformableDETRHead):
    r"""Head of the DINO: DETR with Improved DeNoising Anchor Boxes
    for End-to-End Object Detection

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    More details can be found in the `paper
    <https://arxiv.org/abs/2203.03605>`_ .
    """

    def _init_layers(self) -> None:
        """Initialize classification and 2D point regression branches."""
    
        fc_cls = Linear(
            self.embed_dims,
            self.cls_out_channels)
    
        reg_branch = []
    
        for _ in range(self.num_reg_fcs):
            reg_branch.append(
                Linear(
                    self.embed_dims,
                    self.embed_dims))
            reg_branch.append(nn.ReLU())
    
        # Point-DINO Stage 2:
        # regress only (dx, dy), no (dw, dh).
        reg_branch.append(
            Linear(
                self.embed_dims,
                2))
    
        reg_branch = nn.Sequential(*reg_branch)
    
        if self.share_pred_layer:
            self.cls_branches = nn.ModuleList(
                [
                    fc_cls
                    for _ in range(self.num_pred_layer)
                ])
    
            self.reg_branches = nn.ModuleList(
                [
                    reg_branch
                    for _ in range(self.num_pred_layer)
                ])
    
        else:
            self.cls_branches = nn.ModuleList(
                [
                    copy.deepcopy(fc_cls)
                    for _ in range(self.num_pred_layer)
                ])
    
            self.reg_branches = nn.ModuleList(
                [
                    copy.deepcopy(reg_branch)
                    for _ in range(self.num_pred_layer)
                ])


    def forward(self, hidden_states: Tensor,
                references: List[Tensor]) -> Tuple[Tensor, Tensor]:
        """Forward the DINO head and keep only point coordinates.
    
        The internal DINO regression branches still predict 4D
        (cx, cy, w, h) for decoder refinement and two-stage proposals.
        For the external prediction task, only (cx, cy) is returned.
        """
        all_layers_cls_scores, all_layers_bbox_preds = super().forward(
            hidden_states, references)
    
        all_layers_point_preds = all_layers_bbox_preds[..., :2]
    
        return all_layers_cls_scores, all_layers_point_preds
        
    def _predict_by_feat_single(self,
                                cls_score: Tensor,
                                point_pred: Tensor,
                                img_meta: dict,
                                rescale: bool = True) -> InstanceData:
        """Convert Point-DINO outputs into point predictions."""
    
        assert len(cls_score) == len(point_pred)
    
        # Stage-1 Point-DINO uses sigmoid Focal classification.
        assert self.loss_cls.use_sigmoid
    
        max_per_img = self.test_cfg.get(
            'max_per_img', len(cls_score))
    
        # Classification score of every query.
        cls_score = cls_score.sigmoid()
    
        scores, indexes = cls_score.reshape(-1).topk(
            max_per_img)
    
        det_labels = indexes % self.num_classes
        point_indexes = indexes // self.num_classes
    
        # Select corresponding point predictions.
        det_points = point_pred[point_indexes].clone()
    
        # Normalized (x, y) -> resized image pixel coordinates.
        img_h, img_w = img_meta['img_shape']
    
        det_points[:, 0] *= img_w
        det_points[:, 1] *= img_h
    
        # Resize-space -> original-image space.
        if rescale:
            assert img_meta.get('scale_factor') is not None
    
            scale_factor = det_points.new_tensor(
                img_meta['scale_factor'])
    
            det_points /= scale_factor
    
        results = InstanceData()
        results.points = det_points
        results.scores = scores
        results.labels = det_labels
    
        return results

    def _get_targets_single(self, cls_score: Tensor, point_pred: Tensor,
                            gt_instances: InstanceData,
                            img_meta: dict) -> tuple:
        """Compute classification and point targets for one image."""
    
        num_points = point_pred.size(0)
    
        # Predicted points are normalized (x, y) in [0, 1].
        pred_instances = InstanceData(
            scores=cls_score,
            points=point_pred)
    
        # Hungarian matching:
        # FocalLossCost + PointL1Cost
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            img_meta=img_meta)
    
        gt_points = gt_instances.points
        gt_labels = gt_instances.labels
    
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0,
            as_tuple=False).squeeze(-1).unique()
    
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0,
            as_tuple=False).squeeze(-1).unique()
    
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1
        pos_gt_points = gt_points[pos_assigned_gt_inds.long(), :]
    
        # Classification targets.
        labels = gt_points.new_full(
            (num_points, ),
            self.num_classes,
            dtype=torch.long)
    
        labels[pos_inds] = gt_labels[pos_assigned_gt_inds]
    
        label_weights = gt_points.new_ones(num_points)
    
        # Point regression targets.
        point_targets = torch.zeros_like(
            point_pred,
            dtype=gt_points.dtype)
    
        point_weights = torch.zeros_like(
            point_pred,
            dtype=gt_points.dtype)
    
        point_weights[pos_inds] = 1.0
    
        # GT points are pixel coordinates.
        # Convert them to normalized (x, y) coordinates.
        img_h, img_w = img_meta['img_shape']
        factor = gt_points.new_tensor(
            [img_w, img_h]).unsqueeze(0)
    
        pos_gt_points_normalized = pos_gt_points / factor
    
        point_targets[pos_inds] = pos_gt_points_normalized
    
        return (labels, label_weights,
                point_targets, point_weights,
                pos_inds, neg_inds)

    def loss_by_feat_single(self,
                            cls_scores: Tensor,
                            point_preds: Tensor,
                            batch_gt_instances: InstanceList,
                            batch_img_metas: List[dict]) -> Tuple[Tensor, Tensor]:
        """Compute classification loss and point L1 loss for one decoder layer."""
    
        num_imgs = cls_scores.size(0)
    
        cls_scores_list = [
            cls_scores[i] for i in range(num_imgs)
        ]
        point_preds_list = [
            point_preds[i] for i in range(num_imgs)
        ]
    
        cls_reg_targets = self.get_targets(
            cls_scores_list,
            point_preds_list,
            batch_gt_instances,
            batch_img_metas)
    
        (labels_list,
         label_weights_list,
         point_targets_list,
         point_weights_list,
         num_total_pos,
         num_total_neg) = cls_reg_targets
    
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        point_targets = torch.cat(point_targets_list, 0)
        point_weights = torch.cat(point_weights_list, 0)
    
        # Classification loss: keep original DINO Focal Loss.
        cls_scores = cls_scores.reshape(
            -1, self.cls_out_channels)
    
        cls_avg_factor = (
            num_total_pos * 1.0 +
            num_total_neg * self.bg_cls_weight)
    
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
    
        cls_avg_factor = max(cls_avg_factor, 1)
    
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=cls_avg_factor)
    
        # Average positive point count across GPUs.
        num_total_pos = loss_cls.new_tensor(
            [num_total_pos])
    
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos),
            min=1).item()
    
        # Point L1 loss.
        point_preds = point_preds.reshape(-1, 2)
    
        loss_point = self.loss_bbox(
            point_preds,
            point_targets,
            point_weights,
            avg_factor=num_total_pos)
    
        return loss_cls, loss_point

    def loss(self, hidden_states: Tensor, references: List[Tensor],
             enc_outputs_class: Tensor, enc_outputs_coord: Tensor,
             batch_data_samples: SampleList, dn_meta: Dict[str, int]) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the queries of the upstream network.

        Args:
            hidden_states (Tensor): Hidden states output from each decoder
                layer, has shape (num_decoder_layers, bs, num_queries_total,
                dim), where `num_queries_total` is the sum of
                `num_denoising_queries` and `num_matching_queries` when
                `self.training` is `True`, else `num_matching_queries`.
            references (list[Tensor]): List of the reference from the decoder.
                The first reference is the `init_reference` (initial) and the
                other num_decoder_layers(6) references are `inter_references`
                (intermediate). The `init_reference` has shape (bs,
                num_queries_total, 4) and each `inter_reference` has shape
                (bs, num_queries, 4) with the last dimension arranged as
                (cx, cy, w, h).
            enc_outputs_class (Tensor): The score of each point on encode
                feature map, has shape (bs, num_feat_points, cls_out_channels).
            enc_outputs_coord (Tensor): The proposal generate from the
                encode feature map, has shape (bs, num_feat_points, 4) with the
                last dimension arranged as (cx, cy, w, h).
            batch_data_samples (list[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.

        Returns:
            dict: A dictionary of loss components.
        """
        batch_gt_instances = []
        batch_img_metas = []
        for data_sample in batch_data_samples:
            batch_img_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances)

        outs = self(hidden_states, references)
        loss_inputs = outs + (enc_outputs_class, enc_outputs_coord,
                              batch_gt_instances, batch_img_metas, dn_meta)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
            self,
            all_layers_cls_scores: Tensor,
            all_layers_point_preds: Tensor,
            enc_cls_scores: Tensor,
            enc_bbox_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int],
            batch_gt_instances_ignore: OptInstanceList = None
    ) -> Dict[str, Tensor]:
        """Loss for Point-DINO."""
    
        assert batch_gt_instances_ignore is None
    
        # ---------------------------------------------------------
        # Split matching queries and denoising queries.
        # ---------------------------------------------------------
        (all_layers_matching_cls_scores,
         all_layers_matching_point_preds,
         all_layers_denoising_cls_scores,
         all_layers_denoising_point_preds) = self.split_outputs(
             all_layers_cls_scores,
             all_layers_point_preds,
             dn_meta)
    
        # ---------------------------------------------------------
        # 1. Matching-query losses
        # ---------------------------------------------------------
        losses_cls, losses_point = multi_apply(
            self.loss_by_feat_single,
            all_layers_matching_cls_scores,
            all_layers_matching_point_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas)
    
        loss_dict = dict()
    
        # Last decoder layer.
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_point'] = losses_point[-1]
    
        # Auxiliary losses from previous decoder layers.
        for num_dec_layer, (loss_cls_i, loss_point_i) in enumerate(
                zip(losses_cls[:-1], losses_point[:-1])):
    
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_point'] = loss_point_i
    
        # ---------------------------------------------------------
        # 2. Encoder two-stage point proposal loss
        # ---------------------------------------------------------
        if enc_cls_scores is not None:
    
            # Stage 2:
            # encoder proposal is already pure 2D (x, y).
            # Keeping [:2] here is harmless and avoids a larger
            # variable-renaming cleanup at this stage.
            enc_point_preds = enc_bbox_preds[..., :2]
    
            enc_loss_cls, enc_loss_point = self.loss_by_feat_single(
                enc_cls_scores,
                enc_point_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas)
    
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_point'] = enc_loss_point
    
        # ---------------------------------------------------------
        # 3. Point denoising losses
        # ---------------------------------------------------------
        if all_layers_denoising_cls_scores is not None:
    
            assert all_layers_denoising_point_preds is not None
            assert dn_meta is not None
    
            dn_losses_cls, dn_losses_point = self.loss_dn(
                all_layers_denoising_cls_scores,
                all_layers_denoising_point_preds,
                batch_gt_instances=batch_gt_instances,
                batch_img_metas=batch_img_metas,
                dn_meta=dn_meta)
    
            # Last decoder layer DN loss.
            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_point'] = dn_losses_point[-1]
    
            # Auxiliary DN losses from previous decoder layers.
            for num_dec_layer, (loss_cls_i, loss_point_i) in enumerate(
                    zip(dn_losses_cls[:-1], dn_losses_point[:-1])):
    
                loss_dict[f'd{num_dec_layer}.dn_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_dec_layer}.dn_loss_point'] = loss_point_i
    
        return loss_dict


    def loss_dn(
            self,
            all_layers_denoising_cls_scores: Tensor,
            all_layers_denoising_point_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int]) -> Tuple[List[Tensor]]:
        """Calculate Point-DN losses."""
    
        return multi_apply(
            self._loss_dn_single,
            all_layers_denoising_cls_scores,
            all_layers_denoising_point_preds,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=batch_img_metas,
            dn_meta=dn_meta)

    def _loss_dn_single(
            self,
            dn_cls_scores: Tensor,
            dn_point_preds: Tensor,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int]) -> Tuple[Tensor, Tensor]:
        """Point denoising loss for one decoder layer."""
    
        targets = self.get_dn_targets(
            batch_gt_instances,
            batch_img_metas,
            dn_meta)
    
        (
            labels_list,
            label_weights_list,
            point_targets_list,
            point_weights_list,
            num_total_pos,
            num_total_neg
        ) = targets
    
        labels = torch.cat(
            labels_list,
            0)
    
        label_weights = torch.cat(
            label_weights_list,
            0)
    
        point_targets = torch.cat(
            point_targets_list,
            0)
    
        point_weights = torch.cat(
            point_weights_list,
            0)
    
        # ---------------------------------------------------------
        # Classification loss
        # ---------------------------------------------------------
        cls_scores = dn_cls_scores.reshape(
            -1,
            self.cls_out_channels)
    
        cls_avg_factor = (
            num_total_pos * 1.0
            + num_total_neg
            * self.bg_cls_weight
        )
    
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor(
                    [cls_avg_factor]))
    
        cls_avg_factor = max(
            cls_avg_factor,
            1)
    
        if len(cls_scores) > 0:
            loss_cls = self.loss_cls(
                cls_scores,
                labels,
                label_weights,
                avg_factor=cls_avg_factor)
    
        else:
            loss_cls = torch.zeros(
                1,
                dtype=cls_scores.dtype,
                device=cls_scores.device)
    
        # ---------------------------------------------------------
        # Normalize point regression loss
        # ---------------------------------------------------------
        num_total_pos = loss_cls.new_tensor(
            [num_total_pos])
    
        num_total_pos = torch.clamp(
            reduce_mean(num_total_pos),
            min=1).item()
    
        # ---------------------------------------------------------
        # Point L1 Loss
        # ---------------------------------------------------------
        point_preds = dn_point_preds.reshape(
            -1,
            2)
    
        loss_point = self.loss_bbox(
            point_preds,
            point_targets,
            point_weights,
            avg_factor=num_total_pos)
    
        return loss_cls, loss_point

    def get_dn_targets(
            self,
            batch_gt_instances: InstanceList,
            batch_img_metas: List[dict],
            dn_meta: Dict[str, int]) -> tuple:
        """Get Point-DN targets for a batch."""
    
        (
            labels_list,
            label_weights_list,
            point_targets_list,
            point_weights_list,
            pos_inds_list,
            neg_inds_list
        ) = multi_apply(
            self._get_dn_targets_single,
            batch_gt_instances,
            batch_img_metas,
            dn_meta=dn_meta)
    
        num_total_pos = sum(
            inds.numel()
            for inds in pos_inds_list)
    
        num_total_neg = sum(
            inds.numel()
            for inds in neg_inds_list)
    
        return (
            labels_list,
            label_weights_list,
            point_targets_list,
            point_weights_list,
            num_total_pos,
            num_total_neg)

    def _get_dn_targets_single(
            self,
            gt_instances: InstanceData,
            img_meta: dict,
            dn_meta: Dict[str, int]) -> tuple:
        """Get Point-DN targets for one image."""
    
        gt_points = gt_instances.points
        gt_labels = gt_instances.labels
    
        num_groups = dn_meta['num_denoising_groups']
        num_denoising_queries = \
            dn_meta['num_denoising_queries']
    
        num_queries_each_group = int(
            num_denoising_queries / num_groups)
    
        device = gt_points.device
    
        # ---------------------------------------------------------
        # Positive DN query indices
        # ---------------------------------------------------------
        if len(gt_labels) > 0:
    
            t = torch.arange(
                len(gt_labels),
                dtype=torch.long,
                device=device)
    
            # Which GT each positive DN query corresponds to
            pos_assigned_gt_inds = \
                t.unsqueeze(0).repeat(
                    num_groups, 1).flatten()
    
            # Each DN group:
            #
            # [positive queries | negative queries]
            #
            # Positive queries start at the beginning of each group.
            pos_inds = torch.arange(
                num_groups,
                dtype=torch.long,
                device=device)
    
            pos_inds = (
                pos_inds.unsqueeze(1)
                * num_queries_each_group
                + t
            ).flatten()
    
        else:
            pos_inds = gt_points.new_tensor(
                [],
                dtype=torch.long)
    
            pos_assigned_gt_inds = \
                gt_points.new_tensor(
                    [],
                    dtype=torch.long)
    
        # Negative DN queries occupy the second half of each group.
        neg_inds = (
            pos_inds
            + num_queries_each_group // 2
        )
    
        # ---------------------------------------------------------
        # Classification targets
        # ---------------------------------------------------------
        labels = gt_points.new_full(
            (num_denoising_queries,),
            self.num_classes,
            dtype=torch.long)
    
        labels[pos_inds] = \
            gt_labels[pos_assigned_gt_inds]
    
        label_weights = gt_points.new_ones(
            num_denoising_queries)
    
        # ---------------------------------------------------------
        # Point regression targets
        # ---------------------------------------------------------
        point_targets = torch.zeros(
            num_denoising_queries,
            2,
            device=device,
            dtype=gt_points.dtype)
    
        point_weights = torch.zeros(
            num_denoising_queries,
            2,
            device=device,
            dtype=gt_points.dtype)
    
        # Only positive DN queries regress GT points.
        point_weights[pos_inds] = 1.0
    
        # Pixel coordinates -> normalized point coordinates
        img_h, img_w = img_meta['img_shape']
    
        factor = gt_points.new_tensor(
            [img_w, img_h]).unsqueeze(0)
    
        gt_points_normalized = \
            gt_points / factor
    
        if len(gt_labels) > 0:
            point_targets[pos_inds] = \
                gt_points_normalized[
                    pos_assigned_gt_inds]
    
        return (
            labels,
            label_weights,
            point_targets,
            point_weights,
            pos_inds,
            neg_inds)

    @staticmethod
    def split_outputs(all_layers_cls_scores: Tensor,
                      all_layers_bbox_preds: Tensor,
                      dn_meta: Dict[str, int]) -> Tuple[Tensor]:
        """Split outputs of the denoising part and the matching part.

        For the total outputs of `num_queries_total` length, the former
        `num_denoising_queries` outputs are from denoising queries, and
        the rest `num_matching_queries` ones are from matching queries,
        where `num_queries_total` is the sum of `num_denoising_queries` and
        `num_matching_queries`.

        Args:
            all_layers_cls_scores (Tensor): Classification scores of all
                decoder layers, has shape (num_decoder_layers, bs,
                num_queries_total, cls_out_channels).
            all_layers_bbox_preds (Tensor): Regression outputs of all decoder
                layers. Each is a 4D-tensor with normalized coordinate format
                (cx, cy, w, h) and has shape (num_decoder_layers, bs,
                num_queries_total, 4).
            dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'.

        Returns:
            Tuple[Tensor]: a tuple containing the following outputs.

            - all_layers_matching_cls_scores (Tensor): Classification scores
              of all decoder layers in matching part, has shape
              (num_decoder_layers, bs, num_matching_queries, cls_out_channels).
            - all_layers_matching_bbox_preds (Tensor): Regression outputs of
              all decoder layers in matching part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_matching_queries, 4).
            - all_layers_denoising_cls_scores (Tensor): Classification scores
              of all decoder layers in denoising part, has shape
              (num_decoder_layers, bs, num_denoising_queries,
              cls_out_channels).
            - all_layers_denoising_bbox_preds (Tensor): Regression outputs of
              all decoder layers in denoising part. Each is a 4D-tensor with
              normalized coordinate format (cx, cy, w, h) and has shape
              (num_decoder_layers, bs, num_denoising_queries, 4).
        """
        if dn_meta is not None:
            num_denoising_queries = dn_meta['num_denoising_queries']
            
            all_layers_denoising_cls_scores = \
                all_layers_cls_scores[:, :, : num_denoising_queries, :]
            all_layers_denoising_bbox_preds = \
                all_layers_bbox_preds[:, :, : num_denoising_queries, :]
            all_layers_matching_cls_scores = \
                all_layers_cls_scores[:, :, num_denoising_queries:, :]
            all_layers_matching_bbox_preds = \
                all_layers_bbox_preds[:, :, num_denoising_queries:, :]
        else:
            all_layers_denoising_cls_scores = None
            all_layers_denoising_bbox_preds = None
            all_layers_matching_cls_scores = all_layers_cls_scores
            all_layers_matching_bbox_preds = all_layers_bbox_preds
        return (all_layers_matching_cls_scores, all_layers_matching_bbox_preds,
                all_layers_denoising_cls_scores,
                all_layers_denoising_bbox_preds)
