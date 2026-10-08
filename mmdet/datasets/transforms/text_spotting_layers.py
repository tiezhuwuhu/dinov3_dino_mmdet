import math
from typing import Optional, Sequence, Tuple

import torch
from mmcv.ops import RoIAlign
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures.bbox import (
    bbox2roi,
    bbox_cxcywh_to_xyxy,
)


@MODELS.register_module()
class TaskAwareRecognitionQueryInitializer(nn.Module):
    """Initialize ordered recognition queries from proposal regions.

    This module follows the Task-Aware Query Initialization idea used by
    ESTextSpotter.

    For each text proposal:

        proposal bbox
            -> multi-level ROIAlign
            -> [C, roi_height, num_rec_queries]
            -> average over roi_height
            -> [num_rec_queries, C]

    The recognition sequence length is independent from the character
    vocabulary. Therefore the same module can be reused for English,
    Chinese, multilingual, or other character sets.

    Args:
        feat_channels: Number of channels in input feature maps.
        embed_dims: Output recognition-query dimension.
        num_rec_queries: Number of ordered recognition queries per instance.
        roi_height: Height of the ROIAlign output before height aggregation.
        featmap_strides: Feature strides used for ROI extraction.
        sampling_ratio: ROIAlign sampling ratio.
        aligned: Whether to use aligned ROIAlign. True corresponds to
            Detectron2 ROIAlignV2 behavior.
        canonical_box_size: Canonical box size for FPN level assignment.
        canonical_level: Canonical FPN level for level assignment.
        box_format: Proposal box format. Supported values are "cxcywh"
            and "xyxy".
        normalized_boxes: Whether proposal boxes use normalized coordinates.
        detach_boxes: Whether proposal coordinates are detached before
            ROI extraction.
    """

    def __init__(
        self,
        feat_channels: int = 256,
        embed_dims: int = 256,
        num_rec_queries: int = 25,
        roi_height: int = 8,
        featmap_strides: Sequence[int] = (8, 16, 32),
        sampling_ratio: int = 1,
        aligned: bool = True,
        canonical_box_size: int = 224,
        canonical_level: int = 4,
        box_format: str = 'cxcywh',
        normalized_boxes: bool = True,
        detach_boxes: bool = True,
    ) -> None:
        super().__init__()

        if feat_channels <= 0:
            raise ValueError(
                'feat_channels must be positive.'
            )

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if num_rec_queries <= 0:
            raise ValueError(
                'num_rec_queries must be positive.'
            )

        if roi_height <= 0:
            raise ValueError(
                'roi_height must be positive.'
            )

        if len(featmap_strides) == 0:
            raise ValueError(
                'featmap_strides must not be empty.'
            )

        if canonical_box_size <= 0:
            raise ValueError(
                'canonical_box_size must be positive.'
            )

        if box_format not in {
            'cxcywh',
            'xyxy',
        }:
            raise ValueError(
                'box_format must be "cxcywh" or "xyxy".'
            )

        self.feat_channels = int(
            feat_channels
        )

        self.embed_dims = int(
            embed_dims
        )

        self.num_rec_queries = int(
            num_rec_queries
        )

        self.roi_height = int(
            roi_height
        )

        self.featmap_strides = tuple(
            int(stride)
            for stride in featmap_strides
        )

        self.sampling_ratio = int(
            sampling_ratio
        )

        self.aligned = bool(
            aligned
        )

        self.canonical_box_size = float(
            canonical_box_size
        )

        self.canonical_level = int(
            canonical_level
        )

        self.box_format = box_format

        self.normalized_boxes = bool(
            normalized_boxes
        )

        self.detach_boxes = bool(
            detach_boxes
        )

        self._validate_featmap_strides()

        self.min_level = int(
            math.log2(
                self.featmap_strides[0]
            )
        )

        self.max_level = int(
            math.log2(
                self.featmap_strides[-1]
            )
        )

        self.roi_align_layers = (
            nn.ModuleList(
                [
                    RoIAlign(
                        output_size=(
                            self.roi_height,
                            self.num_rec_queries,
                        ),
                        spatial_scale=(
                            1.0 / stride
                        ),
                        sampling_ratio=(
                            self.sampling_ratio
                        ),
                        pool_mode='avg',
                        aligned=self.aligned,
                    )
                    for stride
                    in self.featmap_strides
                ]
            )
        )

        if (
            self.feat_channels
            == self.embed_dims
        ):
            self.output_proj = (
                nn.Identity()
            )
        else:
            self.output_proj = (
                nn.Linear(
                    self.feat_channels,
                    self.embed_dims,
                )
            )

            nn.init.xavier_uniform_(
                self.output_proj.weight
            )

            if (
                self.output_proj.bias
                is not None
            ):
                nn.init.constant_(
                    self.output_proj.bias,
                    0.0,
                )

    def _validate_featmap_strides(
        self,
    ) -> None:
        levels = []

        for stride in self.featmap_strides:
            if stride <= 0:
                raise ValueError(
                    'All featmap_strides '
                    'must be positive.'
                )

            level = math.log2(
                stride
            )

            if not math.isclose(
                level,
                round(level),
            ):
                raise ValueError(
                    'featmap_strides must '
                    'be powers of two.'
                )

            levels.append(
                int(round(level))
            )

        for previous, current in zip(
            levels[:-1],
            levels[1:],
        ):
            if (
                current
                != previous + 1
            ):
                raise ValueError(
                    'featmap_strides must form '
                    'a contiguous feature pyramid.'
                )

    def _validate_features(
        self,
        feats: Sequence[Tensor],
    ) -> Tuple[Tensor, ...]:
        num_levels = len(
            self.featmap_strides
        )

        if len(feats) < num_levels:
            raise ValueError(
                'Not enough feature maps: '
                f'got {len(feats)}, '
                f'but {num_levels} are required.'
            )

        selected_feats = tuple(
            feats[:num_levels]
        )

        batch_size = (
            selected_feats[0].shape[0]
        )

        for level, feat in enumerate(
            selected_feats
        ):
            if feat.ndim != 4:
                raise ValueError(
                    'Feature maps must have '
                    'shape [B, C, H, W].'
                )

            if (
                feat.shape[0]
                != batch_size
            ):
                raise ValueError(
                    'All feature maps must '
                    'have the same batch size.'
                )

            if (
                feat.shape[1]
                != self.feat_channels
            ):
                raise ValueError(
                    'Unexpected channel count '
                    f'at feature level {level}: '
                    f'got {feat.shape[1]}, '
                    f'expected '
                    f'{self.feat_channels}.'
                )

        return selected_feats

    def _convert_boxes_to_xyxy(
        self,
        proposal_boxes: Tensor,
        image_shapes: Optional[
            Sequence[Sequence[int]]
        ],
    ) -> Tensor:
        if proposal_boxes.ndim != 3:
            raise ValueError(
                'proposal_boxes must have '
                'shape [B, N, 4].'
            )

        if (
            proposal_boxes.shape[-1]
            != 4
        ):
            raise ValueError(
                'The last dimension of '
                'proposal_boxes must be 4.'
            )

        boxes = proposal_boxes

        if self.detach_boxes:
            boxes = boxes.detach()

        if self.box_format == 'cxcywh':
            boxes = (
                bbox_cxcywh_to_xyxy(
                    boxes
                )
            )
        else:
            boxes = boxes.clone()

        if self.normalized_boxes:
            if image_shapes is None:
                raise ValueError(
                    'image_shapes are required '
                    'for normalized boxes.'
                )

            if (
                len(image_shapes)
                != boxes.shape[0]
            ):
                raise ValueError(
                    'image_shapes batch size '
                    'does not match '
                    'proposal_boxes.'
                )

            absolute_boxes = []

            for batch_index in range(
                boxes.shape[0]
            ):
                image_shape = (
                    image_shapes[
                        batch_index
                    ]
                )

                if len(image_shape) < 2:
                    raise ValueError(
                        'Each image shape must '
                        'contain height and width.'
                    )

                image_height = int(
                    image_shape[0]
                )

                image_width = int(
                    image_shape[1]
                )

                if (
                    image_height <= 0
                    or image_width <= 0
                ):
                    raise ValueError(
                        'Image dimensions must '
                        'be positive.'
                    )

                scale = boxes.new_tensor(
                    [
                        image_width,
                        image_height,
                        image_width,
                        image_height,
                    ]
                )

                batch_boxes = (
                    boxes[batch_index]
                    * scale
                )

                batch_boxes = (
                    batch_boxes.clamp_min(
                        0.0
                    )
                )

                absolute_boxes.append(
                    batch_boxes
                )

            boxes = torch.stack(
                absolute_boxes,
                dim=0,
            )

        return boxes

    def _assign_boxes_to_levels(
        self,
        rois: Tensor,
    ) -> Tensor:
        """Assign every ROI to one feature-pyramid level."""

        if rois.numel() == 0:
            return rois.new_empty(
                (0,),
                dtype=torch.long,
            )

        widths = (
            rois[:, 3]
            - rois[:, 1]
        ).clamp_min(0.0)

        heights = (
            rois[:, 4]
            - rois[:, 2]
        ).clamp_min(0.0)

        box_sizes = torch.sqrt(
            widths * heights
        )

        target_levels = torch.floor(
            self.canonical_level
            + torch.log2(
                box_sizes
                / self.canonical_box_size
                + 1e-8
            )
        )

        target_levels = (
            target_levels.clamp(
                min=self.min_level,
                max=self.max_level,
            )
        )

        return (
            target_levels.to(
                dtype=torch.long
            )
            - self.min_level
        )

    def _multi_level_roi_align(
        self,
        feats: Tuple[Tensor, ...],
        boxes_xyxy: Tensor,
    ) -> Tensor:
        batch_size = (
            boxes_xyxy.shape[0]
        )

        num_instances = (
            boxes_xyxy.shape[1]
        )

        box_list = [
            boxes_xyxy[
                batch_index
            ]
            for batch_index
            in range(batch_size)
        ]

        rois = bbox2roi(
            box_list
        )

        rois = rois.to(
            device=feats[0].device,
            dtype=feats[0].dtype,
        )

        num_rois = rois.shape[0]

        output_shape = (
            num_rois,
            self.feat_channels,
            self.roi_height,
            self.num_rec_queries,
        )

        roi_features = (
            feats[0].new_zeros(
                output_shape
            )
        )

        if num_rois == 0:
            return roi_features

        target_levels = (
            self._assign_boxes_to_levels(
                rois
            )
        )

        for level_index, (
            feat,
            roi_align,
        ) in enumerate(
            zip(
                feats,
                self.roi_align_layers,
            )
        ):
            indices = torch.nonzero(
                target_levels
                == level_index,
                as_tuple=False,
            ).squeeze(1)

            if indices.numel() > 0:
                level_rois = rois[
                    indices
                ]

                level_features = (
                    roi_align(
                        feat,
                        level_rois,
                    )
                )

                roi_features[
                    indices
                ] = level_features
            else:
                roi_features = (
                    roi_features
                    + feat.sum() * 0.0
                )

        expected_num_rois = (
            batch_size
            * num_instances
        )

        if (
            roi_features.shape[0]
            != expected_num_rois
        ):
            raise RuntimeError(
                'Unexpected number of '
                'ROI features.'
            )

        return roi_features

    def forward(
        self,
        feats: Sequence[Tensor],
        proposal_boxes: Tensor,
        image_shapes: Optional[
            Sequence[Sequence[int]]
        ] = None,
        valid_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Build ordered recognition queries.

        Args:
            feats:
                Multi-scale feature maps. Each tensor has shape
                [B, C, H, W].

            proposal_boxes:
                Proposal boxes with shape [B, N, 4].

            image_shapes:
                Processed image shapes before batch padding. Required when
                normalized_boxes=True. Each item starts with (height, width).

            valid_mask:
                Optional boolean tensor with shape [B, N]. True means that
                the proposal is valid. Invalid recognition queries are zeroed.

        Returns:
            Tensor:
                Recognition queries with shape
                [B, N, num_rec_queries, embed_dims].
        """

        selected_feats = (
            self._validate_features(
                feats
            )
        )

        batch_size = (
            selected_feats[0].shape[0]
        )

        if (
            proposal_boxes.shape[0]
            != batch_size
        ):
            raise ValueError(
                'Feature batch size and '
                'proposal batch size '
                'do not match.'
            )

        num_instances = (
            proposal_boxes.shape[1]
        )

        boxes_xyxy = (
            self._convert_boxes_to_xyxy(
                proposal_boxes=(
                    proposal_boxes
                ),
                image_shapes=(
                    image_shapes
                ),
            )
        )

        roi_features = (
            self._multi_level_roi_align(
                feats=selected_feats,
                boxes_xyxy=boxes_xyxy,
            )
        )

        recognition_queries = (
            roi_features.mean(
                dim=2
            )
        )

        recognition_queries = (
            recognition_queries
            .permute(
                0,
                2,
                1,
            )
            .contiguous()
        )

        recognition_queries = (
            recognition_queries.reshape(
                batch_size,
                num_instances,
                self.num_rec_queries,
                self.feat_channels,
            )
        )

        recognition_queries = (
            self.output_proj(
                recognition_queries
            )
        )

        if valid_mask is not None:
            if (
                valid_mask.shape
                != (
                    batch_size,
                    num_instances,
                )
            ):
                raise ValueError(
                    'valid_mask must have '
                    'shape [B, N].'
                )

            valid_mask = (
                valid_mask.to(
                    device=(
                        recognition_queries
                        .device
                    ),
                    dtype=torch.bool,
                )
            )

            recognition_queries = (
                recognition_queries
                * valid_mask[
                    ...,
                    None,
                    None,
                ].to(
                    recognition_queries.dtype
                )
            )

        expected_shape = (
            batch_size,
            num_instances,
            self.num_rec_queries,
            self.embed_dims,
        )

        if (
            recognition_queries.shape
            != expected_shape
        ):
            raise RuntimeError(
                'Unexpected recognition '
                'query shape: '
                f'{tuple(recognition_queries.shape)}, '
                f'expected {expected_shape}.'
            )

        return recognition_queries