import math
from typing import Optional, Sequence, Tuple

import torch
from mmcv.ops import RoIAlign, MultiScaleDeformableAttention
from torch import Tensor, nn

from mmdet.registry import MODELS
from mmdet.structures.bbox import (
    bbox2roi,
    bbox_cxcywh_to_xyxy,
)


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
        
class OrderedTextQueryPositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for ordered text-query slots.

    The position sequence covers the complete query group:

        slot 0:
            detection query

        slots 1..T:
            ordered recognition queries

    This follows the role of ESTextSpotter's PositionalEncoding1D while
    keeping the implementation batch-first and independent from sequence
    length or vocabulary size.

    Args:
        embed_dims: Query embedding dimension.
        temperature: Frequency temperature.
        normalize: Whether to normalize slot positions.
        scale: Position scale after normalization. ESTextSpotter uses 1.0.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        temperature: float = 10000.0,
        normalize: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if embed_dims % 2 != 0:
            raise ValueError(
                'embed_dims must be even for '
                'sinusoidal positional encoding.'
            )

        if temperature <= 0:
            raise ValueError(
                'temperature must be positive.'
            )

        if (
            scale is not None
            and not normalize
        ):
            raise ValueError(
                'normalize must be True when '
                'scale is specified.'
            )

        if scale is None:
            scale = 2.0 * math.pi

        self.embed_dims = int(
            embed_dims
        )

        self.temperature = float(
            temperature
        )

        self.normalize = bool(
            normalize
        )

        self.scale = float(
            scale
        )

        frequency_indices = torch.arange(
            0,
            self.embed_dims,
            2,
            dtype=torch.float32,
        )

        inv_freq = 1.0 / (
            self.temperature
            ** (
                frequency_indices
                / self.embed_dims
            )
        )

        self.register_buffer(
            'inv_freq',
            inv_freq,
            persistent=True,
        )

    def forward(
        self,
        query_groups: Tensor,
    ) -> Tensor:
        """Build positional encodings.

        Args:
            query_groups:
                Query groups with shape
                [B, N, S, C], where S = 1 + T.

        Returns:
            Tensor:
                Positional encodings with shape
                [B, N, S, C].
        """

        if query_groups.ndim != 4:
            raise ValueError(
                'query_groups must have shape '
                '[B, N, S, C].'
            )

        batch_size = (
            query_groups.shape[0]
        )

        num_instances = (
            query_groups.shape[1]
        )

        num_slots = (
            query_groups.shape[2]
        )

        channels = (
            query_groups.shape[3]
        )

        if channels != self.embed_dims:
            raise ValueError(
                'Unexpected query embedding '
                f'dimension: got {channels}, '
                f'expected {self.embed_dims}.'
            )

        if num_slots <= 0:
            raise ValueError(
                'Query groups must contain '
                'at least one slot.'
            )

        positions = torch.arange(
            1,
            num_slots + 1,
            device=query_groups.device,
            dtype=self.inv_freq.dtype,
        )

        if self.normalize:
            eps = 1e-6

            positions = (
                positions
                / (
                    positions[-1]
                    + eps
                )
                * self.scale
            )

        sinusoid = torch.einsum(
            'i,j->ij',
            positions,
            self.inv_freq,
        )

        positional_encoding = torch.cat(
            (
                sinusoid.sin(),
                sinusoid.cos(),
            ),
            dim=-1,
        )

        positional_encoding = (
            positional_encoding[
                :,
                :self.embed_dims,
            ]
            .to(
                dtype=query_groups.dtype
            )
        )

        positional_encoding = (
            positional_encoding
            .view(
                1,
                1,
                num_slots,
                self.embed_dims,
            )
            .expand(
                batch_size,
                num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        return positional_encoding


class TaskAwareTextQueryGroupBuilder(nn.Module):
    """Build ESTextSpotter-style detection-recognition query groups.

    Each text instance is represented as:

        [D, R0, R1, ..., R(T-1)]

    where D is one detection query and the recognition queries are ordered.

    The module is independent from character vocabulary size.

    Args:
        embed_dims: Query embedding dimension.
        num_rec_queries: Number of recognition slots per text instance.
        position_temperature: Temperature of sinusoidal position encoding.
        position_normalize: Whether slot positions are normalized.
        position_scale: Position scale. ESTextSpotter uses 1.0.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_rec_queries: int = 25,
        position_temperature: float = 10000.0,
        position_normalize: bool = True,
        position_scale: float = 1.0,
    ) -> None:
        super().__init__()

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if num_rec_queries <= 0:
            raise ValueError(
                'num_rec_queries must be positive.'
            )

        self.embed_dims = int(
            embed_dims
        )

        self.num_rec_queries = int(
            num_rec_queries
        )

        self.num_group_slots = (
            self.num_rec_queries + 1
        )

        self.position_encoding = (
            OrderedTextQueryPositionalEncoding(
                embed_dims=self.embed_dims,
                temperature=(
                    position_temperature
                ),
                normalize=(
                    position_normalize
                ),
                scale=position_scale,
            )
        )

    def forward(
        self,
        detection_queries: Tensor,
        recognition_queries: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Build grouped text queries.

        Args:
            detection_queries:
                Detection queries with shape
                [B, N, C].

            recognition_queries:
                Ordered recognition queries with shape
                [B, N, T, C].

        Returns:
            tuple:
                - query_groups:
                    [B, N, T+1, C]
                - text_positional_encoding:
                    [B, N, T+1, C]
        """

        if detection_queries.ndim != 3:
            raise ValueError(
                'detection_queries must have '
                'shape [B, N, C].'
            )

        if recognition_queries.ndim != 4:
            raise ValueError(
                'recognition_queries must have '
                'shape [B, N, T, C].'
            )

        batch_size = (
            detection_queries.shape[0]
        )

        num_instances = (
            detection_queries.shape[1]
        )

        det_channels = (
            detection_queries.shape[2]
        )

        if (
            recognition_queries.shape[0]
            != batch_size
        ):
            raise ValueError(
                'Detection and recognition '
                'batch sizes do not match.'
            )

        if (
            recognition_queries.shape[1]
            != num_instances
        ):
            raise ValueError(
                'Detection and recognition '
                'instance counts do not match.'
            )

        if (
            recognition_queries.shape[2]
            != self.num_rec_queries
        ):
            raise ValueError(
                'Unexpected number of '
                'recognition queries: '
                f'got '
                f'{recognition_queries.shape[2]}, '
                f'expected '
                f'{self.num_rec_queries}.'
            )

        if (
            det_channels
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected detection query '
                f'dimension: got {det_channels}, '
                f'expected {self.embed_dims}.'
            )

        if (
            recognition_queries.shape[3]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected recognition query '
                'dimension: got '
                f'{recognition_queries.shape[3]}, '
                f'expected {self.embed_dims}.'
            )

        query_groups = torch.cat(
            (
                detection_queries.unsqueeze(
                    dim=2
                ),
                recognition_queries,
            ),
            dim=2,
        )

        expected_shape = (
            batch_size,
            num_instances,
            self.num_group_slots,
            self.embed_dims,
        )

        if (
            query_groups.shape
            != expected_shape
        ):
            raise RuntimeError(
                'Unexpected grouped query '
                f'shape: '
                f'{tuple(query_groups.shape)}, '
                f'expected {expected_shape}.'
            )

        text_positional_encoding = (
            self.position_encoding(
                query_groups
            )
        )

        if (
            text_positional_encoding.shape
            != expected_shape
        ):
            raise RuntimeError(
                'Unexpected text positional '
                'encoding shape.'
            )

        return (
            query_groups,
            text_positional_encoding,
        )
        
class VisionLanguageCommunication(nn.Module):
    """Vision-Language Communication used by ESTextSpotter.

    Each text instance contains one detection slot followed by ordered
    recognition slots:

        [D, R0, R1, ..., R(T-1)]

    Visual query features are converted into recognition class
    distributions. The recognition distributions are projected back into
    the embedding space and used as semantic key/value features.

    The detection slot keeps its visual feature.

    Args:
        embed_dims: Query embedding dimension.
        num_rec_classes: Number of recognition classes.
        num_heads: Number of attention heads.
        ffn_dims: Hidden dimension of the VLC feed-forward network.
        dropout: Dropout probability.
        activation: Feed-forward activation.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_rec_classes: int = 97,
        num_heads: int = 8,
        ffn_dims: int = 2048,
        dropout: float = 0.0,
        activation: str = 'relu',
    ) -> None:
        super().__init__()

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if num_rec_classes <= 1:
            raise ValueError(
                'num_rec_classes must be greater than 1.'
            )

        if num_heads <= 0:
            raise ValueError(
                'num_heads must be positive.'
            )

        if embed_dims % num_heads != 0:
            raise ValueError(
                'embed_dims must be divisible by num_heads.'
            )

        if ffn_dims <= 0:
            raise ValueError(
                'ffn_dims must be positive.'
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                'dropout must be in [0, 1).'
            )

        if activation not in {
            'relu',
            'gelu',
        }:
            raise ValueError(
                'activation must be "relu" or "gelu".'
            )

        self.embed_dims = int(
            embed_dims
        )

        self.num_rec_classes = int(
            num_rec_classes
        )

        self.num_heads = int(
            num_heads
        )

        self.ffn_dims = int(
            ffn_dims
        )

        self.dropout_prob = float(
            dropout
        )

        self.rec_classifier = nn.Linear(
            self.embed_dims,
            self.num_rec_classes,
        )

        self.rec_projection = nn.Linear(
            self.num_rec_classes,
            self.embed_dims,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )

        self.attention_dropout = nn.Dropout(
            self.dropout_prob
        )

        self.attention_norm = nn.LayerNorm(
            self.embed_dims
        )

        self.ffn_linear1 = nn.Linear(
            self.embed_dims,
            self.ffn_dims,
        )

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.ffn_inner_dropout = nn.Dropout(
            self.dropout_prob
        )

        self.ffn_linear2 = nn.Linear(
            self.ffn_dims,
            self.embed_dims,
        )

        self.ffn_output_dropout = nn.Dropout(
            self.dropout_prob
        )

        self.ffn_norm = nn.LayerNorm(
            self.embed_dims
        )

        self._reset_parameters()

    def _reset_parameters(
        self,
    ) -> None:
        """Initialize trainable projections."""

        nn.init.xavier_uniform_(
            self.rec_classifier.weight
        )

        nn.init.constant_(
            self.rec_classifier.bias,
            0.0,
        )

        nn.init.xavier_uniform_(
            self.rec_projection.weight
        )

        nn.init.constant_(
            self.rec_projection.bias,
            0.0,
        )

        nn.init.xavier_uniform_(
            self.ffn_linear1.weight
        )

        nn.init.constant_(
            self.ffn_linear1.bias,
            0.0,
        )

        nn.init.xavier_uniform_(
            self.ffn_linear2.weight
        )

        nn.init.constant_(
            self.ffn_linear2.bias,
            0.0,
        )

    @staticmethod
    def build_location_mask(
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Build the ESTextSpotter diagonal communication mask.

        A slot is prevented from attending to itself, while communication
        with every other slot in the same text instance remains available.
        """

        if num_slots <= 1:
            raise ValueError(
                'VLC requires at least two query slots.'
            )

        mask = torch.zeros(
            (
                num_slots,
                num_slots,
            ),
            device=device,
            dtype=dtype,
        )

        diagonal = torch.arange(
            num_slots,
            device=device,
        )

        mask[
            diagonal,
            diagonal,
        ] = float('-inf')

        return mask

    def compute_recognition_logits(
        self,
        query_groups: Tensor,
    ) -> Tensor:
        """Classify recognition slots.

        Args:
            query_groups:
                Tensor with shape [B, N, T+1, C].

        Returns:
            Tensor:
                Recognition logits with shape
                [B, N, T, num_rec_classes].
        """

        if query_groups.ndim != 4:
            raise ValueError(
                'query_groups must have '
                'shape [B, N, T+1, C].'
            )

        if (
            query_groups.shape[-1]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected embedding dimension: '
                f'{query_groups.shape[-1]}, '
                f'expected {self.embed_dims}.'
            )

        if query_groups.shape[2] <= 1:
            raise ValueError(
                'query_groups must contain one '
                'detection slot and at least one '
                'recognition slot.'
            )

        recognition_features = (
            query_groups[
                :,
                :,
                1:,
                :,
            ]
        )

        recognition_logits = (
            self.rec_classifier(
                recognition_features
            )
        )

        return recognition_logits

    def build_semantic_features(
        self,
        query_groups: Tensor,
        recognition_logits: Tensor,
    ) -> Tensor:
        """Build VLC key/value features.

        Detection keeps its visual feature.

        Recognition slots are replaced by projected recognition
        probabilities.
        """

        expected_logit_shape = (
            query_groups.shape[0],
            query_groups.shape[1],
            query_groups.shape[2] - 1,
            self.num_rec_classes,
        )

        if (
            recognition_logits.shape
            != expected_logit_shape
        ):
            raise ValueError(
                'Unexpected recognition logit '
                f'shape: '
                f'{tuple(recognition_logits.shape)}, '
                f'expected '
                f'{expected_logit_shape}.'
            )

        recognition_probabilities = (
            recognition_logits.softmax(
                dim=-1
            )
        )

        recognition_semantics = (
            self.rec_projection(
                recognition_probabilities
            )
        )

        detection_feature = (
            query_groups[
                :,
                :,
                :1,
                :,
            ]
        )

        semantic_features = torch.cat(
            (
                detection_feature,
                recognition_semantics,
            ),
            dim=2,
        )

        return semantic_features

    def forward(
        self,
        query_groups: Tensor,
        text_positional_encoding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Run one VLC block.

        Args:
            query_groups:
                Current visual query features with shape
                [B, N, T+1, C].

            text_positional_encoding:
                Ordered slot positional encoding with shape
                [B, N, T+1, C].

        Returns:
            tuple:
                - communicated query groups:
                    [B, N, T+1, C]
                - recognition logits used by VLC:
                    [B, N, T, num_rec_classes]
        """

        if query_groups.ndim != 4:
            raise ValueError(
                'query_groups must have '
                'shape [B, N, T+1, C].'
            )

        if (
            text_positional_encoding.shape
            != query_groups.shape
        ):
            raise ValueError(
                'text_positional_encoding must '
                'have the same shape as '
                'query_groups.'
            )

        if (
            query_groups.shape[-1]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected query embedding '
                f'dimension: '
                f'{query_groups.shape[-1]}, '
                f'expected {self.embed_dims}.'
            )

        batch_size = query_groups.shape[0]
        num_instances = query_groups.shape[1]
        num_slots = query_groups.shape[2]

        recognition_logits = (
            self.compute_recognition_logits(
                query_groups
            )
        )

        semantic_features = (
            self.build_semantic_features(
                query_groups=(
                    query_groups
                ),
                recognition_logits=(
                    recognition_logits
                ),
            )
        )

        query_with_position = (
            query_groups
            + text_positional_encoding
        )

        key_with_position = (
            semantic_features
            + text_positional_encoding
        )

        query_flat = (
            query_with_position.reshape(
                batch_size
                * num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        key_flat = (
            key_with_position.reshape(
                batch_size
                * num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        value_flat = (
            semantic_features.reshape(
                batch_size
                * num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        attention_mask = (
            self.build_location_mask(
                num_slots=num_slots,
                device=query_groups.device,
                dtype=query_groups.dtype,
            )
        )

        communicated_flat, _ = (
            self.attention(
                query=query_flat,
                key=key_flat,
                value=value_flat,
                attn_mask=attention_mask,
                need_weights=False,
            )
        )

        communicated = (
            communicated_flat.reshape(
                batch_size,
                num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        output = (
            query_groups
            + self.attention_dropout(
                communicated
            )
        )

        output = self.attention_norm(
            output
        )

        ffn_output = (
            self.ffn_linear2(
                self.ffn_inner_dropout(
                    self.activation(
                        self.ffn_linear1(
                            output
                        )
                    )
                )
            )
        )

        output = (
            output
            + self.ffn_output_dropout(
                ffn_output
            )
        )

        output = self.ffn_norm(
            output
        )

        return (
            output,
            recognition_logits,
        )
        
class TaskAwareIntraInterSelfAttention(nn.Module):
    """ESTextSpotter-style intra/inter query self-attention.

    Query groups use the layout:

        [D, R0, R1, ..., R(T-1)]

    Intra-instance attention operates over the slot dimension inside each
    text instance.

    Inter-instance attention operates over the instance dimension for every
    slot independently.

    This module is independent from recognition vocabulary size and sequence
    length.

    Args:
        embed_dims: Query embedding dimension.
        num_heads: Number of attention heads.
        dropout: Attention and residual dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if num_heads <= 0:
            raise ValueError(
                'num_heads must be positive.'
            )

        if embed_dims % num_heads != 0:
            raise ValueError(
                'embed_dims must be divisible '
                'by num_heads.'
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                'dropout must be in [0, 1).'
            )

        self.embed_dims = int(
            embed_dims
        )

        self.num_heads = int(
            num_heads
        )

        self.dropout_prob = float(
            dropout
        )

        self.attn_intra = nn.MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )

        self.dropout_intra = nn.Dropout(
            self.dropout_prob
        )

        self.norm_intra = nn.LayerNorm(
            self.embed_dims
        )

        self.attn_inter = nn.MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=self.num_heads,
            dropout=self.dropout_prob,
            batch_first=True,
        )

        self.dropout_inter = nn.Dropout(
            self.dropout_prob
        )

        self.norm_inter = nn.LayerNorm(
            self.embed_dims
        )

    def _validate_inputs(
        self,
        query_groups: Tensor,
        text_positional_encoding: Tensor,
        instance_positional_encoding: Tensor,
    ) -> None:
        if query_groups.ndim != 4:
            raise ValueError(
                'query_groups must have '
                'shape [B, N, S, C].'
            )

        if (
            text_positional_encoding.shape
            != query_groups.shape
        ):
            raise ValueError(
                'text_positional_encoding must '
                'have the same shape as '
                'query_groups.'
            )

        expected_instance_shape = (
            query_groups.shape[0],
            query_groups.shape[1],
            self.embed_dims,
        )

        if (
            instance_positional_encoding.shape
            != expected_instance_shape
        ):
            raise ValueError(
                'instance_positional_encoding '
                'must have shape [B, N, C]. '
                f'Got '
                f'{tuple(instance_positional_encoding.shape)}, '
                f'expected '
                f'{expected_instance_shape}.'
            )

        if (
            query_groups.shape[-1]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected query embedding '
                f'dimension: '
                f'{query_groups.shape[-1]}, '
                f'expected {self.embed_dims}.'
            )

    @staticmethod
    def _validate_instance_mask(
        instance_attn_mask: Optional[Tensor],
        num_instances: int,
    ) -> None:
        if instance_attn_mask is None:
            return

        if instance_attn_mask.ndim != 2:
            raise ValueError(
                'instance_attn_mask must have '
                'shape [N, N].'
            )

        if (
            instance_attn_mask.shape
            != (
                num_instances,
                num_instances,
            )
        ):
            raise ValueError(
                'Unexpected instance_attn_mask '
                f'shape: '
                f'{tuple(instance_attn_mask.shape)}, '
                f'expected '
                f'({num_instances}, '
                f'{num_instances}).'
            )

    def forward_intra(
        self,
        query_groups: Tensor,
        text_positional_encoding: Tensor,
    ) -> Tensor:
        """Apply self-attention inside every text instance.

        Args:
            query_groups:
                [B, N, S, C]

            text_positional_encoding:
                [B, N, S, C]

        Returns:
            Tensor:
                [B, N, S, C]
        """

        batch_size = (
            query_groups.shape[0]
        )

        num_instances = (
            query_groups.shape[1]
        )

        num_slots = (
            query_groups.shape[2]
        )

        query_with_position = (
            query_groups
            + text_positional_encoding
        )

        query_flat = (
            query_with_position.reshape(
                batch_size
                * num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        value_flat = (
            query_groups.reshape(
                batch_size
                * num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        intra_output, _ = (
            self.attn_intra(
                query=query_flat,
                key=query_flat,
                value=value_flat,
                need_weights=False,
            )
        )

        intra_output = (
            intra_output.reshape(
                batch_size,
                num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        output = (
            query_groups
            + self.dropout_intra(
                intra_output
            )
        )

        output = self.norm_intra(
            output
        )

        return output

    def forward_inter(
        self,
        query_groups: Tensor,
        instance_positional_encoding: Tensor,
        instance_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Apply self-attention across text instances.

        For every slot independently, all text instances communicate:

            D0 <-> D1 <-> D2 ...

            R0,0 <-> R1,0 <-> R2,0 ...

            R0,1 <-> R1,1 <-> R2,1 ...

        Args:
            query_groups:
                [B, N, S, C]

            instance_positional_encoding:
                Spatial/query positional encoding for every text instance,
                with shape [B, N, C].

            instance_attn_mask:
                Optional attention mask with shape [N, N]. This is reserved
                for DINO denoising isolation and other instance-level masks.

        Returns:
            Tensor:
                [B, N, S, C]
        """

        batch_size = (
            query_groups.shape[0]
        )

        num_instances = (
            query_groups.shape[1]
        )

        num_slots = (
            query_groups.shape[2]
        )

        self._validate_instance_mask(
            instance_attn_mask=(
                instance_attn_mask
            ),
            num_instances=(
                num_instances
            ),
        )

        inter_groups = (
            query_groups.transpose(
                1,
                2,
            )
            .contiguous()
        )

        instance_position = (
            instance_positional_encoding
            .unsqueeze(1)
        )

        query_with_position = (
            inter_groups
            + instance_position
        )

        query_flat = (
            query_with_position.reshape(
                batch_size
                * num_slots,
                num_instances,
                self.embed_dims,
            )
        )

        value_flat = (
            inter_groups.reshape(
                batch_size
                * num_slots,
                num_instances,
                self.embed_dims,
            )
        )

        if instance_attn_mask is not None:
            instance_attn_mask = (
                instance_attn_mask.to(
                    device=query_groups.device
                )
            )

        inter_output, _ = (
            self.attn_inter(
                query=query_flat,
                key=query_flat,
                value=value_flat,
                attn_mask=(
                    instance_attn_mask
                ),
                need_weights=False,
            )
        )

        inter_output = (
            inter_output.reshape(
                batch_size,
                num_slots,
                num_instances,
                self.embed_dims,
            )
        )

        inter_groups = (
            inter_groups
            + self.dropout_inter(
                inter_output
            )
        )

        inter_groups = (
            self.norm_inter(
                inter_groups
            )
        )

        output = (
            inter_groups.transpose(
                1,
                2,
            )
            .contiguous()
        )

        return output

    def forward(
        self,
        query_groups: Tensor,
        text_positional_encoding: Tensor,
        instance_positional_encoding: Tensor,
        instance_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run intra-instance then inter-instance self-attention."""

        self._validate_inputs(
            query_groups=(
                query_groups
            ),
            text_positional_encoding=(
                text_positional_encoding
            ),
            instance_positional_encoding=(
                instance_positional_encoding
            ),
        )

        output = self.forward_intra(
            query_groups=(
                query_groups
            ),
            text_positional_encoding=(
                text_positional_encoding
            ),
        )

        output = self.forward_inter(
            query_groups=output,
            instance_positional_encoding=(
                instance_positional_encoding
            ),
            instance_attn_mask=(
                instance_attn_mask
            ),
        )

        return output
        
class TaskAwareDeformableCrossAttention(nn.Module):
    """Multi-scale deformable cross-attention for grouped text queries.

    Every text instance is represented by:

        [D, R0, R1, ..., R(T-1)]

    All slots belonging to the same instance share the same spatial
    reference box, while each slot keeps its own query content.

    Args:
        embed_dims: Query embedding dimension.
        num_heads: Number of deformable-attention heads.
        num_feature_levels: Number of encoder feature levels.
        num_points: Number of sampling points per head and level.
        dropout: Residual dropout probability inside deformable attention.
        im2col_step: CUDA im2col step used by deformable attention.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_feature_levels: int = 4,
        num_points: int = 4,
        dropout: float = 0.0,
        im2col_step: int = 64,
    ) -> None:
        super().__init__()

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if num_heads <= 0:
            raise ValueError(
                'num_heads must be positive.'
            )

        if embed_dims % num_heads != 0:
            raise ValueError(
                'embed_dims must be divisible by num_heads.'
            )

        if num_feature_levels <= 0:
            raise ValueError(
                'num_feature_levels must be positive.'
            )

        if num_points <= 0:
            raise ValueError(
                'num_points must be positive.'
            )

        if im2col_step <= 0:
            raise ValueError(
                'im2col_step must be positive.'
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                'dropout must be in [0, 1).'
            )

        self.embed_dims = int(
            embed_dims
        )

        self.num_heads = int(
            num_heads
        )

        self.num_feature_levels = int(
            num_feature_levels
        )

        self.num_points = int(
            num_points
        )

        self.cross_attn = (
            MultiScaleDeformableAttention(
                embed_dims=self.embed_dims,
                num_heads=self.num_heads,
                num_levels=(
                    self.num_feature_levels
                ),
                num_points=self.num_points,
                im2col_step=im2col_step,
                dropout=dropout,
                batch_first=True,
            )
        )

        self.norm = nn.LayerNorm(
            self.embed_dims
        )

    def init_weights(
        self,
    ) -> None:
        """Initialize deformable-attention parameters."""

        self.cross_attn.init_weights()

    def _validate_memory_inputs(
        self,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        batch_size: int,
    ) -> None:
        if memory.ndim != 3:
            raise ValueError(
                'memory must have shape '
                '[B, M, C].'
            )

        if (
            memory.shape[0]
            != batch_size
        ):
            raise ValueError(
                'Memory batch size does not '
                'match query batch size.'
            )

        if (
            memory.shape[2]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected memory embedding '
                f'dimension: {memory.shape[2]}, '
                f'expected {self.embed_dims}.'
            )

        if spatial_shapes.ndim != 2:
            raise ValueError(
                'spatial_shapes must have '
                'shape [L, 2].'
            )

        if (
            spatial_shapes.shape
            != (
                self.num_feature_levels,
                2,
            )
        ):
            raise ValueError(
                'Unexpected spatial_shapes '
                f'shape: '
                f'{tuple(spatial_shapes.shape)}, '
                f'expected '
                f'({self.num_feature_levels}, 2).'
            )

        if (
            level_start_index.ndim != 1
        ):
            raise ValueError(
                'level_start_index must have '
                'shape [L].'
            )

        if (
            level_start_index.shape[0]
            != self.num_feature_levels
        ):
            raise ValueError(
                'Unexpected number of '
                'level_start_index entries.'
            )

        num_memory_values = int(
            spatial_shapes.prod(
                dim=1
            ).sum().item()
        )

        if (
            num_memory_values
            != memory.shape[1]
        ):
            raise ValueError(
                'spatial_shapes do not match '
                'memory length: '
                f'{num_memory_values} vs '
                f'{memory.shape[1]}.'
            )

        if memory_mask is not None:
            if (
                memory_mask.shape
                != memory.shape[:2]
            ):
                raise ValueError(
                    'memory_mask must have '
                    'shape [B, M].'
                )

    def build_slot_reference_points(
        self,
        reference_points: Tensor,
        valid_ratios: Tensor,
        num_slots: int,
    ) -> Tensor:
        """Expand instance reference boxes to every query slot.

        Args:
            reference_points:
                Normalized instance boxes with shape [B, N, 4].
                Coordinates are (cx, cy, w, h).

            valid_ratios:
                Valid width/height ratios with shape [B, L, 2].

            num_slots:
                Number of slots in every instance group, normally T + 1.

        Returns:
            Tensor:
                Reference boxes with shape
                [B, N * num_slots, L, 4].
        """

        if reference_points.ndim != 3:
            raise ValueError(
                'reference_points must have '
                'shape [B, N, 4].'
            )

        if (
            reference_points.shape[-1]
            != 4
        ):
            raise ValueError(
                'Only 4D cxcywh reference '
                'boxes are supported.'
            )

        if valid_ratios.ndim != 3:
            raise ValueError(
                'valid_ratios must have '
                'shape [B, L, 2].'
            )

        if (
            valid_ratios.shape[0]
            != reference_points.shape[0]
        ):
            raise ValueError(
                'reference_points and '
                'valid_ratios batch sizes '
                'do not match.'
            )

        if (
            valid_ratios.shape[1]
            != self.num_feature_levels
        ):
            raise ValueError(
                'valid_ratios feature-level '
                'count does not match '
                'num_feature_levels.'
            )

        if (
            valid_ratios.shape[2]
            != 2
        ):
            raise ValueError(
                'The last dimension of '
                'valid_ratios must be 2.'
            )

        if num_slots <= 0:
            raise ValueError(
                'num_slots must be positive.'
            )

        batch_size = (
            reference_points.shape[0]
        )

        num_instances = (
            reference_points.shape[1]
        )

        ratio_scale = torch.cat(
            (
                valid_ratios,
                valid_ratios,
            ),
            dim=-1,
        )

        instance_reference_points = (
            reference_points[
                :,
                :,
                None,
                :,
            ]
            * ratio_scale[
                :,
                None,
                :,
                :,
            ]
        )

        slot_reference_points = (
            instance_reference_points[
                :,
                :,
                None,
                :,
                :,
            ]
            .expand(
                batch_size,
                num_instances,
                num_slots,
                self.num_feature_levels,
                4,
            )
            .reshape(
                batch_size,
                num_instances * num_slots,
                self.num_feature_levels,
                4,
            )
            .contiguous()
        )

        return slot_reference_points

    def build_slot_query_position(
        self,
        instance_positional_encoding: Tensor,
        num_slots: int,
    ) -> Tensor:
        """Repeat each instance spatial position over all query slots."""

        if (
            instance_positional_encoding.ndim
            != 3
        ):
            raise ValueError(
                'instance_positional_encoding '
                'must have shape [B, N, C].'
            )

        if (
            instance_positional_encoding.shape[-1]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected instance positional '
                'embedding dimension.'
            )

        if num_slots <= 0:
            raise ValueError(
                'num_slots must be positive.'
            )

        batch_size = (
            instance_positional_encoding.shape[0]
        )

        num_instances = (
            instance_positional_encoding.shape[1]
        )

        slot_query_position = (
            instance_positional_encoding[
                :,
                :,
                None,
                :,
            ]
            .expand(
                batch_size,
                num_instances,
                num_slots,
                self.embed_dims,
            )
            .reshape(
                batch_size,
                num_instances * num_slots,
                self.embed_dims,
            )
            .contiguous()
        )

        return slot_query_position

    def forward(
        self,
        query_groups: Tensor,
        instance_positional_encoding: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
    ) -> Tensor:
        """Run grouped deformable cross-attention.

        Args:
            query_groups:
                [B, N, S, C]

            instance_positional_encoding:
                [B, N, C]

            memory:
                Encoder memory with shape [B, M, C].

            memory_mask:
                Optional encoder padding mask with shape [B, M].

            reference_points:
                Normalized instance boxes with shape [B, N, 4].

            spatial_shapes:
                Feature shapes with shape [L, 2].

            level_start_index:
                Feature-level start indices with shape [L].

            valid_ratios:
                Valid feature ratios with shape [B, L, 2].

        Returns:
            Tensor:
                Updated grouped queries with shape [B, N, S, C].
        """

        if query_groups.ndim != 4:
            raise ValueError(
                'query_groups must have '
                'shape [B, N, S, C].'
            )

        batch_size = (
            query_groups.shape[0]
        )

        num_instances = (
            query_groups.shape[1]
        )

        num_slots = (
            query_groups.shape[2]
        )

        if (
            query_groups.shape[3]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected query embedding '
                f'dimension: '
                f'{query_groups.shape[3]}, '
                f'expected {self.embed_dims}.'
            )

        if (
            instance_positional_encoding.shape
            != (
                batch_size,
                num_instances,
                self.embed_dims,
            )
        ):
            raise ValueError(
                'instance_positional_encoding '
                'must have shape [B, N, C].'
            )

        if (
            reference_points.shape
            != (
                batch_size,
                num_instances,
                4,
            )
        ):
            raise ValueError(
                'reference_points must have '
                'shape [B, N, 4].'
            )

        self._validate_memory_inputs(
            memory=memory,
            memory_mask=memory_mask,
            spatial_shapes=spatial_shapes,
            level_start_index=(
                level_start_index
            ),
            batch_size=batch_size,
        )

        query_flat = (
            query_groups.reshape(
                batch_size,
                num_instances * num_slots,
                self.embed_dims,
            )
        )

        slot_query_position = (
            self.build_slot_query_position(
                instance_positional_encoding=(
                    instance_positional_encoding
                ),
                num_slots=num_slots,
            )
        )

        slot_reference_points = (
            self.build_slot_reference_points(
                reference_points=(
                    reference_points
                ),
                valid_ratios=(
                    valid_ratios
                ),
                num_slots=num_slots,
            )
        )

        output_flat = self.cross_attn(
            query=query_flat,
            value=memory,
            identity=query_flat,
            query_pos=(
                slot_query_position
            ),
            key_padding_mask=(
                memory_mask
            ),
            reference_points=(
                slot_reference_points
            ),
            spatial_shapes=(
                spatial_shapes
            ),
            level_start_index=(
                level_start_index
            ),
        )

        output_flat = self.norm(
            output_flat
        )

        output = output_flat.reshape(
            batch_size,
            num_instances,
            num_slots,
            self.embed_dims,
        )

        return output

class TaskAwareTextDecoderLayer(nn.Module):
    """One ESTextSpotter-style task-aware decoder layer.

    The layer itself performs:

        intra-instance self-attention
        -> inter-instance self-attention
        -> multi-scale deformable cross-attention
        -> feed-forward network

    Vision-Language Communication is intentionally kept outside this layer,
    matching the original ESTextSpotter decoder structure.

    Args:
        embed_dims: Query embedding dimension.
        num_heads: Number of attention heads.
        num_feature_levels: Number of encoder feature levels.
        num_points: Number of deformable sampling points.
        ffn_dims: Hidden dimension of the feed-forward network.
        dropout: Dropout probability.
        activation: Feed-forward activation.
        im2col_step: CUDA im2col step for deformable attention.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_heads: int = 8,
        num_feature_levels: int = 4,
        num_points: int = 4,
        ffn_dims: int = 2048,
        dropout: float = 0.0,
        activation: str = 'relu',
        im2col_step: int = 64,
    ) -> None:
        super().__init__()

        if embed_dims <= 0:
            raise ValueError(
                'embed_dims must be positive.'
            )

        if ffn_dims <= 0:
            raise ValueError(
                'ffn_dims must be positive.'
            )

        if activation not in {
            'relu',
            'gelu',
        }:
            raise ValueError(
                'activation must be "relu" or "gelu".'
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                'dropout must be in [0, 1).'
            )

        self.embed_dims = int(
            embed_dims
        )

        self.ffn_dims = int(
            ffn_dims
        )

        self.dropout_prob = float(
            dropout
        )

        self.self_attention = (
            TaskAwareIntraInterSelfAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        self.cross_attention = (
            TaskAwareDeformableCrossAttention(
                embed_dims=embed_dims,
                num_heads=num_heads,
                num_feature_levels=(
                    num_feature_levels
                ),
                num_points=num_points,
                dropout=dropout,
                im2col_step=im2col_step,
            )
        )

        self.ffn_linear1 = nn.Linear(
            embed_dims,
            ffn_dims,
        )

        if activation == 'relu':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.GELU()

        self.ffn_inner_dropout = nn.Dropout(
            dropout
        )

        self.ffn_linear2 = nn.Linear(
            ffn_dims,
            embed_dims,
        )

        self.ffn_output_dropout = nn.Dropout(
            dropout
        )

        self.ffn_norm = nn.LayerNorm(
            embed_dims
        )

        self._reset_parameters()

    def _reset_parameters(
        self,
    ) -> None:
        """Initialize FFN parameters."""

        nn.init.xavier_uniform_(
            self.ffn_linear1.weight
        )

        nn.init.constant_(
            self.ffn_linear1.bias,
            0.0,
        )

        nn.init.xavier_uniform_(
            self.ffn_linear2.weight
        )

        nn.init.constant_(
            self.ffn_linear2.bias,
            0.0,
        )

    def init_weights(
        self,
    ) -> None:
        """Initialize deformable-attention parameters."""

        self.cross_attention.init_weights()

    def forward_ffn(
        self,
        query_groups: Tensor,
    ) -> Tensor:
        """Apply the decoder-layer FFN."""

        ffn_output = self.ffn_linear2(
            self.ffn_inner_dropout(
                self.activation(
                    self.ffn_linear1(
                        query_groups
                    )
                )
            )
        )

        output = (
            query_groups
            + self.ffn_output_dropout(
                ffn_output
            )
        )

        output = self.ffn_norm(
            output
        )

        return output

    def forward(
        self,
        query_groups: Tensor,
        text_positional_encoding: Tensor,
        instance_positional_encoding: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        instance_attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Run one complete task-aware decoder layer.

        Args:
            query_groups:
                [B, N, S, C]

            text_positional_encoding:
                Ordered slot positions with shape [B, N, S, C].

            instance_positional_encoding:
                Spatial instance positions with shape [B, N, C].

            memory:
                Encoder memory with shape [B, M, C].

            memory_mask:
                Encoder padding mask with shape [B, M] or None.

            reference_points:
                Normalized cxcywh boxes with shape [B, N, 4].

            spatial_shapes:
                Encoder feature shapes with shape [L, 2].

            level_start_index:
                Encoder level offsets with shape [L].

            valid_ratios:
                Valid width/height ratios with shape [B, L, 2].

            instance_attn_mask:
                Optional mask for inter-instance attention.

        Returns:
            Tensor:
                Updated query groups with shape [B, N, S, C].
        """

        output = self.self_attention(
            query_groups=query_groups,
            text_positional_encoding=(
                text_positional_encoding
            ),
            instance_positional_encoding=(
                instance_positional_encoding
            ),
            instance_attn_mask=(
                instance_attn_mask
            ),
        )

        output = self.cross_attention(
            query_groups=output,
            instance_positional_encoding=(
                instance_positional_encoding
            ),
            memory=memory,
            memory_mask=memory_mask,
            reference_points=(
                reference_points
            ),
            spatial_shapes=(
                spatial_shapes
            ),
            level_start_index=(
                level_start_index
            ),
            valid_ratios=valid_ratios,
        )

        output = self.forward_ffn(
            output
        )

        return output