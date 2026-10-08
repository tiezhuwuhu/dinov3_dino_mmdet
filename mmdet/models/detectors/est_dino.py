from typing import Dict, Optional, Tuple

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import (
    OptSampleList,
    SampleList,
)
from mmdet.utils import OptConfigType

from ..layers.transformer.estext_spotter_decoder import (
    ESTextSpotterDecoder,
)
from ..layers.transformer.text_spotting_layers import (
    TaskAwareRecognitionQueryInitializer,
    TaskAwareTextQueryGroupBuilder,
)
from .dino import DINO


@MODELS.register_module()
class ESTDINO(DINO):
    """DINO with an ESTextSpotter-style task-aware decoder.

    Both matching queries and DINO denoising queries are represented as:

        [D, R0, R1, ..., R(T-1)]

    During training the query order is:

        [denoising instances, matching instances]

    which preserves the original MMDetection DINO query ordering.
    """

    def __init__(
        self,
        *args,
        recognition_query_cfg: OptConfigType = None,
        text_query_group_cfg: OptConfigType = None,
        text_decoder_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        stock_num_decoder_layers = (
            self.decoder.num_layers
        )

        # ------------------------------------------------------------
        # Task-Aware Query Initialization
        # ------------------------------------------------------------

        if recognition_query_cfg is None:
            recognition_query_cfg = {}

        recognition_query_cfg = dict(
            recognition_query_cfg
        )

        feat_channels = (
            recognition_query_cfg.pop(
                'feat_channels',
                self.embed_dims,
            )
        )

        recognition_embed_dims = (
            recognition_query_cfg.pop(
                'embed_dims',
                self.embed_dims,
            )
        )

        if feat_channels != self.embed_dims:
            raise ValueError(
                'TAQI feat_channels must match '
                'DINO embed_dims.'
            )

        if (
            recognition_embed_dims
            != self.embed_dims
        ):
            raise ValueError(
                'TAQI embed_dims must match '
                'DINO embed_dims.'
            )

        self.recognition_query_initializer = (
            TaskAwareRecognitionQueryInitializer(
                feat_channels=feat_channels,
                embed_dims=(
                    recognition_embed_dims
                ),
                **recognition_query_cfg,
            )
        )

        self.num_rec_queries = (
            self.recognition_query_initializer
            .num_rec_queries
        )

        # ------------------------------------------------------------
        # Query grouping
        # ------------------------------------------------------------

        if text_query_group_cfg is None:
            text_query_group_cfg = {}

        text_query_group_cfg = dict(
            text_query_group_cfg
        )

        group_embed_dims = (
            text_query_group_cfg.pop(
                'embed_dims',
                self.embed_dims,
            )
        )

        group_num_rec_queries = (
            text_query_group_cfg.pop(
                'num_rec_queries',
                self.num_rec_queries,
            )
        )

        if (
            group_embed_dims
            != self.embed_dims
        ):
            raise ValueError(
                'Text query group embed_dims '
                'must match DINO embed_dims.'
            )

        if (
            group_num_rec_queries
            != self.num_rec_queries
        ):
            raise ValueError(
                'Text query group '
                'num_rec_queries must match '
                'TAQI num_rec_queries.'
            )

        self.text_query_group_builder = (
            TaskAwareTextQueryGroupBuilder(
                embed_dims=group_embed_dims,
                num_rec_queries=(
                    group_num_rec_queries
                ),
                **text_query_group_cfg,
            )
        )

        # ------------------------------------------------------------
        # Task-aware decoder
        # ------------------------------------------------------------

        if text_decoder_cfg is None:
            raise ValueError(
                'text_decoder_cfg is required.'
            )

        text_decoder_cfg = dict(
            text_decoder_cfg
        )

        if (
            'num_rec_classes'
            not in text_decoder_cfg
        ):
            raise ValueError(
                'text_decoder_cfg must contain '
                'num_rec_classes.'
            )

        decoder_num_layers = (
            text_decoder_cfg.pop(
                'num_layers',
                stock_num_decoder_layers,
            )
        )

        decoder_embed_dims = (
            text_decoder_cfg.pop(
                'embed_dims',
                self.embed_dims,
            )
        )

        decoder_num_feature_levels = (
            text_decoder_cfg.pop(
                'num_feature_levels',
                self.num_feature_levels,
            )
        )

        if (
            decoder_num_layers
            != stock_num_decoder_layers
        ):
            raise ValueError(
                'Task-aware decoder num_layers '
                'must match the original DINO '
                'decoder configuration.'
            )

        if (
            decoder_embed_dims
            != self.embed_dims
        ):
            raise ValueError(
                'Task-aware decoder embed_dims '
                'must match DINO embed_dims.'
            )

        if (
            decoder_num_feature_levels
            != self.num_feature_levels
        ):
            raise ValueError(
                'Task-aware decoder '
                'num_feature_levels must match '
                'DINO num_feature_levels.'
            )

        self.decoder = ESTextSpotterDecoder(
            num_layers=decoder_num_layers,
            embed_dims=decoder_embed_dims,
            num_feature_levels=(
                decoder_num_feature_levels
            ),
            **text_decoder_cfg,
        )

        required_prediction_branches = (
            self.decoder.num_layers + 1
        )

        if (
            len(
                self.bbox_head.reg_branches
            )
            < required_prediction_branches
        ):
            raise RuntimeError(
                'Not enough bbox regression '
                'branches.'
            )

        if (
            len(
                self.bbox_head.cls_branches
            )
            < required_prediction_branches
        ):
            raise RuntimeError(
                'Not enough classification '
                'branches.'
            )

    def loss(
        self,
        batch_inputs: Tensor,
        batch_data_samples: SampleList,
    ) -> Dict[str, Tensor]:
    
        img_feats = self.extract_feat(
            batch_inputs
        )

        (
            head_inputs_dict,
            text_outputs_dict,
        ) = (
            self._forward_transformer_impl(
                img_feats=img_feats,
                batch_data_samples=(
                    batch_data_samples
                ),
            )
        )

        if (
            'recognition_logits'
            not in text_outputs_dict
        ):
            raise RuntimeError(
                'OCRDINO transformer did not '
                'return recognition_logits.'
            )

        recognition_logits = (
            text_outputs_dict[
                'recognition_logits'
            ]
        )

        losses = self.bbox_head.loss(
            **head_inputs_dict,
            recognition_logits=(
                recognition_logits
            ),
            batch_data_samples=(
                batch_data_samples
            ),
        )

        return losses

    @staticmethod
    def _get_image_shapes(
        batch_data_samples: OptSampleList,
    ) -> Tuple[Tuple[int, int], ...]:
        if batch_data_samples is None:
            raise ValueError(
                'batch_data_samples are '
                'required for TAQI.'
            )

        image_shapes = []

        for sample_index, sample in enumerate(
            batch_data_samples
        ):
            if (
                'img_shape'
                not in sample.metainfo
            ):
                raise KeyError(
                    'img_shape is missing from '
                    f'data sample '
                    f'{sample_index}.'
                )

            img_shape = sample.img_shape

            if len(img_shape) < 2:
                raise ValueError(
                    'img_shape must contain '
                    'height and width.'
                )

            image_shapes.append(
                (
                    int(img_shape[0]),
                    int(img_shape[1]),
                )
            )

        return tuple(
            image_shapes
        )

    def _validate_decoder_query_inputs(
        self,
        decoder_inputs_dict: Dict,
    ) -> Tuple[Tensor, Tensor]:
        """Validate complete decoder query tensors.

        During training these tensors contain:

            [DN queries, matching queries]

        During inference they contain matching queries only.
        """

        if (
            'query'
            not in decoder_inputs_dict
        ):
            raise KeyError(
                'decoder_inputs_dict does not '
                'contain query.'
            )

        if (
            'reference_points'
            not in decoder_inputs_dict
        ):
            raise KeyError(
                'decoder_inputs_dict does not '
                'contain reference_points.'
            )

        query = decoder_inputs_dict[
            'query'
        ]

        reference_points = (
            decoder_inputs_dict[
                'reference_points'
            ]
        )

        if query.ndim != 3:
            raise ValueError(
                'query must have shape '
                '[B, N, C].'
            )

        if reference_points.ndim != 3:
            raise ValueError(
                'reference_points must have '
                'shape [B, N, 4].'
            )

        if (
            query.shape[-1]
            != self.embed_dims
        ):
            raise ValueError(
                'Unexpected query embedding '
                'dimension.'
            )

        if (
            reference_points.shape[-1]
            != 4
        ):
            raise ValueError(
                'ESTDINO requires 4D '
                'cxcywh reference boxes.'
            )

        if (
            query.shape[:2]
            != reference_points.shape[:2]
        ):
            raise ValueError(
                'query and reference_points '
                'are not aligned.'
            )

        if (
            query.shape[1]
            < self.num_queries
        ):
            raise ValueError(
                'The total decoder query count '
                'cannot be smaller than the '
                'matching query count.'
            )

        return (
            query,
            reference_points,
        )

    def build_recognition_queries(
        self,
        img_feats: Tuple[Tensor, ...],
        decoder_inputs_dict: Dict,
        batch_data_samples: OptSampleList,
    ) -> Dict[str, Tensor]:
        """Build grouped text queries for all decoder instances.

        During training this includes both the denoising part and the
        matching part.

        During inference this includes the matching part only.
        """

        (
            detection_queries,
            reference_points,
        ) = (
            self._validate_decoder_query_inputs(
                decoder_inputs_dict
            )
        )

        image_shapes = (
            self._get_image_shapes(
                batch_data_samples
            )
        )

        recognition_queries = (
            self.recognition_query_initializer(
                feats=img_feats,
                proposal_boxes=(
                    reference_points
                ),
                image_shapes=image_shapes,
            )
        )


        (
            text_query_groups,
            text_positional_encoding,
        ) = (
            self.text_query_group_builder(
                detection_queries=(
                    detection_queries
                ),
                recognition_queries=(
                    recognition_queries
                ),
            )
        )

        batch_size = (
            detection_queries.shape[0]
        )

        num_total_queries = (
            detection_queries.shape[1]
        )

        expected_recognition_shape = (
            batch_size,
            num_total_queries,
            self.num_rec_queries,
            self.embed_dims,
        )

        if (
            recognition_queries.shape
            != expected_recognition_shape
        ):
            raise RuntimeError(
                'Unexpected recognition query '
                f'shape: '
                f'{tuple(recognition_queries.shape)}, '
                f'expected '
                f'{expected_recognition_shape}.'
            )

        expected_group_shape = (
            batch_size,
            num_total_queries,
            self.num_rec_queries + 1,
            self.embed_dims,
        )

        if (
            text_query_groups.shape
            != expected_group_shape
        ):
            raise RuntimeError(
                'Unexpected text query group '
                f'shape: '
                f'{tuple(text_query_groups.shape)}, '
                f'expected '
                f'{expected_group_shape}.'
            )

        return dict(
            detection_queries=(
                detection_queries
            ),
            recognition_queries=(
                recognition_queries
            ),
            text_query_groups=(
                text_query_groups
            ),
            text_positional_encoding=(
                text_positional_encoding
            ),
            recognition_reference_points=(
                reference_points
            ),
        )

    def forward_decoder(
        self,
        query: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        dn_mask: Optional[Tensor] = None,
        text_query_groups: Tensor = None,
        text_positional_encoding: Tensor = None,
        **kwargs,
    ) -> Dict:
        """Run the task-aware text decoder."""

        if query.ndim != 3:
            raise ValueError(
                'query must have shape '
                '[B, N, C].'
            )

        batch_size = (
            query.shape[0]
        )

        num_total_queries = (
            query.shape[1]
        )

        if (
            reference_points.shape
            != (
                batch_size,
                num_total_queries,
                4,
            )
        ):
            raise ValueError(
                'Unexpected reference_points '
                'shape.'
            )

        if text_query_groups is None:
            raise ValueError(
                'text_query_groups are required.'
            )

        if (
            text_positional_encoding
            is None
        ):
            raise ValueError(
                'text_positional_encoding '
                'is required.'
            )

        expected_group_shape = (
            batch_size,
            num_total_queries,
            self.num_rec_queries + 1,
            self.embed_dims,
        )

        if (
            text_query_groups.shape
            != expected_group_shape
        ):
            raise ValueError(
                'Unexpected text_query_groups '
                f'shape: '
                f'{tuple(text_query_groups.shape)}, '
                f'expected '
                f'{expected_group_shape}.'
            )

        if (
            text_positional_encoding.shape
            != expected_group_shape
        ):
            raise ValueError(
                'Unexpected text positional '
                'encoding shape.'
            )

        if dn_mask is not None:
            expected_mask_shape = (
                num_total_queries,
                num_total_queries,
            )

            if (
                dn_mask.shape
                != expected_mask_shape
            ):
                raise ValueError(
                    'Unexpected dn_mask shape: '
                    f'{tuple(dn_mask.shape)}, '
                    f'expected '
                    f'{expected_mask_shape}.'
                )

        outputs = self.decoder(
            query_groups=(
                text_query_groups
            ),
            text_positional_encoding=(
                text_positional_encoding
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
            reg_branches=(
                self.bbox_head.reg_branches
            ),
            instance_attn_mask=(
                dn_mask
            ),
        )

        # Keep the DINO label embedding in the graph when the current
        # batch contains no denoising queries. This avoids an unused
        # parameter in distributed training.
        if (
            self.training
            and num_total_queries
            == self.num_queries
        ):
            zero_dependency = (
                self.dn_query_generator
                .label_embedding
                .weight[0, 0]
                * 0.0
            )

            outputs[
                'hidden_states'
            ] = (
                outputs[
                    'hidden_states'
                ]
                + zero_dependency
            )

        return outputs

    def _forward_transformer_impl(
        self,
        img_feats: Tuple[Tensor, ...],
        batch_data_samples: OptSampleList,
    ) -> Tuple[Dict, Dict]:
        """Shared train/eval transformer implementation."""

        (
            encoder_inputs_dict,
            decoder_inputs_dict,
        ) = self.pre_transformer(
            img_feats,
            batch_data_samples,
        )

        encoder_outputs_dict = (
            self.forward_encoder(
                **encoder_inputs_dict
            )
        )

        (
            temporary_decoder_inputs,
            head_inputs_dict,
        ) = self.pre_decoder(
            **encoder_outputs_dict,
            batch_data_samples=(
                batch_data_samples
            ),
        )

        text_inputs_dict = (
            self.build_recognition_queries(
                img_feats=img_feats,
                decoder_inputs_dict=(
                    temporary_decoder_inputs
                ),
                batch_data_samples=(
                    batch_data_samples
                ),
            )
        )

        decoder_inputs_dict.update(
            temporary_decoder_inputs
        )

        decoder_inputs_dict.update(
            dict(
                text_query_groups=(
                    text_inputs_dict[
                        'text_query_groups'
                    ]
                ),
                text_positional_encoding=(
                    text_inputs_dict[
                        'text_positional_encoding'
                    ]
                ),
            )
        )

        decoder_outputs_dict = (
            self.forward_decoder(
                **decoder_inputs_dict
            )
        )

        head_inputs_dict.update(
            dict(
                hidden_states=(
                    decoder_outputs_dict[
                        'hidden_states'
                    ]
                ),
                references=(
                    decoder_outputs_dict[
                        'references'
                    ]
                ),
            )
        )

        text_outputs_dict = dict(
            detection_queries=(
                text_inputs_dict[
                    'detection_queries'
                ]
            ),
            recognition_queries=(
                text_inputs_dict[
                    'recognition_queries'
                ]
            ),
            text_hidden_states=(
                decoder_outputs_dict[
                    'text_hidden_states'
                ]
            ),
            recognition_logits=(
                decoder_outputs_dict[
                    'recognition_logits'
                ]
            ),
            references=(
                decoder_outputs_dict[
                    'references'
                ]
            ),
            dn_meta=(
                head_inputs_dict.get(
                    'dn_meta',
                    None,
                )
            ),
        )

        return (
            head_inputs_dict,
            text_outputs_dict,
        )

    def forward_transformer(
        self,
        img_feats: Tuple[Tensor, ...],
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        """Main MMDetection transformer path."""

        (
            head_inputs_dict,
            _,
        ) = self._forward_transformer_impl(
            img_feats=img_feats,
            batch_data_samples=(
                batch_data_samples
            ),
        )

        return head_inputs_dict

    def forward_transformer_with_text_outputs(
        self,
        img_feats: Tuple[Tensor, ...],
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict, Dict]:
        """Expose text outputs for structural testing."""

        return self._forward_transformer_impl(
            img_feats=img_feats,
            batch_data_samples=(
                batch_data_samples
            ),
        )