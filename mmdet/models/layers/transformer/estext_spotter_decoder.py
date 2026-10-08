import copy
from typing import Dict, Optional, Sequence

import torch
from torch import Tensor, nn

from .text_spotting_layers import (
    TaskAwareDeformableCrossAttention,
    TaskAwareIntraInterSelfAttention,
)
from .utils import MLP, coordinate_to_encoding, inverse_sigmoid


class RecognitionSemanticProjector(nn.Module):
    """Shared recognition classifier and semantic projection.

    The recognition classifier and semantic projection are shared by all
    decoder layers, matching ESTextSpotter's decoder-level rec_cls and
    rec_proj design.

    Args:
        embed_dims: Query embedding dimension.
        num_rec_classes: Number of recognition classes.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_rec_classes: int = 97,
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

        self.embed_dims = int(embed_dims)
        self.num_rec_classes = int(
            num_rec_classes
        )

        self.rec_classifier = nn.Linear(
            self.embed_dims,
            self.num_rec_classes,
        )

        self.rec_projection = nn.Linear(
            self.num_rec_classes,
            self.embed_dims,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
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

    def compute_logits(
        self,
        query_groups: Tensor,
    ) -> Tensor:
        """Classify ordered recognition slots.

        Args:
            query_groups:
                [B, N, T+1, C]

        Returns:
            Tensor:
                [B, N, T, num_rec_classes]
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
                'Unexpected embedding dimension.'
            )

        if query_groups.shape[2] <= 1:
            raise ValueError(
                'query_groups must contain '
                'one detection slot and at '
                'least one recognition slot.'
            )

        return self.rec_classifier(
            query_groups[:, :, 1:, :]
        )

    def build_semantic_features(
        self,
        query_groups: Tensor,
        recognition_logits: Tensor,
    ) -> Tensor:
        """Build semantic key/value features for VLC.

        Detection slot keeps its current visual feature.
        Recognition slots use projected character probabilities.
        """

        expected_shape = (
            query_groups.shape[0],
            query_groups.shape[1],
            query_groups.shape[2] - 1,
            self.num_rec_classes,
        )

        if (
            recognition_logits.shape
            != expected_shape
        ):
            raise ValueError(
                'Unexpected recognition logits '
                f'shape: '
                f'{tuple(recognition_logits.shape)}, '
                f'expected {expected_shape}.'
            )

        recognition_semantics = (
            self.rec_projection(
                recognition_logits.softmax(
                    dim=-1
                )
            )
        )

        return torch.cat(
            (
                query_groups[:, :, :1, :],
                recognition_semantics,
            ),
            dim=2,
        )


class ESTextSpotterDecoderLayer(nn.Module):
    """One task-aware ESTextSpotter decoder layer.

    The layer performs:

        VLC attention
        -> VLC FFN
        -> intra-instance self-attention
        -> inter-instance self-attention
        -> deformable cross-attention
        -> decoder FFN

    Recognition classification and semantic projection are intentionally
    outside this class so that they are shared across decoder layers.
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

        if num_heads <= 0:
            raise ValueError(
                'num_heads must be positive.'
            )

        if embed_dims % num_heads != 0:
            raise ValueError(
                'embed_dims must be divisible '
                'by num_heads.'
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

        self.embed_dims = int(embed_dims)
        self.ffn_dims = int(ffn_dims)
        self.dropout_prob = float(dropout)

        # ------------------------------------------------------------
        # Layer-local VLC communication.
        # ------------------------------------------------------------

        self.vlc_attention = nn.MultiheadAttention(
            embed_dim=self.embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.vlc_attention_dropout = nn.Dropout(
            dropout
        )

        self.vlc_attention_norm = nn.LayerNorm(
            self.embed_dims
        )

        self.vlc_ffn_linear1 = nn.Linear(
            self.embed_dims,
            self.ffn_dims,
        )

        self.vlc_activation = self._build_activation(
            activation
        )

        self.vlc_ffn_inner_dropout = nn.Dropout(
            dropout
        )

        self.vlc_ffn_linear2 = nn.Linear(
            self.ffn_dims,
            self.embed_dims,
        )

        self.vlc_ffn_output_dropout = nn.Dropout(
            dropout
        )

        self.vlc_ffn_norm = nn.LayerNorm(
            self.embed_dims
        )

        # ------------------------------------------------------------
        # Intra/inter attention.
        # ------------------------------------------------------------

        self.self_attention = (
            TaskAwareIntraInterSelfAttention(
                embed_dims=self.embed_dims,
                num_heads=num_heads,
                dropout=dropout,
            )
        )

        # ------------------------------------------------------------
        # Multi-scale deformable cross-attention.
        # ------------------------------------------------------------

        self.cross_attention = (
            TaskAwareDeformableCrossAttention(
                embed_dims=self.embed_dims,
                num_heads=num_heads,
                num_feature_levels=(
                    num_feature_levels
                ),
                num_points=num_points,
                dropout=dropout,
                im2col_step=im2col_step,
            )
        )

        # ------------------------------------------------------------
        # Main decoder FFN.
        # ------------------------------------------------------------

        self.ffn_linear1 = nn.Linear(
            self.embed_dims,
            self.ffn_dims,
        )

        self.activation = self._build_activation(
            activation
        )

        self.ffn_inner_dropout = nn.Dropout(
            dropout
        )

        self.ffn_linear2 = nn.Linear(
            self.ffn_dims,
            self.embed_dims,
        )

        self.ffn_output_dropout = nn.Dropout(
            dropout
        )

        self.ffn_norm = nn.LayerNorm(
            self.embed_dims
        )

        self._reset_ffn_parameters()

    @staticmethod
    def _build_activation(
        activation: str,
    ) -> nn.Module:
        if activation == 'relu':
            return nn.ReLU()

        if activation == 'gelu':
            return nn.GELU()

        raise ValueError(
            f'Unsupported activation: {activation}'
        )

    def _reset_ffn_parameters(
        self,
    ) -> None:
        for module in (
            self.vlc_ffn_linear1,
            self.vlc_ffn_linear2,
            self.ffn_linear1,
            self.ffn_linear2,
        ):
            nn.init.xavier_uniform_(
                module.weight
            )
            nn.init.constant_(
                module.bias,
                0.0,
            )

    def init_weights(
        self,
    ) -> None:
        self.cross_attention.init_weights()

    @staticmethod
    def build_vlc_mask(
        num_slots: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        """Prevent every slot from attending directly to itself."""

        if num_slots <= 1:
            raise ValueError(
                'VLC requires at least two slots.'
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

    def forward_vlc(
        self,
        query_groups: Tensor,
        semantic_features: Tensor,
        text_positional_encoding: Tensor,
    ) -> Tensor:
        """Run the layer-local VLC block."""

        if (
            semantic_features.shape
            != query_groups.shape
        ):
            raise ValueError(
                'semantic_features must have '
                'the same shape as query_groups.'
            )

        if (
            text_positional_encoding.shape
            != query_groups.shape
        ):
            raise ValueError(
                'text_positional_encoding must '
                'have the same shape as query_groups.'
            )

        batch_size = query_groups.shape[0]
        num_instances = query_groups.shape[1]
        num_slots = query_groups.shape[2]

        query_with_position = (
            query_groups
            + text_positional_encoding
        )

        key_with_position = (
            semantic_features
            + text_positional_encoding
        )

        query_flat = query_with_position.reshape(
            batch_size * num_instances,
            num_slots,
            self.embed_dims,
        )

        key_flat = key_with_position.reshape(
            batch_size * num_instances,
            num_slots,
            self.embed_dims,
        )

        value_flat = semantic_features.reshape(
            batch_size * num_instances,
            num_slots,
            self.embed_dims,
        )

        vlc_mask = self.build_vlc_mask(
            num_slots=num_slots,
            device=query_groups.device,
            dtype=query_groups.dtype,
        )

        attention_output, _ = (
            self.vlc_attention(
                query=query_flat,
                key=key_flat,
                value=value_flat,
                attn_mask=vlc_mask,
                need_weights=False,
            )
        )

        attention_output = (
            attention_output.reshape(
                batch_size,
                num_instances,
                num_slots,
                self.embed_dims,
            )
        )

        output = (
            query_groups
            + self.vlc_attention_dropout(
                attention_output
            )
        )

        output = self.vlc_attention_norm(
            output
        )

        ffn_output = self.vlc_ffn_linear2(
            self.vlc_ffn_inner_dropout(
                self.vlc_activation(
                    self.vlc_ffn_linear1(
                        output
                    )
                )
            )
        )

        output = (
            output
            + self.vlc_ffn_output_dropout(
                ffn_output
            )
        )

        output = self.vlc_ffn_norm(
            output
        )

        return output

    def forward_ffn(
        self,
        query_groups: Tensor,
    ) -> Tensor:
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

        return self.ffn_norm(
            output
        )

    def forward(
        self,
        query_groups: Tensor,
        semantic_features: Tensor,
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
        output = self.forward_vlc(
            query_groups=query_groups,
            semantic_features=(
                semantic_features
            ),
            text_positional_encoding=(
                text_positional_encoding
            ),
        )

        output = self.self_attention(
            query_groups=output,
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


class ESTextSpotterDecoder(nn.Module):
    """Multi-layer ESTextSpotter task-aware decoder.

    Shared across layers:

        recognition classifier
        recognition semantic projection
        reference-point positional head

    Independent for every layer:

        VLC attention and VLC FFN
        intra/inter self-attention
        deformable cross-attention
        decoder FFN
    """

    def __init__(
        self,
        num_layers: int = 6,
        embed_dims: int = 256,
        num_rec_classes: int = 97,
        num_heads: int = 8,
        num_feature_levels: int = 4,
        num_points: int = 4,
        ffn_dims: int = 2048,
        dropout: float = 0.0,
        activation: str = 'relu',
        im2col_step: int = 64,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(
                'num_layers must be positive.'
            )

        self.num_layers = int(num_layers)
        self.embed_dims = int(embed_dims)
        self.num_rec_classes = int(
            num_rec_classes
        )
        self.num_feature_levels = int(
            num_feature_levels
        )

        base_layer = ESTextSpotterDecoderLayer(
            embed_dims=self.embed_dims,
            num_heads=num_heads,
            num_feature_levels=(
                self.num_feature_levels
            ),
            num_points=num_points,
            ffn_dims=ffn_dims,
            dropout=dropout,
            activation=activation,
            im2col_step=im2col_step,
        )

        self.layers = nn.ModuleList(
            [
                copy.deepcopy(base_layer)
                for _ in range(
                    self.num_layers
                )
            ]
        )

        self.norm = nn.LayerNorm(
            self.embed_dims
        )

        self.ref_point_head = MLP(
            self.embed_dims * 2,
            self.embed_dims,
            self.embed_dims,
            2,
        )

        self.recognition_semantics = (
            RecognitionSemanticProjector(
                embed_dims=self.embed_dims,
                num_rec_classes=(
                    self.num_rec_classes
                ),
            )
        )

    def init_weights(
        self,
    ) -> None:
        for layer in self.layers:
            layer.init_weights()

    def _build_instance_position(
        self,
        reference_points: Tensor,
        valid_ratios: Tensor,
    ) -> Tensor:
        """Build DINO-style spatial position for every instance."""

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

        if (
            valid_ratios.ndim != 3
            or valid_ratios.shape[-1] != 2
        ):
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

        ratio_scale = torch.cat(
            (
                valid_ratios,
                valid_ratios,
            ),
            dim=-1,
        )

        reference_points_input = (
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

        query_sine_embed = (
            coordinate_to_encoding(
                reference_points_input[
                    :,
                    :,
                    0,
                    :,
                ]
            )
        )

        return self.ref_point_head(
            query_sine_embed
        )

    def _validate_reg_branches(
        self,
        reg_branches: Sequence[nn.Module],
    ) -> None:
        if reg_branches is None:
            raise ValueError(
                'reg_branches are required '
                'for iterative bbox refinement.'
            )

        if (
            len(reg_branches)
            < self.num_layers
        ):
            raise ValueError(
                'Not enough bbox regression '
                'branches: '
                f'got {len(reg_branches)}, '
                f'expected at least '
                f'{self.num_layers}.'
            )

    def forward(
        self,
        query_groups: Tensor,
        text_positional_encoding: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor],
        reference_points: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
        reg_branches: Sequence[nn.Module],
        instance_attn_mask: Optional[Tensor] = None,
    ) -> Dict:
        """Run the complete task-aware decoder.

        Args:
            query_groups:
                [B, N, T+1, C]

            text_positional_encoding:
                [B, N, T+1, C]

            memory:
                [B, M, C]

            memory_mask:
                [B, M] or None

            reference_points:
                Initial normalized cxcywh boxes [B, N, 4].

            spatial_shapes:
                [L, 2]

            level_start_index:
                [L]

            valid_ratios:
                [B, L, 2]

            reg_branches:
                DINO bbox regression branches.

            instance_attn_mask:
                Optional [N, N] inter-instance mask.

        Returns:
            dict containing:

                hidden_states:
                    [num_layers, B, N, C]

                text_hidden_states:
                    [num_layers, B, N, T+1, C]

                recognition_logits:
                    [num_layers, B, N, T, V]

                references:
                    list of num_layers + 1 tensors,
                    each [B, N, 4]
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
                'dimension.'
            )

        if (
            reference_points.shape
            != (
                query_groups.shape[0],
                query_groups.shape[1],
                4,
            )
        ):
            raise ValueError(
                'reference_points and '
                'query_groups are not aligned.'
            )

        self._validate_reg_branches(
            reg_branches
        )

        output = query_groups

        current_reference_points = (
            reference_points
        )

        reference_history = [
            current_reference_points
        ]

        intermediate_text_states = []
        intermediate_detection_states = []
        intermediate_recognition_logits = []

        # Initial recognition prediction is used only to build
        # the semantic features for decoder layer 0.
        recognition_logits = (
            self.recognition_semantics
            .compute_logits(output)
        )

        for layer_id, layer in enumerate(
            self.layers
        ):
            semantic_features = (
                self.recognition_semantics
                .build_semantic_features(
                    query_groups=output,
                    recognition_logits=(
                        recognition_logits
                    ),
                )
            )

            instance_position = (
                self._build_instance_position(
                    reference_points=(
                        current_reference_points
                    ),
                    valid_ratios=(
                        valid_ratios
                    ),
                )
            )

            output = layer(
                query_groups=output,
                semantic_features=(
                    semantic_features
                ),
                text_positional_encoding=(
                    text_positional_encoding
                ),
                instance_positional_encoding=(
                    instance_position
                ),
                memory=memory,
                memory_mask=memory_mask,
                reference_points=(
                    current_reference_points
                ),
                spatial_shapes=(
                    spatial_shapes
                ),
                level_start_index=(
                    level_start_index
                ),
                valid_ratios=(
                    valid_ratios
                ),
                instance_attn_mask=(
                    instance_attn_mask
                ),
            )

            # --------------------------------------------------------
            # DINO/ESTextSpotter iterative bbox refinement.
            # Only the detection slot predicts the box.
            # --------------------------------------------------------

            bbox_delta = reg_branches[
                layer_id
            ](
                output[:, :, 0, :]
            )

            if (
                bbox_delta.shape
                != current_reference_points.shape
            ):
                raise RuntimeError(
                    'BBox regression branch '
                    'returned unexpected shape: '
                    f'{tuple(bbox_delta.shape)}.'
                )

            new_reference_points = (
                bbox_delta
                + inverse_sigmoid(
                    current_reference_points,
                    eps=1e-3,
                )
            ).sigmoid()

            # --------------------------------------------------------
            # Recognition after this decoder layer.
            # Shared classifier across all layers.
            # --------------------------------------------------------

            recognition_logits = (
                self.recognition_semantics
                .compute_logits(output)
            )

            intermediate_recognition_logits.append(
                recognition_logits
            )

            normalized_output = self.norm(
                output
            )

            intermediate_text_states.append(
                normalized_output
            )

            intermediate_detection_states.append(
                normalized_output[
                    :,
                    :,
                    0,
                    :,
                ]
            )

            # Keep the non-detached value for auxiliary bbox losses.
            reference_history.append(
                new_reference_points
            )

            # The next decoder layer receives detached reference boxes.
            current_reference_points = (
                new_reference_points.detach()
            )

        return dict(
            hidden_states=torch.stack(
                intermediate_detection_states,
                dim=0,
            ),
            text_hidden_states=torch.stack(
                intermediate_text_states,
                dim=0,
            ),
            recognition_logits=torch.stack(
                intermediate_recognition_logits,
                dim=0,
            ),
            references=reference_history,
        )