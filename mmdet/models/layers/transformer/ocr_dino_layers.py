# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmengine.model import BaseModule, ModuleList
from torch import Tensor, nn

from .deformable_detr_layers import (
    DeformableDetrTransformerDecoderLayer,
)


class BlockVisualExtractor(BaseModule):
    """Extract fixed-length visual features for each DINO block query.

    Each DINO object query corresponds to one document block.

    The block query is expanded into multiple visual queries. These
    visual queries use multi-scale deformable attention to read features
    from DINO encoder memory.

    Args:
        embed_dims (int): Feature dimension.
        num_visual_queries (int): Number of visual slots per block.
        num_layers (int): Number of deformable decoder layers.
        num_heads (int): Number of attention heads.
        num_feature_levels (int): Number of feature levels.
        num_points (int): Number of deformable sampling points.
        ffn_channels (int): FFN hidden dimension.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        num_visual_queries: int = 64,
        num_layers: int = 2,
        num_heads: int = 8,
        num_feature_levels: int = 4,
        num_points: int = 4,
        ffn_channels: int = 1024,
        dropout: float = 0.0,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.embed_dims = embed_dims
        self.num_visual_queries = num_visual_queries
        self.num_layers = num_layers

        self.visual_pos_embed = nn.Embedding(
            num_visual_queries,
            embed_dims,
        )

        layer_cfg = dict(
            self_attn_cfg=dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            ),
            cross_attn_cfg=dict(
                embed_dims=embed_dims,
                num_levels=num_feature_levels,
                num_points=num_points,
                dropout=dropout,
                batch_first=True,
            ),
            ffn_cfg=dict(
                embed_dims=embed_dims,
                feedforward_channels=ffn_channels,
                num_fcs=2,
                ffn_drop=dropout,
                act_cfg=dict(
                    type='ReLU',
                    inplace=True,
                ),
            ),
        )

        self.layers = ModuleList([
            DeformableDetrTransformerDecoderLayer(**layer_cfg)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dims)

    def _build_instance_attn_mask(
        self,
        num_instances: int,
        device,
    ) -> Tensor:
        """Mask attention between different document blocks."""

        length = self.num_visual_queries
        total_queries = num_instances * length

        mask = torch.ones(
            (total_queries, total_queries),
            dtype=torch.bool,
            device=device,
        )

        for i in range(num_instances):
            start = i * length
            end = start + length
            mask[start:end, start:end] = False

        return mask

    def forward(
        self,
        object_queries: Tensor,
        object_boxes: Tensor,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        valid_ratios: Tensor,
    ) -> Tensor:
        """Forward block visual extractor.

        Args:
            object_queries: Shape [B, K, C].
            object_boxes: Shape [B, K, 4].
            memory: Shape [B, S, C].
            memory_mask: Shape [B, S] or None.
            spatial_shapes: Shape [num_levels, 2].
            level_start_index: Shape [num_levels].
            valid_ratios: Shape [B, num_levels, 2].

        Returns:
            Tensor: Shape [B, K, V, C].
        """

        batch_size, num_instances, embed_dims = (
            object_queries.shape
        )

        if embed_dims != self.embed_dims:
            raise ValueError(
                f'Expected embed_dims={self.embed_dims}, '
                f'but got {embed_dims}.'
            )

        if object_boxes.shape[:2] != (
            batch_size,
            num_instances,
        ):
            raise ValueError(
                'object_boxes must match object_queries '
                'in batch and instance dimensions.'
            )

        visual_query = object_queries.unsqueeze(2).expand(
            -1,
            -1,
            self.num_visual_queries,
            -1,
        )

        visual_query = visual_query.reshape(
            batch_size,
            num_instances * self.num_visual_queries,
            self.embed_dims,
        )

        visual_pos = self.visual_pos_embed.weight

        visual_pos = visual_pos.unsqueeze(0).unsqueeze(0).expand(
            batch_size,
            num_instances,
            -1,
            -1,
        )

        visual_pos = visual_pos.reshape(
            batch_size,
            num_instances * self.num_visual_queries,
            self.embed_dims,
        )

        visual_reference_boxes = object_boxes.unsqueeze(2).expand(
            -1,
            -1,
            self.num_visual_queries,
            -1,
        )

        visual_reference_boxes = visual_reference_boxes.reshape(
            batch_size,
            num_instances * self.num_visual_queries,
            4,
        )

        visual_reference_boxes = visual_reference_boxes.clamp(
            min=0.0,
            max=1.0,
        )

        self_attn_mask = self._build_instance_attn_mask(
            num_instances=num_instances,
            device=object_queries.device,
        )

        for layer in self.layers:

            reference_points_input = (
                visual_reference_boxes[:, :, None, :]
                *
                torch.cat(
                    [valid_ratios, valid_ratios],
                    dim=-1,
                )[:, None, :, :]
            )

            visual_query = layer(
                query=visual_query,
                query_pos=visual_pos,
                value=memory,
                key_padding_mask=memory_mask,
                self_attn_mask=self_attn_mask,
                reference_points=reference_points_input,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                valid_ratios=valid_ratios,
            )

        visual_query = self.norm(visual_query)

        block_visual_features = visual_query.reshape(
            batch_size,
            num_instances,
            self.num_visual_queries,
            self.embed_dims,
        )

        return block_visual_features


class BlockTextDecoder(BaseModule):
    """Autoregressive text decoder for OCR-DINO.

    Training uses teacher forcing.

    Example:
        Input:
            BOS H E L L O

        Target:
            H E L L O EOS

    Args:
        vocab_size (int): Vocabulary size.
        embed_dims (int): Hidden dimension.
        max_seq_len (int): Maximum text sequence length.
        num_layers (int): Number of Transformer decoder layers.
        num_heads (int): Number of attention heads.
        ffn_channels (int): FFN hidden dimension.
        dropout (float): Dropout probability.
        pad_token_id (int): PAD token index.
        bos_token_id (int): BOS token index.
        eos_token_id (int): EOS token index.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dims: int = 256,
        max_seq_len: int = 1024,
        num_layers: int = 2,
        num_heads: int = 8,
        ffn_channels: int = 1024,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.vocab_size = vocab_size
        self.embed_dims = embed_dims
        self.max_seq_len = max_seq_len

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

        self.token_embedding = nn.Embedding(
            vocab_size,
            embed_dims,
            padding_idx=pad_token_id,
        )

        self.position_embedding = nn.Embedding(
            max_seq_len,
            embed_dims,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_channels,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dims),
        )

        self.token_classifier = nn.Linear(
            embed_dims,
            vocab_size,
        )

    def _build_causal_mask(
        self,
        seq_len: int,
        device,
    ) -> Tensor:
        """Build autoregressive causal attention mask."""

        return torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=device,
            ),
            diagonal=1,
        )

    def forward(
        self,
        block_visual_features: Tensor,
        input_ids: Tensor,
        block_queries: Tensor = None,
    ) -> Tensor:
        """Teacher-forcing forward.

        Args:
            block_visual_features:
                Shape [B, K, V, C].

            input_ids:
                Shape [B, K, T].

            block_queries:
                Optional shape [B, K, C].

        Returns:
            Tensor:
                Logits with shape [B, K, T, vocab_size].
        """

        if block_visual_features.dim() != 4:
            raise ValueError(
                'block_visual_features must have shape '
                '[B, K, V, C].'
            )

        if input_ids.dim() != 3:
            raise ValueError(
                'input_ids must have shape [B, K, T].'
            )

        batch_size, num_blocks, num_visual, embed_dims = (
            block_visual_features.shape
        )

        input_batch, input_blocks, seq_len = input_ids.shape

        if input_batch != batch_size:
            raise ValueError(
                'Batch size mismatch between visual features '
                'and input_ids.'
            )

        if input_blocks != num_blocks:
            raise ValueError(
                'Number of blocks mismatch between visual features '
                'and input_ids.'
            )

        if embed_dims != self.embed_dims:
            raise ValueError(
                f'Expected embed_dims={self.embed_dims}, '
                f'but got {embed_dims}.'
            )

        if seq_len > self.max_seq_len:
            raise ValueError(
                f'Sequence length {seq_len} exceeds '
                f'max_seq_len={self.max_seq_len}.'
            )

        memory = block_visual_features.reshape(
            batch_size * num_blocks,
            num_visual,
            embed_dims,
        )

        flat_input_ids = input_ids.reshape(
            batch_size * num_blocks,
            seq_len,
        )

        token_features = self.token_embedding(
            flat_input_ids
        )

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        )

        position_features = self.position_embedding(
            positions
        )

        token_features = (
            token_features
            + position_features.unsqueeze(0)
        )

        if block_queries is not None:

            expected_shape = (
                batch_size,
                num_blocks,
                embed_dims,
            )

            if block_queries.shape != expected_shape:
                raise ValueError(
                    'block_queries must have shape '
                    f'{expected_shape}.'
                )

            flat_block_queries = block_queries.reshape(
                batch_size * num_blocks,
                embed_dims,
            )

            token_features = (
                token_features
                + flat_block_queries.unsqueeze(1)
            )

        causal_mask = self._build_causal_mask(
            seq_len=seq_len,
            device=input_ids.device,
        )

        padding_mask = (
            flat_input_ids == self.pad_token_id
        )

        text_features = self.decoder(
            tgt=token_features,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=padding_mask,
        )

        logits = self.token_classifier(
            text_features
        )

        logits = logits.reshape(
            batch_size,
            num_blocks,
            seq_len,
            self.vocab_size,
        )

        return logits

    @torch.no_grad()
    def greedy_decode(
        self,
        block_visual_features: Tensor,
        block_queries: Tensor = None,
        max_length: int = None,
    ) -> Tensor:
        """Greedy autoregressive decoding."""

        batch_size, num_blocks, _, _ = (
            block_visual_features.shape
        )

        if max_length is None:
            max_length = self.max_seq_len

        max_length = min(
            max_length,
            self.max_seq_len,
        )

        generated = torch.full(
            (
                batch_size,
                num_blocks,
                1,
            ),
            fill_value=self.bos_token_id,
            dtype=torch.long,
            device=block_visual_features.device,
        )

        finished = torch.zeros(
            (
                batch_size,
                num_blocks,
            ),
            dtype=torch.bool,
            device=block_visual_features.device,
        )

        generated_tokens = []

        for _ in range(max_length):

            logits = self.forward(
                block_visual_features=block_visual_features,
                input_ids=generated,
                block_queries=block_queries,
            )

            next_token_logits = logits[:, :, -1, :]

            next_tokens = next_token_logits.argmax(
                dim=-1
            )

            next_tokens = torch.where(
                finished,
                torch.full_like(
                    next_tokens,
                    self.eos_token_id,
                ),
                next_tokens,
            )

            generated_tokens.append(
                next_tokens.unsqueeze(-1)
            )

            finished = (
                finished
                | (
                    next_tokens
                    == self.eos_token_id
                )
            )

            generated = torch.cat(
                [
                    generated,
                    next_tokens.unsqueeze(-1),
                ],
                dim=-1,
            )

            if finished.all():
                break

        if not generated_tokens:
            return torch.empty(
                (
                    batch_size,
                    num_blocks,
                    0,
                ),
                dtype=torch.long,
                device=block_visual_features.device,
            )

        return torch.cat(
            generated_tokens,
            dim=-1,
        )